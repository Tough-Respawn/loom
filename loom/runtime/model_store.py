"""Persistance des modèles distants : config/local.toml, source UNIQUE.

Historique : les distants ajoutés via l'UI vivaient dans un store JSON séparé
(var/remote_models.json) pour ne pas clobber le TOML écrit à la main. Depuis que
upsert_remote_in_toml/delete_remote_in_toml éditent local.toml via tomlkit (commentaires
et structure PRÉSERVÉS), cette séparation n'a plus de raison d'être : deux emplacements =
des modèles introuvables. Unification 2026-07-21 : TOUT distant (wizard /add-model,
panneau engrenage, édition) vit dans [[remote_models]] de config/local.toml ; un store
JSON résiduel est replié dedans au chargement (migrate_to_toml) puis supprimé.
Les fonctions JSON (load/save/upsert/delete) ne servent plus qu'à cette migration."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Champs persistés d'un modèle distant géré par l'UI. `api_key` est stocké en clair dans
# var/ (état machine, gitignored) comme dans local.toml — jamais renvoyé tel quel au client.
KEEP = (
    "id",
    "base_url",
    "model",
    "api_key",
    "context",
    "max_tokens",
    "vision",
    "enable_thinking_param",
    "price_in",
    "price_out",
    "price_cached",
)


def load(path: str | Path) -> list[dict]:
    """Liste des modèles distants gérés (vide si absent/illisible). Filtre les entrées sans
    les 3 champs indispensables (id/base_url/model) pour ne jamais monter une route bancale."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [
        {k: m[k] for k in KEEP if k in m}
        for m in data
        if isinstance(m, dict) and m.get("id") and m.get("base_url") and m.get("model")
    ]


def save(path: str | Path, models: list[dict]) -> None:
    """Écrit le store en OWNER-ONLY (0600) : ce fichier porte des clés API en clair. Sur
    POSIX (Loom vise aussi Linux/Mac) un 0644 par défaut exposerait les secrets aux autres
    utilisateurs ; on crée donc le fichier en 0600 dès l'ouverture (pas de fenêtre umask) et
    on rechmod un fichier préexistant. Sur Windows le mode POSIX est ~ignoré (sans effet, sans
    erreur). Écriture atomique (temp + replace) pour ne jamais laisser un JSON tronqué."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(list(models), ensure_ascii=False, indent=2).encode("utf-8")
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    os.replace(
        tmp, p
    )  # atomique : le lecteur voit l'ancien ou le nouveau, jamais un mix
    try:
        os.chmod(p, 0o600)  # verrouille aussi un fichier préexistant (best-effort)
    except OSError:
        pass


def upsert(path: str | Path, model: dict) -> list[dict]:
    """Ajoute (ou remplace par id) un modèle et persiste. Renvoie la liste à jour."""
    mid = model.get("id")
    kept = {k: model[k] for k in KEEP if k in model}
    models = [m for m in load(path) if m.get("id") != mid]
    models.append(kept)
    save(path, models)
    return models


def delete(path: str | Path, model_id: str) -> list[dict]:
    """Retire un modèle par id et persiste. Renvoie la liste à jour."""
    models = [m for m in load(path) if m.get("id") != model_id]
    save(path, models)
    return models


def migrate_to_toml(
    store_path: str | Path, local_path: str | Path | None
) -> list[dict]:
    """Replie un store JSON hérité dans config/local.toml, puis le supprime.

    Renvoie les entrées migrées (dicts KEEP) pour que load_config leur garde la
    priorité par id sur ce chargement, sans relire le TOML. Sans local_path (tests,
    config chargée sans surcharge machine) : lecture seule, fichier laissé en place.
    Un JSON illisible n'est JAMAIS supprimé (load() le lit vide) — on ne détruit pas
    ce qu'on n'a pas migré."""
    records = load(store_path)
    if not records or local_path is None:
        return records
    for rec in records:
        upsert_remote_in_toml(local_path, rec)
    try:
        Path(store_path).unlink()
    except OSError:
        pass  # best-effort : au pire il sera re-migré (idempotent, upsert par id)
    return records


def upsert_remote_in_toml(local_path: str | Path, record: dict) -> None:
    """Écrit/maj un [[remote_models]] dans local.toml par id, via tomlkit (commentaires et
    structure du fichier PRÉSERVÉS). Sert à éditer depuis l'UI un modèle distant DÉFINI EN
    CONFIG (pas dans le store JSON) : url, modèle, contexte, clé... restent dans local.toml,
    là où l'utilisateur les avait mis. Seuls les champs présents dans `record` sont posés."""
    import tomlkit

    p = Path(local_path)
    doc = (
        tomlkit.parse(p.read_text(encoding="utf-8"))
        if p.exists()
        else tomlkit.document()
    )
    arr = doc.get("remote_models")
    if arr is None:
        arr = tomlkit.aot()
        doc["remote_models"] = arr
    # Mêmes champs qu'en store JSON (KEEP) : single source of truth. Les 4 champs
    # enable_thinking_param / price_in / price_out / price_cached sont des champs de
    # local.toml par conception (cf. config.RemoteModelConfig) et doivent donc pouvoir être
    # persistés ici - un écart ferait qu'un modèle édité via l'UI les perdrait silencieusement.
    # NB : api_key_env (TOML-only, géré à la main) reste volontairement exclu de KEEP/fields.
    fields = KEEP
    target = None
    for t in arr:
        if t.get("id") == record.get("id"):
            target = t
            break
    if target is None:
        target = tomlkit.table()
        for k in fields:
            if record.get(k) is not None:
                target[k] = record[k]
        arr.append(target)
    else:
        for k in fields:
            if record.get(k) is not None:
                target[k] = record[k]
    p.write_text(tomlkit.dumps(doc), encoding="utf-8")


def delete_remote_in_toml(local_path: str | Path, model_id: str) -> bool:
    """Retire un [[remote_models]] de local.toml par id (tomlkit : commentaires et
    structure PRÉSERVÉS). Pendant de upsert_remote_in_toml pour /remove-model sur un
    distant déclaré en config. Renvoie False si fichier ou entrée absents (no-op)."""
    import tomlkit

    p = Path(local_path)
    if not p.exists():
        return False
    doc = tomlkit.parse(p.read_text(encoding="utf-8"))
    arr = doc.get("remote_models")
    if arr is None:
        return False
    idx = next((i for i, t in enumerate(arr) if t.get("id") == model_id), None)
    if idx is None:
        return False
    del arr[idx]
    p.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return True
