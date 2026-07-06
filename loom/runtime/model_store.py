"""Store MACHINE-OWNED des modèles distants ajoutés via l'UI (var/remote_models.json).

Séparé de config/local.toml (écrit à la main, avec commentaires) : l'UI possède ce fichier
JSON et le réécrit en entier à chaque changement, sans clobbering du TOML. Fusionné au
démarrage avec cfg.remote_models (les entrées gérées par l'UI l'emportent par id). Objectif :
ajouter/configurer un modèle distant sans jamais ouvrir un TOML à la main (zéro friction).
"""

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
