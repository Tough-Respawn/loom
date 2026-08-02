"""Persistance des modèles distants : UN DOSSIER PAR MODÈLE, comme les locaux.

Un distant vit dans `<racine>/remote/<id>/model.toml` (+ profile.md optionnel), découvert
multi-racines par load_config exactement comme `local/text/<id>/` — première racine
gagnante par id, l'id = le nom du dossier. Le wizard /add-model et le panneau engrenage
écrivent ici (racine prioritaire) ; supprimer un distant = supprimer son dossier.

Historique (deux migrations automatiques, idempotentes, conservées en filet) :
- store JSON var/remote_models.json (pré-2026-07-21) ;
- [[remote_models]] de config/local.toml (unification du 21/07, remplacée le 26/07 :
  les modèles n'ont rien à faire au milieu des réglages machine).
Les deux sont repliés en dossiers au chargement (migrate_into_dirs) puis vidés."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# Les clés restent sur une racine machine-owned et ne sont jamais renvoyées au client.
FIELDS = (
    "base_url",
    "model",
    "api_key",
    "context",
    "max_tokens",
    "vision",
    "strong",
    "enable_thinking_param",
    "price_in",
    "price_out",
    "price_cached",
    "description",
)
# Accepter les champs hérités pendant les migrations.
KEEP = ("id", "api_key_env", *FIELDS)


def remote_dir(root: str | Path, model_id: str) -> Path:
    """Dossier d'un distant sous une racine : <racine>/remote/<id>."""
    return Path(root) / "remote" / model_id


def discover_remote(models_roots: list[Path] | list[str]) -> list[dict]:
    """Découvre un distant par sous-dossier `<racine>/remote/<id>/model.toml` sur CHAQUE
    racine, première racine gagnante par id (même règle que les locaux). L'id = nom du
    dossier (un `id` dans le toml est ignoré). Un toml illisible ou sans base_url/model
    est sauté : on ne monte jamais une route bancale. Tri par id."""
    import tomllib

    out: list[dict] = []
    seen: set[str] = set()
    for root in models_roots:
        base = Path(root) / "remote"
        if not base.is_dir():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            if folder.name.startswith("_") or folder.name in seen:
                continue
            toml_path = folder / "model.toml"
            if not toml_path.exists():
                continue
            try:
                d = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not (d.get("base_url") and d.get("model")):
                continue
            rec = {k: d[k] for k in KEEP if k in d}
            rec["id"] = folder.name
            seen.add(folder.name)
            out.append(rec)
    return sorted(out, key=lambda m: m["id"])


def write_remote_dir(root: str | Path, record: dict) -> Path:
    """Écrit/maj `<racine>/remote/<id>/model.toml` via tomlkit (commentaires d'un fichier
    édité à la main PRÉSERVÉS). Seuls les champs non-None de `record` sont posés — une
    édition partielle (ex. sans re-saisir la clé) ne perd rien. Crée le dossier au besoin."""
    import tomlkit

    d = remote_dir(root, record["id"])
    d.mkdir(parents=True, exist_ok=True)
    p = d / "model.toml"
    doc = (
        tomlkit.parse(p.read_text(encoding="utf-8"))
        if p.exists()
        else tomlkit.document()
    )
    if not p.exists():
        doc.add(
            tomlkit.comment(
                "Modèle DISTANT (API OpenAI-compatible) — généré par Loom, ajuste "
                "librement (cf. _TEMPLATE/remote-model.toml)."
            )
        )
    for k in FIELDS:
        if record.get(k) is not None:
            doc[k] = record[k]
    p.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return p


def delete_remote_dir(models_roots: list[Path] | list[str], model_id: str) -> bool:
    """Supprime le dossier `remote/<id>` sur TOUTES les racines où il existe : la
    découverte étant première-racine-gagnante, un reliquat sur une autre racine
    ressusciterait le modèle au boot suivant. Renvoie True si au moins un supprimé."""
    removed = False
    for root in models_roots:
        d = remote_dir(root, model_id)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed = removed or not d.exists()
    return removed


def migrate_into_dirs(
    local_path: str | Path | None,
    store_path: str | Path | None,
    dest_root: str | Path | None,
) -> list[dict]:
    """Replie les deux emplacements hérités en dossiers `remote/<id>/` sur la racine
    prioritaire : [[remote_models]] de config/local.toml et le store JSON
    var/remote_models.json. Idempotent ; un dossier EXISTANT du même id n'est jamais
    écrasé (le dossier fait foi). Chaque entrée migrée est retirée de sa source ; une
    entrée qui n'a pas pu s'écrire (FS en lecture seule…) y RESTE et est renvoyée pour
    que load_config la fusionne en mémoire ce boot-ci — dégradation douce, zéro perte."""
    leftovers: list[dict] = []
    entries: list[tuple[str, dict]] = []  # (source, record)
    if local_path is not None and Path(local_path).exists():
        import tomllib

        try:
            data = tomllib.loads(Path(local_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for rm in data.get("remote_models", []):
            if isinstance(rm, dict) and rm.get("id"):
                entries.append(("toml", {k: rm[k] for k in KEEP if k in rm}))
    if store_path is not None:
        for rec in load(store_path):
            entries.append(("json", rec))
    if not entries:
        return []
    migrated_toml_ids: list[str] = []
    json_done = True
    for source, rec in entries:
        ok = False
        if dest_root is not None:
            try:
                if remote_dir(dest_root, rec["id"]).is_dir():
                    ok = True  # le dossier existant fait foi — l'entrée héritée saute
                else:
                    write_remote_dir(dest_root, rec)
                    ok = True
            except OSError:
                ok = False
        if ok:
            if source == "toml":
                migrated_toml_ids.append(rec["id"])
        else:
            leftovers.append(rec)
            if source == "json":
                json_done = False
    for mid in migrated_toml_ids:
        delete_remote_in_toml(local_path, mid)
    if json_done and store_path is not None and Path(store_path).exists():
        try:
            Path(store_path).unlink()
        except OSError:
            pass  # best-effort : re-migré au prochain boot (idempotent)
    return leftovers


# Compatibilité avec les anciens stores, uniquement pour migration et suppression.


def load(path: str | Path) -> list[dict]:
    """Store JSON hérité : liste des distants (vide si absent/illisible). Filtre les
    entrées sans les 3 champs indispensables (id/base_url/model)."""
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
    """Écrit le store JSON hérité en OWNER-ONLY (0600, clés API en clair) et en
    atomique (temp + replace). Ne sert plus qu'aux tests des migrations."""
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
    os.replace(tmp, p)  # atomique : l'ancien ou le nouveau, jamais un mix
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def delete(path: str | Path, model_id: str) -> list[dict]:
    """Retire un modèle par id du store JSON hérité et persiste."""
    models = [m for m in load(path) if m.get("id") != model_id]
    save(path, models)
    return models


def delete_remote_in_toml(local_path: str | Path | None, model_id: str) -> bool:
    """Retire un [[remote_models]] hérité de local.toml par id (tomlkit : commentaires
    et structure PRÉSERVÉS). Renvoie False si fichier ou entrée absents (no-op)."""
    import tomlkit

    if local_path is None:
        return False
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
    # Supprimer aussi la clé vide laissée par certaines anciennes configurations.
    if len(arr) == 0:
        del doc["remote_models"]
    p.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return True
