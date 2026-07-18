# loom/setup/steps.py
"""Logique PURE de l'installeur : lecture de la config brute, constats
(binaire présent ? modèles branchés ?) et écriture ciblée de config/local.toml.

Pourquoi « brute » : load_config() lève quand AUCUN modèle n'existe — exactement
l'état post-clone que le setup doit gérer. On lit donc defaults + local fusionnés
SANS validation, et on ne passe par load_config qu'une fois le parc installé."""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

from loom.config import _deep_merge


@dataclass
class StepOutcome:
    """Résultat d'une étape, accumulé pour le bilan final."""

    name: str  # "detection" | "binaire" | "modele"
    status: str  # "ok" | "fait" | "ignore" | "echec" | "manuel"
    detail: str  # phrase courte pour le bilan


def read_raw_config(defaults_path: str | Path, local_path: str | Path) -> dict:
    """defaults.toml + local.toml fusionnés (local écrase), SANS validation.

    Même fusion que load_config (config.py) — juste sans le « aucun modèle »
    fatal, pour pouvoir constater l'état d'une machine vierge."""
    data = tomllib.loads(Path(defaults_path).read_text(encoding="utf-8"))
    local = Path(local_path)
    if local.exists():
        data = _deep_merge(data, tomllib.loads(local.read_text(encoding="utf-8")))
    return data


def server_bin_status(raw_cfg: dict) -> tuple[bool, str]:
    """(présent, chemin/nom) du binaire llama-server configuré.

    Présent = chemin de fichier existant, OU nom résolu par le PATH."""
    bin_name = raw_cfg.get("server", {}).get("bin", "llama-server")
    present = Path(bin_name).is_file() or shutil.which(bin_name) is not None
    return present, bin_name


def models_roots(raw_cfg: dict, package_models: Path) -> list[Path]:
    """Racine(s) des modèles : [storage] models_root (chaîne OU liste), sinon le
    package (loom/models). Même résolution que load_config (config.py:347-353)."""
    raw = raw_cfg.get("storage", {}).get("models_root") or package_models
    if not isinstance(raw, list):
        raw = [raw]
    return [Path(r).resolve() for r in raw]


def _model_folders(raw_cfg: dict, package_models: Path) -> list[tuple[str, Path, dict]]:
    """(id, dossier, contenu du model.toml) de chaque modèle texte branché.

    Lecture TOLÉRANTE, contrairement à _discover_models/_parse_model : un
    model.toml minimal (écrit avant le download, pas encore finalisé — pas de
    n_layers) ne doit pas faire planter l'installeur. _TEMPLATE ignoré."""
    out: list[tuple[str, Path, dict]] = []
    for root in models_roots(raw_cfg, package_models):
        base = root / "local" / "text"
        if not base.is_dir():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            if folder.name.startswith("_"):
                continue
            toml_path = folder / "model.toml"
            if not toml_path.exists():
                continue
            try:
                data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                continue
            out.append((folder.name, folder, data))
    return out


def installed_model_ids(raw_cfg: dict, package_models: Path) -> list[str]:
    """Ids des modèles texte déjà branchés (un dossier <racine>/local/text/<id>/
    avec model.toml = un modèle)."""
    return sorted({mid for mid, _, _ in _model_folders(raw_cfg, package_models)})


def resolve_bin(bin_name: str) -> Path | None:
    """Chemin RÉEL du binaire configuré : fichier direct, sinon résolution PATH.
    None s'il est introuvable (le bench a besoin du dossier de la release)."""
    p = Path(bin_name)
    if p.is_file():
        return p
    which = shutil.which(bin_name)
    return Path(which) if which else None


def first_model_file(raw_cfg: dict, package_models: Path) -> tuple[Path, int] | None:
    """(chemin du GGUF, size_mb) du modèle par défaut (sinon premier) DÉJÀ
    téléchargé sur disque — None si aucun modèle ou GGUF pas encore là.
    C'est le modèle que le bench doit mesurer : celui qui servira."""
    folders = _model_folders(raw_cfg, package_models)
    if not folders:
        return None
    default = raw_cfg.get("chat", {}).get("default_model")
    mid, folder, data = next((f for f in folders if f[0] == default), folders[0])
    filename = data.get("filename")
    if not filename:
        return None
    path = folder / filename
    if not path.is_file():
        return None
    return path, int(data.get("size_mb", 0))


def needs_setup(raw_cfg: dict, package_models: Path) -> bool:
    """Machine pas prête à servir : binaire llama-server introuvable OU aucun
    modèle branché. C'est le déclencheur de l'installeur auto au premier
    `serve.py` (et le critère d'idempotence de loom-setup)."""
    present, _ = server_bin_status(raw_cfg)
    return not present or not installed_model_ids(raw_cfg, package_models)


def set_local_values(local_path: str | Path, values: dict[str, dict]) -> None:
    """Écrit {table: {clé: valeur}} dans config/local.toml en UNE passe. tomlkit :
    crée le fichier s'il n'existe pas, sinon met à jour en préservant commentaires
    et autres tables (même pattern que finalize_model_toml)."""
    import tomlkit

    local_path = Path(local_path)
    if local_path.exists():
        doc = tomlkit.parse(local_path.read_text(encoding="utf-8"))
    else:
        doc = tomlkit.document()
        doc.add(
            tomlkit.comment(
                "Surcharge MACHINE (gitignored) — générée par loom-setup, "
                "complète-la librement (cf. config/local.example.toml)."
            )
        )
    for table, kv in values.items():
        if table not in doc:
            doc[table] = tomlkit.table()
        for key, value in kv.items():
            doc[table][key] = value
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _set_local_value(local_path: str | Path, table: str, key: str, value) -> None:
    set_local_values(local_path, {table: {key: value}})


def set_server_bin(local_path: str | Path, bin_path: str | Path) -> None:
    """[server] bin = <chemin absolu, slashes avant> dans config/local.toml."""
    _set_local_value(
        local_path, "server", "bin", str(Path(bin_path).resolve()).replace("\\", "/")
    )


def set_default_model(local_path: str | Path, model_id: str) -> None:
    """[chat] default_model = <id> dans config/local.toml. Sans ça, le défaut de
    defaults.toml peut pointer un modèle d'une AUTRE machine (le parc) : le
    premier modèle installé ici devient le défaut local."""
    _set_local_value(local_path, "chat", "default_model", model_id)
