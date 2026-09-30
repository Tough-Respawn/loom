"""Journaux de llama-server : un fichier par modèle, archivé avant d'être écrasé.

llama-server ouvre son `--log-file` en écriture (`fopen(path, "w")`) : chaque
(re)démarrage efface le journal précédent. Or c'est précisément le journal d'avant
un changement de modèle ou un redémarrage qu'on veut relire après coup (post-mortem
Bonsai 2 du 2026-09-30 : les décisions de cache n'étaient visibles nulle part). On
copie donc le fichier dans `archive/` au lancement de Loom et juste avant chaque
changement de modèle local, en gardant les plus récents.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LLAMA_LOGS_DIR = REPO_ROOT / "var" / "logs" / "llama"
KEEP_ARCHIVES = 40


def logs_dir() -> Path:
    LLAMA_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LLAMA_LOGS_DIR


def server_log_path(model_id: str) -> str:
    """Chemin du journal courant de `model_id`, en séparateurs POSIX (yaml llama-swap)."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_id)
    return str(logs_dir() / f"{safe}.log").replace("\\", "/")


def archive_server_log(model_id: str) -> Path | None:
    """Copie le journal courant de `model_id` dans archive/ (s'il existe et n'est pas
    vide). Copie et non déplacement : le fichier peut être ouvert par un llama-server
    encore vivant. Ne lève jamais."""
    try:
        src = Path(server_log_path(model_id))
        if not src.is_file() or src.stat().st_size == 0:
            return None
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(src.stat().st_mtime))
        dest_dir = logs_dir() / "archive"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{src.stem}-{stamp}.log"
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            return dest  # déjà archivé tel quel
        shutil.copy2(src, dest)
        _prune(dest_dir)
        return dest
    except OSError:
        return None


def archive_all() -> None:
    """Au lancement : archive tous les journaux courants avant que les serveurs ne
    redémarrent et ne les écrasent."""
    try:
        for path in logs_dir().glob("*.log"):
            archive_server_log(path.stem)
    except OSError:
        pass


def _prune(dest_dir: Path) -> None:
    files = sorted(dest_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    for old in files[:-KEEP_ARCHIVES]:
        try:
            old.unlink()
        except OSError:
            pass
