# loom/setup/llama_release.py
"""Installation du binaire llama.cpp depuis les releases GitHub officielles.

Sélection de l'asset selon OS + arch + GPU (matrice de regex ORDONNÉES : le
nommage des assets llama.cpp a changé plusieurs fois, on matche large et on
préfère le plus récent nommage). Téléchargement httpx en streaming, extraction
dans var/runtime/llama/<tag>/ (gitignoré, self-contained, sans admin), et
vérification `--version` AVANT toute écriture de config."""

from __future__ import annotations

import platform
import re
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# Matrice (os, gpu) -> regex d'assets par ordre de PRÉFÉRENCE. Clé "cudart" :
# compagnon OBLIGATOIRE (DLL CUDA à poser à côté du binaire) — absent -> pas
# d'install CUDA possible, on retombe sur le guidage manuel.
_MATRIX: dict[tuple[str, str, bool], dict] = {
    ("windows", "x64", True): {
        "backend": "cuda",
        "patterns": [r"bin-win-cuda.*x64\.zip$"],
        "companion": r"^cudart-.*win-cuda.*x64\.zip$",
    },
    ("windows", "x64", False): {
        "backend": "vulkan/cpu",
        "patterns": [
            r"bin-win-vulkan-x64\.zip$",
            r"bin-win-cpu-x64\.zip$",
            r"bin-win-avx2-x64\.zip$",  # ancien nommage
        ],
    },
    ("windows", "arm64", False): {
        "backend": "cpu",
        "patterns": [r"bin-win-cpu-arm64\.zip$"],
    },
    ("linux", "x64", True): {
        "backend": "cuda",
        "patterns": [r"bin-ubuntu.*cuda.*x64.*\.(zip|tar\.gz)$"],
        # Pas de build CUDA Linux dans toutes les releases -> select_assets
        # renvoie None et l'appelant guide vers la compilation.
    },
    ("linux", "x64", False): {
        "backend": "cpu/vulkan",
        "patterns": [
            r"bin-ubuntu-x64\.(zip|tar\.gz)$",
            r"bin-ubuntu-vulkan-x64\.(zip|tar\.gz)$",
        ],
    },
    ("macos", "arm64", False): {
        "backend": "metal",
        "patterns": [r"bin-macos-arm64\.zip$"],
    },
    ("macos", "x64", False): {
        "backend": "cpu",
        "patterns": [r"bin-macos-x64\.zip$"],
    },
}


@dataclass
class AssetPlan:
    """Ce que le setup propose de télécharger, montrable tel quel à l'utilisateur."""

    tag: str  # ex. "b5321"
    backend: str  # "cuda" | "vulkan/cpu" | "metal" | ...
    reason: str  # phrase FR expliquant le choix
    assets: list[dict] = field(default_factory=list)  # {name, url, size_mb}

    @property
    def total_mb(self) -> int:
        return sum(a["size_mb"] for a in self.assets)


def local_arch() -> str:
    """ "x64" ou "arm64" (les deux seules archs des releases llama.cpp)."""
    m = platform.machine().lower()
    return "arm64" if m in ("arm64", "aarch64") else "x64"


def fetch_latest_release(client) -> dict:
    """Dernière release stable (une SEULE requête API — quota anonyme 60/h).

    `client` : httpx.Client injecté (fake dans les tests). Lève RuntimeError
    au message montrable sur rate-limit ou réseau coupé."""
    try:
        resp = client.get(
            RELEASES_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "loom-setup",
            },
            follow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - tout devient un message actionnable
        raise RuntimeError(
            f"GitHub injoignable ({type(exc).__name__}) — vérifie la connexion, "
            "ou installe llama.cpp à la main (docs/install-windows.md)."
        ) from exc
    if resp.status_code == 403:
        raise RuntimeError(
            "GitHub a refusé la requête (403, probable rate-limit anonyme) — "
            "réessaie dans une heure, ou installe llama.cpp à la main."
        )
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub a répondu {resp.status_code} sur {RELEASES_URL}.")
    return resp.json()


def _asset_dict(a: dict) -> dict:
    return {
        "name": a["name"],
        "url": a["browser_download_url"],
        "size_mb": max(1, int(a.get("size", 0)) // (1024 * 1024)),
    }


def select_assets(
    release: dict, os_key: str, arch: str, has_nvidia: bool
) -> AssetPlan | None:
    """Choisit le(s) asset(s) à télécharger pour cette machine, ou None si aucun
    ne convient (l'appelant bascule alors sur le guidage manuel)."""
    entry = _MATRIX.get((os_key, arch, has_nvidia))
    if entry is None:
        # Pas de variante GPU pour cette combinaison (ex. macos + nvidia n'existe
        # pas) : retente sans GPU avant d'abandonner.
        entry = _MATRIX.get((os_key, arch, False))
    if entry is None:
        return None
    assets = release.get("assets", [])
    chosen = None
    for pat in entry["patterns"]:
        rx = re.compile(pat, re.IGNORECASE)
        for a in assets:
            if rx.search(a["name"]):
                chosen = a
                break
        if chosen:
            break
    if chosen is None:
        return None
    plan = AssetPlan(
        tag=release.get("tag_name", "?"),
        backend=entry["backend"],
        reason=f"build {entry['backend']} pour {os_key} {arch}",
        assets=[_asset_dict(chosen)],
    )
    companion = entry.get("companion")
    if companion:
        rx = re.compile(companion, re.IGNORECASE)
        comp = next((a for a in assets if rx.search(a["name"])), None)
        if comp is None:
            return None  # DLL CUDA introuvables -> pas d'install fiable
        plan.assets.append(_asset_dict(comp))
    return plan


def download_and_extract(
    plan: AssetPlan, dest_root: str | Path, client, progress_cb=None
) -> Path:
    """Télécharge et extrait tous les assets du plan dans dest_root/<tag>/
    (bin + cudart dans le MÊME dossier : les DLL doivent côtoyer l'exe).
    `progress_cb(asset_name, done_mb, total_mb)` si fourni. Renvoie le dossier."""
    dest = Path(dest_root) / plan.tag
    dest.mkdir(parents=True, exist_ok=True)
    for asset in plan.assets:
        archive = dest / asset["name"]
        _download_one(asset, archive, client, progress_cb)
        _extract(archive, dest)
        archive.unlink()  # l'archive ne sert plus une fois extraite
    return dest


def _download_one(asset: dict, target: Path, client, progress_cb) -> None:
    with client.stream(
        "GET", asset["url"], follow_redirects=True, headers={"User-Agent": "loom-setup"}
    ) as resp:
        if resp.status_code != 200:
            raise RuntimeError(
                f"téléchargement de {asset['name']} : HTTP {resp.status_code}"
            )
        total_mb = int(resp.headers.get("content-length", 0)) // (1024 * 1024)
        done = 0
        with open(target, "wb") as fh:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(asset["name"], done // (1024 * 1024), total_mb)


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest, filter="data")
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)


def find_llama_server(root: str | Path) -> Path | None:
    """Cherche llama-server(.exe) sous root (zip Windows = à plat ; tar ubuntu =
    build/bin/). Renvoie le premier trouvé, exécutable rendu si besoin (POSIX)."""
    root = Path(root)
    if not root.is_dir():
        return None
    for p in sorted(root.rglob("llama-server*")):
        if p.is_file() and p.stem == "llama-server" and p.suffix in ("", ".exe"):
            if p.suffix == "":  # POSIX : zipfile/tarfile ne garantit pas le +x
                p.chmod(p.stat().st_mode | 0o755)
            return p
    return None


def verify_binary(bin_path: str | Path, timeout: int = 15) -> str | None:
    """`llama-server --version` — renvoie la 1re ligne de sortie, ou None si le
    binaire ne se lance pas (DLL manquante, mauvaise arch…)."""
    try:
        res = subprocess.run(
            [str(bin_path), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = (res.stdout or "") + (res.stderr or "")  # --version sort sur stderr
    first = out.strip().splitlines()[0].strip() if out.strip() else ""
    return first or None
