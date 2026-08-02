"""SearXNG géré par Loom — recherche web FIABLE en self-host (pattern ComfyUI : un
service externe que Loom installe et démarre à la demande, jamais indispensable —
`web_search` retombe sur ddgs s'il est absent).

SIMPLE POUR L'UTILISATEUR (exigence produit) :
- installation = UNE commande : `uv run python -m loom.runtime.searxng install`
  (Docker requis ; tire l'image officielle ~200 Mo, écrit la config Loom tout seul) ;
- ensuite ZÉRO maintenance : conteneur en `--restart unless-stopped`, et si on le
  trouve arrêté au moment d'une recherche, `ensure_running` le relance (sans jamais
  télécharger quoi que ce soit à l'insu de l'utilisateur : seul `install` tire l'image).

Détail qui compte : l'API `format=json` (utilisée par web_search) est REFUSÉE par
l'image officielle par défaut -> on fournit notre settings.yml (var/searxng/) qui
l'active, monté dans le conteneur.
"""

from __future__ import annotations

import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SETTINGS_DIR = REPO_ROOT / "var" / "searxng"
CONTAINER = "loom-searxng"
IMAGE = "searxng/searxng"
DEFAULT_PORT = (
    8890  # port distinctif (8080 = llama-swap, 8888 souvent pris par Jupyter)
)


def _docker(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


def docker_available() -> bool:
    """Docker CLI présent ET démon joignable (Docker Desktop lancé)."""
    try:
        return _docker("version", "--format", "{{.Server.Version}}").returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def container_state() -> str:
    """'running' | 'exited' | ... | 'absent' (inspect du conteneur géré)."""
    try:
        r = _docker("inspect", "-f", "{{.State.Status}}", CONTAINER)
    except (OSError, subprocess.TimeoutExpired):
        return "absent"
    return r.stdout.strip() if r.returncode == 0 else "absent"


def is_ready(url: str, timeout: float = 2.0) -> bool:
    """L'instance répond-elle ? /healthz est l'endpoint de vivacité de SearXNG."""
    try:
        with urllib.request.urlopen(
            url.rstrip("/") + "/healthz", timeout=timeout
        ) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - sonde best-effort
        return False


def _write_settings() -> None:
    """settings.yml minimal : défauts upstream + format JSON (requis par web_search)
    + secret aléatoire. Écrit une seule fois (on ne clobber pas une customisation)."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    p = SETTINGS_DIR / "settings.yml"
    if p.exists():
        return
    p.write_text(
        "# Généré par Loom (loom/runtime/searxng.py) — customisable, non-écrasé.\n"
        "use_default_settings: true\n"
        "server:\n"
        f'  secret_key: "{secrets.token_hex(32)}"\n'
        "  limiter: false\n"
        "search:\n"
        "  formats:\n"
        "    - html\n"
        "    - json\n",
        encoding="utf-8",
    )


def _wait_ready(url: str, wait: float) -> bool:
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if is_ready(url):
            return True
        time.sleep(1.0)
    return False


def ensure_running(url: str, wait: float = 12.0) -> bool:
    """Rend l'instance joignable SI POSSIBLE sans rien installer : déjà prête -> True ;
    conteneur géré arrêté -> docker start + attente ; sinon False (l'appelant retombe
    sur ddgs). Ne tire JAMAIS d'image ici (ça, c'est `install`, consenti)."""
    if is_ready(url):
        return True
    if container_state() != "exited" or not docker_available():
        return False
    try:
        if _docker("start", CONTAINER).returncode != 0:
            return False
    except (OSError, subprocess.TimeoutExpired):
        return False
    return _wait_ready(url, wait)


def install(port: int = DEFAULT_PORT) -> str | None:
    """Installe (ou répare) le conteneur géré et CÂBLE la config Loom. Renvoie l'URL
    prête, ou None avec les explications sur stdout (jamais d'exception)."""
    url = f"http://127.0.0.1:{port}"
    if not docker_available():
        print(
            "[searxng] Docker indisponible : lance Docker Desktop (ou installe-le), "
            "puis relance `uv run python -m loom.runtime.searxng install`. En attendant, "
            "web_search continue sur le repli ddgs."
        )
        return None
    _write_settings()
    state = container_state()
    if state == "running":
        print(f"[searxng] conteneur déjà actif ({CONTAINER}).")
    elif state == "exited":
        print(f"[searxng] conteneur présent, démarrage ({CONTAINER})…")
        _docker("start", CONTAINER)
    else:
        print(f"[searxng] création du conteneur (image {IMAGE}, ~200 Mo au 1er pull)…")
        r = _docker(
            "run",
            "-d",
            "--name",
            CONTAINER,
            "--restart",
            "unless-stopped",
            "-p",
            f"127.0.0.1:{port}:8080",
            "-v",
            f"{SETTINGS_DIR.as_posix()}:/etc/searxng",
            IMAGE,
            timeout=300.0,  # inclut le pull de l'image
        )
        if r.returncode != 0:
            print(f"[searxng] échec docker run : {r.stderr.strip()[:300]}")
            return None
    if not _wait_ready(url, 60.0):
        print(
            "[searxng] l'instance ne répond pas (docker logs loom-searxng pour voir)."
        )
        return None
    # En mode auto, SearXNG devient prioritaire et DDGS reste le repli.
    from loom.runtime.config_schema import set_value
    from loom.runtime.serve import CONFIG_PATH, PERSONAL_CONFIG_PATH

    res = set_value(CONFIG_PATH, PERSONAL_CONFIG_PATH, "web_search", "searxng_url", url)
    if not res.get("ok"):
        print(f"[searxng] prêt sur {url} mais écriture config échouée : {res}")
        return None
    print(f"[searxng] PRÊT : {url} (searxng_url écrit dans config/local.toml).")
    return url


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "install"
    if cmd == "install":
        return 0 if install() else 1
    if cmd == "status":
        print(f"conteneur : {container_state()}")
        return 0
    if cmd == "stop":
        _docker("stop", CONTAINER)
        print("arrêté.")
        return 0
    print("usage : python -m loom.runtime.searxng [install|status|stop]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
