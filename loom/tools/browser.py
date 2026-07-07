# loom/tools/browser.py
"""Outils navigateur : check_page (yeux sur une page rendue), check_interactive (preuve de
jouabilité) et serve_and_check (démarre un serveur, vérifie, l'arrête).

Sans ca, le modele edite du HTML/JS a l'aveugle et confabule « ca marche » : il ne voit
ni l'erreur console qui plante le jeu, ni que la grille ne s'affiche pas. check_page charge
la page dans Chromium headless, EXECUTE le JS, et renvoie les ERREURS CONSOLE, le compte
d'éléments (count_selectors) et un extrait du texte visible. Lazy-import de playwright :
message clair et actionnable si la lib (ou le navigateur) n'est pas installee.

Le contenu d'une page est EXTERNE/non fiable -> renvoye via untrusted() : donnee a
analyser, jamais des ordres (une page peut contenir « ignore tes consignes »).
"""

from __future__ import annotations

import atexit
import ipaddress
import os
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from loom.permissions import _is_hard_denied
from loom.runtime.platform_info import detect
from loom.tools._net import categorize_ip
from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.shell import _kill_tree, _shell_argv
from loom.tools.trust import untrusted

# --- Registre des serveurs LAISSÉS VIVANTS par serve_and_check ---------------------------
# serve_and_check démarre un serveur, le vérifie, puis le LAISSE TOURNER (au lieu de le tuer
# aussitôt) pour que le modèle puisse tester PLUSIEURS pages/jeux sur le même serveur, puis
# l'arrête explicitement via stop_server. Sécurités : arrêt de tout à la sortie de Loom
# (atexit) + TTL de secours si le modèle oublie de fermer (pas de serveur orphelin éternel).
_SERVERS_LOCK = threading.Lock()
_LIVE_SERVERS: dict[str, dict] = {}  # id -> {proc, url, logpath, logf, started}
_server_seq = [0]
_SERVER_TTL = 900.0  # 15 min : large pour explorer, borné pour ne pas fuir


def _kill_server_locked(sid: str) -> None:
    """Tue et nettoie un serveur suivi. À appeler AVEC _SERVERS_LOCK détenu."""
    s = _LIVE_SERVERS.pop(sid, None)
    if not s:
        return
    try:
        _kill_tree(s["proc"])
    except Exception:  # noqa: BLE001 - best-effort
        pass
    try:
        s["logf"].close()
    except Exception:  # noqa: BLE001
        pass
    try:
        os.unlink(s["logpath"])
    except OSError:
        pass


def _reap_expired_servers() -> None:
    """Arrête les serveurs plus vieux que le TTL (filet si le modèle oublie stop_server)."""
    now = time.monotonic()
    with _SERVERS_LOCK:
        for sid in [
            k for k, s in _LIVE_SERVERS.items() if now - s["started"] > _SERVER_TTL
        ]:
            _kill_server_locked(sid)


def _register_server(proc, url: str, logpath: str, logf) -> str:
    with _SERVERS_LOCK:
        _server_seq[0] += 1
        sid = f"srv{_server_seq[0]}"
        _LIVE_SERVERS[sid] = {
            "proc": proc,
            "url": url,
            "logpath": logpath,
            "logf": logf,
            "started": time.monotonic(),
        }
    return sid


def _find_server_by_url(url: str) -> str | None:
    with _SERVERS_LOCK:
        for sid, s in _LIVE_SERVERS.items():
            if s["url"] == url and s["proc"].poll() is None:
                return sid
    return None


def _stop_servers(sid: str | None) -> str:
    with _SERVERS_LOCK:
        if sid:
            if sid not in _LIVE_SERVERS:
                actifs = ", ".join(_LIVE_SERVERS) or "(aucun)"
                return f"aucun serveur suivi « {sid} ». Serveurs actifs : {actifs}"
            url = _LIVE_SERVERS[sid]["url"]
            _kill_server_locked(sid)
            return f"serveur {sid} ({url}) arrêté."
        if not _LIVE_SERVERS:
            return "aucun serveur actif à arrêter."
        n = len(_LIVE_SERVERS)
        for k in list(_LIVE_SERVERS):
            _kill_server_locked(k)
        return f"{n} serveur(s) arrêté(s)."


def _alive_hint(sid: str) -> str:
    """Message accolé à un serveur laissé vivant : dit au modèle qu'il reste lancé, comment
    tester d'autres pages, comment le fermer, et POURQUOI ne pas bricoler avec Start-Process."""
    return (
        f"[serveur TOUJOURS ACTIF - id={sid}] Il reste lance : teste d'AUTRES pages de ce "
        "serveur avec check_page/check_interactive (ou re-appelle serve_and_check sur une "
        "autre url du meme site, sans relancer). QUAND TU AS TA REPONSE, ferme-le avec "
        f"serve_and_check(action='stop', id='{sid}'). Ne lance JAMAIS un serveur toi-meme via "
        "Start-Process / start / Invoke-Item (ca ouvre le .ps1 dans un éditeur et ne survit "
        "pas) : serve_and_check s'occupe du cycle de vie."
    )


@atexit.register
def _kill_all_servers() -> None:  # pragma: no cover - filet de sortie process
    with _SERVERS_LOCK:
        for k in list(_LIVE_SERVERS):
            _kill_server_locked(k)


_INSTALL_HINT = (
    "playwright non installe. Lance une fois : `uv add playwright` puis "
    "`uv run playwright install chromium`."
)

# check_page REND une page dans un navigateur (execute son JS) ; ce n'est PAS un lecteur de
# fichiers. On le borne aux formats reellement web-rendables : sinon `check_page` sert de
# contournement de lecture (file://.../id_rsa rendu -> contenu fuite dans le « texte visible »).
# La vraie lecture passe par read_file/read_document, pas par le navigateur.
_WEB_EXT = frozenset({".html", ".htm", ".xhtml", ".svg"})
_NOT_WEB_MSG = (
    "check_page ne rend que des pages web (.html/.htm/.xhtml/.svg). Pour lire un autre "
    "fichier, utilise read_file (texte) ou read_document (pdf/xlsx/docx)."
)


def _browser_http_blocked(url: str) -> str | None:
    """Garde anti-SSRF DEDIE au navigateur. Contrairement a fetch_url (contenu externe
    arbitraire), check_page/serve_and_check sont des outils de VERIF LOCALE : leur usage
    normal EST de viser un serveur de dev sur localhost (127.0.0.1/::1) ou le LAN prive
    (192.168.x). On AUTORISE donc loopback + prive, et on ne bloque que les cibles sans
    usage dev legitime : metadonnees cloud (link-local 169.254.x), reserve, multicast, non
    specifie. Renvoie la raison de blocage, ou None si la cible est acceptable."""
    host = urlparse(url).hostname
    if not host:
        return "url sans hote valide"
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return f"hote introuvable : {host}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        cat = categorize_ip(ip)
        # Loopback (127.0.0.1/::1) et LAN privé (192.168.x) AUTORISÉS : c'est l'usage
        # même de l'outil. On ne bloque que ce qui n'a aucun usage dev : métadonnées
        # cloud (link-local 169.254.x / fe80::), multicast, adresse non spécifiée.
        if cat in ("link-local", "multicast", "unspecified"):
            return f"hote interdit (adresse speciale : {ip})"
    return None


def _browser_url(root: Path, target: str) -> str:
    """Resout et VALIDE la cible d'un outil navigateur ; leve ToolError si refusee.

    - http(s):// -> garde de navigateur (`_browser_http_blocked`) : loopback/LAN prive
      AUTORISES (c'est l'usage : vérifier un dev server), seules les adresses speciales
      (metadonnees cloud) sont bloquees ;
    - file:// ou chemin local -> confine aux extensions web (`_WEB_EXT`) : le navigateur ne
      peut pas servir a exfiltrer un fichier arbitraire rendu en page.
    """
    target = target.strip()
    low = target.lower()
    if low.startswith(("http://", "https://")):
        reason = _browser_http_blocked(target)
        if reason:
            raise ToolError(f"url refusee : {reason}")
        return target
    if low.startswith("file://"):
        ext = Path(unquote(urlparse(target).path)).suffix.lower()
        if ext not in _WEB_EXT:
            raise ToolError(_NOT_WEB_MSG)
        return target
    # chemin local (relatif au dossier de travail ou absolu) -> file:// (Path.as_uri()).
    path = _resolve_in_root(root, target)
    if not path.exists():
        raise ToolError(f"fichier introuvable : {target}")
    if path.suffix.lower() not in _WEB_EXT:
        raise ToolError(_NOT_WEB_MSG)
    return path.as_uri()


def _render_page(
    url: str, wait_selector: str | None, count_selectors: list[str]
) -> str:
    """Charge `url` dans Chromium headless, EXECUTE son JS et renvoie un rapport TEXTE :
    erreurs console, comptes d'éléments, extrait visible, diagnostic de localisation. Ne
    leve PAS pour une page injoignable (renvoie un diagnostic) ; leve ToolError seulement
    si Playwright (lib ou navigateur) est absent. Partage par check_page et serve_and_check."""
    try:
        from playwright.sync_api import TimeoutError as PWTimeout
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ToolError(_INSTALL_HINT) from exc

    console: list[tuple[str, str]] = []
    page_errors: list[str] = []
    title = ""
    body_text = ""
    counts: dict[str, int] = {}
    note = ""  # diagnostic de localisation (timeout, lecture interrompue, echec)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(5000)  # borne les lectures (pas de hang 30s)
            page.on("console", lambda m: console.append((m.type, m.text)))
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            try:
                page.goto(url, wait_until="load", timeout=15000)
                loaded = True
            except PWTimeout:
                # Chargement non termine : on NE LIT PAS la page (le thread JS peut etre
                # gele -> nouveaux timeouts). On garde les preuves deja captees + un indice.
                loaded = False
                note = (
                    "chargement non termine en 15s -> script bloquant probable "
                    "(ex. boucle infinie a l'init). Desactive les scripts un par un pour "
                    "bisecter ; les erreurs console ci-dessus pointent souvent la cause."
                )
            if loaded:
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=5000)
                    except Exception:  # noqa: BLE001 - absence = info, pas un crash
                        pass
                page.wait_for_timeout(1200)  # laisse le JS d'init s'executer
                try:
                    title = page.title()
                    body = page.query_selector("body")
                    body_text = body.inner_text()[:2000] if body else ""
                    counts = {
                        sel: len(page.query_selector_all(sel))
                        for sel in count_selectors
                    }
                except Exception as exc:  # noqa: BLE001 - lecture partiellement bloquee
                    note = f"lecture de la page interrompue : {str(exc)[:120]}"
            browser.close()
    except Exception as exc:  # noqa: BLE001 - navigateur absent / page injoignable
        msg = str(exc)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            raise ToolError(_INSTALL_HINT) from exc
        # On NE jette PAS les preuves : diagnostic structure plutot qu'une exception seche.
        note = note or f"echec du chargement : {msg[:200]}"

    errors = [t for (k, t) in console if k == "error"] + page_errors
    warnings = [t for (k, t) in console if k == "warning"]
    lines = [
        f"page : {title!r} ({url})" if title else f"page : ({url})",
        f"console : {len(errors)} erreur(s), {len(warnings)} warning(s)",
    ]
    for e in errors[:8]:
        lines.append(f"  [erreur] {e[:200]}")
    if counts:
        lines.append("éléments : " + " - ".join(f"{s} x{n}" for s, n in counts.items()))
    visible = " ".join(body_text.split())
    if visible:
        lines.append(f"texte visible : {visible[:400]}")
    if note:
        lines.append(f"DIAGNOSTIC : {note}")
    elif not errors and not page_errors:
        lines.append("(aucune erreur console - la page s'est chargée et exécutée)")
    return "\n".join(lines)


def make_check_page(workspace_dir: str) -> ToolSpec:
    """Outil check_page borne au workspace pour les chemins relatifs (absolus acceptes)."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        target = (args.get("url") or "").strip()
        if not target:
            raise ToolError(
                "argument 'url' manquant (URL http(s):// OU chemin d'un fichier .html)"
            )
        # Cible validee : loopback/LAN prive AUTORISES (verif locale), extensions web sur
        # file:// (le navigateur ne sert pas a exfiltrer un fichier arbitraire).
        url = _browser_url(root, target)
        wait_selector = (args.get("wait_selector") or "").strip() or None
        count_selectors = [
            s.strip()
            for s in (args.get("count_selectors") or "").split(",")
            if s.strip()
        ]
        return untrusted(
            _render_page(url, wait_selector, count_selectors), f"page {url}"
        )

    return ToolSpec(
        name="check_page",
        description=(
            "Loads a web page (http(s):// URL OR path to a local .html file) in a headless "
            "browser, EXECUTES its JavaScript, and returns: the console ERRORS, the number of "
            "elements matching count_selectors (e.g. '.cell,#board'), and an excerpt of the "
            "visible text. USE IT to VERIFY that an HTML page you just wrote renders and works "
            "(0 console errors, expected elements present) INSTEAD of assuming it works. For an "
            "app served by a SERVER (Next.js, Vite, Flask) that isn't started yet, use "
            "serve_and_check instead. If errors show up, fix them then rerun until 0 errors."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "http(s):// URL or path to a .html file (relative to the working "
                        "directory or absolute)."
                    ),
                },
                "wait_selector": {
                    "type": "string",
                    "description": (
                        "CSS selector to wait for before reading the page (optional)."
                    ),
                },
                "count_selectors": {
                    "type": "string",
                    "description": (
                        "CSS selectors to count, comma-separated (e.g. "
                        "'.cell,.flag') - to verify that elements are actually rendered."
                    ),
                },
            },
            "required": ["url"],
        },
        run=run,
    )


def _wait_for_port(proc: subprocess.Popen, host: str, port: int, timeout: int) -> bool:
    """Attend qu'un port TCP accepte une connexion (serveur prêt). Renvoie False si le
    delai expire OU si le process serveur meurt avant (crash au démarrage)."""
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False  # le serveur s'est arrête tout seul -> démarrage en echec
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def make_serve_and_check(workspace_dir: str) -> ToolSpec:
    """Outil serve_and_check : démarre un serveur en arrière-plan, attend son port, charge
    la page (comme check_page), PUIS tue tout l'arbre de process. run_shell ne peut pas
    garder un serveur vivant (il le tue au timeout) et check_page seul n'a rien a viser tant
    que rien n'ecoute : cet outil ferme ce trou pour Next.js/Vite/Flask."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        _reap_expired_servers()  # filet : referme les serveurs oublies > TTL
        action = (args.get("action") or "start").strip().lower()

        # --- action='stop' : ferme un serveur laissé vivant (ou tous), sans command/url ---
        if action == "stop":
            return _stop_servers((args.get("id") or "").strip() or None)

        command = (args.get("command") or "").strip()
        if not command:
            raise ToolError(
                "argument 'command' manquant (commande qui démarre le serveur, ex. "
                "'npm run dev -- --port 3000'). Pour ARRETER un serveur, action='stop' (+ id)."
            )
        target = (args.get("url") or "").strip()
        if not target:
            raise ToolError(
                "argument 'url' manquant (URL du serveur a vérifier, "
                "ex. 'http://127.0.0.1:3000')"
            )
        if not target.lower().startswith(("http://", "https://")):
            raise ToolError("'url' doit etre une URL http(s):// du serveur local")
        # Barriere de securite (comme run_shell) : commande destructrice refusee avant lancement.
        if _is_hard_denied(command, []):
            raise ToolError("commande interdite par la politique de securite")
        # localhost resout en ::1 (IPv6) chez certains serveurs lies en IPv4 seulement (et
        # l'inverse) -> faux negatifs. On normalise sur 127.0.0.1 pour l'attente ET la verif.
        url = target.replace("://localhost", "://127.0.0.1")
        reason = _browser_http_blocked(url)
        if reason:
            raise ToolError(f"url refusee : {reason}")
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        cwd = (args.get("cwd") or "").strip()
        workdir = _resolve_in_root(root, cwd) if cwd else root
        if not Path(workdir).is_dir():
            raise ToolError(f"dossier de travail introuvable : {workdir}")

        try:
            ready_timeout = int(args.get("ready_timeout") or 45)
        except (TypeError, ValueError):
            ready_timeout = 45
        ready_timeout = max(5, min(ready_timeout, 120))

        wait_selector = (args.get("wait_selector") or "").strip() or None
        count_selectors = [
            s.strip()
            for s in (args.get("count_selectors") or "").split(",")
            if s.strip()
        ]

        # Un serveur suivi tourne DEJA sur cette url (lance a un tour precedent) ? On ne
        # relance PAS (le double bind echouerait) : on re-vérifie simplement la page demandée
        # sur le serveur vivant. C'est le cas « tester une 2e page du meme site ».
        existing = _find_server_by_url(url)
        if existing:
            report = _render_page(url, wait_selector, count_selectors)
            return untrusted(
                f"serveur deja actif (id={existing}) sur {url} - verification :\n{report}"
                f"\n\n{_alive_hint(existing)}",
                f"page {url}",
            )

        # Serveur DETACHE : sa sortie va dans un fichier temporaire. On NE communicate() PAS
        # (il ne rend jamais la main) : on poll le port.
        fd, logpath = tempfile.mkstemp(prefix="loom-serve-", suffix=".log")
        os.close(fd)
        logf = open(logpath, "w", encoding="utf-8", errors="replace")
        proc = None
        handed_off = False  # True une fois logf confié au registre (serveur vivant)
        try:
            popen_kwargs: dict = {}
            if not detect().is_windows:
                popen_kwargs["start_new_session"] = True
            try:
                proc = subprocess.Popen(
                    _shell_argv(command),
                    cwd=str(workdir),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    **popen_kwargs,
                )
            except OSError as exc:
                raise ToolError(f"impossible de lancer le serveur : {exc}") from exc

            ready = _wait_for_port(proc, host, port, ready_timeout)
            if ready:
                report = _render_page(url, wait_selector, count_selectors)
                # On LAISSE le serveur vivre (registre) : le modele peut tester d'autres pages,
                # puis le fermer avec action='stop'. Le log reste ouvert (le serveur y ecrit).
                sid = _register_server(proc, url, logpath, logf)
                handed_off = True
                body = (
                    f"serveur démarre, port {port} prêt - verification de {url} :\n{report}"
                    f"\n\n{_alive_hint(sid)}"
                )
                return untrusted(body, f"page {url}")

            # Echec de démarrage : diag depuis le log, puis on tue et on nettoie.
            try:
                logf.flush()
                tail = Path(logpath).read_text(encoding="utf-8", errors="replace")[
                    -1200:
                ]
            except OSError:
                tail = ""
            if proc.poll() is not None:
                diag = (
                    f"le serveur s'est ARRETE tout seul (exit {proc.returncode}) avant "
                    f"d'ecouter sur {host}:{port} - démarrage en echec."
                )
            else:
                diag = (
                    f"le port {host}:{port} n'a pas repondu en {ready_timeout}s. Verifie que la "
                    "commande démarre bien un serveur sur CE port (option --port), ou augmente "
                    "ready_timeout."
                )
            return untrusted(
                f"serve_and_check : {diag}\n--- sortie serveur ---\n{tail or '(aucune sortie)'}",
                f"page {url}",
            )
        finally:
            # Sur TOUT chemin non-remis (erreur OU echec de démarrage) : tue le proc,
            # ferme le log et supprime le temp. Le chemin 'ready' a transfere la propriete
            # au registre (handed_off) -> on ne touche pas a logf/logpath.
            if not handed_off:
                if proc is not None:
                    try:
                        _kill_tree(proc)
                    except Exception:  # noqa: BLE001 - best-effort
                        pass
                try:
                    logf.close()
                except OSError:
                    pass
                try:
                    os.unlink(logpath)
                except OSError:
                    pass

    return ToolSpec(
        name="serve_and_check",
        description=(
            "Manages the LIFECYCLE of a local server to prove that a SERVER-backed app "
            "(Next.js, Vite, Flask...) renders/works. run_shell CANNOT keep a server alive "
            "(it kills it at the timeout) and NEVER start a server yourself via "
            "Start-Process/start (that opens the .ps1 in an editor and doesn't survive): ALWAYS "
            "go through this tool.\n"
            "action='start' (default): starts 'command' in the background, waits for 'url' to "
            "respond, loads the page in a headless browser (console errors, elements, text) "
            "and LEAVES THE SERVER ALIVE -> you can then test OTHER pages of the same server "
            "(check_page/check_interactive, or serve_and_check on another url). "
            "action='stop': stops the server with the given id (or ALL if no id) -> do it "
            "WHEN YOU HAVE YOUR ANSWER. For a STATIC .html page (no server), prefer "
            "check_page."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop"],
                    "description": (
                        "'start' (default) starts+checks+leaves alive; 'stop' closes a "
                        "server left alive (via id, or all if id absent)."
                    ),
                },
                "id": {
                    "type": "string",
                    "description": (
                        "For action='stop': id of the server to close (returned by 'start', "
                        "e.g. 'srv1'). Absent = closes ALL servers left alive."
                    ),
                },
                "command": {
                    "type": "string",
                    "description": (
                        "action='start': command that starts the server (e.g. 'npm run dev "
                        "-- --port 3000'). Launched in the background, left alive."
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "action='start': http(s):// URL where to reach the server (e.g. "
                        "'http://127.0.0.1:3000')."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Directory where to run the command (relative to the working "
                        "directory or absolute). Default: the working directory."
                    ),
                },
                "wait_selector": {
                    "type": "string",
                    "description": "CSS selector to wait for before reading the page (optional).",
                },
                "count_selectors": {
                    "type": "string",
                    "description": (
                        "CSS selectors to count, comma-separated (e.g. '.card,nav')."
                    ),
                },
                "ready_timeout": {
                    "type": "integer",
                    "description": (
                        "Max seconds to wait for the port to respond (default 45, max 120)."
                    ),
                },
            },
            "required": [],
        },
        run=run,
    )


def _eval_expect(page, expect: dict) -> tuple[bool, str]:
    """Evalue une post-condition DANS le DOM courant. Renvoie (ok, observe)."""
    sel = (expect.get("selector") or "").strip()
    check = (expect.get("check") or "").strip().lower()
    val = expect.get("value")
    if not sel or not check:
        return True, "(aucune post-condition)"
    try:
        if check == "count":
            try:
                target = int(val)
            except (TypeError, ValueError):
                return (
                    False,
                    f"{sel} count : 'value' doit etre un entier (recu {val!r})",
                )
            n = len(page.query_selector_all(sel))
            cmp = (expect.get("cmp") or "min").lower()
            ok = n >= target if cmp == "min" else n == target
            return ok, f"{sel} x{n} (attendu {cmp} {target})"
        el = page.query_selector(sel)
        if check == "absent":
            return el is None, f"{sel} {'absent' if el is None else 'present'}"
        if el is None:
            return False, f"{sel} introuvable"
        if check == "class":
            classes = (el.get_attribute("class") or "").split()
            return str(val) in classes, f"{sel} classes={classes}"
        if check == "text":
            txt = el.inner_text()
            return str(val).lower() in txt.lower(), f"{sel} texte~{txt[:60]!r}"
        return False, f"check inconnu '{check}'"
    except Exception as exc:  # noqa: BLE001 - une eval ratee = step en echec, pas un crash
        return False, f"evaluation echouee : {str(exc)[:120]}"


def _run_step(page, step: dict) -> dict:
    """Joue UNE action puis evalue sa post-condition. Ne leve jamais."""
    op = (step.get("op") or "none").strip().lower()
    selector = (step.get("selector") or "").strip()
    expect = step.get("expect") if isinstance(step.get("expect"), dict) else {}
    # Une etape n'est une PREUVE que si elle porte une post-condition reelle (selector +
    # check). Sans ca, l'etape passe « pour rien » -> traquee pour interdire la preuve vide.
    asserted = bool(
        (expect.get("selector") or "").strip() and (expect.get("check") or "").strip()
    )
    res = {
        "op": op,
        "selector": selector,
        "ok": False,
        "asserted": asserted,
        "observed": "",
    }
    try:
        if op == "click":
            page.click(selector, timeout=4000)
        elif op == "rightclick":
            page.click(selector, button="right", timeout=4000)
        elif op == "dblclick":
            page.dblclick(selector, timeout=4000)
        elif op == "hover":
            page.hover(selector, timeout=4000)
        elif op == "type":
            page.fill(selector, step.get("text") or "", timeout=4000)
        elif op in ("none", "load", ""):
            pass
        else:
            res["observed"] = f"op inconnu '{op}'"
            return res
        page.wait_for_timeout(300)  # laisse le JS reagir a l'action
    except Exception as exc:  # noqa: BLE001 - action ratee = step en echec
        res["observed"] = f"action '{op}' echouee : {str(exc)[:120]}"
        return res
    res["ok"], res["observed"] = _eval_expect(page, expect)
    return res


def run_interactive(workspace_dir: str, target: str, steps: list[dict]) -> dict:
    """Charge une page, JOUE `steps` (clics/saisie reels) et evalue une post-condition DOM
    apres chaque action. Renvoie un dict STRUCTURE lu par le harnais (jamais par le modele) :
    {url, ok, console_errors, steps:[{op,selector,ok,observed}], error}. `ok` global = 0 erreur
    console ET toutes les etapes ok. Ne leve jamais (toute panne -> ok=False + error)."""
    root = Path(workspace_dir)
    try:
        url = _browser_url(root, target)
    except ToolError as exc:
        return {
            "url": target,
            "ok": False,
            "error": str(exc),
            "console_errors": [],
            "steps": [],
        }
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "url": url,
            "ok": False,
            "error": _INSTALL_HINT,
            "console_errors": [],
            "steps": [],
        }

    console: list[tuple[str, str]] = []
    page_errors: list[str] = []
    results: list[dict] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("console", lambda m: console.append((m.type, m.text)))
            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.goto(url, wait_until="load", timeout=15000)
            page.wait_for_timeout(800)
            for step in steps:
                results.append(_run_step(page, step))
            browser.close()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "Executable doesn't exist" in msg or "playwright install" in msg:
            return {
                "url": url,
                "ok": False,
                "error": _INSTALL_HINT,
                "console_errors": [],
                "steps": results,
            }
        return {
            "url": url,
            "ok": False,
            "error": f"echec du chargement : {msg[:200]}",
            "console_errors": [],
            "steps": results,
        }

    errors = [t for (k, t) in console if k == "error"] + page_errors
    asserted = sum(1 for r in results if r.get("asserted"))
    steps_ok = bool(results) and all(r["ok"] for r in results)
    # PREUVE NON VIDE : `ok` global exige au moins une post-condition reelle qui passe.
    # Sinon une suite de clics sans `expect` se declarerait « jouable » a tort.
    ok = (not errors) and steps_ok and asserted > 0
    note = (
        ""
        if asserted
        else "preuve vide : aucune etape n'a de post-condition reelle (expect)"
    )
    return {
        "url": url,
        "ok": ok,
        "console_errors": errors[:8],
        "steps": results,
        "asserted_steps": asserted,
        "note": note,
        "error": "",
    }


def make_check_interactive(workspace_dir: str) -> ToolSpec:
    """Outil check_interactive : joue une sequence d'actions sur une page et vérifie le DOM
    apres chaque action. Pour PROUVER qu'une page est jouable (pas seulement « 0 erreur »)."""

    def run(args: dict) -> str:
        target = (args.get("url") or "").strip()
        if not target:
            raise ToolError("argument 'url' manquant (page HTML a tester)")
        steps = args.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ToolError(
                "argument 'steps' : liste non vide d'actions {op, selector, expect}"
            )
        res = run_interactive(workspace_dir, target, steps)
        lines = [f"page : {res['url']}"]
        if res.get("error"):
            lines.append(f"erreur: {res['error']}")
        lines.append(f"console : {len(res.get('console_errors', []))} erreur(s)")
        for e in res.get("console_errors", [])[:5]:
            lines.append(f"  [erreur] {e[:160]}")
        for i, s in enumerate(res.get("steps", []), 1):
            mark = "ok" if s["ok"] else "ECHEC"
            lines.append(
                f"  etape {i} [{mark}] {s['op']} {s['selector']} -> {s['observed']}"
            )
        if res.get("note"):
            lines.append(f"NOTE : {res['note']}")
        if res["ok"]:
            verdict = "toutes les actions passent, 0 erreur"
        elif res.get("note"):
            verdict = (
                "preuve INSUFFISANTE - ajoute un `expect` testable (selector + check) sur "
                "au moins une etape pour prouver le comportement"
            )
        else:
            verdict = "au moins une action/post-condition echoue"
        lines.append("VERDICT : " + verdict)
        # Le texte observe vient d'une page (potentiellement hostile) : donnee, pas des ordres.
        return untrusted("\n".join(lines), f"page {res['url']}")

    return ToolSpec(
        name="check_interactive",
        description=(
            "Proves that an HTML page is PLAYABLE: plays a sequence of real actions "
            "(click, rightclick, dblclick, hover, type) on CSS selectors and checks, "
            "AFTER each action, a post-condition in the DOM. Goes further than check_page "
            "(which only loads). Use it to prove 'clicking a cell reveals it', 'right-click "
            "places a flag', 'restart resets'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTML page (.html path or URL).",
                },
                "steps": {
                    "type": "array",
                    "description": "Actions to play in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": [
                                    "click",
                                    "rightclick",
                                    "dblclick",
                                    "hover",
                                    "type",
                                    "none",
                                ],
                            },
                            "selector": {
                                "type": "string",
                                "description": "CSS target of the action.",
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to enter (op=type).",
                            },
                            "expect": {
                                "type": "object",
                                "description": "DOM post-condition after the action.",
                                "properties": {
                                    "selector": {"type": "string"},
                                    "check": {
                                        "type": "string",
                                        "enum": ["count", "class", "text", "absent"],
                                    },
                                    "value": {"type": "string"},
                                    "cmp": {"type": "string", "enum": ["min", "eq"]},
                                },
                            },
                        },
                        "required": ["op"],
                    },
                },
            },
            "required": ["url", "steps"],
        },
        run=run,
    )
