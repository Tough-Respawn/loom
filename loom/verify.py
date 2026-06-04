# loom/verify.py
"""Vérificateur déterministe : exécute des checks RÉELS sur les artefacts produits
et renvoie un rapport de défauts factuel (PAS une opinion du LLM).

Pierre angulaire de la boucle fermée observe→agis→VÉRIFIE (cf. docs/harness-strategy.md).
Offline-friendly : `compile()` (Python) et `json` sont natifs ; `node --check` (syntaxe
JS) n'est lancé que si `node` est présent. Tous les checks sont READ-ONLY : `--check` et
`compile()` n'EXÉCUTENT pas le code, ils le parsent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Détection déterministe des ES modules : ils ne se chargent PAS en file:// (CORS) dans
# aucun navigateur -> pour un jeu auto-suffisant (double-clic index.html), c'est une
# erreur, pas un choix. On la signale comme défaut actionnable (cf. plan-harness-robustesse).
_HTML_MODULE_RE = re.compile(r"""type\s*=\s*["']?module""", re.IGNORECASE)
_JS_ESM_RE = re.compile(
    r"""^\s*(?:export\s|export\{|import\s+[\w*{].*?\sfrom\s|import\s+["'])""",
    re.MULTILINE,
)


def _check_no_es_modules(files: list[Path]) -> list["Defect"]:
    """Refuse les ES modules (import/export, <script type=module>) : non chargeables en
    file://. Défaut clair -> la boucle de fix convertit en scripts classiques globaux."""
    defects: list[Defect] = []
    for f in files:
        ext = f.suffix.lower()
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if ext in (".html", ".htm") and _HTML_MODULE_RE.search(text):
            defects.append(
                Defect(
                    f.name,
                    "modules",
                    "<script type=module> INTERDIT : les ES modules ne se chargent pas "
                    "en file:// (CORS). Charge le JS via <script src> CLASSIQUE.",
                )
            )
        if ext in (".js", ".mjs") and _JS_ESM_RE.search(text):
            defects.append(
                Defect(
                    f.name,
                    "modules",
                    "import/export ES INTERDIT (incompatible file://). Expose les "
                    "fonctions en GLOBAL (window.X ou fonctions globales), sans "
                    "import/export, et charge-les via <script src> classiques.",
                )
            )
    return defects


@dataclass
class Defect:
    """Un défaut localisé et prouvé (un défaut = une cible à corriger)."""

    location: str  # "game.js:199" ou "data.json"
    kind: str  # 'syntax' | 'json' | 'error'
    evidence: str


@dataclass
class VerifyReport:
    ok: bool
    defects: list[Defect] = field(default_factory=list)


def _check_python(path: Path) -> list[Defect]:
    """Compile (sans exécuter) un .py : capture les SyntaxError."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        compile(src, str(path), "exec")
    except SyntaxError as exc:
        line = exc.lineno or "?"
        return [Defect(f"{path.name}:{line}", "syntax", (exc.msg or "SyntaxError"))]
    return []


def _check_json(path: Path) -> list[Defect]:
    try:
        json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return [Defect(f"{path.name}:{exc.lineno}", "json", exc.msg)]
    return []


def _check_js(path: Path) -> list[Defect]:
    """`node --check` (parse, n'exécute pas). No-op si node absent (offline sans node)."""
    node = shutil.which("node")
    if not node:
        return []
    try:
        proc = subprocess.run(
            [node, "--check", str(path)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [Defect(path.name, "error", str(exc)[:200])]
    if proc.returncode != 0:
        return [Defect(path.name, "syntax", (proc.stderr or "").strip()[:300])]
    return []


def _check_web(html_path: Path) -> list[Defect]:
    """Charge la page + ses scripts via jsdom (node) : capture les erreurs RUNTIME
    (TypeError…) et vérifie que le plateau se rend (conteneur non vide). C'est l'œil
    « ça tourne » que `node --check` n'a pas. No-op si node/jsdom absents (offline)."""
    node = shutil.which("node")
    script = Path(__file__).with_name("verify_web.js")
    if not node or not script.exists():
        return []
    try:
        proc = subprocess.run(
            [node, str(script), str(html_path)],
            cwd=str(script.parent.parent),  # racine repo -> trouve node_modules/jsdom
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return [Defect(html_path.name, "error", str(exc)[:200])]
    out = (proc.stdout or "").strip()
    if not out:
        # jsdom non installé => on dégrade silencieusement (pas de faux défaut).
        if "Cannot find module" in (proc.stderr or ""):
            return []
        return [
            Defect(html_path.name, "error", f"verify_web: {(proc.stderr or '')[:200]}")
        ]
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return [
            Defect(html_path.name, "error", f"verify_web sortie invalide: {out[:200]}")
        ]
    return [
        Defect(
            d.get("location", html_path.name),
            d.get("kind", "runtime"),
            d.get("evidence", ""),
        )
        for d in data.get("defects", [])
    ]


_CHECKERS = {
    ".py": _check_python,
    ".json": _check_json,
    ".js": _check_js,
    ".mjs": _check_js,
}

# Dossiers jamais scannés en récursif : ils contiennent des milliers de fichiers
# (et `node --check` spawnerait un process par .js → étouffement). Cf. le hang
# observé quand le workspace = un dossier géant (ex. C:\Users\...\Documents).
_SKIP_DIRS = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".next",
        ".cache",
    }
)
# Plafond de fichiers vérifiables sur un scan de DOSSIER : au-delà, on refuse plutôt
# que de lancer des centaines de sous-process (garde-fou anti-blocage de l'outil verify).
_MAX_DIR_SCAN = 300


def verify_path(path: str) -> VerifyReport:
    """Vérifie un fichier, ou tous les fichiers vérifiables d'un dossier (récursif).

    `ok=True` si aucun défaut détecté. Les fichiers non vérifiables (html, css…) sont
    ignorés. Les dossiers lourds (_SKIP_DIRS) sont exclus et un plafond (_MAX_DIR_SCAN)
    évite de scanner un arbre géant. Ne lève jamais : une erreur de check devient un Defect.
    """
    p = Path(path)
    files: list[Path] = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        capped = False
        for ext in _CHECKERS:
            for f in p.rglob(f"*{ext}"):
                if _SKIP_DIRS.intersection(f.parts):
                    continue
                files.append(f)
                if len(files) > _MAX_DIR_SCAN:
                    capped = True
                    break
            if capped:
                break
        if capped:
            return VerifyReport(
                ok=False,
                defects=[
                    Defect(
                        p.name or str(p),
                        "error",
                        f"dossier trop vaste (> {_MAX_DIR_SCAN} fichiers vérifiables) — "
                        "cible un sous-dossier ou les fichiers précis, pas la racine",
                    )
                ],
            )
    defects: list[Defect] = []
    for f in sorted(set(files)):
        checker = _CHECKERS.get(f.suffix.lower())
        if checker is None:
            continue
        try:
            defects.extend(checker(f))
        except Exception as exc:  # noqa: BLE001 - un check ne casse jamais le rapport
            defects.append(Defect(f.name, "error", str(exc)[:200]))
    # Vérif RUNTIME web : si un index.html est présent, charger la page (jsdom) pour
    # prouver que ça TOURNE (pas juste que la syntaxe est valide).
    html_entry = None
    if p.is_file() and p.suffix.lower() in (".html", ".htm"):
        html_entry = p
    elif p.is_dir() and (p / "index.html").exists():
        html_entry = p / "index.html"
    # check ES modules sur tous les fichiers (HTML inclus) ; check runtime web seulement
    # si une page est présente.
    targets = sorted(set(files))
    if html_entry is not None:
        targets.append(html_entry)
    defects.extend(_check_no_es_modules(targets))
    if html_entry is not None:
        defects.extend(_check_web(html_entry))
    return VerifyReport(ok=not defects, defects=defects)


def verify_files(paths: list[str]) -> VerifyReport:
    """Vérifie une LISTE EXPLICITE de fichiers (syntaxe), + check runtime web si un
    index.html en fait partie. Borné au set fourni : ne rglob JAMAIS un dossier (donc
    ne peut pas étouffer sur un arbre géant ni spawn node par fichier). C'est le
    vérificateur du hard-gate P0.4 — on sait exactement ce que le développeur a écrit."""
    defects: list[Defect] = []
    html_entries: list[Path] = []
    for raw in paths:
        f = Path(raw)
        if f.suffix.lower() in (".html", ".htm"):
            html_entries.append(f)
        checker = _CHECKERS.get(f.suffix.lower())
        if checker is None:
            continue
        try:
            defects.extend(checker(f))
        except Exception as exc:  # noqa: BLE001 - un check ne casse jamais le rapport
            defects.append(Defect(f.name, "error", str(exc)[:200]))
    defects.extend(_check_no_es_modules([Path(raw) for raw in paths]))
    # Check runtime web sur CHAQUE page (site multi-pages) : sinon un asset cassé ou une
    # erreur JS propre à blog.html/dashboard.html passe sous le radar (seul index était vu).
    for h in html_entries:
        if h.exists():
            defects.extend(_check_web(h))
    return VerifyReport(ok=not defects, defects=defects)


def verify_syntax_content(content: str, filename: str) -> VerifyReport:
    """Vérifie la SYNTAXE d'un CONTENU sans dépendre d'une écriture disque préalable :
    écrit un fichier TEMPORAIRE de même extension et applique le checker. Pour le portique
    syntaxe au fil de l'eau (le contenu peut ne pas encore être sur le disque cible, et le
    `write` du caller peut être indirect). Les défauts pointent le vrai nom de fichier."""
    suffix = Path(filename).suffix.lower()
    checker = _CHECKERS.get(suffix)
    if checker is None:
        return VerifyReport(ok=True)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        defects = checker(Path(tmp))
        real = Path(filename).name
        for d in defects:
            d.location = d.location.replace(Path(tmp).name, real)
        return VerifyReport(ok=not defects, defects=defects)
    except Exception as exc:  # noqa: BLE001 - un check ne casse jamais le rapport
        return VerifyReport(
            ok=False, defects=[Defect(filename, "error", str(exc)[:200])]
        )
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def verify_syntax_file(path: str) -> VerifyReport:
    """Vérifie la SYNTAXE d'un seul fichier (sans check runtime web). Pour le
    contrôle d'intégrité post-write (P1.3) : une troncature de contenu devient un
    défaut ciblé au lieu d'un faux succès."""
    p = Path(path)
    checker = _CHECKERS.get(p.suffix.lower())
    if checker is None:
        return VerifyReport(ok=True)
    try:
        defects = checker(p)
    except Exception as exc:  # noqa: BLE001
        defects = [Defect(p.name, "error", str(exc)[:200])]
    return VerifyReport(ok=not defects, defects=defects)


def format_report(report: VerifyReport) -> str:
    """Rend le rapport en texte exploitable par le modèle (un défaut par ligne)."""
    if report.ok:
        return "VERIFY OK : aucun défaut détecté (syntaxe JS / Python / JSON)."
    lines = [f"VERIFY: {len(report.defects)} défaut(s) détecté(s) :"]
    for d in report.defects:
        lines.append(f"- {d.location} [{d.kind}] {d.evidence}")
    return "\n".join(lines)
