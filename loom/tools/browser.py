# loom/tools/browser.py
"""Outil check_page : donne des YEUX à Loom sur une page web rendue (Playwright headless).

Sans ça, le modèle édite du HTML/JS à l'aveugle et confabule « ça marche » : il ne voit
ni l'erreur console qui plante le jeu, ni que la grille ne s'affiche pas. check_page charge
la page dans Chromium headless, EXÉCUTE le JS, et renvoie les ERREURS CONSOLE, le compte
d'éléments (count_selectors) et un extrait du texte visible — le modèle VOIT le crash / la
grille manquante et peut itérer jusqu'à « 0 erreur ». Lazy-import de playwright : message
clair et actionnable si la lib (ou le navigateur) n'est pas installée.

Le contenu d'une page est EXTERNE/non fiable -> renvoyé via untrusted() : donnée à
analyser, jamais des ordres (une page peut contenir « ignore tes consignes »).
"""

from __future__ import annotations

from pathlib import Path

from loom.tools.base import ToolError, ToolSpec, _resolve_in_root
from loom.tools.trust import untrusted

_INSTALL_HINT = (
    "playwright non installé. Lance une fois : `uv add playwright` puis "
    "`uv run playwright install chromium`."
)


def make_check_page(workspace_dir: str) -> ToolSpec:
    """Outil check_page borné au workspace pour les chemins relatifs (absolus acceptés)."""
    root = Path(workspace_dir)

    def run(args: dict) -> str:
        target = (args.get("url") or "").strip()
        if not target:
            raise ToolError(
                "argument 'url' manquant (URL http(s):// OU chemin d'un fichier .html)"
            )
        # URL telle quelle, sinon chemin local -> file:// (Path.as_uri()).
        if target.startswith(("http://", "https://", "file://")):
            url = target
        else:
            path = _resolve_in_root(root, target)
            if not path.exists():
                raise ToolError(f"fichier introuvable : {target}")
            url = path.as_uri()

        wait_selector = (args.get("wait_selector") or "").strip() or None
        count_selectors = [
            s.strip()
            for s in (args.get("count_selectors") or "").split(",")
            if s.strip()
        ]

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ToolError(_INSTALL_HINT) from exc

        console: list[tuple[str, str]] = []
        page_errors: list[str] = []
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                page.on("console", lambda m: console.append((m.type, m.text)))
                page.on("pageerror", lambda e: page_errors.append(str(e)))
                page.goto(url, wait_until="load", timeout=15000)
                if wait_selector:
                    try:
                        page.wait_for_selector(wait_selector, timeout=5000)
                    except Exception:  # noqa: BLE001 - absence = info, pas un crash
                        pass
                page.wait_for_timeout(1200)  # laisse le JS d'init s'exécuter
                title = page.title()
                body = page.query_selector("body")
                body_text = body.inner_text()[:2000] if body else ""
                counts = {
                    sel: len(page.query_selector_all(sel)) for sel in count_selectors
                }
                browser.close()
        except Exception as exc:  # noqa: BLE001 - navigateur absent / page injoignable
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                raise ToolError(_INSTALL_HINT) from exc
            raise ToolError(f"échec du chargement de {url} : {msg[:200]}") from exc

        errors = [t for (k, t) in console if k == "error"] + page_errors
        warnings = [t for (k, t) in console if k == "warning"]
        lines = [
            f"page : {title!r} ({url})",
            f"console : {len(errors)} erreur(s), {len(warnings)} warning(s)",
        ]
        for e in errors[:8]:
            lines.append(f"  [erreur] {e[:200]}")
        if counts:
            lines.append(
                "éléments : " + " · ".join(f"{s} ×{n}" for s, n in counts.items())
            )
        visible = " ".join(body_text.split())
        if visible:
            lines.append(f"texte visible : {visible[:400]}")
        if not errors and not page_errors:
            lines.append("(aucune erreur console — la page s'est chargée et exécutée)")
        return untrusted("\n".join(lines), f"page {url}")

    return ToolSpec(
        name="check_page",
        description=(
            "Charge une page web (URL http(s):// OU chemin d'un fichier .html local) dans "
            "un navigateur headless, EXÉCUTE son JavaScript, et renvoie : les ERREURS de "
            "la console, le nombre d'éléments correspondant à count_selectors (ex "
            "'.cell,#board'), et un extrait du texte visible. SERS-T'EN pour VÉRIFIER "
            "qu'une page HTML que tu viens d'écrire s'affiche et fonctionne (0 erreur "
            "console, éléments attendus présents) AU LIEU de supposer que ça marche. Si des "
            "erreurs apparaissent, corrige puis relance check_page jusqu'à 0 erreur."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "URL http(s):// ou chemin d'un fichier .html (relatif au dossier "
                        "de travail ou absolu)."
                    ),
                },
                "wait_selector": {
                    "type": "string",
                    "description": (
                        "Sélecteur CSS à attendre avant de lire la page (optionnel)."
                    ),
                },
                "count_selectors": {
                    "type": "string",
                    "description": (
                        "Sélecteurs CSS à compter, séparés par des virgules (ex "
                        "'.cell,.flag') — pour vérifier que des éléments sont bien rendus."
                    ),
                },
            },
            "required": ["url"],
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
    if target.startswith(("http://", "https://", "file://")):
        url = target
    else:
        path = _resolve_in_root(root, target)
        if not path.exists():
            return {
                "url": target,
                "ok": False,
                "error": f"fichier introuvable : {target}",
                "console_errors": [],
                "steps": [],
            }
        url = path.as_uri()
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
    """Outil check_interactive : joue une séquence d'actions sur une page et vérifie le DOM
    après chaque action. Pour PROUVER qu'une page est jouable (pas seulement « 0 erreur »)."""

    def run(args: dict) -> str:
        target = (args.get("url") or "").strip()
        if not target:
            raise ToolError("argument 'url' manquant (page HTML à tester)")
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
            mark = "ok" if s["ok"] else "ÉCHEC"
            lines.append(
                f"  étape {i} [{mark}] {s['op']} {s['selector']} -> {s['observed']}"
            )
        if res.get("note"):
            lines.append(f"NOTE : {res['note']}")
        if res["ok"]:
            verdict = "toutes les actions passent, 0 erreur"
        elif res.get("note"):
            verdict = (
                "preuve INSUFFISANTE — ajoute un `expect` testable (selector + check) sur "
                "au moins une étape pour prouver le comportement"
            )
        else:
            verdict = "au moins une action/post-condition échoue"
        lines.append("VERDICT : " + verdict)
        return "\n".join(lines)

    return ToolSpec(
        name="check_interactive",
        description=(
            "Prouve qu'une page HTML est JOUABLE : joue une séquence d'actions réelles "
            "(click, rightclick, dblclick, hover, type) sur des sélecteurs CSS et vérifie, "
            "APRÈS chaque action, une post-condition dans le DOM. Va plus loin que check_page "
            "(qui ne fait que charger). Utilise-le pour prouver « cliquer une cellule la "
            "révèle », « clic droit pose un drapeau », « restart réinitialise »."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Page HTML (chemin .html ou URL).",
                },
                "steps": {
                    "type": "array",
                    "description": "Actions à jouer dans l'ordre.",
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
                                "description": "Cible CSS de l'action.",
                            },
                            "text": {
                                "type": "string",
                                "description": "Texte à saisir (op=type).",
                            },
                            "expect": {
                                "type": "object",
                                "description": "Post-condition DOM après l'action.",
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
