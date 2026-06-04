# loom/parallel.py
"""Génération PARALLÈLE de fichiers : décompose une tâche web en fichiers, puis
génère leur contenu via des requêtes CONCURRENTES que llama-server batche en continu
(GPU saturé ~100 tok/s sur RTX 2060, cf. docs/perf-gpu.md).

Pourquoi : en séquentiel, le petit modèle sur-raisonne et P1.1 sérialise les écritures
→ il ne finit jamais les N fichiers. En parallèle, chaque fichier est généré par un
appel FOCALISÉ (thinking off, sortie = contenu brut), garanti complet, et la cohérence
inter-fichiers vient de notes d'architecture partagées (`design`).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from loom.prompts import (
    EDIT_SYSTEM,
    GEN_SYSTEM,
    PLAN_SYSTEM,
    REVIEW_SYSTEM,
    edit_prompt,
    file_prompt,
    fix_prompt,
    plan_prompt,
    review_prompt,
)
from loom.tools.base import ToolError
from loom.tools.fs import make_edit_file

# Compat : les prompts vivent dans loom/prompts/ (source de vérité). On ré-expose les
# anciens noms privés pour l'orchestrateur et les tests qui les importent d'ici.
_PLAN_SYS = PLAN_SYSTEM
_GEN_SYS = GEN_SYSTEM
_EDIT_SYS = EDIT_SYSTEM
_REVIEW_SYS = REVIEW_SYSTEM
_plan_prompt = plan_prompt
_file_prompt = file_prompt
_fix_prompt = fix_prompt
_edit_prompt = edit_prompt


@dataclass
class FileSpec:
    path: str
    role: str


@dataclass
class PlannedFile:
    """Un fichier planifié + son mode de génération (dérivé, jamais parsé du plan)."""

    spec: FileSpec
    mode: str  # 'create' | 'patch' | 'rewrite'


def derive_modes(specs: list[FileSpec], workspace: str, verifier) -> list[PlannedFile]:
    """Dérive le mode de chaque fichier SANS LLM (cf. spec §5) :
    - absent du disque                 -> create
    - présent et verify échoue déjà    -> rewrite (déclencheur OBJECTIF)
    - présent et verify OK (ou None)    -> patch  (le moins destructeur)

    `verifier(list[str_abspath]) -> VerifyReport | None`. N'est appelé que pour un
    fichier EXISTANT (un fichier absent est create sans vérification).
    """
    root = Path(workspace)
    planned: list[PlannedFile] = []
    for spec in specs:
        abspath = root / spec.path
        if not abspath.exists():
            mode = "create"
        else:
            report = verifier([str(abspath)])
            mode = "rewrite" if (report is not None and not report.ok) else "patch"
        planned.append(PlannedFile(spec=spec, mode=mode))
    return planned


def cap_rewrites(
    planned: list[PlannedFile], workspace: str, *, max_lines: int = 200
) -> list[PlannedFile]:
    """Dégrade rewrite -> patch pour les fichiers existants > max_lines (cf. spec §5) :
    réécrire intégralement un gros fichier sur un 4B risque la troncature."""
    root = Path(workspace)
    out: list[PlannedFile] = []
    for pf in planned:
        if pf.mode == "rewrite":
            abspath = root / pf.spec.path
            try:
                n_lines = (
                    abspath.read_text(encoding="utf-8", errors="replace").count("\n")
                    + 1
                )
            except OSError:
                n_lines = 0
            if n_lines > max_lines:
                out.append(PlannedFile(pf.spec, "patch"))
                continue
        out.append(pf)
    return out


def compute_budget(
    context: int, n_parallel: int, n_files: int, *, reserve_prompt_tokens: int = 2048
):
    """Dérive (max_workers, gen_max_tokens, file_char_cap) du budget contexte du serveur,
    pour NE JAMAIS déborder le pool KV partagé (kv_unified : -c divisé entre slots
    concurrents). Avant on hand-tunait -c/max_tokens/workers à chaque run → désormais
    calculé. cf. docs/plan-harness-robustesse.md.

    Idée : chaque requête concurrente consomme (prompt + génération) du pool. On borne la
    concurrence et la taille de génération pour que `workers × (prompt+gen) <= 0.9·context`.
    """
    context = max(2048, int(context or 8192))
    n_parallel = max(1, int(n_parallel or 1))
    n_files = max(1, int(n_files or 1))
    slot = max(1536, context // n_parallel)  # part de contexte par slot serveur
    # Fenêtre de génération = part de slot RESTANTE après la réserve prompt MESURÉE :
    # une réserve plus grande (prompt plus gros) laisse mécaniquement moins pour la gen.
    # On borne la part exploitée du slot à 6144 (plafond de fenêtre) AVANT de soustraire
    # la réserve, pour que la réserve influence toujours gen (cf. budget mesuré).
    gen_max_tokens = max(1024, min(slot, 6144) - reserve_prompt_tokens)
    per_req = reserve_prompt_tokens + gen_max_tokens
    fit = max(1, int(context * 0.9) // per_req)  # combien tiennent ensemble (marge 10%)
    max_workers = max(1, min(n_parallel, n_files, fit))
    file_char_cap = reserve_prompt_tokens * 4  # budget prompt en chars (~4 chars/token)
    return max_workers, gen_max_tokens, file_char_cap


def extract_code(text: str) -> str:
    """Enlève une éventuelle clôture markdown ```lang ... ``` ; sinon renvoie tel quel.

    Les modèles encadrent souvent le contenu malgré la consigne. On récupère le bloc
    s'il existe, on strip les espaces de bord, et on garantit un \\n final.
    """
    if text is None:
        return ""
    fence = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    body = body.strip("\n")
    return body + "\n" if body and not body.endswith("\n") else body


def _extract_json(text: str) -> dict:
    """Extrait le 1er objet JSON {...} d'une réponse (tolérant au texte autour)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("aucun objet JSON trouvé dans la réponse du planificateur")
    return json.loads(text[start : end + 1])


_FILENAME_RE = re.compile(r"[\w\-/]+\.(?:html|css|js|mjs|json)\b")


def _parse_plan(raw: str):
    """Parse le plan, TOLÉRANT à la variance du modèle. 3 niveaux :
    1) format délimité ===FILES=== (préféré, supporte les snippets de code) ;
    2) JSON {design, files} (ancien format) ;
    3) dernier recours : extraire les noms de fichiers (regex) du texte brut.
    Renvoie (design, list[FileSpec]). Ne lève jamais si un fichier est trouvable."""
    # 1) format délimité
    if "===FILES===" in raw:
        design, _, flist = raw.partition("===FILES===")
        design = design.split("===DESIGN===", 1)[-1].strip()
        specs: list[FileSpec] = []
        for line in flist.splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip().strip("`")
            if not line or line.startswith("==="):
                continue
            path, _, role = line.partition("|")
            path = path.strip().strip("`")
            if path and "." in path and " " not in path and len(path) < 120:
                specs.append(FileSpec(path=path, role=role.strip()))
        if specs:
            return design, specs
    # 2) JSON
    try:
        data = _extract_json(raw)
        specs = [
            FileSpec(path=f["path"], role=f.get("role", ""))
            for f in data.get("files", [])
            if f.get("path")
        ]
        if specs:
            return data.get("design", ""), specs
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    # 3) dernier recours : noms de fichiers trouvés dans le texte (ordre d'apparition)
    seen: list[str] = []
    for m in _FILENAME_RE.finditer(raw):
        p = m.group(0).lstrip("./")
        if p not in seen:
            seen.append(p)
    design = raw.split("===FILES===", 1)[0].replace("===DESIGN===", "").strip()
    return design, [FileSpec(path=p, role="") for p in seen]


def plan_files(
    client,
    task: str,
    *,
    model: str | None,
    max_tokens: int = 2048,
    explore_summary: str = "",
):
    """Un appel : produit un CONTRAT D'IMPLÉMENTATION détaillé + la liste de fichiers.

    Renvoie (design: str, specs: list[FileSpec]). `design` n'est PAS quelques lignes :
    c'est le contrat complet (state, signatures, snippets, boucle, clavier, rendu DOM,
    sélecteurs) qui permet de générer chaque fichier SÉPARÉMENT sans qu'ils divergent.
    Format DÉLIMITÉ (pas JSON) pour pouvoir inclure des snippets de code sans casser le
    parsing (le JSON devient invalide dès qu'on met du code dans le champ design).
    """
    raw = client.complete(
        [{"role": "user", "content": _plan_prompt(task, explore_summary)}],
        _PLAN_SYS,
        max_tokens=max_tokens,
        model=model,
        thinking=False,
        temperature=0.2,  # format strict -> peu de variance
    )
    return _parse_plan(raw)


def generate_one(
    client,
    design: str,
    spec: FileSpec,
    all_paths: list[str],
    *,
    model: str | None,
    max_tokens: int = 2048,
    file_char_cap: int | None = None,
    stories_text: str = "",
) -> tuple[str, str]:
    """Génère UN fichier (appel isolé, non-streamé, thinking off). Brique unitaire
    réutilisée par le batch ET par l'orchestrateur (events live par fichier).
    `file_char_cap` (optionnel) borne le `design` injecté (anti-overflow KV en fan-out).
    `stories_text` (optionnel) : les US qui touchent ce fichier, pour un dev guidé."""
    raw = client.complete(
        [
            {
                "role": "user",
                "content": _file_prompt(
                    spec, design, all_paths, file_char_cap, stories_text
                ),
            }
        ],
        _GEN_SYS,
        max_tokens=max_tokens,
        model=model,
        thinking=False,
    )
    return spec.path, extract_code(raw)


def edit_one(
    client,
    design: str,
    spec: FileSpec,
    workspace: str,
    all_paths: list[str] | None = None,
    *,
    model: str | None,
    max_tokens: int,
    file_char_cap: int,
    defects: str = "",
) -> tuple[str, str]:
    """PATCH ciblé en 2 temps DANS le harness (cf. spec §5) :
    1) read déterministe du fichier (borné file_char_cap/2, byte-exact pour le match) ;
    2) le modèle renvoie {old_string, new_string} ;
    3) application via make_edit_file (erreurs exploitables) ;
    4) FALLBACK generate_one borné si JSON invalide / introuvable / ambigu.
    Renvoie (path, CONTENU COMPLET relu) — contrat d'état identique à generate_one.
    """
    root = Path(workspace)
    abspath = root / spec.path

    def _fallback() -> tuple[str, str]:
        return generate_one(
            client,
            design,
            spec,
            all_paths or [spec.path],
            model=model,
            max_tokens=max_tokens,
            file_char_cap=file_char_cap,
        )

    try:
        content = abspath.read_bytes().decode("utf-8")  # byte-exact (match edit_file)
    except (OSError, UnicodeDecodeError):
        return _fallback()
    cap = max(256, file_char_cap // 2)
    injected = content if len(content) <= cap else content[:cap] + "\n…[tronqué]"
    raw = client.complete(
        [{"role": "user", "content": _edit_prompt(spec, design, injected, defects)}],
        _EDIT_SYS,
        max_tokens=max_tokens,
        model=model,
        thinking=False,
    )

    try:
        data = _extract_json(raw)
        old_string = data["old_string"]
        new_string = data["new_string"]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return _fallback()
    if (
        not isinstance(old_string, str)
        or not isinstance(new_string, str)
        or not old_string
    ):
        return _fallback()
    editor = make_edit_file(str(root))
    try:
        editor.run(
            {"path": spec.path, "old_string": old_string, "new_string": new_string}
        )
    except ToolError:
        return _fallback()
    # Relit byte-exact puis normalise les fins de ligne en \n : contrat d'état
    # identique à generate_one (qui renvoie du \n via extract_code), indépendant du
    # CRLF que l'OS/Git a pu introduire sur le fichier existant.
    final = abspath.read_bytes().decode("utf-8").replace("\r\n", "\n")
    return spec.path, final


def generate_files(
    client,
    design: str,
    specs: list[FileSpec],
    *,
    model: str | None,
    max_tokens: int = 2048,
    max_workers: int = 4,
) -> list[tuple[str, str]]:
    """Génère le contenu de chaque fichier EN PARALLÈLE (requêtes concurrentes →
    continuous batching). Renvoie [(path, content), ...] dans l'ordre des specs.
    """
    all_paths = [s.path for s in specs]
    if not specs:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(specs))) as ex:
        return list(
            ex.map(
                lambda s: generate_one(
                    client, design, s, all_paths, model=model, max_tokens=max_tokens
                ),
                specs,
            )
        )


def fix_one(
    client,
    design: str,
    spec: FileSpec,
    current: list[tuple[str, str]],
    defects: str,
    *,
    model: str | None,
    max_tokens: int = 2048,
    file_char_cap: int = 4000,
) -> tuple[str, str]:
    """Régénère UN fichier corrigé en lui donnant tous les fichiers actuels + les défauts.
    `file_char_cap` borne le contexte embarqué (dérivé du budget serveur, anti-overflow)."""
    raw = client.complete(
        [
            {
                "role": "user",
                "content": _fix_prompt(
                    spec, design, current, defects, file_char_cap=file_char_cap
                ),
            }
        ],
        _GEN_SYS,
        max_tokens=max_tokens,
        model=model,
        thinking=False,
    )
    return spec.path, extract_code(raw)


def fix_files(
    client,
    design: str,
    specs: list[FileSpec],
    current: list[tuple[str, str]],
    defects: str,
    *,
    model: str | None,
    max_tokens: int = 2048,
    max_workers: int = 4,
    file_char_cap: int = 4000,
) -> list[tuple[str, str]]:
    """Régénère EN PARALLÈLE chaque fichier en lui donnant TOUS les fichiers actuels +
    le rapport de défauts, pour une correction cohérente. Boucle fermée du fan-out."""

    def one(spec: FileSpec) -> tuple[str, str]:
        return fix_one(
            client,
            design,
            spec,
            current,
            defects,
            model=model,
            max_tokens=max_tokens,
            file_char_cap=file_char_cap,
        )

    if not specs:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(specs))) as ex:
        return list(ex.map(one, specs))


def review_semantic(
    client,
    design: str,
    current_files: list[tuple[str, str]],
    *,
    model: str | None,
    max_tokens: int = 1024,
    acceptance: str = "",
) -> list:
    """Défauts SÉMANTIQUES (comportement), que le verify déterministe ne voit pas.
    `acceptance` (optionnel) : critères observables des US -> vérification ORIENTÉE
    INTENTION (le code permet-il vraiment d'accomplir chaque critère ?). Renvoie
    list[Defect] (kind='semantic'). Robuste : [] si JSON invalide / aucun défaut."""
    from loom.verify import Defect

    files_txt = "\n\n".join(f"----- {p} -----\n{c}" for p, c in current_files)
    raw = client.complete(
        [{"role": "user", "content": review_prompt(design, files_txt, acceptance)}],
        _REVIEW_SYS,
        max_tokens=max_tokens,
        model=model,
        thinking=False,
    )
    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError, TypeError):
        return []
    items = data.get("defects", []) if isinstance(data, dict) else []
    defects = []
    for it in items:
        if isinstance(it, dict) and it.get("location"):
            defects.append(
                Defect(str(it["location"]), "semantic", str(it.get("evidence", "")))
            )
    return defects


def _content_valid(path: str, content: str) -> bool:
    """True si `content` passe le check syntaxique déterministe pour ce type de fichier.
    Pour un type sans checker (.html/.css) -> verify_syntax_file renvoie ok=True."""
    from loom.verify import verify_syntax_file

    fd, tmp = tempfile.mkstemp(suffix=Path(path).suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        return verify_syntax_file(tmp).ok
    finally:
        os.unlink(tmp)


def best_of(make_fn, n: int) -> tuple[str, str]:
    """Joue make_fn() jusqu'à n fois (SÉQUENTIEL), garde le 1er candidat valide, sinon
    le dernier. n=1 => un seul appel. Cf. spec §8 : best-of-N en réparation."""
    last: tuple[str, str] | None = None
    for _ in range(max(1, n)):
        last = make_fn()
        if _content_valid(last[0], last[1]):
            return last
    return last
