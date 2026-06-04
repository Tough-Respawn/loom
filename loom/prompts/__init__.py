# loom/prompts/__init__.py
"""Tous les prompts du framework Loom, regroupés ici (jamais éparpillés dans le code).

Le TEXTE vit dans des fichiers `.md` voisins (éditables sans toucher au code, comme les
skills) ; ce module ne fait que les charger, substituer les placeholders `__NOM__` et
borner la taille. Les prompts sont VERBEUX et EXPLICITES : un petit modèle (4B) a besoin
de règles écrites noir sur blanc pour être robuste.

Convention : `<role>.system.md` = prompt système (chargé tel quel) ; `<role>.<usage>.md`
= gabarit avec placeholders `__MAJUSCULES__` remplis par les fonctions builder."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from loom.prompts.util import clip

_DIR = Path(__file__).resolve().parent
_TOKEN = re.compile(r"__[A-Z_]+__")


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    """Charge un fichier prompt `.md` du dossier (mis en cache). strip les bords."""
    return (_DIR / name).read_text(encoding="utf-8").strip()


def _fill(template: str, mapping: dict[str, str]) -> str:
    """Remplit les placeholders `__NOM__` en UNE passe : le contenu injecté n'est jamais
    re-scanné (un fichier qui contiendrait `__DESIGN__` ne casse pas la substitution)."""
    return _TOKEN.sub(lambda m: mapping.get(m.group(0), m.group(0)), template)


# --- Prompts système (texte figé, chargé tel quel) ---
CHAT_SYSTEM = _load("chat.system.md")
CLASSIFY_SYSTEM = _load("classify.system.md")
PLAN_SYSTEM = _load("plan.system.md")
GEN_SYSTEM = _load("gen.system.md")
EDIT_SYSTEM = _load("edit.system.md")
REVIEW_SYSTEM = _load("review.system.md")
CRITIQUE_SYSTEM = _load("critique.system.md")
DECOMPOSE_SYSTEM = _load("decompose.system.md")


# --- Gabarits (assemblés + bornés en Python) ---
def plan_prompt(
    task: str, explore_summary: str = "", existing_files: list[str] | None = None
) -> str:
    """Prompt utilisateur du PLAN. `existing_files` (brownfield) force la réutilisation
    EXACTE des fichiers existants (tue la divergence d'archi : 3 fichiers -> 4)."""
    prompt = _fill(_load("plan.user.md"), {"__TASK__": task})
    if existing_files:
        listing = ", ".join(existing_files)
        prompt += (
            "\n\nCONTRAINTE BROWNFIELD (IMPÉRATIVE) : ce projet existe déjà. Ton plan DOIT "
            f"réutiliser EXACTEMENT ces fichiers, sans en inventer d'autres : {listing}. "
            "N'introduis PAS de nouveau fichier (pas de main.js/renderer.js si ce n'est "
            "pas dans la liste). Tu corriges/complètes l'existant, tu ne refais pas l'archi."
        )
    if explore_summary:
        prompt += (
            "\n\nCODE EXISTANT (à MODIFIER de façon ciblée, ne réécris PAS ce qui "
            f"marche) :\n{explore_summary}\n"
        )
    return prompt


def file_prompt(spec, design: str, all_paths: list[str], file_char_cap=None) -> str:
    """Prompt de génération d'UN fichier. `file_char_cap` borne le design injecté."""
    if file_char_cap is not None:
        design = clip(design, file_char_cap)
    return _fill(
        _load("gen.file.md"),
        {
            "__ALL_PATHS__": ", ".join(all_paths),
            "__PATH__": spec.path,
            "__ROLE__": spec.role,
            "__DESIGN__": design,
        },
    )


def fix_prompt(
    spec, design: str, current, defects: str, *, file_char_cap: int = 4000
) -> str:
    """Prompt de correction d'UN fichier (tout le contexte borné à ~file_char_cap : le
    pool KV est partagé en fan-out). Le fichier ciblé garde la plus grosse part ; les
    voisins sont clippés dur (le design porte la cohérence)."""
    design_cap = file_char_cap // 2
    target_cap = file_char_cap // 2
    sib_cap = max(200, file_char_cap // 16)
    defects_cap = file_char_cap // 4
    others = "\n\n".join(
        f"----- {p} -----\n{clip(c, target_cap if p == spec.path else sib_cap)}"
        for p, c in current
    )
    return _fill(
        _load("gen.fix.md"),
        {
            "__PATH__": spec.path,
            "__FILES__": others,
            "__DEFECTS__": clip(defects, defects_cap),
            "__DESIGN__": clip(design, design_cap),
        },
    )


def edit_prompt(spec, design: str, content: str, defects: str = "") -> str:
    """Prompt de patch ciblé (renvoie {old_string, new_string})."""
    defects_block = f"\nDÉFAUTS à corriger :\n{defects}\n" if defects else ""
    return _fill(
        _load("edit.user.md"),
        {
            "__DESIGN__": design,
            "__PATH__": spec.path,
            "__ROLE__": spec.role,
            "__CONTENT__": content,
            "__DEFECTS_BLOCK__": defects_block,
        },
    )


def review_prompt(design: str, files_txt: str) -> str:
    """Prompt de relecture sémantique (renvoie le JSON des défauts comportementaux)."""
    return _fill(
        _load("review.user.md"), {"__DESIGN__": design, "__FILES__": files_txt}
    )


def critique_prompt(design: str, task: str) -> str:
    """Prompt d'auto-critique du plan (renvoie une liste de trous/risques)."""
    return _fill(_load("critique.user.md"), {"__DESIGN__": design, "__TASK__": task})


def decompose_prompt(design: str, task: str, files: list[str]) -> str:
    """Prompt de découpe en user stories (renvoie le JSON {stories: [...]})."""
    return _fill(
        _load("decompose.user.md"),
        {"__DESIGN__": design, "__TASK__": task, "__FILES__": ", ".join(files)},
    )
