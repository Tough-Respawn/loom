# loom/runtime/models_profile.py
"""Profils par modèle : correctifs DÉTERMINISTES propres à chaque modèle, activés par un
profile.md dans loom/models/<id>/. Le .md ne contient PAS de logique : son frontmatter
ACTIVE des fixes déjà codés ici (registre curaté). Chaque modèle a ses travers ; on les
corrige sans dépendre du prompt (ex. Qwen3.5 ré-émet des guillemets typographiques)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Ce module vit dans loom/runtime/ : remonter de DEUX niveaux jusqu'à la racine loom/
# où se trouve models/ (sinon on chercherait loom/runtime/models/, inexistant -> profils muets).
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Fichiers de PROSE : on n'y touche pas (les guillemets typographiques peuvent y être voulus).
_PROSE_EXT = frozenset({".md", ".markdown", ".txt", ".rst"})
_SMART_QUOTES = {"’": "'", "‘": "'", "“": '"', "”": '"'}


def _normalize_quotes(content: str, suffix: str) -> str:
    """Remplace ’ ‘ ” “ par ' et " — sauf dans les fichiers de prose."""
    if suffix.lower() in _PROSE_EXT:
        return content
    for bad, good in _SMART_QUOTES.items():
        content = content.replace(bad, good)
    return content


# Registre curaté : nom de fix -> fonction (content, suffix) -> content.
FIXES = {
    "normalize_quotes": _normalize_quotes,
}

# Outils d'écriture et les clés d'arguments qui portent du contenu à corriger.
_CONTENT_KEYS = {
    "write_file": ("content",),
    "append_file": ("content",),
    "replace_lines": ("content",),
    "edit_file": ("old_string", "new_string"),
}


@dataclass(frozen=True)
class Profile:
    model_id: str
    fixes: tuple[str, ...]

    def apply(self, tool_name: str, args: dict, suffix: str) -> dict:
        """Applique les fixes actifs au contenu des arguments d'un outil d'écriture."""
        keys = _CONTENT_KEYS.get(tool_name)
        if not keys or not self.fixes:
            return args
        out = dict(args)
        for k in keys:
            v = out.get(k)
            if isinstance(v, str):
                for name in self.fixes:
                    fn = FIXES.get(name)
                    if fn:
                        v = fn(v, suffix)
                out[k] = v
        return out


_EMPTY = Profile("", ())


def _parse_frontmatter_fixes(text: str) -> list[str]:
    """Lit les fixes ACTIFS (valeur vraie) sous 'fixes:' dans le frontmatter YAML simple
    d'un profile.md. Parsing minimal (sans dépendance YAML)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return []
    fm: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        fm.append(line)
    active: list[str] = []
    in_fixes = False
    for line in fm:
        stripped = line.strip()
        if stripped.startswith("fixes:"):
            in_fixes = True
            continue
        if in_fixes:
            if line and not line[0].isspace():  # fin du bloc indenté
                in_fixes = False
                continue
            if ":" in stripped:
                name, _, val = stripped.partition(":")
                if name.strip() in FIXES and val.strip().lower() in {
                    "true",
                    "yes",
                    "on",
                    "1",
                }:
                    active.append(name.strip())
    return active


def load_profile(model_id: str, models_dir: Path | None = None) -> Profile:
    """Charge le profil d'un modèle depuis loom/models/<id>/profile.md. Absent/illisible
    -> profil vide (aucun fix). Ne lève jamais."""
    if not model_id:
        return _EMPTY
    path = (models_dir or MODELS_DIR) / model_id / "profile.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return Profile(model_id, ())
    return Profile(model_id, tuple(_parse_frontmatter_fixes(text)))
