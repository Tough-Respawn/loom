# loom/prompts/__init__.py
"""Prompts de Loom. Le TEXTE vit dans des fichiers `.md` voisins (éditables sans toucher
au code) ; ce module ne fait que les charger. Loom est un agent tool-use : un seul prompt
système, celui qui lui dit d'agir avec les outils."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    """Charge un fichier prompt `.md` du dossier (mis en cache), strip les bords."""
    return (_DIR / name).read_text(encoding="utf-8").strip()


CHAT_SYSTEM = _load("chat.system.md")
# Variante ALLÉGÉE pour un modèle FORT (API distante) : identité + outils (bruts) + mémoire
# soulignée + sécurité, SANS le scaffolding de comportement (impératifs, séquences, règles
# d'or) qui ne sert qu'à un petit modèle local. On laisse le frontier se piloter seul.
CHAT_SYSTEM_STRONG = _load("chat.system.strong.md")
SUBAGENT_SYSTEM = _load("subagent.system.md")
REFLECT_SYSTEM = _load("reflect.system.md")
# Affinage des prompts image : réécrit la demande utilisateur (toute langue) en UN prompt
# de diffusion anglais propre. Utilisé par la branche image de /chat quand le modèle image
# déclare un `refiner` dans son model.toml.
IMAGE_REFINE_SYSTEM = _load("image_refine.system.md")
