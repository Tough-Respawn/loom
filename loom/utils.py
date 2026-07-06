"""Utilitaires partagés : timestamps ISO et estimation de tokens.

Centralise les petites fonctions utilitaires dupliquées auparavant dans plusieurs
modules (agent/session.py, agent/reflect.py, extend/plugins.py, agent/context.py,
memory/identity.py).
"""

from __future__ import annotations

from datetime import datetime, timezone

#: Heuristique partagée : ~4 caractères par token (prose moyenne).
#: Utilisée par agent/context.py et memory/identity.py pour borner les budgets.
CHARS_PER_TOKEN = 4


def now_iso() -> str:
    """Horodatage ISO 8601 UTC tronqué à la seconde (ex. ``2025-01-02T15:04:05+00:00``).

    Variante SECONDS : pour les métadonnées persistées (timestamps de session,
    d'installation de plugin...) où la sous-seconde est du bruit.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now() -> str:
    """Horodatage ISO 8601 UTC complet (avec microsecondes).

    Variante par défaut (sans timespec) : pour les marqueurs internes qui
    veulent la résolution maximale (ex. ``created_at``/``updated_at`` des skills
    appris, où deux écritures dans la même seconde doivent rester ordonnables).
    """
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(text: str) -> int:
    """Estimation grossière du nombre de tokens : ~1 token pour 4 caractères.

    Renvoie au moins 1 pour un texte non vide. Heuristique partagée avec
    ``memory/identity.py`` (bornage du bloc identité) - garde un seul seuil.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)
