"""Point de sortie unique des appels de modèle en session privée.

Le tour d'une session privée pose `PRIVATE` (variable de contexte) ; le client des
modèles refuse alors toute route distante, quel que soit le code qui la demande (tour,
titre, réflexion, synthèse de mémoire, raffineur d'image, futur appel oublié). Les fils
lancés pendant ce tour doivent hériter du contexte (`contextvars.copy_context().run`).
"""

from __future__ import annotations

import contextvars

PRIVATE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "loom_private_session", default=False
)


class PrivateEgressError(RuntimeError):
    """Un appel vers une API distante a été tenté pendant une session privée."""


def check_remote_allowed(model: str) -> None:
    """Lève si `model` (route distante) est appelé dans le contexte d'une session privée."""
    if PRIVATE.get():
        raise PrivateEgressError(
            f"session privée : le modèle « {model} » est distant, rien ne doit sortir "
            "de la machine. Choisis un modèle local."
        )
