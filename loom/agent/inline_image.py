# loom/agent/inline_image.py
"""Protocole d'image INLINE entre un outil (read_image) et la boucle tool-use.

Problème : un message de rôle `tool` ne transporte que du TEXTE — on ne peut pas y
glisser une image pour que le modèle multimodal la « voie ». La solution : l'outil
read_image renvoie une chaîne sentinelle (texte) encapsulant le data-URL ; la boucle
(`stream_chat_tools`) la reconnaît, met un accusé court dans le message `tool`, et
ajoute APRÈS les résultats d'outils un message `user` multimodal portant l'image.

Ce module isole ce contrat pour que `tools/read.py` et `client.py` s'accordent sans
se coupler l'un à l'autre. Le caractère NUL sert de séparateur (absent d'un data-URL).
"""

from __future__ import annotations

_SENTINEL = "\x00LOOM_INLINE_IMAGE\x00"
_SEP = "\x00"


def wrap_image(data_url: str, caption: str) -> str:
    """Encode (caption, data_url) en chaîne sentinelle renvoyée comme résultat d'outil."""
    return f"{_SENTINEL}{caption}{_SEP}{data_url}"


def is_inline_image(result: str) -> bool:
    """Vrai si `result` est une image inline produite par read_image."""
    return isinstance(result, str) and result.startswith(_SENTINEL)


def parse_inline_image(result: str) -> tuple[str, str]:
    """Décode -> (caption, data_url). À n'appeler que si is_inline_image est vrai."""
    caption, data_url = result[len(_SENTINEL) :].split(_SEP, 1)
    return caption, data_url


def image_user_message(caption: str, data_url: str) -> dict:
    """Construit le message `user` multimodal qui fait VOIR l'image au modèle."""
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Image « {caption} » (demandée via read_image) :",
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }
