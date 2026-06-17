"""Identité always-on de Loom : trois fichiers markdown TOUJOURS LOCAUX, lisibles et
éditables à la main, injectés au system prompt à chaque tour (design §5.2).

- SOUL.md   : persona/caractère de l'agent
- USER.md   : profil de l'utilisateur
- MEMORY.md : mémoire générale durable (conventions, environnement, consignes)

IO pures (fichiers <-> texte), append dédup ligne par ligne, et `identity_block` qui
concatène les trois en un bloc BORNÉ (jamais délégué à un service externe). Sans Flask
ni modèle -> testable en isolation.
"""

from __future__ import annotations

from pathlib import Path

# ~4 caractères par token (même heuristique que loom/agent/context.py).
_CHARS_PER_TOKEN = 4


def read_md(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def append_unique(path: str, line: str) -> None:
    """Ajoute `line` au fichier si absente (dédup ligne par ligne). Crée le fichier au besoin."""
    line = (line or "").strip()
    if not line:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = read_md(path)
    lines = {ln.strip() for ln in existing.splitlines()}
    if line in lines:
        return
    body = (existing + "\n" + line).strip() if existing else line
    p.write_text(body + "\n", encoding="utf-8")


def identity_block(
    soul_path: str, user_path: str, memory_path: str, *, max_tokens: int = 400
) -> str:
    """Concatène SOUL/USER/MEMORY en un bloc borné pour le system prompt. Vide si rien.

    Bornage simple par caractères (max_tokens * 4) : on tronque le bloc concaténé en
    gardant l'ordre SOUL -> USER -> MEMORY. Le bornage fin (resserrage) est l'affaire de
    `reflect` au Plan 2 ; ici on protège juste le budget de contexte.
    """
    sections = [
        ("# Mon identité (SOUL)", read_md(soul_path)),
        ("# L'utilisateur (USER)", read_md(user_path)),
        ("# Mémoire durable (MEMORY)", read_md(memory_path)),
    ]
    parts = [f"{title}\n{content}" for title, content in sections if content]
    if not parts:
        return ""
    block = "\n\n".join(parts)
    cap = max_tokens * _CHARS_PER_TOKEN
    if len(block) > cap:
        block = block[:cap].rstrip() + "\n[…tronqué]"
    return block
