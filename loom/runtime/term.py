# loom/runtime/term.py
"""Couleurs ANSI pour les sorties terminal de Loom (installeur, serve).

Règles d'hygiène : couleur UNIQUEMENT si la sortie est un TTY (jamais dans un
fichier de log ni un pipe), respect de NO_COLOR (https://no-color.org), et
activation du mode VT sur les vieilles consoles Windows (conhost) — Windows
Terminal le fait déjà. Tout est best-effort : sans couleur, le texte reste
identique."""

from __future__ import annotations

import os
import re
import sys

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"

_vt_enabled = False


def _enable_windows_vt() -> None:
    """Active ENABLE_VIRTUAL_TERMINAL_PROCESSING sur conhost (no-op ailleurs)."""
    global _vt_enabled
    if _vt_enabled or sys.platform != "win32":
        _vt_enabled = True
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32
        for std in (-11, -12):  # stdout, stderr
            h = k32.GetStdHandle(std)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(h, ctypes.byref(mode)):
                k32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:  # noqa: BLE001 - le confort ne casse jamais rien
        pass
    _vt_enabled = True


def supports_color(stream) -> bool:
    """Couleur seulement sur un vrai terminal, et jamais si NO_COLOR est posé."""
    if os.environ.get("NO_COLOR"):
        return False
    try:
        if not (stream and stream.isatty()):
            return False
    except (AttributeError, ValueError):
        return False
    _enable_windows_vt()
    return True


def paint(text: str, *codes: str) -> str:
    return f"{''.join(codes)}{text}{RESET}"


# Règles d'auto-coloration d'une ligne, testées DANS L'ORDRE : (regex, codes).
# Élues d'après les écrans réels de loom-setup et serve — les émojis/préfixes
# portent déjà la sémantique, la couleur ne fait que l'amplifier.
_RULES: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(r"^──"), (BOLD, CYAN)),  # bannières ── Loom setup ── / ── Bilan ──
    (re.compile(r"^\[\d/\d\]"), (BOLD,)),  # têtes d'étape [1/3]
    (re.compile(r"^\s*(✅|OK\b)"), (GREEN,)),
    (re.compile(r"^\s*(❌|.*\bERREUR\b)"), (RED,)),
    (re.compile(r"^\s*⏭️"), (DIM,)),
    (re.compile(r"^\s*🔧"), (YELLOW,)),
    (re.compile(r"^\s*→"), (CYAN,)),  # résultats intermédiaires (repo retenu…)
    (
        re.compile(r"^\s*\d+\.\s"),
        (),
    ),  # items de menu : laissés bruts
]


def colorize(line: str) -> str:
    """Ligne colorée selon les règles Loom (identique si aucune ne matche)."""
    for rx, codes in _RULES:
        if rx.search(line):
            return paint(line, *codes) if codes else line
    return line
