# Couleurs terminal (loom/runtime/term.py) : règles d'auto-coloration et
# détection TTY/NO_COLOR — le texte reste IDENTIQUE hors codes ANSI.
import io

from loom.runtime import term
from loom.runtime.term import BOLD, CYAN, DIM, GREEN, RED, colorize, supports_color


def _plain(s: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def test_colorize_regles_loom():
    assert colorize("── Bilan ──").startswith(BOLD + CYAN)
    assert colorize("[2/3] Binaire llama-server").startswith(BOLD)
    assert colorize("  [ok] Modèle installé").startswith(GREEN)
    assert colorize("  [échec] Téléchargement raté").startswith(RED)
    assert colorize("[loom] ERREUR : binaire introuvable").startswith(RED)
    assert colorize("  [passé] Ignoré").startswith(DIM)
    assert colorize("  [manuel] llama-bench introuvable").startswith(term.YELLOW)
    assert colorize("  → repo retenu : x").startswith(CYAN)
    # ligne quelconque : intacte
    assert colorize("  RAM : 10156 Mo disponibles") == "  RAM : 10156 Mo disponibles"


def test_colorize_preserve_le_texte():
    for line in ["── Bilan ──", "  [ok] ok", "  [échec] raté", "texte neutre"]:
        assert _plain(colorize(line)) == line


def test_supports_color_pas_de_tty_ni_no_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert supports_color(io.StringIO()) is False  # pas un TTY (fichier/pipe)
    assert supports_color(None) is False

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(term, "_enable_windows_vt", lambda: None)
    assert supports_color(Tty()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert supports_color(Tty()) is False  # convention no-color.org
