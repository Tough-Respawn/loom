---
fixes:
  normalize_quotes: true
---
Qwen3.5-4B émet des guillemets typographiques (’ ‘ ” “) au lieu d'ASCII -> casse la
syntaxe Python, et il les ré-émet malgré le prompt. On normalise le contenu écrit dans
les fichiers de code (les fichiers de prose .md/.txt sont épargnés).
