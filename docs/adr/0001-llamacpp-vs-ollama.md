# ADR 0001 — Runtime d'inférence : llama.cpp plutôt qu'Ollama

- Statut : Accepté
- Date : 2026-05-31

## Contexte
Besoin d'un runtime local servant un modèle GGUF via API OpenAI-compatible, sur
laptop Windows (RTX 2060 6 Go) ET VPS Linux CPU 64 Go. Contraintes de l'utilisateur :
agnosticisme (pas de techno « installe un modèle » type LM Studio), ne pas
réinventer la roue.

## Options
- **llama.cpp (`llama-server`)** : moteur fondamental, GGUF brut, GBNF, contrôle
  total des flags, API OpenAI-compatible native. Setup manuel un peu plus long.
- **Ollama** : installeur simple, mais registre/format propriétaire, wrapper
  par-dessus llama.cpp, sortie contrainte limitée au JSON-schema.

## Décision
llama.cpp. Un GGUF est un fichier nu (agnostique) ; llama.cpp EST la roue de
référence (Ollama en est un wrapper) ; GBNF est nécessaire pour fiabiliser les
petits modèles. L'abstraction API OpenAI-compatible garde Ollama swappable plus tard.

## Conséquences
- + Contrôle total (offload GPU, contexte), GBNF, zéro lock-in.
- − ~15-20 min de setup binaire en plus, à documenter (install-windows/linux.md).
