"""Loom — agent IA local, multimodal et offline (boucle tool-use sur llama.cpp).

Carte du package (pour s'y retrouver d'un coup d'œil) :

  config.py        Réglages (chargés depuis loom.config.toml) — transverse à tout.
  permissions.py   Politique de sécurité : autorise / demande / refuse chaque appel d'outil.

  agent/           Le moteur : boucle tool-use (client), conversation, contexte, sessions, images.
  tools/           Les capacités appelées par le modèle (lire, écrire, exécuter, web, sous-agent…).
  extend/          Extensions : skills (catalogue + use_skill) et store de plugins (compatible Claude Code).
  prompts/         Prompts système (.md).
  runtime/         Service llama.cpp : lanceur (serve), swap, hardware, args, téléchargement, profils.
  web/             Serveur d'interface (Flask + SSE + templates/static) — la couche humain↔app.

  models/  skills/  plugins/  data/   Données : poids GGUF, contenu des skills, store de plugins, runtime.
"""
