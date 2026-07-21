"""Loom — agent IA local, multimodal et offline (boucle tool-use sur llama.cpp).

Carte du package (pour s'y retrouver d'un coup d'œil) :

  config.py        Schéma + chargement de la config (config/defaults.toml + config/local.toml).
  permissions.py   Politique de sécurité : autorise / demande / refuse chaque appel d'outil.

  agent/           Le moteur : boucle tool-use (client), conversation, contexte, sessions, images.
  tools/           Les capacités appelées par le modèle (lire, écrire, exécuter, web, sous-agent…).
  extend/          Extensions : skills (catalogue + use_skill) et store de plugins (compatible Claude Code).
  prompts/         Prompts système livrés (.md) + prompts/identity/ (gabarits SOUL/USER/MEMORY).
  runtime/         Service llama.cpp : lanceur (serve), swap, hardware, args, téléchargement, profils.
  web/             Serveur d'interface (Flask + SSE + templates/static) — la couche humain↔app.
  models/ skills/ plugins/   Assets livrés : poids GGUF (gitignorés), skills, store de plugins.

Hors package (racine du repo) :
  config/          defaults.toml (versionné) + local.toml (surcharge machine, gitignored).
  var/             État machine (gitignored) : identity/, memory/, sessions/, skills_learned/, logs/, cache/.
"""

# Confiance TLS via le magasin de certificats de l'OS (même mécanisme que pip) :
# transparent sur machine perso, indispensable derrière un proxy d'entreprise à
# inspection TLS (son certificat racine est déployé par l'IT dans le magasin
# Windows/macOS, jamais dans le bundle certifi). Injecté ICI — l'import du package
# précède toute création de contexte SSL (httpx, huggingface_hub, urllib), qui se
# fait à l'usage, pas à l'import.
try:
    import truststore as _truststore

    _truststore.inject_into_ssl()
    TRUSTSTORE_ACTIVE = True
except ImportError:  # dep absente (env minimal) : comportement certifi inchangé
    TRUSTSTORE_ACTIVE = False
