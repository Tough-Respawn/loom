"""Lancement et gestion du runtime llama.cpp : détection matériel, args llama-server,
génération llama-swap, téléchargement et profils de modèles.

- serve.py          : lanceur auto-adaptatif (point d'entrée `uv run loom/runtime/serve.py`)
- hardware.py       : détection GPU/VRAM (nvidia-smi)
- server_args.py    : construction des args llama-server
- swap.py           : génération de llama-swap.yaml (multi-modèles)
- models_fetch.py   : téléchargement des GGUF
- models_profile.py : chargement du profile.md par modèle
- model_store.py    : registre/chemins des modèles installés
- config_schema.py  : introspection + édition de la config (SPEC, console UI)
- platform_info.py  : détection OS/shell (source de vérité unique)
- sysmon.py         : supervision système (mémoire, charge) en cours de run
- ngl.py            : calcul du nombre de couches GPU à offloader

(`config.py` reste à la racine du package : config transverse, pas runtime-spécifique.)
"""
