# loom/agent/ — le MOTEUR de la boucle tool-use et son état.
"""Cœur agentic de Loom : la boucle qui appelle le modèle, exécute les outils et tient l'état.

- client.py        : SDK openai + boucle stream_chat / stream_chat_tools + garde-fous
- conversation.py  : mémoire de conversation (messages, modèle, thinking, outils) persistée
- session.py       : un fil persistant par projet (conversation + métadonnées)
- context.py       : budget de tokens + résumé auto
- inline_image.py  : injection d'images (multimodal) dans la boucle

(`config.py` et `permissions.py` restent à la racine : transverses à toutes les couches.)
"""
