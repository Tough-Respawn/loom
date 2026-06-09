# loom/extend/ — extensions de connaissance/capacités du modèle.
"""Ce qui étend le modèle au-delà de la boucle + des outils de base :

- skills.py   : catalogue de skills (locaux + plugins) déclenchés par le modèle (use_skill)
- plugins.py  : store de plugins compatible Claude Code (+ CLI `python -m loom.extend.plugins`)

Les DOSSIERS de données restent à la racine du package : `loom/skills/` (contenu des skills
locaux) et `loom/plugins/` (store installé, gitignoré). Séparer le code (ici) des données
(là-bas) lève la collision de noms module/dossier.
"""
