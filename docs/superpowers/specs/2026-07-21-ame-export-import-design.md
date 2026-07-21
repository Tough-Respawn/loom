# L'Âme — export/import portable de l'état de Loom

Date : 2026-07-21
Statut : validé (fusion à l'import ; pas de clés API ; sélection des sessions à l'export)

## Problème

L'état qui fait la « personnalité » d'un Loom (sessions, mémoire, identité, skills
appris) est enfermé dans `var/` d'une machine. Impossible aujourd'hui de reprendre
ses échanges et son contexte sur une autre machine ou un serveur sans copie manuelle
hasardeuse. On veut un export en un fichier chiffré, transportable par n'importe quel
support (clé USB, disque, dossier cloud synchronisé), et un import sans friction de
l'autre côté.

## Périmètre

### Ce qui part dans l'export

| Élément | Source | Règle |
|---|---|---|
| Sessions | `var/sessions/<id>/session.json` + `timeline.jsonl` | Sélectionnables à l'export (tout par défaut). Les logs `serve.log`/`debug.log` sont EXCLUS (bruit machine-local). |
| Mémoire | `var/memory/memory.db` | Toujours incluse. |
| Identité | `var/identity/` (SOUL.md, USER.md, MEMORY.md) | Toujours incluse. |
| Skills | `var/skills_learned/`, `var/skills_user/` | Toujours inclus. |

### Ce qui ne part JAMAIS

- `var/remote_models.json` — contient des clés API. Décision ferme : aucun secret ne
  quitte la machine, même chiffré. Sur la machine cible, les modèles distants se
  re-déclarent à la main (panneau engrenage).
- Modèles locaux (GGUF, model.toml) — liés au matériel, pas à la personnalité.
- `var/cache/`, `var/logs/`, `var/last_model`, états transitoires.

Le chiffrement protège la **confidentialité des données** (conversations, mémoire,
identité) pendant le transport — pas des secrets, puisqu'il n'y en a pas dedans.

## Format du fichier

- Nom par défaut : `ame-loom-<YYYY-MM-DD>.soul` (un seul fichier).
- Contenu : archive tar.gz de l'arborescence exportée + `manifest.json` à la racine :
  `{version: 1, date, machine, sessions: [<id>...], counts: {skills, memoires}}`.
- Chiffrement : AES-256-GCM, clé dérivée de la passphrase par scrypt
  (sel + nonce aléatoires stockés en tête de fichier, en clair, comme d'usage).
- La passphrase est saisie à l'export, resaisie à l'import, jamais stockée nulle part.
- GCM authentifie : passphrase fausse ou fichier corrompu → échec propre AVANT toute
  écriture, message clair, rien n'est touché.
- Dépendance nouvelle : `cryptography`.

## UX

Panneau engrenage, nouvelle section « Âme ».

### Export

1. Bouton **Exporter** → la section montre la liste des sessions avec cases à cocher
   (titre + date), « tout sélectionner » en tête, tout coché par défaut. Le socle
   (mémoire, identité, skills) part toujours et est affiché comme tel (non décochable).
2. Champ chemin de destination (n'importe quel dossier monté : USB, disque, dossier
   Dropbox/Drive/OneDrive) + champ passphrase (avec confirmation).
3. Écriture du fichier, puis récap : chemin, taille, inventaire.

### Import

1. Bouton **Importer** → champ chemin du `.soul` + champ passphrase.
2. Loom déchiffre, lit le manifest, fusionne, puis affiche un rapport :
   « N sessions ajoutées, M ignorées (déjà présentes), X skills, mémoire fusionnée ».
3. Toute erreur (passphrase, corruption, version de manifest inconnue) → message
   clair, état local intact.

## Règles de fusion (import)

- **Sessions** : ajout par id. Id déjà présent des deux côtés = même session déplacée
  deux fois → la version la plus récente (horodatage interne) gagne.
- **Skills** : même nom + même contenu → ignoré ; même nom + contenu différent →
  importé avec suffixe (`<nom>-importe`).
- **Mémoire** : fusion par enregistrement avec déduplication (clé selon le schéma de
  `memory.db`, à préciser au plan).
- **Identité** : si la cible a déjà un SOUL.md/USER.md/MEMORY.md différent, on garde
  le sien et on pose la version importée à côté (`SOUL.imported-<date>.md`) — jamais
  d'écrasement silencieux d'une âme par une autre. Cible vierge → les fichiers
  importés prennent leur place normale.

Rien n'est jamais supprimé côté cible par un import.

## Hors périmètre v1

- Connecteurs cloud natifs (API S3/Drive…) — un dossier synchronisé fait le boulot.
- Sync automatique/bidirectionnelle, import auto à la détection d'un support.
- Export des modèles (locaux ou distants) et de leurs configs.

## Architecture (indicative)

- `loom/web/soul.py` : logique pure export/import (constitution de l'archive,
  chiffrement/déchiffrement, règles de fusion) — testable sans Flask.
- Routes dans `routes.py` : `GET /soul/sessions` (liste pour les cases à cocher),
  `POST /soul/export`, `POST /soul/import`.
- UI : section « Âme » dans le panneau engrenage (app.js + templates), même style que
  le gestionnaire de modèles.

## Tests

- Unitaires sur `soul.py` : aller-retour export→import sur un `var/` de fixture
  (fusion, conflits de skills, identité divergente, session en double, passphrase
  fausse, fichier tronqué, absence de `remote_models.json` dans l'archive — test
  explicite anti-fuite de clés).
- E2E réel : export depuis une instance, import dans une instance vierge (autre
  `var/`), vérifier que les sessions reprennent (timeline visible, workspace).
