# Export / import de session

**Date** : 2026-07-26 · **Statut** : validé (demande user : « exporter ou importer une
session » depuis le menu contextuel d'onglet / le bouton Nouvelle session)

## Problème

Une session (fil de travail persistant : session.json + timeline.jsonl) vit dans
var/sessions/<sid>/ et ne sort jamais de la machine. On veut l'emporter (archiver,
partager, changer de machine) et la faire renaître ailleurs — sans toucher à l'Âme
(qui porte l'identité globale, chiffrée) : une session est un fichier banal qui se
donne, pas un coffre.

## Décisions

1. **Format** : un `.zip` clair contenant `session.json`, `timeline.jsonl` (si
   présente) et un manifeste `loom-session.json` (`format = "loom-session"`,
   `version = 1`, `exported_at`). `debug.log` exclu (log runtime machine). Pas de
   chiffrement : contenu = la conversation elle-même, l'utilisateur sait ce qu'il
   partage ; les clés API n'y vivent pas (la saisie wizard est déjà masquée).
2. **Export** : `GET /session/<sid>/export` → téléchargement
   `loom-session-<titre-slug>-<sid>.zip`. UI : entrée « Exporter la conversation
   (.zip) » dans le menu contextuel d'onglet/bandeau (à côté du chemin réel) —
   libellé « conversation », le mot « Sessions » désignant l'onglet de config du
   coffre .soul chiffré (état portable global).
3. **Import** : `POST /session/import` (multipart `file`) → session recréée sous un
   **id NEUF** (jamais d'écrasement), activée et ouverte en onglet comme une
   session neuve. UI : clic droit sur le bouton « + » (tooltip l'annonce) → sélecteur
   de fichier. Modèle d'origine inconnu sur cette machine → repli sur le défaut
   local (sinon 404 llama-swap au premier tour), signalé par un toast.
4. **Garde-fous** : zip invalide ou sans `session.json` → 400 avec message
   actionnable ; archive > 200 Mo (déclaré OU décompressé) → refus ; seuls les
   membres connus sont extraits (aucun chemin du zip n'est suivi — pas de traversal).

## Hors périmètre

Chiffrement (c'est l'Âme), export multi-sessions, fusion dans une session
existante, transport des images générées (var/generated est global, pas par session).

## Validation

Pytest (round-trip export→import : titre/historique préservés, id neuf, remap
modèle inconnu, rejets zip invalide) + E2E runtime réel sur serveur live.
