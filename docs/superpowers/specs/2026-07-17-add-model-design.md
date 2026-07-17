# /add-model — ajout et paramétrage d'un modèle en une commande

**Date** : 2026-07-17
**Statut** : validé (brainstorm avec Amine)

## Besoin

Ajouter un modèle à Loom doit être zéro-friction : une seule commande `/add-model` dans le
chat qui prend en charge TOUT le parcours — local (recherche Hugging Face, choix du quant,
téléchargement, config générée) ou distant (config API collectée et persistée). C'est le
« local = v2 » du gestionnaire de modèles (le panneau engrenage v1 couvre déjà les distants),
plus une entrée unique conversationnelle.

## Décisions d'architecture (actées)

1. **Wizard déterministe, zéro LLM** — comme `/goal` et `/init`, la commande est interceptée
   côté serveur (`routes.py`). Aucun modèle n'est sollicité : ça marche même sans modèle
   servi (indispensable pour ajouter son PREMIER modèle), et pas de risque qu'un petit
   modèle local se plante sur un download de 20 Go.
2. **Interaction dans le chat** — les étapes se déroulent dans le fil de discussion
   (questions, shortlists numérotées, progression du téléchargement). Pas de nouvel écran.

## 1. Commande et machine à états

- `/add-model` seul → première question : local ou distant.
- `/add-model <recherche>` → saute direct au flux local, recherche lancée avec l'argument.
- Contrairement à `/goal` (one-shot), le wizard est **multi-étapes** : un état est posé sur
  la session (étape courante + données collectées). Tant qu'il est actif, les messages
  entrants sont routés vers lui (réponse par numéro ou texte libre) au lieu du flux LLM.
- Sortie : `/cancel` ou « annuler » à toute étape ; l'état est purgé, retour au chat normal.
- **Chaque étape est VISIBLE dans le chat** : toute question, shortlist, choix retenu et
  progression est un message du fil (pas d'action silencieuse côté serveur). Chaque message
  du wizard porte un indicateur d'étape (ex. « [add-model 2/4] Choix du quant ») pour que
  l'utilisateur sache toujours où il en est, et la progression du téléchargement se met à
  jour en direct dans le fil.
- L'état du wizard est **persisté avec la session** (survit à un refresh de page), champ
  sérialisé dans `Conversation` comme `goal`/`skill_overrides`.
- Une seule commande : `/add-model` (pas d'alias `/install_model`).

## 2. Flux distant

Réutilise la plomberie du panneau engrenage v1 — même source de vérité, deux portes d'entrée.

- Étapes : `id` → `base_url` → `model` → `api_key` (vide accepté) → « réglages avancés ? »
  (contexte, max_tokens, vision, prix — sinon défauts).
- Persistance : `model_store.upsert()` dans `var/remote_models.json` (déjà atomique, 0600).
- Le modèle est monté à chaud comme le fait déjà le panneau engrenage, et apparaît dans le
  sélecteur immédiatement.

## 3. Flux local (le cœur)

### 3.1 Recherche HF
- `GET https://huggingface.co/api/models?search=<q>&filter=gguf&sort=downloads&limit=8`
  (API publique, pas de clé pour les repos publics).
- Shortlist numérotée dans le chat : nom du repo, téléchargements, likes. L'utilisateur
  répond par numéro (ou relance une autre recherche).

### 3.2 Choix du quant
- API `tree` du repo choisi → fichiers `.gguf` avec tailles réelles (les `mmproj-*` sont
  repérés à part pour la vision).
- **Recommandation matérielle** : VRAM libre via `nvidia-smi` (`hardware.detect_hardware()`,
  notre méthode actée sur Windows) + RAM (`ram_available_mb()`). Le plus gros quant qui
  tient est marqué « recommandé » ; ceux qui ne tiennent pas sont listés quand même,
  marqués « ne tiendra pas » — l'override utilisateur reste roi.
- Les GGUF multi-parties (`-00001-of-0000N.gguf`) sont regroupés en une entrée (taille
  cumulée) ; le téléchargement récupère toutes les parties.

### 3.3 Installation
- `id` proposé (dérivé du nom du repo, modifiable) ; création de
  `<models_root>/local/text/<id>/model.toml` avec `repo`, `filename`, `size_mb`.
- Téléchargement **immédiat en arrière-plan** (thread), progression affichée dans le chat
  (Mo reçus / total, via la taille du fichier sur disque vs taille annoncée par l'API).
- Si ça coupe (fermeture, réseau) : filet existant — `ensure_model()` reprend/complète au
  premier serve. Le wizard le dit explicitement.

### 3.4 Post-download
- Lecture des métadonnées GGUF (header local) pour compléter `model.toml` : `n_layers`,
  contexte max du modèle, détection MoE (`expert_count > 0` → proposer `cpu_moe = true`,
  cohérent avec la règle du parc « MoE 24B+, experts en RAM »).
- **Re-scan à chaud** : la découverte des modèles (`config._discover_models`) tourne au
  démarrage ; le wizard déclenche un re-scan pour que le modèle apparaisse dans le
  sélecteur SANS redémarrer loom.web.

## 4. Hors périmètre v1

- Modèles image/vidéo (ComfyUI = moteur séparé, autre parcours).
- Génération automatique du `profile.md` (dossier créé avec stub vide ; un LLM pourra le
  remplir plus tard — piste « hybride » notée pour v2).
- Gated repos HF (nécessitent un token : message actionnable, pas de gestion de login v1).

## 5. Erreurs

Pattern `ModelUnavailable` existant : tout échec (réseau, repo privé/introuvable, HF down)
est ramené à un message ACTIONNABLE dans le chat (quoi faire, où poser le fichier), jamais
une stacktrace. Le wizard reste dans son étape courante après une erreur récupérable
(ex. recherche sans résultat → reformuler).

## 6. Tests

- Unitaires : machine à états du wizard (transitions, cancel, persistance), parsing des
  réponses API HF (fixtures JSON enregistrées), génération du `model.toml`, heuristique de
  recommandation quant (cas VRAM/RAM connus). Pas de download réel dans les tests.
- Runtime : validation E2E réelle (ajouter un vrai modèle local + un distant de bout en
  bout) AVANT de déclarer la feature terminée — règle absolue.
