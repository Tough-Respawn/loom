# Modèles IMAGE dans Loom — sélection directe dans l'UI (v1 sans outil)

Date : 2026-07-08 — Statut : validé par le user (design présenté et approuvé en session).

## But
Générer des images depuis l'UI de Loom sans ouvrir ComfyUI : un modèle image apparaît
dans le sélecteur comme les modèles texte ; le sélectionner fait qu'un message user =
un prompt d'image = une image affichée dans la conversation. PAS d'outil `generate_image`
en v1 (le LLM n'appelle rien) : le futur outil reste en roadmap.

## Principe : un troisième type de modèle
- **local** (llama-swap), **distant** (routes API), désormais **image** (ComfyUI).
- Déclaration PAR DOSSIER, patron des LLM : `loom/models/_IMAGE/<id>/`
  - `model.toml` : `label`, `width`/`height` par défaut, options ComfyUI (dossier, port).
  - `workflow.json` : graphe ComfyUI **au format API** avec les placeholders `{PROMPT}`
    et `{SEED}`. Ajouter un modèle image = copier un dossier, éditer deux fichiers.
- Le préfixe `_` exclut ces dossiers de la découverte llama-swap (convention _TEMPLATE/_REMOTE).

## Flux d'une génération (branche dédiée dans /chat)
1. `conv.model` est un id image → PAS de boucle tool-use, pas de prompt système LLM.
2. Garde-éveil (`stay_awake`) + **libération VRAM** : `unload_local()` (existant) décharge
   le LLM local le cas échéant.
3. **ComfyUI géré par Loom** : instance démarrée à la demande via le patron
   `ModelServerManager` (Job Object kill-on-close — fermer Loom tue ComfyUI).
   Attente du port (health `GET /system_stats`), timeout net et message clair sinon.
4. **Soumission** : le message user est injecté dans `workflow.json` (`{PROMPT}`),
   seed aléatoire (`{SEED}`), `POST /prompt`, puis poll `GET /history/<prompt_id>`
   (label d'activité SSE pendant ce temps : « génération de l'image… »).
5. **Résultat** : PNG récupéré (`GET /view`), écrit dans le WORKSPACE de la session
   (`images/loom_<horodatage>.png`), affiché INLINE dans la conversation (mécanique
   d'images existante) ; le chemin est cité en texte.
6. Erreur ComfyUI (nœud manquant, OOM) → message d'interruption lisible dans le chat,
   jamais de stacktrace.

## Bascule texte ↔ image
- Vers image : étape 2 ci-dessus (unload LLM).
- Retour texte : `POST /free` à ComfyUI (best-effort) pour rendre la VRAM ; le LLM se
  recharge à la première requête (llama-swap). ComfyUI reste UP (processus léger sans
  modèle chargé) tant que loom.web vit.

## UI
- Sélecteur : entrées image listées avec les autres, suffixe/badge « image ».
- Un message user = une image. Pas de réglages v1 (taille/seed/steps = défauts du
  workflow validés au terrain le 2026-07-08 : Q4_K_M + LoRA réalisme 0.8, 8 steps, cfg 1).
- Multi-onglets : la génération image suit la même sérialisation que le LOCAL
  (même GPU) : verrou `_local_gen_lock`.

## Premier modèle livré
`loom/models/_IMAGE/krea2-turbo/` : Krea-2-Turbo GGUF Q4_K_M + LoRA Krea2-realism-V2
(0.8) + text encoder qwen3vl fp8 + VAE qwen_image — la chaîne validée en réel ce jour
(installée sous C:\tools\ComfyUI, chemins portés par la config, pas en dur dans le code).

## Hors périmètre v1 (assumé)
- Outil `generate_image` appelable par le LLM (roadmap).
- Réglages d'image dans l'UI ; img2img ; multi-LoRA dynamique.
- Téléchargement automatique des poids image (ComfyUI et ses modèles sont installés à part).

## Vérification
- Smokes : découverte _IMAGE, injection prompt/seed dans le workflow, manager start/stop.
- E2E réel obligatoire avant de dire « ça marche » : depuis l'UI Loom, sélectionner
  krea2-turbo, envoyer un prompt, constater l'image dans le chat et le PNG dans le
  workspace ; puis rebasculer sur le 35B et vérifier qu'il répond (VRAM rendue).
