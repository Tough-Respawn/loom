# Rationalisation de la surface d'outils — design (2026-07-15)

## Constat mesuré

- 27 schémas d'outils = ~21 300 car. ≈ 7 100 tokens = **29 % de la fenêtre locale (24 576)**, payés avant tout message. Problème LOCAL (négligeable sur distant 1M).
- Usage réel (~350 appels, 3 sessions + évals) : 8 outils font ~90 % du trafic (run_shell, read_file, web_search, fetch_url, edit_file, check_page/interactive, calculate, write_file). Jamais observés : search_text, append_file, serve_and_check, read_note, recall, remember, plugins ×3.
- Coût inversement corrélé à l'usage : serve_and_check 704 tok / 0 appel ; calculate 523 tok (documente des mécaniques que la coercition gère déjà) ; remember 362 tok / 0 appel.
- Toutes les descriptions/schémas sont déjà en ANGLAIS (décision consolidation 2026-07) — la compression reste en EN.

## Décisions (validées user 2026-07-15)

1. **Lot C — fetch_url `params`** (objet optionnel → query string encodée par l'outil ; les valeurs dict/list sont JSON-sérialisées puis URL-encodées). Motif : 44+ échecs d'encodage manuel PowerShell dans la session chasse-invest (API Bien'ici `filters=<json>`). GET only, anti-SSRF inchangé. Pas de nouvel outil http_request.
2. **B3 — check_interactive fusionné dans check_page** : check_page gagne `steps` (optionnel, même schéma). Sans steps = chargement+console+comptages (comportement actuel) ; avec steps = actions+post-conditions. check_interactive retiré du catalogue, des prompts (chat.system.md, subagent.system.md), des frozensets client (_VERIFY_TOOLS, _BROWSER_CHECKS, anti-sur-vérification), du défaut config. serve_and_check reste séparé (danger=True, cycle de vie serveur ≠ niveau de permission).
3. **B2 — plugins hors du set par défaut** : list_plugins/add_marketplace/install_plugin retirés de `[tools] enabled` (defaults.toml). Le code reste ; activables par config/UI. Les sessions existantes gardent leur liste.
4. **B4 — read_image gaté vision** : absent du registre si le modèle actif n'a pas la vision (`active_is_vision`), sous-registre dispatch compris. La décision 2026-07-09 (pas de repli vers un autre modèle) est préservée ; ce qui disparaît est le message « je ne vois pas » du modèle texte.
5. **current_date GARDÉ** (décision user) — description compressée seulement.
6. **Lot A — compression des descriptions (EN)** des schémas les plus verbeux (serve_and_check, calculate, read_file, check_page fusionné, remember, read_image, edit_file, write_note, manage_todos, current_date, format_code, append_file). Règles : garder les impératifs comportementaux (ALWAYS calculate, check don't assume, LOCATE before READ) — leçon éval : l'emphase compte pour un petit modèle ; couper les mécaniques désormais gérées par coercition/erreurs actionnables (ex. calculate : ^, exposants unicode) ; couper les redites du prompt système.

**Gardés délibérément** : list_dir (modèle mental simple), append_file (plafond de sortie local), write_note/read_note/recall/remember (closed-loop), search_text (échantillon biaisé web ; outil « qui appelle X » du mode code).

## Cible

7 100 → ~4 200 tokens de schémas (−40 %), 27 → 23 outils par défaut.

## Validation

- TDD sur chaque changement de comportement (fetch_url params, check_page steps, gating vision, catalogue).
- Suite pytest complète verte.
- Banc d'évals A/B avant/après (evals/run_eval.py, baseline 26/27) — les descriptions sont des prompts : on mesure, on ne suppose pas.
- Compatibilité sessions : une session dont active_tools contient check_interactive/plugins les perd silencieusement du registre (check_page couvre ; plugins réactivables) — pas de migration.

## Livraison

3 commits : (1) fetch_url params, (2) surface (B2+B3+B4), (3) compression descriptions. Évals après le lot 3 (et après le lot 2 si le temps le permet).
