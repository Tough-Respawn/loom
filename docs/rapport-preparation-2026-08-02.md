# Rapport de préparation — 2026-08-02

## Verdict

**Loom est prêt pour un usage local supervisé et peut être considéré comme une
release candidate.** Aucun défaut bloquant n'a été observé sur la suite automatisée,
le harnais d'évaluation, le parcours navigateur principal ou la construction du paquet.

Loom n'est toutefois **pas « terminé à 100 % » ni durci pour un environnement non
fiable** : des fonctions restent explicitement en backlog, les intégrations dépendantes
du matériel et de services externes sont moins couvertes, et le mode `allow` exige un
environnement isolé ou de confiance.

## Périmètre contrôlé

- Runtime Python, boucle tool-use, outils, sessions, configuration et routes Flask.
- Harnais d'évaluation hors ligne et garde-fous d'injection.
- Interface réelle dans Chromium : chargement des modules, modale clavier, envoi d'un
  message, appel OpenAI-compatible simulé, streaming SSE et rendu de la réponse.
- Qualité statique, couverture Python et construction des distributions.
- Cohérence des documents de démarrage, de statut et de validation.

## Corrections réalisées pendant l'audit

1. Le self-test d'évaluation échouait depuis l'éclatement de `client.py` : il
   importait `_scan_repeat` et `_verify_streak_update` depuis une ancienne façade.
   Les imports ciblent maintenant les modules propriétaires (`compaction`, `guards`,
   `streaming`, `toolrun`).
2. Un test pytest appelle désormais le self-test complet, afin qu'un futur refactoring
   ne puisse plus casser silencieusement la commande documentée.
3. Le banc E2E manuel dispose maintenant d'un smoke autonome
   (`python -m tests.e2e.smoke_ui`) qui démarre et arrête ses serveurs lui-même.
4. `README.md`, `loom.md`, `ETAT_PROJET.md`, `CHANGELOG.md` et la documentation E2E ont
   été remis en cohérence avec l'état réel du projet.
5. Les commentaires du code ont été rendus plus sobres sur l'ensemble de `loom/`,
   `tests/`, `evals/` et `scripts/`. L'inventaire est passé de **3 584 à 1 655 lignes**
   de commentaires (**-54 %**) sans modification fonctionnelle : restent surtout les
   raisons de sécurité, contraintes de concurrence/cache, compatibilité OS et invariants
   de régression. Les historiques, séparateurs décoratifs et descriptions du code évident
   ont été supprimés.

## Preuves rejouées

| Contrôle | Commande | Résultat |
|---|---|---|
| Suite principale | `uv run pytest tests/ -q` | **533 tests passés** en 63,78 s |
| Couverture | `uv run pytest tests/ --cov=loom --cov-report=term -q` | **60 %** (10 573 instructions, 4 226 non couvertes) |
| Analyse statique | `uv run ruff check loom tests scripts evals` | **All checks passed** |
| Harnais d'éval hors ligne | `uv run python -m evals.run_eval --self-test` | **SELF-TEST VERT** |
| Navigateur E2E | `uv run python -m tests.e2e.smoke_ui` | **VERT** : HTTP 200, clavier, chat SSE, rendu, zéro erreur console |
| Distribution | `uv build --out-dir <temp>` | wheel `loom-0.1.0-py3-none-any.whl` et source `loom-0.1.0.tar.gz` construits |
| Propreté des patchs | `git diff --check` | aucune erreur d'espace ou de patch |

En couverture, les chemins centraux sont notamment bien exercés : configuration 97 %,
wizard 96 %, workflows 91 %, sessions 89 %, streaming 84 % et application Flask 82 %.
Le total à 60 % reflète surtout les chemins d'intégration qui demandent un vrai matériel
ou service (serveur llama.cpp, ComfyUI, SearXNG, télémétrie GPU, certains plugins).

## Limites et risques restant ouverts

### Avant une diffusion « production non supervisée »

- **Sécurité** : le mode livré `allow` autorise écriture et shell sans confirmation.
  La deny-list limite les accidents mais n'est pas un sandbox. Utiliser `ask` hors VM,
  conteneur ou machine explicitement dédiée.
- **CI absente** : aucune pipeline versionnée ne rejoue encore automatiquement pytest,
  Ruff, self-test, build et smoke navigateur à chaque changement.
- **Couverture d'intégration** : les modules liés au matériel et aux services restent
  moins couverts. Le présent audit n'a pas chargé un vrai GGUF et n'a pas lancé de
  comparaison A/B avec juge LLM.
- **E2E long** : le smoke automatise le parcours principal ; les scénarios lents
  image-pendant-génération et note en vol restent décrits pour validation manuelle.

### Limitations produit connues

- Les hooks et agents fournis par des plugins ne sont pas encore pris en charge ; le
  store charge actuellement leurs skills.
- RAG pour catalogues de skills volumineux et audio restent au backlog.
- OCR précis des petits chiffres dans des scans : la vision aide mais ne garantit pas
  une transcription fiable.
- L'activité interne d'un `dispatch_agent` manque encore d'observabilité détaillée.
- Une modification manuelle de `model.toml` exige encore la régénération de
  `llama-swap.yaml` ou le passage par la console ; le re-scan à chaud n'est pas complet.
- Le save/restore de slot KV multimodal dépend du support de la version de llama.cpp ;
  Loom se replie proprement sur le re-prefill quand cette primitive répond 501.
- Les améliorations UX restantes (broadcast multi-panneaux, séparateurs redimensionnables,
  mobile, accessibilité et lisibilité avancée des compteurs) sont une roadmap, pas des
  fonctions annoncées comme livrées.

## Décision recommandée

- **Usage personnel/local supervisé** : feu vert.
- **Démonstration ou recette interne** : feu vert, avec le smoke E2E avant livraison.
- **Publication comme version stable générale** : feu orange jusqu'à l'ajout d'une CI,
  une recette avec au moins un vrai modèle cible et une politique de permission adaptée.
- **Environnement hostile ou multi-utilisateur** : feu rouge sans isolation supplémentaire.

## Commande de recette courte

```powershell
uv run pytest tests/ -q
uv run ruff check loom tests scripts evals
uv run python -m evals.run_eval --self-test
uv run python -m tests.e2e.smoke_ui
```
