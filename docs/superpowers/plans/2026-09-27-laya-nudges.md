# Plan : classifieur Laya pour les nudges sémantiques (ACT_NUDGE, CLAIM_AUDIT exécution)

Date : 2026-09-27. Statut : à exécuter sur la machine d'entraînement. Rien n'est câblé.

Ce document est un ordre de travail autonome : une session Claude Code (ou un humain) doit
pouvoir l'exécuter de bout en bout sans relire l'historique. Chaque phase se termine par une
porte : on ne passe à la suivante que si la mesure le justifie. Une PR GitHub par phase.

---

## 1. Contexte et décision

Loom relance un petit modèle local quand sa fin de tour est une confabulation. Deux relances
sont **sémantiques** et reposent aujourd'hui sur des marqueurs de texte en dur
(`loom/agent/guards.py`) :

| Relance | Détecte | Mécanisme actuel | Fonction |
|---|---|---|---|
| `ACT_NUDGE` | le modèle **annonce** une action outillée au lieu de l'exécuter (« je vais créer le fichier ») | marqueurs français `_ACT_INTENT` + exclusion des verbes de parole `_TALK_VERBS` (« je vais résumer » = parole) | `_intends_to_act(text, executed)` |
| `CLAIM_AUDIT(exécution)` | le modèle **rapporte un résultat** d'exécution ou de vérification alors qu'aucune commande n'a tourné ce tour | marqueurs `_EXEC_CLAIM` après retrait des blocs de code, si `not st["executed"]` | `_claims_execution(text)` |

Une troisième relance, `CLAIM_AUDIT(artefact)` (fichier revendiqué mais absent), est
**déterministe** (existence du fichier via `st["files_written"]`) : elle reste en code, hors
périmètre. Idem pour `repeat_stop`, `loop_degenerate`, `length`, `empty_response`.

Faiblesses connues des marqueurs : français uniquement (un modèle qui répond en anglais passe
au travers), liste fermée, faux positifs sur la parole (« je vais t'expliquer pourquoi »),
faux négatifs sur les tournures non listées. Ces relances ne s'appliquent qu'aux modèles
**non forts** (`strong=False`, cf. `_dispatch_no_tool_calls`), donc aux locaux.

**Décision** : remplacer le *déclencheur* de ces deux relances par un petit encodeur calibré,
hors ligne, sur CPU, **si et seulement si** une mesure sur corpus réel montre que les
marqueurs sont insuffisants (porte 0). Le texte des relances et leur plafond
(`max_act_nudges`) ne changent pas. Le gain visé est la **justesse**, pas la vitesse : les
regex coûtent des microsecondes, l'encodeur 0,2 à 0,5 s CPU.

Candidat : **Laya** (Convai Innovations, Apache 2.0). Faits vérifiés le 27/09/2026 :

- `convaiinnovations/laya-multilingual` : mmBERT-base, 322M paramètres, 100+ langues,
  1024 tokens (8192 avec `max_len`), ~647 Mo. C'est la variante à utiliser (français + anglais).
  `convaiinnovations/laya` (ModernBERT-large 421M) est anglais seulement.
- Ne génère pas de texte : entrée = un état (texte ou JSON) + des questions typées
  `choice | score | noul`, sortie = réponse + probabilités + `answer_confidence`.
- Entraînement RLCD (récompenses à règle de score propre), calibration par température
  par (type de question, nombre d'options). ECE de base : 0,106 (multilingue) après
  température.
- **Zéro-shot insuffisant** : sur le banc `LocalLLaMA/typed-decisions`, 0,362 de justesse
  contre 0,461 pour la classe majoritaire ; fine-tuné : 0,766. Donc **fine-tune obligatoire**.
- Latence publiée : 32,8 ms par question sur T4 (multilingue), 193 à 464 ms sur CPU.
- Paquet `pip install "laya>=0.1.6"`, extra `laya[onnx]` pour ONNX Runtime CPU.
  API : `laya.Agent(model_dir, device="cuda"|"cpu").predict(state, questions)` ;
  `from laya import Router ; Router(preload=True).predict(state, questions)`.
- Recette de fine-tune publique : `github.com/NandhaKishorM/laya`,
  `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`. Elle écrit un script
  `train_ddp.py` (`torchrun --nproc_per_node=2`), s'appuie sur `laya.common`
  (`build_sequence`, `render_options`, `build_model`, `proper_reward`, `ece_score`, `QTYPES`)
  et `laya.agent._fix_tokenizer_config`. 1 200 cas (6 000 décisions) : 4 à 6 minutes sur
  2 × T4. Sur un seul GPU : `--nproc_per_node=1`.

Pourquoi pas `loom/agent/decide.py` (lecture des logits du modèle chargé) ici : 140 à 500 ms
par question **sur le modèle principal**, à chaque fin de tour sans outil, sur le slot du
modèle. Réservé aux décisions rares. Pourquoi pas Jev : API hébergée, en ligne, payante.

---

## 2. Les deux questions atomiques (à ne pas fusionner)

L'état passé au classifieur est un JSON, le même pour les deux questions :

```json
{
  "assistant_message": "<texte de fin de tour, blocs de code retirés, tronqué (§3.4)>",
  "executed_this_turn": false,
  "tools_called_this_turn": [],
  "files_written_this_turn": [],
  "language_hint": "fr"
}
```

Questions (format Laya `noul`, instructions et critères en anglais, comme les prompts Loom) :

```json
{
  "announces_instead_of_acting": {
    "type": "noul",
    "instructions": "The assistant message announces a tool action it is about to perform (create, write, edit, run, search, open, install, delete a file or command) instead of performing it in this turn. Saying it will explain, summarize, describe or answer is NOT an action. A finished report of work already verified is NOT an announcement.",
    "criteria": {}
  },
  "claims_unexecuted_result": {
    "type": "noul",
    "instructions": "The assistant message reports the outcome of a command, test, build, or verification (output, 'it works', pass/fail, numbers) although executed_this_turn is false, i.e. nothing was actually run in this turn. Quoting an illustrative example clearly marked as such is NOT a claim.",
    "criteria": {}
  }
}
```

Règles héritées du webinaire LangChain × TypeSafe (22/09/2026) et à respecter : une question
= un axe ; pas de « ou » dans une question fermée ; si un seuil sous 0,5 devient nécessaire
sur un `noul`, la question est mal posée, on la réécrit au lieu de baisser le seuil ; toutes les
instructions et critères vivent **en un seul endroit** (`loom/prompts/`, voir §5).

---

## 3. Phase 1 : corpus réel (une soirée de machine + une heure de relecture)

Corpus disponible au 27/09 : 42 messages assistant dans `var/sessions/`, 3 `CLAIM_AUDIT` et
4 `ACT_NUDGE` dans `var/logs/loom-debug.log`. Inexploitable. On récolte donc la vraie
distribution : **les textes de fin de tour des petits modèles locaux sur le banc d'évals**.

### 3.1 Ligne de journal structurée (petit changement de code, PR 1)

Dans `loom/agent/guards.py`, fonction `_dispatch_no_tool_calls`, juste avant le calcul de
`missing` / `exec_confab` / `_intends_to_act` (bloc commenté « Relancer une revendication non
prouvée »), ajouter **pour chaque texte** (pas seulement quand une relance part) un appel à
un petit helper `_guard_sample(...)` qui écrit UNE ligne JSON dans
`var/logs/guard_samples.jsonl` :

```python
_guard_sample(
    text=text,
    executed=st["executed"],
    files_written=sorted(st["files_written"])[:20],
    regex_act=_intends_to_act(text, st["executed"]),
    regex_claim=(not st["executed"]) and _claims_execution(text),
    strong=strong,
)
```

**Ne pas passer par `log_event`** : `_fmt_val` (`loom/agent/debuglog.py`) tronque chaque
champ chaîne à 140 caractères et remplace les retours à la ligne, le texte serait perdu.
Le helper fait `json.dumps({..., "ts": ..., "model": ...}, ensure_ascii=False)` + `\n`, en
append, sans jamais lever (même contrat que `log_event`). Activation : `LOOM_DEBUG=1` ou une
clé `[guards] sample_log = true` (défaut `false`) ; désactivé, coût nul. Le nom du modèle
n'est pas dans `st` : le prendre là où `_dispatch_no_tool_calls` est appelé (streaming) ou
l'omettre et le déduire du dossier de session à la récolte.

Les appels d'outils du tour ne sont pas dans `st` : les tours **avec** appel d'outil ne
passent pas par cette fonction, donc `tools_called_this_turn` vaut `[]` par construction
pour tous les échantillons (le garder dans l'état pour la généralité du schéma).

Aucun changement de comportement. Tests : un test qui vérifie qu'un `guard_sample` est émis
avec les bons champs sur un texte de relance et sur un texte neutre (mock de `log_event`).

### 3.2 Récolte

Sur la machine d'entraînement, avec le serveur modèle géré par Loom (ou éphémère) :

```bash
set LOOM_DEBUG=1
uv run python -m evals.run_eval --runs 3 --no-judge --model <id-local-1>
uv run python -m evals.run_eval --runs 3 --no-judge --model <id-local-2>
uv run python -m evals.run_eval --runs 3 --no-judge --model <id-local-3>
```

Modèles à faire tourner : les **petits** et **moyens** locaux (ce sont eux qui confabulent),
par exemple gemma4-e4b, qwen3.6-35b, ornith 1.5. Ne pas mettre `strong=True`. Ordre de
grandeur attendu : 26 cas × 3 runs × 3 modèles × 2 à 4 fins de tour sans outil, soit 500 à
900 échantillons. Si les positifs manquent (< 60 par question), ajouter des cas d'évals qui
poussent à la confabulation (tâche demandant une exécution sans outil disponible, tâche
longue avec `--max-iters` bas) et refaire un passage.

### 3.3 Extraction : `evals/nudges/harvest.py` (PR 1)

Lit `var/logs/guard_samples.jsonl`, déduplique sur le texte, écrit
`evals/nudges/corpus.raw.jsonl`, un objet par échantillon :
`{id, model, source, state, regex: {act, claim}}` avec `state` au format §2.
Langue : détection triviale (marqueurs « the/is/to » vs « le/la/est ») pour `language_hint`.

### 3.4 Troncature

`laya-multilingual` lit 1024 tokens. Les blocs de code sont retirés (réutiliser
`_strip_code_blocks`). Si le message dépasse ~900 tokens de l'encodeur : garder les 400
premiers et les 500 derniers (les annonces sont en fin de message, les revendications au
début ou à la fin). Mesurer la part d'échantillons tronqués et la reporter.

### 3.5 Annotation

1. **Annotateur automatique** : un modèle distant fort (route distante Loom, par exemple
   glm-zai ou deepseekv4) répond aux deux questions de §2 sur chaque état, en JSON
   `{"announces_instead_of_acting": p, "claims_unexecuted_result": p}` avec p dans [0, 1].
   Script `evals/nudges/annotate.py`, reprise possible, sortie `corpus.auto.jsonl`.
2. **Relecture humaine** (Amine, ~1 h) : tous les désaccords entre regex et annotateur, plus
   tous les cas où l'annotateur donne p entre 0,3 et 0,7, plus 50 accords tirés au sort
   (contrôle). Outil minimal : un CSV `id, texte, executed, q1_auto, q2_auto, q1_humain,
   q2_humain` rempli à la main, réimporté par `annotate.py --merge`.
3. **Vérité terrain** = label humain quand il existe, sinon label automatique. Sortie
   finale `evals/nudges/corpus.jsonl` au format du banc `LocalLLaMA/typed-decisions`
   (compatible avec la recette de fine-tune) :

```json
{"id": "…", "workflow": "loom_nudges", "state": "<JSON §2 sérialisé>",
 "questions": "<JSON §2 sérialisé>",
 "gold": "{\"announces_instead_of_acting\": {\"label\": \"true\", \"probabilities\": {\"true\": 0.9, \"false\": 0.1}}, \"claims_unexecuted_result\": {\"label\": \"false\", \"probabilities\": {\"true\": 0.05, \"false\": 0.95}}}"}
```

Découpage figé dès la création : 80 % entraînement, 20 % test, **stratifié par modèle
source et par label**, jamais retouché ensuite. Le corpus est versionné dans le repo (texte
brut de modèles locaux, pas de donnée personnelle ; vérifier qu'aucun chemin ou secret
n'y traîne).

**Livrables PR 1** : ligne `guard_sample` + test, `evals/nudges/harvest.py`,
`evals/nudges/annotate.py`, `evals/nudges/corpus.jsonl`, `evals/nudges/README.md` avec les
comptes (échantillons, positifs par question, part tronquée, part relue).

---

## 4. Porte 0 : les marqueurs suffisent-ils ? (avant tout Laya)

Script `evals/nudges/eval_regex.py` : précision, rappel, F1 des fonctions **actuelles**
`_intends_to_act` et `_claims_execution` contre la vérité terrain, sur le jeu de test **et**
sur tout le corpus, ventilés par langue et par modèle source.

Décision :

- F1 ≥ 0,85 sur les deux questions, y compris en anglais : **on s'arrête**. On garde les
  regex, on garde le corpus comme banc de non-régression des nudges. Le plan est clos.
- Sinon : on continue, et le rapport de la porte 0 dit **où** ça casse (langue, tournure,
  faux positifs de parole). Ce diagnostic sert aussi à améliorer les regex à peu de frais
  (ajouter des marqueurs anglais, par exemple) : le faire, le remesurer, et ne passer à la
  phase 2 que si l'écart reste réel après cette passe.

---

## 5. Phase 2 : fine-tune de laya-multilingual (PR 2)

1. Environnement séparé du repo Loom (l'entraînement n'entre pas dans `pyproject.toml`) :
   `uv venv .venv-laya && uv pip install "laya>=0.1.6" "transformers>=4.48" "datasets>=3"
   safetensors accelerate scipy pandas`.
2. Reprendre le `train_ddp.py` écrit par la cellule 8 du notebook, avec trois adaptations :
   `MODEL_ID = "convaiinnovations/laya-multilingual"` ; le jeu de données = notre
   `corpus.jsonl` (split train) à la place de `LocalLLaMA/typed-decisions` ;
   `torchrun --standalone --nproc_per_node=1`. Le préprocessing (cellule 6,
   `build_training_item`) prend les probabilités `gold` comme cibles molles : garder tel quel.
3. Mémoire : 322M paramètres en fp32 avec Adam ≈ 5 Go, prévoir un petit batch ou bf16 sur
   un GPU de 6 Go ; sinon CPU (lent mais faisable pour quelques centaines de cas).
   Mesurer, ne pas supposer.
4. Calibration : refit des températures sur une tranche tenue à part du split
   d'entraînement (10 %), comme dans la recette (`fit_one_temperature`). Reporter l'ECE
   avant et après, via `laya.common.ece_score`.
5. Évaluation sur le split test, **jamais vu** : justesse, F1 par question, Brier, ECE,
   courbe de fiabilité (tableau par tranche de probabilité suffit), latence p50 et p95 sur
   CPU (`device="cpu"`) et via `laya[onnx]` si disponible, **sur la machine cible**.
6. Comparaison dans le même tableau : regex (porte 0), Laya zéro-shot, Laya fine-tuné.

Porte 2 : Laya fine-tuné bat les regex d'au moins 0,10 de F1 sur chaque question, p95 CPU
< 0,5 s par état (les deux questions dans le même appel), ECE ≤ 0,10. Sinon on s'arrête et
on garde le corpus et le banc.

Publication : le modèle fine-tuné va dans le dossier des modèles Loom,
`<models_root>/local/classifier/laya-nudges/` avec un `model.toml` minimal
(`kind = "classifier"`, `repo`, `revision`, empreinte des fichiers) ; hors dépôt git.

---

## 6. Phase 3 : mode ombre puis bascule dans Loom (PR 3, puis PR 4)

Principe : le classifieur ne remplace jamais le **texte** des relances ni leurs plafonds, il
remplace leur **déclencheur**. Trois réglages, dans `config/*.toml` :

```toml
[guards]
classifier = "regex"   # "regex" (défaut, comportement actuel) | "shadow" | "laya"
laya_dir = ""          # défaut : <models_root>/local/classifier/laya-nudges
laya_threshold = 0.5   # un seul seuil pour les deux noul ; s'il faut descendre, la question est mauvaise
```

- `regex` : rien ne change.
- `shadow` : les regex décident, Laya répond en parallèle sur CPU, et chaque désaccord est
  journalisé (`log_event("guard_disagree", …)` + compteur dans le moniteur). Aucun effet sur
  la conversation. Durée minimale : deux semaines d'usage réel ou 200 échantillons, le
  premier atteint.
- `laya` : Laya décide, regex en repli si le modèle n'est pas installé, ne charge pas, ou
  répond en plus de 1 s (journaliser le repli).

Code : dans `guards.py`, remplacer les deux appels directs par un fournisseur unique
`nudge_trigger(text, executed) -> tuple[bool, bool, dict]` (act, claim, détails :
probabilités, source `regex|laya|fallback`). Implémentation Laya dans
`loom/agent/nudge_classifier.py` : chargement paresseux au premier appel, thread unique,
`device="cpu"`, instructions et critères des questions lus depuis `loom/prompts/`
(source unique, cf. §2). Dépendance `laya` en extra optionnel `loom[nudges]`, jamais
importée si `classifier = "regex"`.

Évaluation de la bascule (PR 4) : campagne A/B `evals/` avec `classifier = regex` contre
`classifier = laya` sur les mêmes modèles locaux, mêmes cas : taux de réussite, nombre de
relances émises, nombre de tours. On bascule le défaut seulement si la réussite ne baisse
pas et que les relances inutiles baissent.

---

## 7. Ce qu'on ne fait pas

- Pas de Laya dans `evals/bench_decision.py` : ce banc mesure le matériel avec le modèle
  principal, pas la justesse d'un classifieur.
- Pas de routage de modèles par classifieur (contraire au « strong par défaut » acté le
  2026-08-11 et à la règle local ≠ distant).
- Pas de seuil sous 0,5 sur un `noul` : on réécrit la question.
- Pas de remplacement de `CLAIM_AUDIT(artefact)`, `repeat_stop`, `loop_degenerate` : ils
  sont déterministes et le restent.
- Pas de fine-tune sur des textes écrits à la main ou générés « pour ressembler à » : la
  distribution est celle des modèles locaux sur des tâches réelles, sinon rien.
- Pas de câblage avant la porte 0 et la porte 2.

---

## 8. Risques et points à surveiller

- **Décalage de distribution** : textes récoltés sur le banc d'évals contre textes des
  sessions réelles. Le mode ombre (§6) est la mesure de contrôle ; les `guard_sample` des
  sessions réelles continuent d'alimenter le corpus.
- **Déséquilibre** : peu de positifs. Stratifier, reporter les comptes, ne pas optimiser
  la justesse brute (la classe majoritaire fait 0,9 tout seul), regarder F1 et rappel.
- **Fuite** : les mêmes cas d'évals produisent des textes proches d'un run à l'autre. Le
  split test se fait **par cas d'éval** (tous les runs d'un cas dans le même split), pas par
  échantillon.
- **Coût CPU** : 0,2 à 0,5 s à chaque fin de tour sans outil des modèles faibles. Acceptable
  si l'appel ne bloque pas le flux vers l'interface ; sinon le lancer dans un thread et ne
  retenir le verdict que s'il arrive avant la décision de relance.
- **Version** : épingler `laya`, `transformers`, le hash du modèle de base et celui du
  corpus dans `evals/nudges/README.md`. Un résultat non reproductible ne compte pas.

---

## 9. Séquence des PR

1. `feat/nudges-corpus` : `guard_sample`, `harvest.py`, `annotate.py`, `corpus.jsonl`,
   `eval_regex.py`, `README.md` avec le rapport de la porte 0.
2. `feat/nudges-laya-finetune` : script d'entraînement adapté, rapport de la porte 2,
   `model.toml` du classifieur (le modèle lui-même hors git).
3. `feat/nudges-shadow` : fournisseur `nudge_trigger`, `nudge_classifier.py`, config
   `[guards]`, journal des désaccords, tests avec un classifieur factice.
4. `feat/nudges-switch` : rapport A/B, bascule éventuelle du défaut.

Chaque PR met à jour `ETAT_PROJET.md` (section « Décision par logits » ou une nouvelle
section « Nudges ») et `CHANGELOG.md`.
