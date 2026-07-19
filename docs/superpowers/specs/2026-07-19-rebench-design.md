# /rebench : recalibration d'un modèle local texte depuis le chat

Date : 2026-07-19 · Statut : validé (design approuvé dans le chat)

## Problème

La calibration topologique du contexte (loom-setup 4/4, topology.py) ne vise que
le default_model/premier modèle local et se saute si `[bench]` existe déjà. Pas
de moyen de re-calibrer UN modèle précis, tel qu'il est configuré (le 49152
d'ornith a été produit par des sondes manuelles). Objectif : `/rebench <id>`
dans le chat.

## Périmètre

- **Locaux texte UNIQUEMENT** (S.local_model_specs). Image/vidéo (ComfyUI) et
  distants (fenêtre du provider) : refus avec message explicite.
- Le modèle est benché **tel que configuré** : cpu_moe / n_cpu_moe / mmproj de
  son model.toml, threads/ngl de la table `[bench]` machine (doctrine MoE+GPU :
  ngl 99). PAS de sweep threads/ngl (= loom-setup), pas de sweep n_cpu_moe.
- Précondition : `[bench]` présente dans config/local.toml (machine calibrée),
  binaire llama-server en place — sinon message « lance uv run loom-setup ».

## Flux wizard (étapes `b_*`, machine à états pure)

1. `/rebench` → liste numérotée des locaux texte avec leur contexte actuel ;
   `/rebench <id>` → direct. Id connu mais non calibrable → « seuls les locaux
   texte-à-texte se calibrent (image/vidéo = ComfyUI, distant = provider) ».
2. `b_confirm` : annonce ~5-20 min, extinction du serveur modèle local (le banc
   charge le modèle lui-même), locaux indisponibles pendant la mesure (distants
   utilisables). « oui » lance → action `{"kind": "rebench", "id"}`.
3. Job en ARRIÈRE-PLAN (pattern download /add-model) : progression SSE via le
   callback `progress` de `topology.calibrate` (« calibration : sonde 32768… »),
   résultat POSTÉ ET PERSISTÉ à la fin (pattern _finish_install), onglet fermé
   compris.
4. Verdict :
   - contexte calibré ≠ actuel (plus grand OU plus petit — une réduction évite
     le spill silencieux) → message avec les chiffres (actuel → calibré, t/s
     validés en profondeur, mécanisme) + « Tape "oui" pour appliquer » ; l'état
     wizard `b_apply` est posé par le callback de fin.
   - égal → « déjà au top (actuel X, mesuré X, Y t/s) — rien à changer ».
5. `b_apply` + « oui » → action `{"kind": "rebench_apply"}` : `context` + note
   mécanisme écrits dans le model.toml (réutilise `_set_model_context` de
   loom.setup.cli), registres à chaud (S.local_model_specs[..]["context"],
   S.model_contexts), regen llama-swap.yaml — effet au prochain chargement.
   Toute autre réponse : rien n'est touché.

## Mécanique de mesure (réutilisée, pas dupliquée)

`read_gguf_meta` → `discover_topology(meta, has_gpu_backend, vram_total)` →
`memory_budget_mb(topo, vram, ram_total, headroom cfg)` →
`ServerProbe(server_bin, gguf, threads=[bench], ngl=doctrine, topology,
mmproj, cpu_moe, n_cpu_moe)` → `calibrate(probe, meta, topology, budget_mb,
progress=cb)`. Avant lancement : `S.server_manager.stop()`.

## Erreurs

- Serveur/binaire/`[bench]` manquants → message actionnable, pas de job.
- `calibrate` lève (RuntimeError/ValueError) → message d'échec persisté,
  contexte inchangé, wizard terminé.
- Un seul rebench à la fois (verrou/job global) : un second /rebench pendant un
  job répond « calibration déjà en cours ».

## Tests

- Unit wizard : liste filtrée, refus par type (deps.model_kind), flux confirm →
  action, b_apply oui/non.
- Routes (calibrate/probe stubbés via monkeypatch) : job → message final +
  état b_apply posé ; apply → model.toml écrit + registres + yaml ; refus
  quand `[bench]` absente ; verrou anti-double-job.
- E2E réel : /rebench gemma4-e4b-heretic (plus petit du parc) sur instance
  éphémère — mesure réelle, verdict affiché, application réelle si amélioration,
  vérif model.toml + sélection dans l'UI ensuite.
