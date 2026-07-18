# Sonde contexte/VRAM — Ornith 35B MoE, RTX 2060 6 Go (2026-07-18)

**Question** : la calibration de juin (« contexte 24576, borné par 6 Go de VRAM ») est-elle
encore la meilleure config possible pour cette machine, après le passage à 64 Go de RAM ?

**Méthode** : serveurs éphémères, ligne de commande construite par `server_args.py`
lui-même (le conseilleur simule l'exécutant) — flags réels : `-ngl 99`, `--n-cpu-moe 40`,
flash-attn, KV q8_0, `--no-mmap`, mmproj chargé. Mesures : VRAM nvidia-smi + `timings`
natifs de llama-server. Config de production intouchée pendant toute la campagne.

## Sonde 1 — capacité (échelle de contextes, prompt ~8 k)

| ctx | chargement | VRAM après gen | prefill t/s | décode t/s |
|---|---|---|---|---|
| 24 576 | 111 s (froid) | 3 294 / 6 144 | 136,8 * | 7,5 * |
| 32 768 | 52 s | 3 333 / 6 144 | 257,6 | 14,4 |
| 49 152 | 49 s | 3 477 / 6 144 | 257,2 | 13,5 |
| 65 536 | 50 s | 3 663 / 6 144 | 257,3 | 14,1 |

\* Premier barreau = artefact de démarrage à froid (cache disque, alloc pinnée, clocks GPU) —
confirmé par la sonde 2 qui, avec un warmup jetable, retrouve les débits pleins d'entrée.
La baseline VRAM (3,3/6 Go) reproduit la mesure du 2026-07-10 : sonde validée.

## Sonde 2 — profondeur (ctx 49 152, warmup, un seul serveur)

| profondeur réelle | prefill t/s | décode t/s | VRAM |
|---|---|---|---|
| 18 901 tokens | 198,8 | 13,6 | 3 497 / 6 144 |
| 31 921 | 191,4 | 13,0 | 3 543 / 6 144 |
| 44 101 | 195,1 | 12,5 | 3 569 / 6 144 |

Érosion du décode : **−13 % entre ~6 k et ~44 k de profondeur** (14,4 → 12,5 t/s).
Pas de falaise, pas de spill (2,6 Go de VRAM libres au pire). Prefill en profondeur :
~195 t/s, soit ~3,8 min pour remplir 44 k à froid (amorti par le prompt cache et la
prime au chargement en usage réel).

## Les trois verdicts

1. **Le mur de 24576 n'existe pas (pour ce modèle).** 65 536 tient dans 3,66/6,14 Go
   sans perte de vitesse. La borne de juin décrivait une autre époque de la config
   (avant flash-attn/KV q8_0/no-mmap systématiques) ou une hypothèse jamais re-testée.

2. **Le coût VRAM marginal mesuré est ~9 Ko/token — 5× sous la formule.** La formule
   « header GGUF » donne 43,5 Ko/token (q8_0) : l'architecture qwen35moe n'a
   vraisemblablement qu'une minorité de couches en attention pleine (le reste en
   fenêtre glissante, KV borné par la fenêtre). Conséquence de méthode : **aucune
   formule à base de header ne prédit le KV réel de ce parc — seule la pente mesurée
   entre deux barreaux fait foi.** C'est la pierre angulaire du futur bench v2.

3. **La justesse-par-coïncidence du bench actuel, constatée sur pièces** : avec la RAM
   vidée (52,8 Go dispo), `compute_context` donne 198 656 tokens… ramenés à 24 576 par
   le cap. Le bon chiffre, par le mauvais mécanisme — le cap masque une formule fausse
   (hypothèse f16, topologie RAM seule, entrée volatile `ram_available`).

## Recommandations

- **Ornith : épingler `context = 49152`** dans son `model.toml` (×2 de fenêtre,
  2,6 Go de marge VRAM conservés, −13 % de décode en profondeur maximale — soutenable).
  65 536 tient aussi mais mange la marge vision/buffers pour un gain d'usage marginal.
- **Par modèle, jamais global** : la pente KV est propre à chaque architecture
  (gemma/denses = attention pleine, pente ~pleine formule). Chaque modèle du parc
  mérite sa sonde de 10 min avant tout changement.
- **Bench v2 (`loom-setup`)** : remplacer la formule par la méthode des deux barreaux
  (pente VRAM mesurée), découvrir la topologie (dense/cpu-moe/nkvo) depuis le matériel
  et `expert_count`, importer `server_args.py`, écrire la décision AVEC son mécanisme
  dans `local.toml`, fail-loud au boot sur valeur de repli. Fixture « machine dorée » :
  les tables ci-dessus.

## Patterns d'audit ayant mené à cette campagne (2026-07-18)

P1 topologie unique modélisée sur trois exploitées · P2 le conseilleur n'importe pas
l'exécutant (`server_args.py` ignoré) · P3 calibration sur `ram_available` volatile ·
P4 cap masquant l'erreur (justesse-par-coïncidence) · P5 sweep threads×ngl alors que le
parc est 100 % MoE (n_cpu_moe absent) · P6 aucun test « machine dorée ».
Méta-pattern de la semaine : le plausible-silencieux-faux — l'antidote est fail-loud +
valeurs effectives + mesure.
