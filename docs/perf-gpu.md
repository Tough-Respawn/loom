# Performance GPU — Gemma 3n E4B sur RTX 2060 (6 Go)

Investigation menée le 2026-06-03. Machine : i7-10750H (6c/12t), RTX 2060 mobile 6 Go, 32 Go RAM.

## Le problème

Génération lente (~9-12 tok/s), **GPU à ~35 %** pendant l'inférence alors que le CPU
sature. Cause racine : **Gemma 3n garde ses Per-Layer Embeddings (PLE) en RAM système
par design** (c'est ce qui permet à E4B ≈ 6.9 B params réels de tenir dans ~4 B de VRAM).
Résultat : à chaque token, une étape d'embedding tourne sur CPU → le GPU attend
(contrôleur mémoire à ~11 % = il ne lit même pas beaucoup la VRAM). Voir llama.cpp
issues #22243 et #22926. **Pas de flag magique** : le plancher single-stream est en
partie incompressible. Le levier, c'est le **continuous batching**.

## Les réglages appliqués (`server_args.py`, `gpu_tuning=True` quand GPU)

```
-ngl 99            # offload TOUT (couches + output/embeddings), pas seulement 35
-fa on             # Flash-Attention : attention plus rapide + cache KV ~÷2
-ctk q8_0 -ctv q8_0  # cache KV quantifié (libère de la VRAM)
-b 2048 -ub 512    # gros batch de prompt (prefill ~×2)
-t 6               # cœurs PHYSIQUES (≈ logiques/2) — au-delà, contention HT
--prio 2           # priorité process (Windows)
```
+ `n_parallel=4` (auto) = continuous batching déjà actif côté llama-server.
Override `n_gpu_layers = 99` dans `config/local.toml`.

## Benchmark (avant → après), prompt 256 tokens

| | single-stream | 4 requêtes concurrentes | prompt |
|---|---|---|---|
| **Baseline** (`-ngl 35`, pas de FA, `-t 12`) | 12 tok/s | 32 tok/s cumulés | 31 tok/s |
| **Tuned** | **42 tok/s** (×3.5) | **100 tok/s cumulés** (×3) | 73 tok/s |
| GPU util pendant 4 flux | — | **82-83 %** (sans spill, ~4 Go/6) | — |

Conclusion : **single-stream ×3.5** (le CPU libre fait voler Gemma 3n), et **le GPU ne
se sature (83 %) qu'en batch**. Pour exploiter la carte à fond → générer en parallèle.

## Fan-out (`loom/parallel.py`) — exploiter les 100 tok/s

En séquentiel, le petit modèle sur-raisonne (×4 reasoning vs content) et P1.1 sérialise
les écritures → il ne finit jamais N fichiers (morpion : 360 s, 2/5 fichiers, incomplet).

Le fan-out : `plan_files` (décompose en design partagé + liste de fichiers) → `generate_files`
(une requête FOCALISÉE par fichier, `thinking=False`, lancées EN PARALLÈLE → batchées par
llama-server) → écriture → `verify_files` → `fix_files` (boucle fermée si défauts).

Résultat morpion : **complet, vérifié, JOUABLE en ~62 s** (vs 360 s incomplet).
Cohérence inter-fichiers via le `design` (contrat strict : id/classes/sélecteurs exacts).

## Verify d'interaction (`verify_web.js`)

Le rendu ne prouve pas la jouabilité. Le vérificateur joue maintenant **2 coups
successifs** (clic case 1 → marque ; clic case 2 → re-query → doit aussi changer) : ça
attrape le sélecteur incohérent (button vs div → 0 écouteur) ET le « figé après le 1er
coup » (re-rendu sans ré-attacher les écouteurs).

## Piste si on veut saturer le GPU en single-stream

Gemma 3n est plafonné par le PLE-CPU. Un dense 3-4 B (Qwen2.5-3B, Llama-3.2-3B) Q4 tient
entièrement en VRAM, sans étape CPU par token → ~50 tok/s et GPU saturé. À considérer si
le single-stream rapide prime sur le côté « non censuré » de ce Gemma.
