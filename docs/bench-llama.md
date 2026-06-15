# Benchmark llama-bench — tuner les configs MoE au lieu de deviner

Campagne du 2026-06-11. Machine : i7-10750H (6c/12t), RTX 2060 mobile 6 Go, 32 Go RAM.
Build llama.cpp : 9442 (d4c8e2c29).

## Le principe

`llama-bench` (livré avec llama.cpp, `C:\tools\llama\llama-bench.exe`) mesure le prefill
(`pp512`) et la génération (`tg128`) pour le **produit cartésien** des valeurs passées.
On ne devine plus `--n-cpu-moe` : on balaye et on lit le tableau.

```powershell
llama-bench.exe -m <model.gguf> `
  -ngl 999 -ncmoe 48,44,40,36,32 -fa on -ctk q8_0 -ctv q8_0 -t 6 --prio 2 `
  -p 512 -n 128 -r 2 -o md --progress
```

- `-ncmoe N` = `--n-cpu-moe` : N couches d'experts en RAM, le reste sur GPU.
- Mêmes flags que `server_args.py` (`-fa on`, KV q8_0, `-t 6`) pour que les chiffres
  soient transposables au serveur réel.
- Chaque config recharge le modèle : ~2 min par valeur, lancer en arrière-plan.

## Résultats — Gemma 4 26B-A4B (sweep ncmoe)

### QAT UD-Q4_K_XL (13,26 GiB, unsloth)

| ncmoe | pp512 (t/s) | tg128 (t/s) |
|------:|------------:|------------:|
| 48 (= `--cpu-moe`) | 199,8 ± 17,5 | 13,90 ± 1,73 |
| 44 | 218,2 ± 17,2 | 13,71 ± 0,09 |
| **40** | **234,2 ± 8,0** | **14,05 ± 0,35** |
| 36 | 230,5 ± 10,6 | 13,75 ± 0,78 |
| 32 | 232,4 ± 12,3 | 14,18 ± 0,48 |

### Q4_K_M uncensored (15,63 GiB, mradermacher) — baseline

| ncmoe | pp512 (t/s) | tg128 (t/s) |
|------:|------------:|------------:|
| 48 (config historique) | 204,3 ± 12,4 | 13,61 ± 0,38 |
| 40 | 193,7 ± 21,3 | **11,81** ± 0,62 (régression) |

## Lecture

- **QAT à ncmoe 40 = la meilleure config** : +15 % de prefill vs la config historique
  (234 vs 204), tg légèrement meilleur. Appliqué dans son `model.toml`.
- Le Q4_K_M **régresse** quand on met des experts sur GPU (ses experts sont ~18 % plus
  gros → pression VRAM). Le balayage évite précisément ce genre de fausse bonne idée.
- Sous 40, le gain plafonne (bruit) : inutile de tasser la VRAM davantage — en usage
  réel le KV 16K + buffers ont besoin de la marge.
- Le QAT est par ailleurs **meilleur en qualité** (quantisation apprise à l'entraînement,
  ~BF16 à 4 bits) et apporte la vision (mmproj).

## MTP (décodage spéculatif) — testé puis ABANDONNÉ (2026-06-11)

Essayé : drafter `mtp-gemma-4-26B-A4B-it.gguf` (240 Mo) via
`--spec-draft-model <gguf> --spec-type draft-mtp --spec-draft-n-max 4`. **Retiré du code.**
Le micro-bench greedy+code montrait ×1,95 (10,4 → 20,3 t/s), mais en usage réel le gain
de génération de tokens est **médiocre** et ne justifie pas les contraintes :

- **Build incompatible.** L'arch `gemma4-assistant` du drafter exige llama.cpp ≥ b9587
  (mergé le 2026-06-07). Notre build de référence b9442 refuse de charger le drafter. Le
  b9596 installé pour l'essayer **régresse de -18 % pp / -7 % tg** sur cette RTX 2060 pour
  TOUS les modèles sans drafter (uncensored, qwen) → on reste sur b9442.
- **S'exclut du multimodal.** Spéculation et `--mmproj` s'excluent dans llama.cpp
  (issue ggml-org/llama.cpp#19712) : avec vision, le MTP est désactivé en silence. On ne
  peut pas avoir vitesse ET vision sur le même modèle.
- **MoE non séparables.** Le gain réel ne tient pas avec l'offload des experts (`--cpu-moe`)
  qui est notre contrainte de base sur 6 Go : le drafter ne compense pas le coût.

Verdict : pour un gain de tg décevant en conditions réelles + un build qui pénalise tout
le reste, MTP n'est pas retenu. À ré-évaluer si une future release llama.cpp lève la
régression et l'exclusion multimodale.
