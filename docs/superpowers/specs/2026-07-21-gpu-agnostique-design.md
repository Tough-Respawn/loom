# GPU agnostique — détecté quoi qu'il arrive, utilisé par défaut

**Date** : 2026-07-21 · **Statut** : validé (design approuvé en session)

## Problème

`detect_hardware()` ne connaît que NVIDIA (`nvidia-smi`) : sur une machine AMD
(Radeon 860M iGPU, vécu) ou Intel, le setup affiche « GPU : aucun », le budget
ignore la VRAM, et surtout le profil runtime a `has_gpu=False` → la
recommandation auto d'offload de `resolve_ngl` (priorité 4) ne s'applique
jamais : le GPU présent n'est pas utilisé, sauf si le bench tourne ET gagne.
Exigences : (1) le GPU doit être détecté quelle que soit la techno — Loom est
agnostique ; (2) s'il est trouvé, il est pris par défaut, sans dépendre d'un
bench qu'on ne relancera pas forcément.

## Décision

**Source de vérité = le binaire llama.cpp installé** : `llama-server
--list-devices` liste exactement les GPU que *ce* binaire sait exploiter
(Vulkan AMD/Intel/NVIDIA, CUDA, Metal…). Si llama le liste, on le prend ;
s'il ne le liste pas, le « détecter » autrement ne servirait à rien.
Alternatives écartées : sondes par vendor (nvidia-smi + WMI/DXGI +
vulkaninfo — fragile, OS-spécifique, peut voir un GPU inexploitable) ;
binding Vulkan Python (dépendance lourde).

### hardware.py

- `parse_list_devices(output)` : parse les lignes
  `Vulkan0: AMD Radeon(TM) 860M Graphics (36682 MiB, 34848 MiB free)` →
  `{backend, index, name, total_mb, free_mb}`.
- `detect_hardware(server_bin=None)` : avec `server_bin`, interroge
  `--list-devices` (1er device = profil ; sortie sans device = CPU-only, le
  binaire fait foi). Sans binaire, ou si l'appel échoue : repli nvidia-smi
  (inchangé — seul cas où il faut savoir *avant* d'avoir le binaire, pour
  choisir le build CUDA).
- `HardwareProfile` gagne `vram_total_mb`, `backend`, `vram_is_discrete` et la
  propriété `budget_vram_mb` : la VRAM n'est ADDITIONNÉE à la RAM dans un
  budget que si elle est prouvée discrète (nvidia-smi/CUDA). Un iGPU Vulkan
  déclare la RAM partagée comme sa mémoire → l'additionner double-compterait.
  Conséquence assumée : un dGPU AMD/Intel a un budget conservateur (RAM seule,
  comme aujourd'hui) mais un offload correct.

### serve.py (runtime — « s'il trouve, il prend »)

`detect_hardware(cfg.server_bin)` aux deux sites (boot + regenerate_swap_yaml).
Le profil agnostique irrigue `resolve_ngl` : la recommandation auto
proportionnelle (priorité 4, code existant) s'applique enfin sur AMD/Intel —
offload par défaut à chaque démarrage, pour chaque modèle, sans bench et sans
valeur figée. La précédence existante reste : model.toml > override global >
auto.

### loom-setup

- Étape 1 : affichage honnête sans NVIDIA (« détection complète après l'étape
  binaire »), sonde NVIDIA conservée pour le choix CUDA vs Vulkan.
- Après l'étape 2 : re-détection via le binaire (profil mis à jour + affiché).
- Budget étape 3 : `hw.budget_vram_mb` partout (anti-double-comptage).
- Bench : candidats `-ngl` via `bench.ngl_candidates()` — `[0]` sans backend
  GPU ; sinon `{0, 99}` + la recommandation proportionnelle intermédiaire si la
  VRAM ne couvre qu'une partie du modèle. Le résultat mesuré s'écrit dans
  `[override] n_gpu_layers` **même quand c'est 0** (dès lors que le GPU a été
  testé) : la mesure a toujours le dernier mot, y compris « CPU plus rapide ».

### Hors périmètre (assumé)

- `gpu_vram_total_mb` (topologie) reste NVIDIA : la topologie VRAM raisonne en
  mémoire discrète ; un iGPU unifié EST la RAM → TOPO_RAM est correct.
- Le chemin curl_cffi/web et le NPU ne sont pas concernés.

## Tests / preuve

- `parse_list_devices` : ligne Vulkan réelle, ligne CUDA, multi-devices,
  sortie vide/malformée.
- `detect_hardware` : binaire répond → profil agnostique ; binaire répond sans
  device → CPU-only même si nvidia-smi existe ; pas de binaire → repli NVIDIA.
- `budget_vram_mb` : discret vs unifié. `ngl_candidates` : sans GPU, offload
  total, offload partiel (candidat intermédiaire).
- Preuve réelle : `loom-setup` sur la machine Radeon 860M doit afficher le GPU,
  et le serve doit lancer llama-server avec `-ngl > 0`.
