# Glossaire — inférence locale et matériel

Les définitions s'appuient sur les chiffres réels du projet (poste Dell G7 : RTX 2060 6 Go,
64 Go DDR4 ; ornith Q8 35B MoE calibré `n_cpu_moe = 40`).

## Les deux phases d'une génération

**Prefill** : la phase de « lecture » du prompt. Avant d'écrire le moindre mot, le modèle doit ingérer tout ce qu'on lui donne (prompt système, historique, résultats d'outils) et calculer son état interne. C'est massivement parallèle : gros calcul matriciel, borné par la puissance de calcul (le GPU excelle). Se mesure en tokens/seconde ; le poste fait 142 t/s en prefill sur ornith Q8. C'est le prefill qui cause l'attente avant le premier mot, celle qu'on a combattue avec le cache KV.

**Décode** : la phase d'écriture, un token à la fois. Chaque token exige de relire une part des poids du modèle : c'est borné par la bande passante mémoire, pas par le calcul. Les 14,4 t/s d'ornith Q8 en local, c'est ça. Règle simple : prefill = muscle de calcul, décode = tuyau mémoire.

## Le vocabulaire modèle

**Token** : l'unité de texte du modèle (un mot court, un bout de mot, un signe). En français, ~1,5-2 tokens par mot.

**Paramètres (B)** : les « poids » appris du modèle, en milliards (B = billions). Ornith = 35B, DeepSeek R1 = 671B.

**Dense vs MoE** : un modèle *dense* utilise tous ses paramètres à chaque token (Llama 405B lit 405B de poids par token, intenable). Un **MoE** (Mixture of Experts) découpe ses couches en dizaines d'« experts » et n'en réveille que quelques-uns par token : un routeur choisit. DeepSeek 671B n'active que 37B par token. C'est ce qui rend le quasi-frontier local possible.

**Paramètres actifs** : la part du MoE réellement lue à chaque token (le « 37B actifs » de DeepSeek, les ~3B d'ornith). C'est *ça* qui détermine la vitesse de décode, pas la taille totale.

**Frontier** : les modèles au sommet du classement mondial du moment (GPT, Claude, DeepSeek, Kimi...). « Quasi-frontier » = les versions ouvertes qui s'en approchent.

**Quantization (Q4, Q8)** : compression des poids. Le modèle est entraîné en 16 bits par paramètre ; on les stocke sur 8 bits (Q8, quasi sans perte) ou 4 bits (Q4, perte légère). Divise la mémoire et le débit nécessaires : ornith Q8 = 37 Go, Q4 = ~20 Go.

**GGUF** : le format de fichier des modèles quantizés pour llama.cpp. L'`ornith-q8.gguf` de `C:/loom-models`.

**Contexte** : tout ce que le modèle « voit » en entrée (fenêtre maximale de tokens). Ornith est configuré à 24 576.

**KV cache** : la mémoire de travail de la conversation. Pendant le prefill, le modèle calcule pour chaque token des vecteurs Key/Value ; les garder évite de tout recalculer au message suivant. Perdre ce cache = re-prefill complet = le « re-prefill interminable » qu'on a fixé. Le save/restore de slots, le warm_context : tout ça protège ce cache.

## Le vocabulaire matériel

**VRAM** : la mémoire embarquée sur la carte graphique. La 2060 en a 6 Go. Très rapide, très chère, c'est la denrée rare de tout ce sujet.

**Bande passante mémoire (Go/s)** : le débit auquel on peut lire la mémoire. LA grandeur qui gouverne le décode : t/s ≈ bande passante ÷ Go lus par token. L'échelle : DDR4 laptop ~40 Go/s, VPS OVH mesuré ~30, Strix Halo 256, RTX 3090 936, RTX 5090 1792, H100 ~3350.

**GDDR / HBM / (LP)DDR** : les familles de mémoire, par débit croissant de prix et de vitesse. DDR = la RAM classique d'un PC ; LPDDR = version soudée basse conso (celle qui a flambé en 2026) ; GDDR = celle des cartes graphiques ; HBM = l'empilée ultra-rapide des GPU datacenter (H100).

**Mémoire unifiée** : CPU et GPU partagent la même mémoire physique rapide (Mac M3 Ultra, Strix Halo). Plus de frontière VRAM/RAM : un gros modèle tient « en un seul morceau ».

**Offload / `--cpu-moe` / `n_cpu_moe`** : répartir un modèle trop gros pour la VRAM : les experts du MoE vont en RAM (lus par le CPU), l'attention et les couches partagées restent sur GPU. `n_cpu_moe = 40` : le réglage calibré au banc (nombre de couches dont les experts partent en RAM). C'est l'architecture actuelle du poste.

**TFLOPS** : les « chevaux » de calcul d'une puce (mille milliards d'opérations/s). Détermine le prefill. 2060 ~6,5 ; 3090 ~35 ; L40S ~90.

**eGPU** : carte graphique externe reliée au laptop en Thunderbolt.

**Headless** : une machine sans écran ni clavier, pilotée par le réseau. La tour à 3090 dans un coin, le laptop comme interface.

**VPS / bare-metal** : serveur loué virtualisé et partagé vs serveur physique entier dédié. Le VPS partage la bande passante mémoire de l'hôte, d'où les 30 Go/s mesurés sur le VPS Lille.

## Le vocabulaire logiciel

**llama.cpp / llama-server** : le moteur d'inférence que Loom utilise. Optimisé pour le matériel modeste, roi du GGUF et de l'offload CPU/GPU. llama-server = son mode serveur HTTP (l'API OpenAI-compatible sur :8080).

**llama-swap** : le proxy qui démarre/éteint le bon llama-server selon le modèle demandé (la bascule ornith/qwen/image).

**vLLM** : l'autre grand moteur d'inférence, pensé datacenter : il sert *plusieurs* requêtes en parallèle sur *plusieurs* GPU avec un débit agrégé maximal (continuous batching, PagedAttention pour gérer les KV caches de dizaines d'utilisateurs). C'est le choix naturel dès qu'on a 4+ GPU et du tensor parallelism ; sur mono-GPU avec offload RAM, llama.cpp reste le bon outil. Résumé : llama.cpp = un utilisateur sur petit matos, vLLM = du débit industriel sur gros matos.

**Tensor parallelism / expert parallelism** : découper un modèle *entre plusieurs GPU* : chaque carte calcule sa tranche de chaque couche (tensor) ou héberge ses experts (expert). C'est ce qui ferait tourner une ferme de 3090 ou des 4× RTX 6000.

**Batch / batching** : traiter plusieurs requêtes (ou plusieurs tokens de prefill) d'un coup pour saturer le matériel. Le fan-out Gemma de juin, c'était exactement ça.

**Prompt caching** : réutiliser le KV cache d'un préfixe déjà vu (le prompt système identique d'un tour à l'autre). Côté API distantes, c'est la remise de 94-98 % mesurée chez Z.ai ; côté local, c'est le slot KV.
