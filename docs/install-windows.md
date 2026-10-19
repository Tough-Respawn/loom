# Install llama-server (Windows)

## Voie rapide (recommandée)

```powershell
uv sync
uv run loom-setup              # installeur guidé (binaire llama.cpp + modèle)
uv run python -m loom.web      # l'UI démarre le serveur modèle toute seule
```

(`serve.py` lancé sur machine vierge enchaîne aussi tout seul sur l'installeur.)
`loom-setup` détecte la machine (GPU/VRAM/RAM), télécharge la dernière release
stable de llama.cpp (build CUDA si NVIDIA, sinon Vulkan/CPU) dans
`var/runtime/llama/<tag>/`, écrit `[server] bin` dans `config/local.toml`,
propose un premier modèle qui tient dans ta mémoire, puis **benche le matériel**
(llama-bench sur TON modèle : threads, offload GPU, contexte qui tient en RAM)
et écrit les meilleurs réglages. Chaque action est confirmée avant d'être
faite ; bilan dans `var/logs/setup.log`. Relançable : ne refait que ce qui
manque (bench re-mesurable en supprimant la table `[bench]` de local.toml).

**Mettre à jour llama.cpp** : relancer `uv run loom-setup`. Si la dernière release
stable de llama.cpp désigne une build plus récente que le binaire configuré, le setup
la propose, l'installe à côté dans `var/runtime/llama/<tag>/` et repointe `[server] bin` ;
l'ancien binaire reste en place (revenir = remettre son chemin dans `[server] bin`).
Le bandeau de l'UI suit la même release et affiche cette commande. Un build maison
(`BUILD.txt` à côté du binaire) se met à jour en recompilant (`[server] rebuild_hint`).

**Anciennes builds** (~1,2 Go chacune) : à chaque passage, le setup propose de supprimer
celles de `var/runtime/llama/` devenues inutiles. Il garde toujours la build en service,
la plus récente des autres (retour arrière) et toute build citée par un `server_bin` de
model.toml. Réponse par défaut « non », jamais de suppression sous `--yes`. Les binaires
installés à la main dans un autre répertoire ne sont jamais touchés.

## Lanceur du Bureau (fenêtre d'application liée)

`python -m loom.web --app` (ou `[chat] app_window = true` dans `config/local.toml`) ouvre
l'interface dans une fenêtre d'application d'un profil navigateur dédié
(`var/browser-profile` ; Brave, Edge ou Chrome détectés, `[chat] browser` pour imposer un
exécutable). La fenêtre est liée à loom.web : la fermer arrête Loom et décharge le modèle ;
fermer la console ferme aussi la fenêtre (même Job Object « kill-on-close » que le serveur
modèle).

Raccourci : `docs/loom-launcher/start-loom.ps1` (à copier où l'on veut, le raccourci
pointe dessus avec `powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File
<chemin>`). Il lance la console minimisée avec `--app` ; si Loom tourne déjà, il rouvre
seulement une fenêtre sur l'instance existante.

## À la main (repli)

Remplacer `<repertoire-outils>` par le dossier d'installation choisi dans les
commandes et la configuration ci-dessous.

1. Récupérer la release Windows CUDA de llama.cpp (`...-bin-win-cuda-x64.zip`
   **et** le zip `cudart-...` qui porte les DLL CUDA).
2. Extraire les deux dans `<repertoire-outils>\llama\`.
3. Vérifier : `& "<repertoire-outils>\llama\llama-server.exe" --version`.
4. Renseigner `config/local.toml` (à copier depuis `config/local.example.toml`) :
   ```toml
   [server]
   bin = "<repertoire-outils>/llama/llama-server.exe"
   ```
5. Lancer : `uv run loom/runtime/serve.py` (télécharge le GGUF au 1er run).
