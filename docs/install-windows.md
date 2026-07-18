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

## À la main (repli)

1. Récupérer la release Windows CUDA de llama.cpp (`...-bin-win-cuda-x64.zip`
   **et** le zip `cudart-...` qui porte les DLL CUDA).
2. Extraire les deux dans `C:\tools\llama\`.
3. Vérifier : `& "C:\tools\llama\llama-server.exe" --version`.
4. Renseigner `config/local.toml` (à copier depuis `config/local.example.toml`) :
   ```toml
   [server]
   bin = "C:/tools/llama/llama-server.exe"
   ```
5. Lancer : `uv run loom/runtime/serve.py` (télécharge le GGUF au 1er run).
