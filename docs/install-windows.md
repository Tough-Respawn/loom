# Install llama-server (Windows, laptop GPU)

1. Récupérer la release Windows CUDA de llama.cpp (`...-bin-win-cuda-x64.zip`).
2. Extraire dans `C:\tools\llama\`.
3. Vérifier : `& "C:\tools\llama\llama-server.exe" --version`.
4. Créer `loom/loom.config.local.toml` :
   ```toml
   [server]
   bin = "C:/tools/llama/llama-server.exe"
   ```
5. Lancer : `uv run loom/serve.py` (télécharge le GGUF au 1er run).
