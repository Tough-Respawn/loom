# Install llama-server (Linux, VPS CPU)

1. Build CPU ou release Linux de llama.cpp ; placer `llama-server` dans le PATH
   ou un dossier connu.
2. Vérifier : `llama-server --version`.
3. (Optionnel) `loom/loom.config.local.toml` si le binaire n'est pas dans le PATH :
   ```toml
   [server]
   bin = "/opt/llama/llama-server"
   ```
4. Lancer : `uv run loom/serve.py`. Pas de GPU -> `serve.py` bascule
   automatiquement en CPU (`n_gpu_layers = 0`) et règle les threads.
