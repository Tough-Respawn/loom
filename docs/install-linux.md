# Install llama-server (Linux, VPS CPU)

## Voie rapide (recommandée)

```bash
uv sync
uv run loom-setup              # installeur guidé (binaire llama.cpp + modèle)
uv run python -m loom.web      # l'UI démarre le serveur modèle toute seule
```

(`serve.py` lancé sur machine vierge enchaîne aussi tout seul sur l'installeur.)
`loom-setup` télécharge la release Linux (ubuntu-x64) de llama.cpp dans
`var/runtime/llama/<tag>/`, écrit `[server] bin` dans `config/local.toml`, et
propose un premier modèle adapté à la RAM. Confirmations avant chaque action ;
bilan dans `var/logs/setup.log`. NVIDIA sous Linux : les releases n'embarquent
pas toujours de build CUDA — le setup guide alors vers la compilation.

## À la main (repli)

1. Build CPU ou release Linux de llama.cpp ; placer `llama-server` dans le PATH
   ou un dossier connu.
2. Vérifier : `llama-server --version`.
3. (Optionnel) `config/local.toml` si le binaire n'est pas dans le PATH :
   ```toml
   [server]
   bin = "/opt/llama/llama-server"
   ```
4. Lancer : `uv run loom/runtime/serve.py`. Pas de GPU -> `serve.py` bascule
   automatiquement en CPU (`n_gpu_layers = 0`) et règle les threads.
