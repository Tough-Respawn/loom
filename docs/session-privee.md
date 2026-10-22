# Session privée

Case « Session privée (local) » dans la barre latérale. But : aucune donnée de la session
(prompts, fichiers, résultats d'outils) ne quitte la machine.

## Ce qui est garanti aujourd'hui (couche applicative, 2026-10-21)

| Chemin de sortie | Comportement en session privée |
|---|---|
| Modèle de la conversation | local obligatoire : cocher avec un distant, choisir un distant, ou lancer un tour sur un distant est refusé |
| Historique | décocher la case une fois l'historique écrit est refusé : une session privée le reste |
| Appels automatiques | titre et maintenance sur le modèle local de la session ; réflexion et synthèse de mémoire coupées ; raffineur d'image distant sauté |
| Client des modèles | point de sortie unique (`loom/agent/egress.py`) : toute route distante lève `PrivateEgressError` pendant un tour privé, tâches différées comprises |
| Mémoire partagée | lecture seule (`recall`), aucune écriture (`remember` indisponible, réflexion coupée) ; `write_note` pour la session |
| Outils réseau | `web_search`, `fetch_url`, `watch_video`, `add_marketplace`, `install_plugin`, MCP : retirés, avec un message explicite |
| Code exécuté dans Loom | `run_workflow` (exec() du script dans le processus de Loom) : retiré |
| Commandes (`run_shell`, `monitor`, `serve_and_check`) | proxy mort (`127.0.0.1:9`) et `UV_OFFLINE` : bloque curl, uv, pip, git, Python, Node ; **consigne, pas une isolation** (voir limites) |
| `check_page` | URL strictement locales (antislash, identifiants, hôtes encodés refusés) ; Chromium : proxy mort, WebRTC limité au proxy, prélecture DNS coupée, chaque requête vérifiée sur l'URL normalisée et interrompue si elle n'est pas locale |
| Pages de Loom | CSP : aucune ressource externe (images Markdown comprises) |
| Images sans vision | `read_image` refuse explicitement, aucun repli vers un autre modèle |

## Limites connues

- **Ce n'est pas une isolation imposée par l'OS.** Le proxy mort est une consigne que
  respectent curl, uv, pip, git, Python et Node, pas un pare-feu : un programme qui ouvre
  ses sockets lui-même, ssh, ou `Invoke-WebRequest` de PowerShell 5.1 passent.
- **Le serveur du modèle local** n'est pas encore confiné par l'OS (il n'écoute que sur
  127.0.0.1, mais rien n'empêche un processus qu'il lancerait de sortir).
- **Les fichiers** écrits par la session dans un dossier synchronisé (OneDrive, Dropbox…)
  partent avec ce service : hors du périmètre de la session privée.

## Étape suivante

Isolation imposée par l'OS (prototype Windows avec élévation administrateur, puis
service privé isolé et cages Linux/macOS), décrite dans un document d'architecture à part.
