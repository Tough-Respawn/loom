# CHANGELOG — Loom

<!-- RÔLE : historique des versions (commits, fonctionnalités, corrections, refactoring). Pitch public : README.md. Suivi interne : ETAT_PROJET.md. Carte technique : loom.md. -->

> Commits du 2026-04 à 2026-06 — `feat/loom-memory-foundation` (HEAD), `master` à `be69fb6`.

---

## 2026-10-19 : chemin du bac à sable en barres obliques, chemins unix absolus refusés

- Session b64dda17d7e9 : Bonsai a bien utilisé le bac à sable, mais a réécrit son chemin
  (donné avec des antislashs) en `/var/sessions/<id>/scratch/` ; sous Windows pathlib ne
  tient pas ce chemin pour absolu et le joint au lecteur : les SVG sont partis dans
  `C:\var\…`. Le prompt donne désormais le bac à sable en barres obliques, comme le
  dossier de travail.
- Les outils fichiers refusent sous Windows un chemin unix absolu (`/var/x`, `\var\x`,
  hors UNC) avec le geste attendu (chemin complet `C:/…` ou relatif) au lieu d'écrire au
  mauvais endroit.

## 2026-10-19 : une session neuve prend le modèle affiché par le sélecteur

- « + new » créait la session avec le dernier modèle choisi (`var/last_model`), qui
  pouvait différer de celui que le sélecteur affichait encore (onglet précédent) : bonsai
  à l'écran, glm-flash invoqué, et le sélecteur restait faux car la réponse de
  `/session/activate` n'était pas répercutée. Le client envoie maintenant le modèle
  affiché à `/session/new` (il devient aussi le défaut), et le sélecteur se recale sur le
  modèle réel de la session dès la réponse d'activation.

## 2026-10-19 : une mise à jour du harnais atteint les sessions ouvertes

- Le system prompt d'une session est figé (préfixe KV stable) et ne se refigeait que sur
  un choix explicite : une session ouverte avant une mise à jour de Loom gardait l'ancien
  prompt pour toujours (vécu : la règle « bibliothèque manquante → `uv run --with` »
  livrée, invisible pour le modèle de la session 99c4225ceb05, qui a continué à répondre
  « je ne peux pas générer d'images »). L'empreinte du harnais (gabarits de prompt,
  consigne du bac à sable) entre dans la clé du prompt figé : au tour suivant une mise à
  jour, le prompt est refigé et le prompt de base de la session est rafraîchi depuis le
  gabarit livré (`prompt.fige raison=maj_harnais` dans le journal). Un seul re-prefill,
  comme après tout redémarrage.

## 2026-10-19 : bac à sable par session et environnement Python jetable

- Chaque session a un bac à sable, `var/sessions/<id>/scratch/`, créé au premier tour et
  supprimé avec la session. Le system prompt (section « Bac à sable de la session ») y
  envoie les essais, fichiers temporaires et images produites pour être montrées, au lieu
  du dossier de travail.
- Bibliothèque Python manquante : le modèle ne doit plus répondre « je ne peux pas » ni
  installer sur la machine ; il exécute dans un environnement jetable
  `uv run --with <paquet> python script.py` (créé et mis en cache par uv). Vécu session
  99c4225ceb05 : « montre-moi graphiquement Pythagore » a échoué sur matplotlib puis PIL
  absents.
- Séquence « figure, schéma, graphique » ajoutée au prompt : SVG ou page HTML dans le bac à
  sable puis check_page, ou PNG via `uv run --with matplotlib` puis read_image.

## 2026-10-19 : passe esthétique sur le fil et la barre latérale

- Les actions d'une réponse (envoyer, copier) forment une barre sous le texte, révélée au
  survol ; elles ne recouvrent plus la première ligne.
- Bloc d'outil : l'accent ne porte plus que le nom et l'état ; entrées, sorties et carets
  passent en gris texte, un échec en rouge ressort à nouveau.
- Réflexion repliée : ses premières lignes restent lisibles en italique sous un fondu,
  clic pour tout lire ; pendant la génération le texte défile en entier.
- Titres markdown des réponses : trois tailles distinctes (h1 1,3 em, h2 1,15, h3 1,05) et
  marges nettes ; chaque réponse porte le nom du modèle en libellé discret au-dessus.
- Le moniteur système n'est plus une boîte flottante déplaçable sur le fil : il est docké en
  bas de la barre latérale, au-dessus du bouton Reset (position mémorisée supprimée).
- Outils « dangereux » de la barre latérale : le « ! » texte devient un triangle d'alerte
  SVG aligné à droite, même trait que le rail d'icônes.
- Les glyphes texte du fil (carets ▸ ▾ des entrées/sorties et des sous-agents, ✓ ✕ d'état,
  caret des sections repliables) deviennent des SVG au même trait que les icônes.
- Liste des sessions : un point de couleur par type de modèle (local, distant, image,
  vidéo), l'heure relative de dernière activité à droite, la croix de suppression en SVG.
  `/sessions`, la création et l'import renvoient désormais `model`, `kind` et `updated_at`.
- Sélecteur de modèle : les groupes s'appellent « sur cette machine » et « distant (API) »
  (au lieu de « home » et « distant · api »), template et reconstruction à chaud en phase.
- Compteur d'en-tête : une ligne courte (total de tokens, jauge de contexte et son
  pourcentage) ; entrée, sortie, taux de cache et nombre d'appels passent dans l'infobulle ;
  « compacter » n'apparaît qu'à partir de 70 % de la fenêtre.
- Session vide : trois demandes d'exemple cliquables (remplissent le composer) et le rappel
  des raccourcis (« / », « ? », Entrée) à la place de la phrase seule.
- Les blocs du fil (étapes, messages, sorties d'outils dépliées, chronologies d'agents)
  apparaissent en 150 ms (fondu + glissement) ; tout est coupé sous `prefers-reduced-motion`.
- Pendant la génération, la réponse est rendue en markdown à chaque mise à jour (gras,
  listes, titres, code) et MathJax repasse toutes les 400 ms : plus de `**` ni de `\[ \]`
  bruts qui ne se mettent en forme qu'à la fin. Le curseur suit le dernier bloc écrit.
- Barre d'onglets : plus de petite barre de défilement verticale à flèches en haut à droite
  (défilement vertical coupé, horizontal conservé et discret).

## 2026-10-19 : un toast signale la fin d'une recalibration ou d'un téléchargement

- Le panneau « Gérer les modèles » peut être fermé pendant les 5 à 20 minutes d'un
  `/rebench` : rien ne signalait la fin. Un toast persistant (« Recalibration terminée
  « id » : verdict dans Gérer les modèles », boutons Voir / Fermer) apparaît à la fin du
  job, idem pour un téléchargement d'ajout. Une page rechargée pendant une calibration
  reprend le suivi et montre le panneau.

## 2026-10-19 : le démarrage du serveur modèle s'affiche comme un terminal dans le fil

- Les lignes « serveur modèle éteint… », « serveur modèle démarré », « chargement du
  modèle en mémoire… » ne flottent plus dans le fil : elles ouvrent un bloc « serveur
  modèle » (icône terminal) qui montre aussi ce que voit le terminal, c'est-à-dire le
  journal de llama-server (chargement du GGUF, ajustement mémoire, avancement du prefill)
  jusqu'au premier signe du modèle. Source : `GET /logs/stream/upstream` de llama-swap,
  suivi en fond par `loom/runtime/upstream_log.py` (tampon de 2 000 lignes, rejeu
  dédupliqué à la reconnexion) et servi au fil par `GET /server/log?since=N`.

## 2026-10-19 : texte en sans-serif, mono réservé au code et aux commandes

- L'interface passe en sans-serif système (Segoe UI, Inter, system-ui ; aucune fonte
  téléchargée, offline intact) à 13,5 px pour le texte, les titres et la réflexion. Le
  mono (`--mono` : Cascadia Code, SF Mono, Consolas) reste pour le code et les blocs
  `pre`, le nom et les entrées/sorties des outils, le chemin du dossier de travail, le
  compteur de tokens et le moniteur système.

## 2026-10-19 : rail d'icônes dans le fil (réflexion, outil, agent, Loom, réponse)

- Chaque bloc du fil porte une pastille-icône à gauche, reliée par un filet, à la
  manière d'une console d'agent : ampoule pour la réflexion, icône d'outil selon sa
  famille (terminal, lecture, recherche, écriture, web, agent, mémoire, skill, cadran,
  clé pour un outil inconnu), personne(s) pour les sous-agents et l'arène parallèle,
  bouclier pour le garde-fou Loom, bulle pour la réponse, triangle pour une erreur. La
  pastille pulse tant que l'étape est en cours. SVG monochromes locaux
  (`static/icons.js`), teinte discrète par type ; aucune donnée ne change, seulement le
  rendu (`components.js`, composant `Step`).

## 2026-10-19 : /rebench ouvre un panneau de recalibration, progression visible

- `/rebench [id]`, tapé dans le chat ou en note, ouvre « Gérer les modèles » →
  « Recalibrer un modèle » (aussi : bouton « Recalibrer… » sur chaque local installé).
  Choix du modèle, confirmation, verdict comparé et application (« oui ») s'y déroulent
  hors du fil et du journal de la conversation, comme l'ajout depuis le 2026-10-18 ;
  plus aucun dialogue de commande ne vit dans une conversation. Route :
  `GET/POST /models/rebench` (panneau unique pour la machine, une calibration à la fois).
- Pendant la mesure, chaque étape (précontrôle, sonde d'isolation, placement, threads,
  pente, vitesse, réglage final, cache) est horodatée dans un journal affiché par le
  panneau ET imprimée sur la console de loom.web (`[rebench <id>] …`), verdict compris.
  Avant : seul le dernier libellé apparaissait, en bas du chat, et la console ne disait
  rien.

## 2026-10-18 : /add-model ouvre un panneau d'ajout hors conversation

- `/add-model`, tapé dans le chat ou en note, même pendant une génération, ouvre
  « Gérer les modèles » → « Ajouter un modèle ». Les étapes (recherche Hugging Face,
  quant recommandé, id, disque, clé d'un distant) s'y déroulent hors du fil et du
  journal de la conversation ; la fin d'un téléchargement n'y écrit rien non plus.
  `/cancel` annule sans appeler le modèle. Les anciens dialogues restés dans une
  conversation sont effacés au message suivant (`/rebench` a suivi le 2026-10-19).

## 2026-10-18 : un appel d'outil répété est signalé au modèle avant l'arrêt

- Hors modèle « fort », le 2e appel identique d'affilée n'est plus réexécuté : son
  résultat est déjà dans la conversation. Une note `[LOOM]` cite l'appel répété et
  demande de changer d'arguments ou de passer à la suite ; le 3e arrête toujours le
  tour (`repeat_stop`). Avant, le doublon était réexécuté puis le tour coupé sans
  explication au modèle.

## 2026-10-17 : loom-setup propose la mise à jour de SearXNG

- Le conteneur SearXNG géré n'était jamais mis à jour : celui de cette machine datait du
  2026-07-10 et ne tirait plus rien de Yahoo, corrigé en amont depuis. Comme pour
  llama.cpp, `uv run loom-setup` compare l'empreinte de l'image en service à celle de
  `latest` sur Docker Hub et propose la mise à jour (aussi :
  `uv run python -m loom.runtime.searxng update`). Le conteneur est recréé sur le même
  port avec le même `settings.yml` ; si le téléchargement échoue, il n'est pas touché.
  Hors ligne ou sans Docker : rien, jamais bloquant.

## 2026-10-17 : web_search et fetch_url distinguent l'échec de l'absence de résultat

- SearXNG répondait HTTP 200 avec zéro résultat, ses moteurs en CAPTCHA ou en 429 :
  l'outil disait « hors-ligne ou aucun résultat », sans essayer ddgs, et le modèle
  relançait en boucle. Désormais, en mode auto, ddgs prend le relais (aussi quand
  SearXNG reste injoignable après relance) et le modèle reçoit la cause. Si tout échoue,
  l'outil renvoie une erreur qui la nomme ; « aucun résultat » est réservé au cas où les
  moteurs ont cherché.
- `fetch_url` : un lot d'URL entièrement en échec est un échec ; une page courte de
  blocage servie en HTTP 200 (« JavaScript is disabled », captcha…) devient une erreur
  explicite au lieu d'un contenu.

## 2026-10-17 : /add-model signale les alternatives « dérivé tiers »

- Le « Qwen3.8-35B-A3B » proposé en alternative était un affinage d'Empero sur
  Qwen3.6-35B-A3B, pris pour une sortie officielle de Qwen. Chaque alternative affiche
  désormais « dérivé tiers de <modèle officiel> » quand un tiers a modifié le modèle :
  filiation `base_model` lue sur Hugging Face, ou nom de dépôt différent du modèle qu'il
  dit quantifier (variantes abliterated, Bonsai). Les simples quantifications ne sont
  pas marquées.

## 2026-10-17 : recommandation alignée sur le lancement réel, évaluations locales

- Recommandation partagée entre l'installation et le wizard : budget aligné sur
  le lancement GPU (cache `q8_0`, projecteur vision en RAM, embeddings d'entrée
  séparés en RAM), contexte configuré et slots. Les drafters auxiliaires sont
  exclus ; un en-tête incompatible avec la taille annoncée n'est pas recommandé.
- Les alternatives ne sont recherchées que si aucun quant ne convient, dans la
  famille sélectionnée et celle du modèle de base déclaré. Le catalogue de première
  installation conserve les MoE et propose d'abord les modèles adaptés au budget
  le plus élevé. Sans évaluation comparable, le classement reste une estimation.
- Les évaluations locales distinguent la réussite des tâches de la vitesse,
  rattachent les résultats au fichier, au moteur et aux réglages exacts, et exigent
  plusieurs répétitions. Les nouvelles tâches vérifient la complétude des sources
  et la réponse face à un accès bloqué. Aucune campagne qualité n'est préremplie.

## 2026-10-17 : panneau « Gérer les modèles »

- `/remove-model` ouvre « Gérer les modèles » sans envoyer de message au modèle ni
  interrompre sa génération. La suppression refuse les modèles utilisés et
  conserve la liste des modèles distants restants. Les raccourcis restent initialisés.
- La suppression n'efface que des dossiers `<racine>/local/{text,image,video}/<id>`.

## 2026-10-17 : journal des outils téléchargeable

- Export du journal des outils en Markdown (menu de l'onglet, et `journal.md` dans l'export
  ZIP), pour consulter les entrées et sorties enregistrées indépendamment des affirmations
  du modèle. En console : `uv run python -m loom.agent.transcript var/sessions/<id>`.

## 2026-10-17 : garde-fou d'exécution sans faux positif sur « Sortie : »

- Le garde-fou d'affirmation d'exécution conserve « Résultat : » et écarte « sortie : »,
  ambigu pour les dates de sortie d'albums (« Sortie : 2019 » relançait le tour).
- Tests d'annulation des sous-agents : leurs faux clients implémentent `annex_slot`
  (3 échecs préexistants sur master).

## 2026-10-17 : fetch_url passe les anti-bots courants et explique les refus

- 7 échecs sur 10 de `fetch_url` dans les sessions étaient des 403 anti-bot : par défaut,
  httpx se présentait en Chrome 125 avec l'empreinte TLS d'un client Python. Le modèle
  ne relançait pas toujours avec `impersonate`.
- Désormais : User-Agent honnête `Loom/1.0 (local assistant)` (sur 24 sites mesurés, il
  ne perd aucun site que le faux Chrome passait, et gagne songlyrics et amazon), repli
  automatique sur l'empreinte navigateur curl_cffi après un 403 (discogs, allmusic,
  medium, leboncoin), en-têtes de curl_cffi laissés intacts.
- 401 (captcha ou authentification), 403 persistant et 429 disent au modèle de chercher
  une autre source au lieu de relancer la même URL.

## 2026-10-17 : /add-model ne recommande plus un modèle dense qui déborde de la VRAM

- Un dense relit tous ses poids à chaque token : ce qui déborde passe par le CPU. L'IQ2_XXS
  de 6,8 Go d'un 27B était recommandé sur 6 Go de VRAM. Désormais seul un quant qui tient
  entier en VRAM est recommandé ; sinon, alternatives de la famille. Les MoE gardent les
  experts en RAM.
- Alternatives aussi quand le recommandé est sous 4 bits (hors ternaire natif, Q2_0 de
  Bonsai) ; taille d'un dépôt lue dans son nom quand Hugging Face a lu un drafter MTP.

## 2026-10-17 : /add-model met la VRAM d'abord, estime la vitesse et propose des alternatives

- La règle précédente bannissait la pleine précision même quand elle tenait en VRAM, et
  recommandait pour un dense qui déborde le quant le moins lu en RAM sans se demander
  s'il tournait assez vite.
- Désormais : le plus gros quant d'au moins 4 bits qui tient entier en VRAM avec 16k de
  contexte (KV lu dans l'en-tête, couches à fenêtre glissante bornées), F16/BF16
  compris ; sinon le plus gros estimé à 12 t/s, à défaut 8 t/s. Vitesse estimée par
  quant (débits fixes calés sur ornith Q8 mesuré à 15,1 t/s, non vérifiée sur un dense).
- Rien ne convient, ou recommandé sous 12 t/s : alternatives de la même famille
  cherchées sur Hugging Face, choisies par lettre dans le wizard.

## 2026-10-17 : /add-model recommande selon le modèle et le llama.cpp installé

- L'ancienne règle (« le plus gros quant qui tient en VRAM + RAM ») recommandait le F16 de
  50 Go d'un dense 27B, comptait 0 Mo de VRAM sur une RTX 2060 (détection sans le
  binaire) et pouvait proposer un format de fork que llama.cpp refuse.
- Nouveau : en-tête GGUF lu à distance (`loom/runtime/gguf_remote.py`, ~11 Mio, ~3 s, un
  seul par dépôt), formats déclarés par le `llama-quantize` installé, règle dense / MoE,
  pleine précision et classe 1 bit écartées, part lue en RAM affichée par quant.
- Le catalogue écarte aussi les fichiers auxiliaires `imatrix` et drafters `DFlash`.

## 2026-10-17 : informations personnelles retirées de la doc, des gabarits et des tests

- Chemins de la machine de développement, noms de GPU et détails personnels remplacés par
  des valeurs génériques (`<racine-des-modeles>`, « GPU intégré Vulkan »…) dans la doc,
  les specs et plans, les gabarits `_TEMPLATE/`, `config/*.toml` et les fixtures de tests.
  Aucun changement de comportement.

## 2026-10-17 : modèle refusé à l'installation, supprimable et jamais remonté

- Un GGUF que llama-server refuse (vécu : Bonsai Heretic en PQ2_0, format du fork Prism)
  restait sur le disque (~8 Go) sans apparaître dans `/remove-model`, et rien n'empêchait
  sa découverte au démarrage suivant : il serait entré au sélecteur et aurait planté au
  premier message. `/add-model` le marque maintenant `unloadable` dans son model.toml,
  la découverte (runtime et setup) l'ignore, et `/remove-model` liste aussi les dossiers
  présents sur disque mais non montés (non chargeable, interrompu, illisible).

## 2026-10-17 : loom-setup fait le ménage des anciennes builds llama.cpp

- Chaque mise à jour ajoute ~1,2 Go dans `var/runtime/llama/<tag>/` et rien ne les
  supprimait. Le setup propose maintenant de supprimer les builds inutiles : il garde la
  build en service, la plus récente des autres (retour arrière) et toute build citée par
  un `server_bin` de model.toml. Défaut « non », jamais sous `--yes`, ne touche que
  `var/runtime/llama/`.

## 2026-10-17 : loom-setup propose la mise à jour de llama.cpp

- `loom-setup` sortait dès que le binaire configuré existait (« rien à faire ») : il ne
  proposait jamais de mise à jour, alors que le bandeau de l'UI recommandait justement
  `uv run loom-setup`. Il compare maintenant la build installée à celle de la dernière
  release stable de llama.cpp et propose la nouvelle, installée À CÔTÉ dans
  `var/runtime/llama/<tag>/` (l'ancien binaire reste, chemin affiché pour revenir).
  Hors ligne : non bloquant. Build maison (`BUILD.txt`) : renvoyé vers la recompilation.
- Bandeau de veille : même source que le setup (release stable, plus le dernier nightly
  qui sort plusieurs fois par jour) ; ce qu'il annonce est ce que le setup installe.
  « Plus besoin de recompiler » n'apparaît plus que pour une sortie de build maison.

## 2026-10-11 : règle de choix des modèles révisée (ternaire admis)

- La règle « seulement MoE 24B+ avec experts en RAM » (2026-06-09) devient : **critère = octets
  lus par token**. Deux familles viables sur 6 Go : MoE 24B+ (experts en RAM) et dense 27B
  ternaire (~1,7-2 bits/poids, ~6-7 Go, quasi tout en VRAM). Déclencheur : Bonsai 2 27B
  (Qwen3.8-27B, AA Index 34 contre 18 pour la classe 35B-A3B), support upstream llama.cpp
  #29600 mergé le 2026-10-11. Docs : README, ETAT_PROJET, loom.md.

## 2026-10-11 : étude CLM (édition de contexte) et hook de capture des requêtes

- **Étude CLM** (`docs/clm-chiffrage-2026-10-08.md`) : coût mesuré d'une édition de contexte sur
  llama-server avec une vraie requête Loom de 22 311 tokens (ornith). Fin de contexte 0,4 à 24 s,
  milieu 52 à 77 s (recul au dernier checkpoint, ~12k tokens refaits), system prompt 97 s. Verdict :
  pas de mode CLM tant que llama-server ne pose pas de checkpoint par résultat d'outil. Script et
  séries de mesures dans `evals/clm/`.
- **`LOOM_DUMP_REQUESTS=<dossier>`** (`loom/agent/calltrace.py`) : copie JSON de chaque requête
  envoyée au modèle, pour la rejouer hors de Loom. Inactif sans la variable. Le dump contient le
  prompt complet (données personnelles) : hors dépôt.

## 2026-10-03 : outil watch_video (lire une vidéo)

- **`watch_video`** (`loom/tools/video.py`, PR #34) : lit ce qu'une vidéo **dit**, à partir
  d'une URL (YouTube et tout site géré par yt-dlp) ou d'un fichier local. Renvoie titre,
  chaîne, durée, chapitres, extrait de description et transcript en paragraphes horodatés
  d'environ 30 s. Pas d'images, par choix : la vision coûte cher en contexte local et exige
  un modèle vision.
- Sous-titres existants d'abord, sans télécharger la vidéo (manuels dans la langue demandée,
  puis langue d'origine, puis auto d'origine, puis auto traduits) ; doublons des sous-titres
  auto « roulants » retirés. Sinon audio seul + faster-whisper `small` int8 sur CPU, batch 8
  beam 1, VAD Silero (6,9x le temps réel mesuré, contre 2,9x en beam 5 séquentiel).
- Cache `var/cache/videos` ; au-delà de 24 000 caractères, transcript tronqué avec le
  `read_file(start_line=…)` qui donne la suite. Sortie marquée frontière de confiance.
- Anti-SSRF (PR #35) : hôtes internes refusés sur l'URL donnée ET sur les URL de sous-titres
  et d'audio fournies par yt-dlp.
- Dépendances : `yt-dlp[default,deno]`, `faster-whisper`, `av<19` (faster-whisper 1.2.1
  incompatible PyAV 19, SYSTRAN/faster-whisper#1589).
- Validé de bout en bout : ornith-1.5-35b-a3b (local) appelle l'outil seul et résume une
  vidéo TED de 14 min (outil 6,8 s, sous-titres manuels).

## 2026-10-02 : llama.cpp à jour, veille des mises à jour, --load-mode, setup v0.x

- **Veille llama.cpp** (`loom/runtime/llama_update.py`) : au démarrage puis toutes les
  heures (GitHub interrogé au plus une fois par 24 h, cache `var/cache/llama_update.json`),
  compare la build du binaire configuré au dernier build nocturne et suit `[server]
  track_prs`. Bandeau dans l'interface : recompiler (`rebuild_hint`) tant qu'une PR suivie
  est ouverte, `uv run loom-setup` quand toutes sont mergées. N'installe jamais rien.
  Un build maison est comparé sur sa base officielle (`upstream_build=` dans `BUILD.txt`) :
  le numéro de build compte aussi les commits des PR.
- `--no-mmap` retiré par llama.cpp (2026-09-09) : `--load-mode none` selon le binaire (PR #30).
- loom-setup : releases `v0.x` sans binaire et `cudart` pris pour llama-server (PR #31).

## 2026-09-21 — décision par logits (mode « Jev » natif, llama-server stock)

- `loom/agent/decide.py` : questions fermées (enum / booléen / entier) répondues sans
  générer de JSON : une requête `/completion` (`n_predict=1`, `n_probs`, probabilités
  PRÉ-sampling) par nœud de divergence des options, préfixe en cache, slot au choix.
  Valeur toujours dans le schéma, probabilité renormalisée par champ, `coverage` = masse
  que le schéma couvrait (signal faible visible, jamais masqué). Probabilité NON calibrée.
- Pièges mesurés et traités : coupe là où les TOKENS des options divergent (`":` + ` true`,
  pas `": ` + `true`) ; booléens et entiers acceptés nus ET cités (Gemma 4 E4B pose ` "`
  après `":` dans 15 à 99 % des cas) ; clé `top_logprobs` en pré-sampling.
- Banc réel `evals/bench_decision.py` (serveur éphémère, flags Loom, 12 champs, 6 tickets
  annotés) : gemma4-e4b 3,05 s vs 5,39 s JSON à chaud (justesse 50/54 vs 54/54) ;
  ornith 1.5 35B cpu-moe 9,8 s vs 13,2 s (27/27 des deux côtés), `--no-op-offload`
  n'apporte rien (10,4 s). Coût = ~140 ms/requête serveur sur E4B, ~500 ms sur le MoE :
  le gain vient des tokens non générés, pas du batch (fork non adopté).
- Évals : `--judge logits` (2 questions fermées, sans texte généré) à côté du juge JSON ;
  smoke runtime sur E4B : trajectoire réussie p(pass)=1,00, ratée p(pass)=0,01, ~1 s.

## 2026-09-13 — audit timings : slots explicites, save vérifié, binaire UMA

### Audit (session ornith Q8, chiffres du log llama-server)
- Amorçage 88 s pour un préfixe de 9 585 tokens (26 schémas d'outils ≈ 5,6k, system
  3 966) ; premier octet 6,4 à 23,5 s par tour même cache plein ; décode 8,4-8,7 t/s.
- Cause n°1 (hors repo) : le binaire maison du 27/07 n'avait PAS le fix de lecture UMA
  (`de63a6dcf`) — ~3,3 s par opération de checkpoint, deux par requête. Rebuild
  `build-vulkan-uma` (branche `fix/kv-restore-checkpoint-uma`) : cache plein +4 tokens
  3,3 s → 0,22 s, +18 tokens 7,0 s → 0,77 s, décode inchangé. Sonde à prompt identique.
- Cause n°2 : llama-server plaçait la conversation sur le slot 1 (LRU au premier appel)
  pendant que Loom sauvait `/slots/0` — `n_saved = 114` = le prompt de titre. Chaque
  redémarrage repayait l'amorçage complet.

### Fonctionnalités
- **Slot llama-server explicite** (`id_slot` dans `extra_body`, local uniquement) : fil
  principal et warm sur le slot 0 (= celui du save/restore de la reprise à chaud) ;
  titre, résumé, `stream_chat` (reflect, synthèse recall) et sous-agents sur le slot 1
  quand le modèle a deux slots (`client.slot_counts` posé au boot depuis
  `resolve_parallel`), sinon 0. L'isolation `cache_isolation` devient déterministe.
- **Save vérifié** : un save qui répond `n_saved = 0` est refusé ET invalide la meta du
  save précédent (le serveur a déjà écrasé le fichier) ; un restore qui répond
  `n_restored = 0` est un échec. La reprise à chaud ne restaure jamais un slot vide.
- **Reprise à chaud depuis `/chat`** : après un déchargement (slots froids), le premier
  message tente le restore one-shot sous le verrou local AVANT de générer — la reprise
  n'existait que dans l'amorçage (boot, changement de modèle/session, fin de tour).
- **`id_slot` jamais vers une route distante**, même si elle active les extras natifs
  (`enable_thinking_param`) ; `slot_counts` resynchronisé à chaque régénération du yaml.
  (Revue croisée du 2026-09-13.)

- **Titre hors du chemin critique** (étape 3) : titre PROVISOIRE (début du message)
  posé et publié dès le début du tour ; `done` n'attend plus jamais le modèle de
  titrage (avant : 21 s de slot local avant `done`, annulé au timeout, ou jusqu'à 8 s
  d'attente du titre distant). Le vrai titre s'infère après coup dans un thread dédié,
  hors verrou, sur le slot annexe — jamais sur un local à un seul slot (il évincerait
  le cache de la conversation). `infer_title` n'envoie plus de `temperature` (refusée
  par certains providers : 400 « only 1 is allowed ») et, en local, coupe le thinking
  dès le premier essai.
  *Révision après validation réelle* : en local, le vrai titre ne tourne plus en
  parallèle sur le slot annexe (les deux slots partagent le matériel : warm tombé à
  6,5 t/s et titre annulé au timeout de 20 s, mesuré) mais dans la maintenance
  SÉQUENTIELLE, après le warm, sous le verrou, en flux et INTERRUPTIBLE par un message
  via le même porte-flux que le warm. Timeout inchangé (20 s) tant que l'interruption
  en plein calcul n'est pas prouvée côté serveur. Distant : thread dédié sans verrou.

- **Warm préemptible** (étape 4) : l'amorçage (warm de fin de tour, keep-warm, reflect)
  publie son flux HTTP dans un porte-flux partagé `S.warm_holder` ; un message qui trouve
  le verrou local tenu par un amorçage FERME ce flux (llama-server annule la tâche à la
  déconnexion et garde le cache déjà calculé), l'amorçage rend la main aussitôt et le tour
  préfille lui-même ce qui manque, avec une notice explicite. Avant : le message attendait
  la fin du prefill d'amorçage (54 s vécus). Une vraie génération d'une autre session
  n'est jamais interrompue.
  *Revue croisée n°2* : le signal d'annulation porte sur TOUTE la durée du verrou —
  `warm_context` / `infer_title` ne le consomment plus, la maintenance n'enchaîne ni
  restore, ni warm, ni titre après une interruption, le keep-warm ne « ping » plus
  après un warm interrompu, et le signal est revérifié dès l'ouverture du flux HTTP
  (un message arrivé pendant l'ouverture ferme le flux sans le lire). Seul le
  détenteur du verrou (maintenance, keep-warm, amorçage, ou le `/chat` qui le prend)
  remet le signal à zéro.

### Tests
- 5 tests (`tests/test_warm_preemptible.py`) : flux publié et abandon signalé, abandon
  déjà demandé, fermeture par le helper, passage du porte-flux à reflect, parcours web
  complet (second message pendant un warm : `done` en < 2 s, flux fermé, notice).
- 12 non-régressions (`tests/test_slot_pinning.py`) : payload local/distant, choix du slot
  annexe, fil principal, warm, appels annexes, sous-agent, save vide/plein.
- 5 parcours web complets (`tests/test_title_off_critical_path.py`) : provisoire avant le
  premier texte, vrai titre après coup hors flux, un seul slot = pas d'appel, distant sans
  attente, titre existant jamais retouché ; `test_infer_title.py` réécrit.

### Constats laissés en pistes (cf. ETAT_PROJET)
- Titre local tenté avant le `done` puis 400 « invalid temperature » ; warm de fin de
  tour non annulable ; `deferred_tools` jamais validé par A/B ; artefacts `{'text': …}`
  dans USER.md ; `regenerate_swap_yaml` oublie ub/b/checkpoint_min_step ;
  `--cache-reuse` inapplicable à ornith (multimodal + hybride).

---

## 2026-08-03 — calibration : repli machine, sonde d'ubatch sur llama-server

### Fonctionnalités
- **Repli MACHINE des réglages mesurés** (`[server].ubatch` / `batch` /
  `checkpoint_min_step`), sur le modèle de `context` : un modèle ajouté par
  `/add-model` n'est jamais benché et tombait sur les constantes aveugles de
  llama-server (ubatch 512 : 61 % de prefill perdus, mesuré 2026-07-21). Le
  `model.toml` reste prioritaire ; la machine n'est qu'un défaut informé.
- **La sonde d'ubatch tourne sur le VRAI llama-server** (via `ServerProbe`, comme la
  calibration du contexte) et non plus sur `llama-bench` — absent des builds maison
  qui ne compilent que la cible serveur (vécu : `build-vulkan` sans `llama-bench`,
  la sonde ne tournait jamais). Mesure avec les flags exacts de l'exécutant, sur un
  prompt de 4096 tokens (à 128, tout tient dans un micro-batch : aucun effet).
  Élue au setup ET au `/rebench` — un modèle déjà installé se calibre sans
  réinstaller ; verdict appliqué dans son `model.toml` sur confirmation.
- Le setup écrit le couple élu en repli machine, et `checkpoint_min_step = 2048`
  comme **défaut raisonné explicitement étiqueté NON mesuré**, réservé aux modèles
  à mémoire hybride détectés par la sonde d'isolation.

### Corrections
- **Des batchs explicites étaient ignorés hors profil GPU** : `build_server_args`
  n'émettait `-ub`/`-b` que sous `gpu_tuning`. Un `ubatch` mesuré ou posé dans
  `model.toml` était silencieusement perdu sur une topologie CPU — sonde comprise,
  qui aurait mesuré deux configs identiques.
- **L'amorçage du cache KV ne tourne plus pendant une calibration** : le banc arrête
  le serveur modèle pour récupérer la VRAM, mais `running_local()` répond « vivant »
  dès que llama-swap (le routeur) décroche — le garde passait, `warm_context`
  échouait en rafale (`Connection error`), et pire, les amorçages à `wait_server`
  REDÉMARRAIENT un serveur contre le banc, faussant la mesure.

### Tests
- 16 non-régressions : repli machine (priorité modèle, repli partiel, comportement
  inchangé sans repli), sonde d'ubatch (élection, gain, base gagnante, sonde muette
  sur échec, flags transmis au serveur, écriture model.toml idempotente), garde de
  calibration (détection, saut avant tout démarrage).

---

## 2026-08-03 — observabilité et robustesse au démarrage

### Corrections
- **Un modèle local incomplet n'empêche plus Loom de démarrer.** `/add-model` écrit
  `model.toml` AVANT la fin du téléchargement (`n_layers` se lit dans le GGUF) ; une
  installation interrompue laissait un `KeyError` remonter jusqu'au chargement de la
  config, donc plus d'interface du tout — y compris pour réparer le modèle fautif.
  `_discover_models` ignore désormais un dossier illisible en disant sur stderr **lequel**
  et **pourquoi**. Loom démarre même si tous les modèles locaux sont cassés.
- **`log_event` ne peut plus interrompre un tour de génération** : l'horodatage passe par
  `_ts_prefix()`, qui ne lève jamais — le contrat best-effort du module (`_emit`) n'était
  pas respecté par les appels à `_ts()`.
- **La reprise à chaud n'est plus perdue au démarrage du serveur.** `try_hot_resume`
  consommait son essai unique AVANT la tentative : un refus de connexion (serveur pas
  encore à l'écoute) était compté comme un vrai échec et la reprise perdue pour toute la
  période froide. Le refus est désormais distingué — le slot redevient froid et le tour
  suivant retente. Un vrai échec continue de consommer l'essai (le disjoncteur qui évite
  une tempête de retries reste intact).

### Diagnostic
- **Un téléchargement de modèle est enfin visible.** Il se faisait entièrement en tâche de
  fond, serveur allumé, sans le moindre signal — ni interface, ni journal. Une ligne `DDL`
  apparaît maintenant dans le moniteur système (barre + Go + modèle), **uniquement pendant
  un transfert**, et `download.progress` est tracé par pas de 5 %. Toute l'information
  existait déjà : octets reçus dans le `.incomplete` de Hugging Face, cible dans `size_mb`
  — personne ne faisait la division.
- **`serve.log` reçoit enfin la sortie du serveur modèle.** llama-swap lance llama-server
  lui-même et retient sa sortie dans ses propres tampons : le fichier ne contenait que les
  lignes de llama-swap, alors que l'interface y renvoyait l'utilisateur en cas de panne.
  `capture_upstream_log()` rapatrie la fin de `/logs` dans `serve.log` au moment où le
  serveur est constaté mort, et le message indique combien de lignes ont été copiées — ou
  dit franchement que le journal est injoignable.

### Diagnostic
- **Les blocs de dump sont horodatés** (`REQUETE`, `HOT_RESUME`, `SLOT_RESTORE_ERR`…).
  Sans date, ils étaient impossibles à replacer sur une chronologie : un post-mortem
  (serveur modèle arrêté pendant un téléchargement) s'est heurté à des blocs indatables.
  L'horodatage ouvre la ligne, ce qui permet de **fusionner par tri** le log de session et
  le repli global — les deux se partagent la trace selon le thread émetteur.

### Tests
- 8 non-régressions : découverte tolérante (dossier incomplet, TOML invalide, tout cassé)
  et chronologie (bloc horodaté et greppable, tri conjoint blocs/événements, silence sous
  `LOOM_DEBUG=0`, horloge défaillante non bloquante).

---

## 2026-08-02 — audit de préparation

### Corrections
- **Self-test du banc d'éval réparé après le refactoring de `client.py`** : les
  garde-fous sont importés depuis leurs modules propriétaires (`compaction`, `guards`,
  `streaming`, `toolrun`) au lieu de dépendre de réexports privés de la façade.
- **Non-régression pytest ajoutée** pour garantir que le self-test documenté reste
  exécutable hors ligne, sans serveur ni modèle.

### Documentation
- Commandes de validation, statut du banc d'éval/SearXNG et prise en charge GPU remis
  en cohérence dans `README.md` et `loom.md`.
- Rapport de préparation ajouté sous `docs/` avec preuves, limites et verdict.

### Maintenance
- **Commentaires rendus lean dans tout le code** : inventaire réduit de **3 584 à
  1 655 lignes (-54 %)**. Les commentaires restants expliquent une raison, une contrainte
  ou un piège réel ; les récits historiques, répétitions du code et séparateurs décoratifs
  ont été retirés sans changement de comportement.

### Validation
- **533 tests verts**, Ruff clean, self-test d'éval vert, couverture globale **60 %**.
- Smoke Chromium autonome vert (HTTP 200, raccourcis, chat SSE, rendu, zéro erreur
  console) ; wheel et source `loom 0.1.0` construits avec succès.

---

## 2026-07-31 — éclatement des god-files

### Refactoring
- **Trois god-files éclatés par domaine, à comportement constant** (branche `refactor/god-files`, mergée en fast-forward) :
  - `loom/agent/client.py` **3014 → 1353 L** : fonctions libres sorties en modules cohésifs (`debuglog`, `errors`, `toolsets`, `compaction`, `guards`, `streaming`, `toolrun`), layering acyclique. `LoomClient` + réexports gardés en façade dans `client.py`.
  - `loom/web/routes.py` **4003 L → package `loom/web/routes/`** : `helpers` (cœur partagé) + `config`/`skills`/`chat`/`models`/`misc`/`sessions`/`soul`. Cycle `models↔chat` rompu par import tardif dans `_register_model_routes` ; `_index_context` basculé en `skills`.
  - `loom/web/static/app.js` **4144 → 1313 L** : ES modules `render`/`state`/`components`/`sse`/`chat`/`tabs`/`panes`/`panels` + `shared.js` (5 variables mutables partagées via setters, seule façon de réassigner un binding importé en ESM).
- **Méthode** : découpage mécanique piloté par AST (Python) / acorn (JS), imports croisés calculés, cycles détectés avant écriture. Filet : 501 tests pytest verts après chaque split ; app.js vérifié en navigateur réel (chat, palette, drawer, compteurs, 0 erreur console).
- **Hygiène** : `audit-dette.md` déplacé dans `docs/`, 2 lambdas assignées converties en `def` (ruff `E731`), ruff `loom` clean.
- **Vague 2 (god-files de 2e génération)** : `routes/chat.py` **1582 → 1098 L** (helpers sortis en `system_prompt`/`priming`/`maintenance`/`commands`, la boucle SSE `_register_chat_routes` reste) ; `routes/models.py` **1182 → 639 L** (`model_admin.py` = helpers remote/image/install, `rebench.py` = calibration ; le cœur patché par les tests reste dans `models.py`). 501 tests verts, ruff clean.

---

## 2026-07-18 — boot remote-only

### Améliorations
- **`loom.web` boote sans modèle local** quand un modèle DISTANT existe (`[[remote_models]]` de `config/local.toml` OU store UI `var/remote_models.json`) : `load_config` tolère `models=[]` dans ce cas (le serveur llama.cpp local ne sert que les locaux et démarre à la demande), `default_model` retombe sur le premier distant si le défaut configuré pointe un local disparu. Machine vraiment vierge (ni local ni distant) : comportement inchangé — installeur guidé (`maybe_bootstrap`) ou `ValueError` claire.
- La fusion du store UI (`var/remote_models.json`) dans `cfg.remote_models` vit désormais dans `load_config` (plus dans `build_app`) : une seule source pour tous les consommateurs, y compris le reload à chaud ; dérivation du chemin centralisée (`config.remote_store_path`).
- `serve.py` (moteur llama.cpp) continue d'exiger un modèle local : garde explicite « aucun modèle LOCAL à servir » (code 1) au lieu d'un crash, `cfg.model`/`model_by_id` lèvent une erreur claire sur parc local vide.

---

## 2026-07-18 — loom-setup

### Nouvelles fonctionnalités
- **`loom-setup`** : installeur console guidé (`uv run loom-setup`) — détecte OS/GPU/VRAM/RAM, télécharge la dernière release stable de llama.cpp (build CUDA/Vulkan/CPU selon la machine, DLL cudart incluses) dans `var/runtime/llama/<tag>/`, écrit `[server] bin` dans `config/local.toml`, puis propose un premier modèle qui fit le budget mémoire. Chaque action lourde est expliquée et confirmée (`--yes` pour tout accepter) ; bilan ✅/⏭️/❌/🔧 + journal `var/logs/setup.log`. Idempotent : relançable, ne refait que ce qui manque.
- **Shortlist de modèles par palier de budget** : le catalogue porte des FAMILLES éprouvées en tool-use (du 35B MoE au ~1.5B), pas des repos figés — le repo réel est résolu en live sur HF au moment du choix (top téléchargements qui fit), donc la liste ne périme pas. Chaque machine a toujours au moins une proposition ; recherche libre et `/add-model` restent pour les initiés.
- **Recherche filtrée par la machine** : les résultats HF impossibles (taille estimée depuis le nom, « 397B » sur 1,6 Go de budget…) sont masqués avec un compteur, les jouables annotés `~X Mo mini` ; la taille réelle des quants (`recommend_quant`) tranche après le choix.
- **« Libère ta RAM » avant le choix du modèle** : le setup liste les plus gros consommateurs (groupés par nom, via psutil), laisse l'utilisateur fermer LUI-MÊME ce qu'il n'utilise pas (rien n'est jamais tué automatiquement), re-mesure sur Entrée et recalcule le budget — la shortlist en profite immédiatement.

### Améliorations
- **Bootstrap auto au premier run** : `serve.py` détecte la machine vierge (binaire ou modèle manquant) et enchaîne tout seul sur l'installeur guidé dans le terminal ; hors terminal interactif (service), il guide vers `uv run loom-setup` et sort proprement.
- Hint de `serve.py` quand le binaire manque : pointe vers `loom-setup`.
- **Couleurs terminal** (`loom/runtime/term.py`) : bannières/étapes en gras, ✅ vert, ❌/ERREUR rouge, ⏭️ estompé, 🔧 jaune — uniquement sur un vrai TTY (jamais de codes ANSI dans les fichiers de log ni les pipes), `NO_COLOR` respecté, mode VT activé sur les vieilles consoles Windows. Branché sur loom-setup et serve.
- Le premier modèle installé devient `[chat] default_model` dans `config/local.toml` (sinon le défaut de defaults.toml peut pointer un modèle d'une autre machine).
- Boucle « libère ta RAM » avant le choix du modèle : liste les plus gros consommateurs (groupés par nom), l'utilisateur ferme lui-même, re-mesure — le budget et la shortlist en profitent.
- **Étape 4 : bench du matériel** (`loom/setup/bench.py`) — après l'install, `llama-bench` (livré avec la release) MESURE la vitesse réelle sur le modèle installé : combinaisons de threads (physiques/logiques/moitié — machines hybrides P+E) et d'offload GPU (`-ngl` si backend Vulkan/CUDA présent). La génération (tg) tranche, le prefill départage. Le contexte est CALCULÉ (KV/token lu du header GGUF → le plus grand cache qui tient en RAM sans swap ; vécu : 24576 par défaut sur 16 Go → swap → 0,8 t/s). Résultats écrits dans `config/local.toml` (`[override] threads`/`n_gpu_layers`, `[server] context`, table `[bench]` pour l'idempotence).
- **Debug actif par défaut** : le trace des échanges modèle (stderr + `sessions/<id>/debug.log`) n'exige plus `LOOM_DEBUG=1` — c'est le premier outil de diagnostic, coût négligeable. `LOOM_DEBUG=0` pour couper.
- **Terminal tenable en debug** : les blocs volumineux (dump complet de la requête avec system prompt, save/restore de slot KV) ne vont plus QUE dans `debug.log` — le terminal garde les lignes compactes (`turn.request`, `usage`, premier byte) et la RÉPONSE du modèle. Les polls `GET /machine_state` rejoignent `/sysmon` dans le filtre werkzeug.
- Docs d'install (Windows/Linux) : voie rapide `loom-setup` + correction du chemin de config obsolète (`loom.config.personnel.toml` → `config/local.toml`).

---

## 2026-07-17 — /add-model

### Nouvelles fonctionnalités
- **`/add-model`** : ajout d'un modèle en une commande (wizard chat déterministe, zéro LLM) — local : recherche Hugging Face, choix du quant recommandé selon VRAM/RAM, téléchargement avec progression, `model.toml` généré (n_layers/MoE lus du GGUF), montage à chaud dans le sélecteur ; distant : URL + clé, persisté via le store du panneau engrenage.

---

## 2026-06-09 — Mémoire & Reflection

### Nouvelles fonctionnalités
- **Mémoire persistante** : provider local SQLite FTS5 (`remember`/`recall`), interface `MemoryProvider` (local v1), bloc identité SOUL/USER/MEMORY toujours injecté au prompt, cache mtime des fichiers identité.
- **Étape `reflect`** : capitalisation post-tour (validation pure + écritures), cles config dédiées (`reflect`, `skills_learned`, `summarization_recall`), trace visible (ran/retenu/erreur).
- **Catalogue skills appris** : namespace `learned:`, frontmatter étendu, câblage au catalogue web.
- **Outils mémoire** : `recall`, `remember`, `write_note`, `read_note` (permission `evaluate()` autorisée d'office).
- **Pastille outil IN/OUT** dans l'UI (nom du tool, commande + sortie dépliables bloc par bloc).

### Améliorations
- Injecte le dossier de travail courant au prompt (anti-tatonnement git/chemins).
- Adopte un repo cité par son nom (sous-dossier du workspace racine).
- Rendu LaTeX offline (MathJax) + compteur live tool-calls.
- Défaut = dernier modèle + robustesse vision.
- Compteur live tokens envoyés/reçus + débit.
- Badge d'alerte quand le mode permission est `allow`.
- Keep-warm du modèle pour éviter le cold start.

### Corrections
- Valide le nom des skills appris (anti traversée de chemin dans `reflect`).
- Autorise d'office `recall`/`remember`/`write_note`/`read_note`.
- Rattrapage des tool-calls textuels + règle 0 assouplie.
- Termine l'arbre de process sur Ctrl+C.
- Durcit les outils web + frontière de confiance native.

### Refactoring
- Range le cœur dans `loom/agent/` et les extensions dans `loom/extend/`.
- Regroupe l'infra de service dans `loom/runtime/`.
- Renomme `loom.config.local.toml` en `.personnel.toml` (suivi par git).

### Évaluation
- Harnais d'eval des prompts (eval set + double grader, A/B git).
- Harnais d'eval du skill code-review (rappel/FP/verdict).

### Modèles
- Ne garder que les MoE 24B+ (retire Gemma E4B + Qwen 4B).
- Template `_TEMPLATE/model.toml` + README.

### Documentation
- Présentation de Loom générée par le modèle local (Qwen3.6).
- Actualise README + ETAT_PROJET (tool-use pur, plugins, skills déclenchés, MoE 24B+).
- Archive les specs/plans de code retirés.
- Design closed-learning-loop + plans mémoire/reflect.
- Spec skill de debug généraliste + `check_page` localisant.

### Debug
- `debug` : imprime le système prompt en entier (LOOM_DEBUG), messages bornes.
- Skill de debug généraliste + `check_page` localisant.

---

## Commits détaillés

| Date | Commit | Catégorie | Description |
|------|--------|-----------|-------------|
| 06-09 | 6ac1207 | debug | imprime le system prompt en entier (LOOM_DEBUG) |
| 06-09 | 89e4c1f | feat(ui) | pastille outil en IN/OUT (nom du tool, commande + sortie dépliables) |
| 06-09 | 7db1867 | feat(web) | injecte le dossier de travail courant au prompt |
| 06-09 | 1027862 | fix(permissions) | autorise d'office recall/remember/write_note/read_note |
| 06-09 | 4840ca9 | feat(web) | trace visible de reflect au lieu du except: pass muet |
| 06-09 | cbc7b88 | perf(memory) | cache mtime des fichiers identite |
| 06-09 | d602a83 | feat(web) | adopte un repo cite par son nom |
| 06-09 | d0ec7a1 | docs(prompt) | presente les skills appris (learned:) au modele |
| 06-09 | 365b281 | feat(web) | cable catalogue skills appris + resumeur recall |
| 06-09 | 7e6bbaf | fix(security) | valide le nom des skills appris (anti traversée) |
| 06-09 | a211918 | feat(agent) | etape reflect (validation pure + ecritures) |
| 06-09 | 1d66ab2 | feat(prompts) | prompt de reflexion (capitalisation post-tour) |
| 06-09 | da8d291 | feat(config) | cles reflect + skills appris + summarization recall |
| 06-09 | 40a527d | feat(skills) | agrege skills_learned (namespace learned) |
| 06-09 | 25d3dd5 | feat(evals) | harnais d'eval du skill code-review |
| 06-09 | 7d62a43 | docs(plans) | design closed-learning-loop + plans memoire |
| 06-09 | bdc844e | docs(prompt) | presente remember/recall + memoire persistante |
| 06-09 | 561d10f | feat(web) | cable la memoire + injecte le bloc identite |
| 06-09 | a909032 | feat(tools) | outils recall/remember (memoire persistante) |
| 06-09 | 0add967 | feat(config) | bloc [memory] + identity_max_tokens |
| 06-09 | 08240a7 | feat(memory) | identite markdown SOUL/USER/MEMORY |
| 06-09 | 6f72662 | feat(memory) | provider local SQLite FTS5 |
| 06-09 | a8f9e6e | feat(memory) | interface MemoryProvider + selection |
| — | be69fb6 | feat(evals) | harnais d'eval des prompts (master) |
| — | d133b46 | feat(prompts) | durcit chat/subagent + corrige la liste d'outils |
| — | a9151bb | refactor(config) | renomme .local.toml en .personnel.toml |
| — | 6447316 | feat(web) | rendu LaTeX offline (MathJax) + compteur live |
| — | 0712b47 | feat(web) | défaut = dernier modèle + robustesse vision |
| — | 0b408fa | feat(agent) | rattrapage des tool-calls textuels |
| — | 4832d8b | docs | présentation de Loom générée par Qwen3.6 |
| — | 88a04f1 | feat(web) | compteur live tokens envoyés/reçus |
| — | 05fdbbe | docs(bench) | abandonne le drafter MTP, garde le sweep MoE |
| — | 74a4cb2 | feat(web) | badge d'alerte mode permission |
| — | 88f86b0 | feat(skills) | ajoute le skill code-review |
| — | 3c1a706 | fix(securite) | durcit les outils web |
| — | 3116b42 | feat(tools) | write_note/read_note + fix du thrash |
| — | 5263016 | feat(web) | keep-warm du modèle |
| — | a85b479 | fix(runtime) | termine l'arbre de process sur Ctrl+C |
| — | 36b6fae | loom | correction des chemins relatifs |
| — | 92ba919 | docs(loom) | carte du package dans \_\_init\_\_ |
| — | 38d68dd | refactor | range le coeur dans loom/agent/ |
| — | 72f63e0 | refactor | regroupe l'infra de service dans loom/runtime/ |
| — | 165766d | docs | actualise README + ETAT_PROJET |
| — | febc20a | chore(models) | template \_TEMPLATE/model.toml |
| — | 4021291 | docs | archive les specs/plans de code retirés |
| — | 67c87e1 | chore(models) | ne garder que les MoE 24B+ |
| — | 773e8fd | feat(debug) | skill de debug + check_page |
| — | 3b70c8a | docs(debug) | spec skill de debug |
| — | 8584d0c | docs(eval) | design du banc d'eval |
| — | fd2ea61 | chore(skills) | retire le skill exemple |
