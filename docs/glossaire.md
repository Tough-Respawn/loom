# Glossaire

**Partie 1** : inférence locale et matériel (les chiffres s'appuient sur le poste réel :
Dell G7, RTX 2060 6 Go, 64 Go DDR4 ; ornith Q8 35B MoE calibré `n_cpu_moe = 40`).
**Partie 2** : concepts généraux LLM/agents, alignés sur les 5 domaines de la certification
**Anthropic CCA-F** (Claude Certified Architect – Foundations), définitions FR + EN.

# Partie 1 — Inférence locale et matériel

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

# Partie 2 — Concepts généraux & certification CCA-F

Chaque terme : définition en français, puis en anglais (*EN:*). Organisé selon les
5 domaines de l'examen CCA-F (Claude Certified Architect – Foundations), précédés
des fondamentaux.

## Fondamentaux LLM & API

**LLM (Large Language Model)** : modèle de langage entraîné sur d'immenses corpus de texte à prédire le token suivant ; toutes ses capacités (raisonnement, code, dialogue) émergent de cet objectif simple à très grande échelle. *EN: a language model trained on massive text corpora to predict the next token; its capabilities (reasoning, code, dialogue) emerge from that simple objective at very large scale.*

**Inférence (inference)** : la phase d'*utilisation* d'un modèle entraîné (générer des réponses), par opposition à l'entraînement. C'est ce que fait une API ou un serveur local. *EN: the phase of using a trained model to generate outputs, as opposed to training it.*

**System prompt** : les instructions de cadrage placées avant la conversation, qui définissent le rôle, les règles et le style de l'assistant. L'utilisateur ne le voit pas ; il a priorité sur les messages ordinaires. *EN: framing instructions placed before the conversation that define the assistant's role, rules and style; it takes precedence over ordinary messages.*

**Rôles (user / assistant / tool)** : chaque message d'une conversation API porte un rôle : `user` (l'humain ou le harnais), `assistant` (le modèle), et les résultats d'outils. L'alternance des rôles structure le dialogue. *EN: every API message carries a role — user, assistant, or tool result — and the alternation of roles structures the dialogue.*

**Temperature** : paramètre de créativité de l'échantillonnage (0 = quasi déterministe, choisit toujours le token le plus probable ; ~1 = varié). Basse pour l'extraction et le code, plus haute pour la créativité. *EN: sampling creativity parameter (0 = near-deterministic, always the most likely token; ~1 = varied). Low for extraction and code, higher for creative tasks.*

**Top-p / top-k (échantillonnage)** : autres réglages d'échantillonnage : top-p limite le choix aux tokens couvrant p % de probabilité cumulée, top-k aux k plus probables. On règle généralement temperature OU top-p, pas les deux. *EN: sampling controls — top-p restricts choice to tokens covering p% of cumulative probability, top-k to the k most likely. Tune temperature or top-p, not both.*

**max_tokens** : plafond de tokens que le modèle peut générer dans sa réponse. Atteint = la réponse est coupée net (stop_reason `max_tokens`). *EN: the ceiling on tokens the model may generate; hitting it truncates the answer (stop_reason max_tokens).*

**Stop sequence** : chaîne définie par le développeur qui, si le modèle la génère, arrête immédiatement la génération. *EN: a developer-defined string that immediately halts generation when the model produces it.*

**stop_reason** : champ de la réponse API indiquant *pourquoi* la génération s'est arrêtée : `end_turn` (fin naturelle), `tool_use` (le modèle appelle un outil), `max_tokens` (coupé), `stop_sequence`. Une boucle agent se pilote en lisant ce champ. *EN: the API response field saying why generation stopped — end_turn, tool_use, max_tokens, stop_sequence. Agent loops are driven by reading this field.*

**Streaming** : recevoir la réponse token par token (SSE) au lieu d'attendre la fin, pour afficher au fil de l'eau et réduire la latence perçue. *EN: receiving the response token by token (SSE) instead of waiting for completion, cutting perceived latency.*

**Fenêtre de contexte (context window)** : la quantité maximale de tokens que le modèle peut voir en une requête (prompt + réponse). Claude : 200 000 tokens en standard. Tout ce qui dépasse doit être résumé, tronqué ou découpé. *EN: the maximum tokens a model can see in one request (prompt + response) — 200K for Claude standard; anything beyond must be summarized, truncated or chunked.*

**Multimodal / vision** : capacité d'un modèle à recevoir autre chose que du texte (images, PDF) dans ses messages. *EN: a model's ability to take non-text inputs (images, PDFs) in its messages.*

**Hallucination** : le modèle affirme avec assurance quelque chose de faux ou d'inventé (fait, citation, API inexistante). Se mitige par le grounding (RAG, outils, citations) et la vérification. *EN: the model confidently asserts something false or invented; mitigated by grounding (RAG, tools, citations) and verification.*

**Zero-shot / few-shot** : demander une tâche sans exemple (zero-shot) ou en fournissant quelques exemples entrée→sortie dans le prompt (few-shot), ce qui améliore fortement le respect du format. *EN: asking a task with no examples (zero-shot) or with a few input→output examples in the prompt (few-shot), which strongly improves format adherence.*

**Chain-of-thought (CoT)** : inciter le modèle à raisonner étape par étape avant de conclure (« think step by step »), ce qui améliore les tâches complexes. *EN: prompting the model to reason step by step before answering, improving complex tasks.*

**Extended thinking** : mode où Claude produit un raisonnement interne long et budgétisé avant sa réponse visible ; on règle un budget de tokens de réflexion. *EN: a mode where Claude produces long, budgeted internal reasoning before its visible answer; you set a thinking-token budget.*

**Embeddings** : représentation d'un texte en vecteur numérique tel que des textes proches en sens sont proches en distance ; la brique de la recherche sémantique. *EN: numeric vector representations of text where semantically similar texts are close in distance; the building block of semantic search.*

**RAG (Retrieval-Augmented Generation)** : architecture où l'on recherche d'abord les documents pertinents (souvent via embeddings) et où on les injecte dans le prompt pour que le modèle réponde à partir de sources plutôt que de mémoire. *EN: an architecture that first retrieves relevant documents (often via embeddings) and injects them into the prompt so the model answers from sources rather than memory.*

**Fine-tuning** : ré-entraîner partiellement un modèle sur des données spécifiques pour spécialiser son comportement, par opposition au prompt engineering qui ne touche pas aux poids. *EN: partially retraining a model on specific data to specialize its behavior, as opposed to prompt engineering, which never touches the weights.*

**RLHF (Reinforcement Learning from Human Feedback)** : étape d'alignement où le modèle est optimisé selon des préférences humaines (réponses jugées meilleures) après le pré-entraînement. *EN: an alignment stage where the model is optimized against human preference judgments after pretraining.*

**Constitutional AI** : méthode d'alignement d'Anthropic où le modèle critique et corrige ses propres réponses selon une « constitution » de principes explicites, réduisant la dépendance au feedback humain. *EN: Anthropic's alignment method where the model critiques and revises its own outputs against an explicit "constitution" of principles, reducing reliance on human feedback.*

**Évals (evaluations)** : tests systématiques et reproductibles de la qualité d'un système IA (jeux de cas, notation automatique ou par modèle juge). On ne modifie pas un prompt sans éval avant/après. *EN: systematic, repeatable quality tests for an AI system (case sets, automatic or model-judged grading). Never change a prompt without a before/after eval.*

**Latence vs débit (latency vs throughput)** : temps de la première réponse pour UN utilisateur vs volume total servi par seconde pour TOUS. Les optimisations diffèrent (streaming et cache pour la latence ; batching pour le débit). *EN: time-to-first-answer for one user vs total volume served per second for all; optimizations differ (streaming and caching for latency, batching for throughput).*

**Batch API** : soumettre en masse des requêtes non urgentes traitées sous 24 h à prix réduit (-50 %), par opposition au temps réel. *EN: submitting bulk, non-urgent requests processed within 24h at reduced cost, as opposed to real-time calls.*

## Domaine 1 — Architecture agentique & orchestration

**Agent** : un LLM placé dans une boucle avec des outils et un objectif : il décide lui-même des actions, observe les résultats et itère jusqu'à accomplir la tâche — par opposition à un simple appel question→réponse. *EN: an LLM placed in a loop with tools and a goal: it decides its own actions, observes results, and iterates until the task is done — as opposed to a single question→answer call.*

**Boucle agentique (agentic loop)** : le cycle fondamental : prompt → le modèle répond ou appelle un outil (`stop_reason: tool_use`) → le harnais exécute l'outil → le résultat est réinjecté → le modèle continue, jusqu'à `end_turn`. *EN: the fundamental cycle — prompt → the model answers or calls a tool (stop_reason: tool_use) → the harness executes it → the result is fed back → the model continues, until end_turn.*

**Harnais (harness / scaffolding)** : le code non-IA autour du modèle : boucle, exécution des outils, gestion du contexte, permissions, erreurs. La qualité d'un agent dépend autant du harnais que du modèle. *EN: the non-AI code around the model — loop, tool execution, context management, permissions, errors. Agent quality depends on the harness as much as on the model.*

**Orchestrateur-travailleurs / hub-and-spoke** : pattern multi-agents où un agent superviseur décompose la tâche et délègue à des sous-agents spécialisés, puis agrège leurs synthèses. Le superviseur garde la compréhension ; les workers gardent le détail. *EN: a multi-agent pattern where a supervisor agent decomposes the task, delegates to specialized subagents, then aggregates their syntheses; the supervisor keeps the understanding, workers keep the detail.*

**Sous-agent (subagent)** : agent enfant lancé avec un contexte isolé et une mission autonome ; il ne voit pas la conversation principale et ne renvoie que son résultat final. Protège le contexte du principal. *EN: a child agent launched with an isolated context and a self-contained mission; it never sees the main conversation and returns only its final result, protecting the main context.*

**Prompt chaining** : enchaîner plusieurs appels au modèle, la sortie de l'un devenant l'entrée du suivant (extraire → transformer → rédiger). Plus simple et prévisible qu'un agent quand le flux est connu d'avance. *EN: chaining several model calls, each output feeding the next (extract → transform → write). Simpler and more predictable than an agent when the flow is known in advance.*

**Séquentiel vs parallèle** : exécuter les étapes l'une après l'autre (dépendances) ou simultanément (tâches indépendantes). Paralléliser les sous-agents indépendants réduit le temps total ; séquencer est obligatoire dès qu'une étape consomme le résultat d'une autre. *EN: running steps one after another (dependencies) or simultaneously (independent tasks). Parallelize independent subagents to cut wall-clock time; sequence whenever a step consumes another's result.*

**Décomposition de tâche (task decomposition)** : découper un objectif complexe en sous-tâches autonomes et vérifiables avant d'agir — le cœur du travail d'un orchestrateur. *EN: splitting a complex goal into self-contained, verifiable subtasks before acting — the core job of an orchestrator.*

**Boucle infinie de tool-use / limites d'itération** : risque qu'un agent répète indéfiniment le même appel d'outil qui échoue. Mitigation : compteurs d'itérations, plafonds stricts, détection de non-progrès, et erreurs d'outils informatives qui disent quoi changer. *EN: the risk of an agent endlessly repeating a failing tool call. Mitigated by iteration counters, hard caps, no-progress detection, and informative tool errors that say what to change.*

**Agrégation de résultats** : condenser les sorties de plusieurs outils/sous-agents en synthèses avant de les réinjecter, pour ne pas saturer le contexte de l'orchestrateur. *EN: condensing multiple tool/subagent outputs into syntheses before feeding them back, to avoid flooding the orchestrator's context.*

**Human-in-the-loop (HITL)** : point de contrôle où un humain valide avant qu'une action risquée ou irréversible ne s'exécute, ou après N échecs successifs (escalade). *EN: a checkpoint where a human approves before a risky or irreversible action executes, or after N consecutive failures (escalation).*

**Propagation et escalade d'erreurs** : politique de fiabilité : une erreur d'outil est d'abord rendue au modèle pour auto-correction ; au-delà d'un seuil déterministe (répétitions, gravité), on escalade vers l'humain au lieu de réessayer à l'infini. *EN: a reliability policy — a tool error is first returned to the model for self-correction; past a deterministic threshold (repeats, severity), escalate to a human instead of retrying forever.*

## Domaine 2 — Claude Code & environnement d'exécution

**Claude Code** : l'agent de développement d'Anthropic en ligne de commande/IDE : il lit, édite, exécute et vérifie du code sur la machine avec des outils, sous un régime de permissions. *EN: Anthropic's command-line/IDE coding agent: it reads, edits, runs and verifies code on the machine through tools, under a permission regime.*

**CLAUDE.md** : fichier d'instructions projet chargé automatiquement dans le contexte : commandes de build, conventions, règles. Hiérarchie : global utilisateur (`~/.claude/CLAUDE.md`) puis projet, le plus spécifique complétant le plus général. *EN: a project instruction file auto-loaded into context — build commands, conventions, rules. Hierarchy: user-global then project, the more specific complementing the more general.*

**Plan mode vs exécution directe** : mode où l'agent doit d'abord présenter un plan et obtenir l'approbation avant de modifier quoi que ce soit, par opposition à l'exécution immédiate. Pour les changements larges ou risqués. *EN: a mode where the agent must present a plan and get approval before changing anything, vs immediate execution; for broad or risky changes.*

**Modes de permission** : régimes contrôlant ce que l'agent exécute sans demander : lecture libre, confirmation pour les écritures/commandes (ask), listes d'autorisation/refus, jusqu'au tout-autorisé. Principe : friction proportionnelle au risque. *EN: regimes controlling what the agent may run unprompted — free reads, ask-before-write, allow/deny lists, up to fully permissive. Principle: friction proportional to risk.*

**Mode headless / CI** : exécuter l'agent sans humain devant (pipeline CI/CD, cron) : aucune question interactive possible, tout doit être pré-autorisé et les blocages d'entrée utilisateur prévenus. *EN: running the agent with no human present (CI/CD, cron): no interactive questions possible, everything must be pre-authorized and user-input hangs prevented.*

**Hooks** : commandes déclenchées automatiquement par le harnais autour des actions de l'agent (avant/après un outil, à la fin d'un tour) pour imposer des règles déterministes que le prompt seul ne garantit pas. *EN: commands the harness fires automatically around agent actions (before/after a tool, at turn end) to enforce deterministic rules a prompt alone cannot guarantee.*

## Domaine 3 — Prompt engineering & sorties structurées

**Prompt engineering** : concevoir les entrées du modèle (structure, exemples, contraintes, ordre) pour obtenir des sorties fiables — la première optimisation à faire, avant tout fine-tuning. *EN: designing model inputs (structure, examples, constraints, ordering) to get reliable outputs — the first optimization to try, before any fine-tuning.*

**Balises XML** : structurer le prompt avec des balises (`<rules>`, `<context>`, `<examples>`, `<document>`) pour séparer sans ambiguïté instructions, données et exemples — le format que Claude suit le mieux. *EN: structuring prompts with XML tags to unambiguously separate instructions, data and examples — the format Claude follows best.*

**Placement du contenu (system vs user)** : les règles durables et le rôle vont dans le system prompt ; les données de la tâche immédiate vont dans le message user. Mélanger les deux dégrade l'obéissance et casse le cache. *EN: durable rules and role go in the system prompt; immediate task data goes in the user message. Mixing them degrades adherence and breaks caching.*

**Sortie structurée / JSON Schema enforcement** : contraindre la réponse à un schéma JSON valide (via tool-use forcé ou mode dédié) pour qu'un programme puisse la consommer sans parsing fragile. *EN: constraining the response to a valid JSON schema (via forced tool use or a dedicated mode) so a program can consume it without fragile parsing.*

**Boucle de validation/retry** : vérifier programmatiquement la sortie (schéma, contraintes) et, en cas d'échec, renvoyer au modèle l'erreur précise pour qu'il corrige — plutôt que réessayer à l'aveugle. *EN: programmatically validating output (schema, constraints) and, on failure, sending the model the precise error to fix — rather than blind retries.*

**Prefill de réponse (assistant prefill)** : commencer soi-même le message assistant (par ex. `{` ou `<analysis>`) pour forcer le format de sortie dès le premier token. À ne pas confondre avec le prefill d'inférence de la partie 1. *EN: starting the assistant message yourself (e.g. `{`) to lock the output format from the first token. Not to be confused with inference prefill from part 1.*

**Role prompting** : assigner un rôle précis au modèle (« tu es un auditeur fiscal ») pour cadrer le ton, le niveau et les priorités de ses réponses. *EN: assigning the model a precise role ("you are a tax auditor") to frame tone, level and priorities.*

## Domaine 4 — Conception d'outils & MCP

**Tool use / function calling** : donner au modèle un catalogue de fonctions décrites en JSON Schema ; il répond par des appels structurés (nom + arguments) que le harnais exécute réellement. Le modèle décide *quand* et *quoi* appeler. *EN: giving the model a catalog of functions described in JSON Schema; it responds with structured calls (name + arguments) the harness actually executes. The model decides when and what to call.*

**Schéma d'outil (tool schema)** : la définition JSON d'un outil : nom, description, paramètres typés, champs requis. C'est un mini-prompt : sa clarté détermine la qualité d'usage par le modèle. *EN: a tool's JSON definition — name, description, typed parameters, required fields. It is a mini-prompt: its clarity determines how well the model uses it.*

**Tool misrouting** : le modèle choisit le mauvais outil parce que deux descriptions se chevauchent. Prévention : une intention par outil, descriptions distinctes et disjointes. *EN: the model picks the wrong tool because two descriptions overlap. Prevention: one intent per tool, distinct and disjoint descriptions.*

**MCP (Model Context Protocol)** : protocole ouvert standardisant la connexion entre applications IA et sources externes : un serveur MCP expose outils, ressources et prompts qu'importe quel client compatible peut consommer. « L'USB-C des applications IA ». *EN: an open protocol standardizing how AI apps connect to external sources: an MCP server exposes tools, resources and prompts any compatible client can consume — "the USB-C of AI apps".*

**MCP Host / Client / Server** : le host est l'application qui exécute la boucle agent (Claude Code, Claude Desktop) ; elle embarque un client MCP par serveur ; le serveur est le service qui expose les capacités (GitHub, base de données, navigateur). *EN: the host is the application running the agent loop; it embeds one MCP client per server; the server is the service exposing capabilities (GitHub, database, browser).*

**Primitives MCP (tools / resources / prompts)** : les trois choses qu'un serveur MCP expose : des outils (actions appelables par le modèle), des ressources (données lisibles adressées par URI), des prompts (gabarits réutilisables proposés à l'utilisateur). *EN: the three things an MCP server exposes — tools (model-callable actions), resources (readable URI-addressed data), prompts (reusable templates offered to the user).*

**Moindre privilège (least privilege)** : ne donner à l'agent que les outils et portées strictement nécessaires à sa tâche (lecture seule si possible, périmètre réduit), pour borner le rayon des dégâts d'une erreur ou d'une injection. *EN: granting the agent only the tools and scopes strictly needed for its task (read-only where possible, narrow perimeter), bounding the blast radius of an error or injection.*

## Domaine 5 — Gestion du contexte & fiabilité

**Gestion du contexte (context management)** : l'art de tenir dans la fenêtre : choisir quoi garder, résumer, tronquer ou externaliser (notes, fichiers) à mesure que la conversation grossit. *EN: the craft of fitting within the window — choosing what to keep, summarize, truncate or externalize (notes, files) as the conversation grows.*

**Compaction / résumé de conversation** : remplacer les tours anciens par un résumé dense pour libérer des tokens tout en préservant les faits décisifs. Risque : perdre un détail dont une étape future a besoin. *EN: replacing old turns with a dense summary to free tokens while preserving decisive facts. Risk: losing a detail a future step needs.*

**Prompt caching / cache_control** : marquer les préfixes stables du prompt (system, gros documents, schémas d'outils) avec des points de cache : les requêtes suivantes qui partagent ce préfixe coûtent ~10 % du prix et accélèrent. Toute modification du préfixe invalide le cache en aval. *EN: marking stable prompt prefixes (system, large documents, tool schemas) with cache breakpoints: subsequent requests sharing that prefix cost ~10% and run faster. Any prefix change invalidates the cache downstream.*

**État de session (session state)** : tout ce qui doit survivre entre les tours ou les redémarrages : historique, todos, fichiers de travail. Le modèle est sans mémoire propre ; l'état vit dans le harnais. *EN: everything that must survive across turns or restarts — history, todos, working files. The model has no memory of its own; state lives in the harness.*

**Guardrails (garde-fous)** : contrôles déterministes autour du modèle : validation des entrées/sorties, listes d'interdits, filtres, plafonds. Complément indispensable du prompt, qui n'est jamais une garantie. *EN: deterministic controls around the model — input/output validation, deny-lists, filters, caps. The indispensable complement to prompting, which is never a guarantee.*

**Injection de prompt (prompt injection)** : attaque où un contenu ingéré (page web, PDF, résultat d'outil) contient des instructions déguisées que le modèle risque de suivre. Défense : frontière de confiance (contenu externe = données, jamais instructions) + verrou sur les effets de bord. *EN: an attack where ingested content (web page, PDF, tool output) carries disguised instructions the model may follow. Defense: a trust boundary (external content = data, never instructions) plus gating of side effects.*

**Jailbreak** : tentative de contourner les règles de sécurité du modèle par la formulation (jeux de rôle, hypothétiques, encodages). Distinct de l'injection : le jailbreak vient de l'utilisateur, l'injection du contenu. *EN: attempts to bypass the model's safety rules through phrasing (role-play, hypotheticals, encodings). Distinct from injection: jailbreaks come from the user, injections from content.*

**Red teaming** : tester offensivement et délibérément un système IA (injections, jailbreaks, cas limites) avant ses utilisateurs malveillants. *EN: deliberately attacking your own AI system (injections, jailbreaks, edge cases) before malicious users do.*

**PII (données personnelles)** : informations identifiant une personne (nom, email, adresse) : à minimiser dans les prompts, les logs et les caches, par conception. *EN: personally identifiable information (name, email, address) — to be minimized in prompts, logs and caches by design.*

**Dégradation gracieuse (graceful degradation)** : concevoir le système pour qu'une panne d'un composant (cache, outil, serveur) réduise la performance au lieu de tout casser : replis en échelle, disjoncteurs, valeurs par défaut sûres. *EN: designing the system so a component failure (cache, tool, server) degrades performance instead of breaking everything — fallback ladders, circuit breakers, safe defaults.*
