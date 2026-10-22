# Architecture : session privée isolée par l'OS

Statut : proposition (2026-10-21). La couche applicative est décrite dans
[session-privee.md](session-privee.md) ; ce document vise l'étape suivante.

## Promesse

En session privée, **rien ne quitte la machine où Loom est installé** (Loom, ses
modèles, sa configuration) : ni internet, ni le reste du réseau local. Aucune donnée de
la session (prompts, fichiers, résultats d'outils) ne sort. La garantie doit venir de
l'OS, pas de la coopération du code ou des programmes lancés.

**Périmètre, déduit de la configuration, sans réglage utilisateur** : la machine, plus
les serveurs de modèles locaux que la configuration de Loom déclare (s'ils tournent sur
une autre machine, ils font partie de l'installation). Tout autre hôte, du LAN comme
d'internet, est hors périmètre : une machine du LAN peut retransmettre ce qu'elle reçoit.

**DNS** : aucun dans la cage. Un résolveur (celui de la box notamment) relaie les noms
vers internet, et le DNS sur HTTPS passe par internet. Loom résout lui-même, au
démarrage et hors de la cage, l'adresse de ses serveurs de modèles déclarés.

## Menaces couvertes

Bug de Loom, modèle manipulé par un contenu piégé, programme lancé par un outil (et ses
descendants), plugin ou serveur MCP, serveur du modèle. Hors périmètre : fichiers que
l'utilisateur place lui-même dans un dossier synchronisé, compromission de l'OS.

## Composants

| Composant | Rôle | Réseau autorisé |
|---|---|---|
| Service connecté | sessions normales, API distantes | libre |
| Service privé | interface et boucle des sessions privées, outils, relais vers le modèle local | aucun hors machine |
| Serveur du modèle | inférence locale | écoute sur 127.0.0.1, aucune connexion sortante |

Le service privé est un processus à part, sous une identité que l'OS confine. Le service
connecté ne reçoit jamais de données privées : un bug de routage ne peut donc pas les
envoyer dehors. Les échanges entre processus passent par un protocole limité (pas de
commande du type « télécharge cette URL », qui contournerait la cage).

**Relais sur la machine** : l'API web de Loom n'a aujourd'hui aucune authentification
(seul un garde CSRF basé sur un en-tête de navigateur) ; un programme de la session
pourrait la solliciter pour joindre une API distante. Une clé ne suffit pas seule (un
processus de même identité peut la lire : fichiers, environnement, mémoire) : la
séparation des identités est la vraie barrière, la clé un renfort. Tout code du modèle
s'exécute hors du processus de Loom (`run_workflow` faisait `exec()` dans Loom : coupé
en session privée par #138).

## Interface commune

`spawn(argv, policy)` avec deux politiques :
- `OFFLINE` : aucune connexion hors machine, descendants compris ;
- `SERVE_LOCAL` : écoute sur une interface locale autorisée, aucune connexion sortante.

Chaque OS fournit son implémentation ; Loom ne voit que cette interface.

## Implémentations candidates (à valider, pas encore un contrat)

| OS | Piste | Droits | État |
|---|---|---|---|
| Windows | compte local dédié + règles de pare-feu par compte (mode « elevated » de Codex) | admin à l'installation | à prototyper |
| Windows | MXC (isolation native, sans compte ni admin, builds récents) | aucun | disponibilité à vérifier sur la machine |
| Windows | AppContainer | aucun | mesuré 2026-10-21 : internet, LAN et DNS bloqués, mais Git Bash ne démarre pas et les fichiers du profil sont inaccessibles |
| Linux | Landlock (TCP dès l'ABI 4, UDP dès l'ABI 10) + seccomp, ou espace réseau vide (bubblewrap) | aucun | détection des capacités obligatoire |
| macOS | Seatbelt (`sandbox-exec`) | aucun | déprécié, langage de profils non public : dépendance à assumer |

## Validation (avant tout contenu privé)

Un échec de connexion ne prouve pas l'isolation (Wi-Fi coupé, serveur muet). À chaque
démarrage du service privé :
1. vérifier que la restriction est installée (règle présente, identité du processus) ;
2. tests contrôlés depuis la cage : IPv4, IPv6, UDP et DNS, descendants, relais locaux,
   avec un témoin hors cage qui, lui, joint la cible ;
3. si un contrôle échoue ou ne peut pas être fait : la session privée ne démarre pas.

## Étapes

1. Prototype Windows minimal (élévation admin, étape séparée, chaque action montrée
   avant d'être lancée). À démontrer : Python, Git Bash, sous-processus et modèle local
   fonctionnent ; internet ET réseau local coupés en IPv4, IPv6, UDP et DNS ; identités
   séparées (le processus confiné ne lit ni la clé, ni la mémoire, ni les fichiers du
   service connecté) ; relais sur la machine bloqués (l'API de Loom, et tout service
   local ; le filtrage du trafic vers 127.0.0.1 par une règle ordinaire est à tester,
   WFP sait le faire mais rien n'est supposé).
2. Interface `spawn` fixée sur les contraintes vérifiées.
3. Extraction du service privé.
4. Implémentations Linux et macOS.
