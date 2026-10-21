# Architecture : session privée isolée par l'OS

Statut : proposition (2026-10-21). La couche applicative est décrite dans
[session-privee.md](session-privee.md) ; ce document vise l'étape suivante.

## Promesse

En session privée, **Loom et le modèle local n'atteignent jamais internet**, et aucune
donnée de la session (prompts, fichiers, résultats d'outils) ne quitte la machine. La
garantie doit venir de l'OS, pas de la coopération du code ou des programmes lancés.

Décision ouverte : réseau local autorisé ou non. Autoriser la box laisse un canal DNS
(la box relaie les requêtes vers internet). Recommandation : aucun réseau hors machine.

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

1. Prototype Windows minimal (élévation admin, étape séparée) : Python, Git Bash,
   sous-processus et modèle local fonctionnent ; internet coupé en IPv4, IPv6, UDP, DNS.
2. Interface `spawn` fixée sur les contraintes vérifiées.
3. Extraction du service privé.
4. Implémentations Linux et macOS.
