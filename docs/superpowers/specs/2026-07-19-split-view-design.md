# Split view : 2 à 4 sessions visibles en même temps

Date : 2026-07-19 · Statut : design validé par le user (référence UX : split éditeur VS Code)

## Besoin

Afficher simultanément 2, 3 ou 4 onglets de session, chacun en chat complet (sa timeline,
sa barre de saisie, son compteur), et pouvoir déplacer les onglets d'un panneau à l'autre.
Déclenchement par clic droit sur un onglet, glisser-déposer, et raccourcis clavier.

## Décisions

- Chaque panneau = un chat complet interactif (pas de mode lecture seule).
- Layouts automatiques : 2 → côte à côte ; 3 → 2 haut + 1 bas ; 4 → grille 2×2.
  Pas de redimensionnement des séparateurs en v1.
- Un onglet est soit affiché dans un panneau, soit en arrière-plan (comme aujourd'hui).
  Fermer un panneau ne ferme pas la session.
- Le panneau « focus » porte un liseré ; les raccourcis globaux ciblent le focus.
- Backend inchangé : la génération concurrente par session existe déjà. La disposition
  des panneaux est persistée côté client (localStorage).

## Architecture

- **État** : `state.panes = [sid, …]` (1 à 4) + `state.focused`. La vue simple actuelle
  est le cas dégénéré `panes.length === 1`.
- **Phase 1 (invisible)** : extraire la colonne chat en composant `ChatView(sid)`
  instanciable N fois — suppression des singletons (`activeTab()`, zone chat unique,
  saisie unique, compteur unique). Zéro changement visuel ; non-régression par le banc
  E2E existant.
- **Phase 2 (visible)** : grille CSS 1-4 panneaux ; clic droit sur onglet (« Ouvrir à
  droite », « Ouvrir en dessous », « Remettre en vue simple ») ; glisser un onglet sur un
  panneau (déplacer) ou vers un bord (scinder) ; raccourcis `Ctrl+\` (scinder),
  `Ctrl+1..4` (focus panneau), `Ctrl+Maj+\` (vue simple).

## Validation

- Phase 1 : pytest complet + banc E2E Playwright existant (aucun changement attendu).
- Phase 2 : scénario E2E nouveau — 2 sessions génèrent en parallèle dans 2 panneaux,
  saisie indépendante dans chacun, déplacement d'un onglet entre panneaux.

## Risque principal

`activeTab()` est utilisé à des dizaines d'endroits de app.js (3 149 lignes) : la
phase 1 concentre le risque, d'où sa validation isolée avant tout changement visuel.
