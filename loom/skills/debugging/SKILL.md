---
name: debugging
description: Utilise dès qu'un bug, un échec de test, un comportement inattendu ou une page/programme qui ne marche pas apparaît — AVANT de proposer un correctif. Méthode systématique : reproduire, localiser avec les outils, trouver la cause racine, changement minimal, preuve forte, réécrire si la base est pourrie. Généraliste (web, CLI, python, script).
---

# Debug systématique

Les correctifs au hasard font perdre du temps et créent de nouveaux bugs. On ne corrige
JAMAIS un symptôme : on trouve la **cause racine** d'abord, avec les outils, pas en devinant.

**Règle d'or : aucun correctif tant que la cause n'est pas localisée.**

## La méthode, en 6 temps

### 1. Reproduire
Déclenche le bug de façon **fiable** avant de toucher au code :
- une commande qui échoue → `run_shell` (lis le message d'erreur EN ENTIER : fichier, ligne, code) ;
- une page web → `check_page` (erreurs console) puis `check_interactive` (le geste qui casse) ;
- pas de repro fiable → rassemble plus de données, **ne devine pas**.

Aucun fix sans repro.

### 2. Localiser AVEC les outils
Tu as des yeux et des mains — sers-t'en, ne suppose rien :
- `read_file` : lis l'**état réel** du fichier (jamais de mémoire ou d'hypothèse sur son contenu) ;
- `search_text` / `find_files` : retrouve la définition, les appels, le point d'entrée ;
- `run_shell` : exécute pour **voir** l'erreur réelle, ajoute un log/print temporaire si besoin ;
- `check_page` : **vois** la page rendue (erreurs console, éléments manquants, diagnostic de hang).

Remonte la donnée jusqu'à sa **source** : où la mauvaise valeur naît-elle ? qui l'a produite ?
Corrige à la source, pas là où elle explose.

### 3. Cause racine unique
Formule **une** hypothèse vérifiable : « la grille ne s'affiche pas PARCE QUE `placeMines`
boucle à l'infini, car les cellules valent `null` (falsy) et la condition ne place jamais de
mine ». Pas un vague « il y a un souci de rendu ». Si plusieurs causes possibles, instrumente
(logs aux frontières) pour trancher avant de corriger.

### 4. Changement minimal
Corrige **la cause**, **une seule chose** à la fois. Pas de refactor opportuniste pendant un
debug — ça brouille la preuve.

### 5. Preuve forte
Relance la repro et **CONSTATE** le succès, preuve runnable à l'appui :
- `check_interactive` avec une **post-condition réelle** (un `expect` testable : la cellule
  cliquée porte la classe `open`, etc.) — pas une suite de clics sans assertion ;
- ou la sortie / le code de sortie d'une commande `run_shell`.

« Ça devrait marcher » n'est pas une preuve. Une preuve sans assertion réelle ne compte pas.

### 6. Réécris si la base est pourrie
Si le code de départ cumule des bugs liés, n'a pas de structure cohérente, ou que chaque fix
en révèle un autre : **repars de zéro**. Réécrire proprement est souvent plus sûr et plus
rapide que rapiécer un code cassé en profondeur — c'est une force, pas un aveu d'échec.
Décide-le explicitement au lieu d'empiler les rustines.

## Garde-fous (anti-acharnement)
- Un fix qui ne marche pas → **retour à l'étape 2 (localiser)**, surtout pas un autre patch au hasard.
- Ne change qu'**une** chose entre deux preuves : sinon tu ne sais pas ce qui a agi.
- Ne déclare jamais « corrigé » sans avoir **relancé la repro** et vu la preuve passer.
- Un `check_page` qui renvoie « chargement non terminé / script bloquant » = boucle infinie
  probable à l'init : cherche une boucle (`while`/`for`) sans condition de sortie, ou désactive
  les scripts un par un pour bisecter.
