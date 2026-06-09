# Loom — Skill de debug généraliste + localisation check_page — Design

**Date :** 2026-06-09
**Statut :** design validé (relecture utilisateur faite en brainstorming).
**Branche :** `feat/debug-skill`.

## Objectif

Donner aux modèles locaux une **méthode de debug systématique, généraliste**, déclenchée
automatiquement (via le catalogue de skills + `use_skill`), pour qu'ils **utilisent leurs
outils** au lieu de deviner — et rendre `check_page` **utile à la localisation** quand une page
ne charge pas (au lieu d'un timeout muet). Diagnostic du terrain : ce n'est pas une limite
modèle, c'est l'usage des outils + un `check_page` opaque sur les hangs.

Deux livrables, ensemble :
1. **Skill** `loom/skills/debugging/SKILL.md` — connaissance déclenchée par le modèle.
2. **Outillage** — `check_page` (`loom/tools/browser.py`) ne perd plus les preuves et donne un
   indice de localisation sur timeout.

## 1. Skill `loom/skills/debugging/SKILL.md`

Frontmatter :
- `name: debugging`
- `description:` ciblé déclenchement — « Utilise dès qu'un bug, échec de test, comportement
  inattendu ou page qui ne marche pas apparaît, AVANT de proposer un correctif. Méthode :
  reproduire, localiser avec les outils, cause racine, changement minimal, preuve forte,
  réécrire si la base est pourrie. »

Corps (généraliste, centré sur les VRAIS outils de Loom) — la méthode en 6 temps :

1. **Reproduire** — déclenche le bug de façon fiable avant tout fix (commande via `run_shell`,
   `check_page`/`check_interactive`). Pas de repro fiable → rassembler des données, ne pas
   deviner. **Aucun fix sans repro.**
2. **Localiser avec les outils** — lis l'état RÉEL (`read_file` ; ne suppose jamais le
   contenu), cherche (`search_text`/`find_files`), exécute pour voir l'erreur réelle
   (`run_shell` : lis le message d'erreur EN ENTIER, ligne/fichier), **vois** la page
   (`check_page` → erreurs console + diagnostic). Remonte la donnée jusqu'à sa **source** (où
   la mauvaise valeur naît), pas le symptôme.
3. **Cause racine unique** — formule UNE hypothèse vérifiable de la cause (pas « ça doit
   être… » vague). Si plusieurs candidates, instrumente pour trancher (logs aux frontières).
4. **Changement minimal** — corrige la **cause**, une seule chose à la fois ; pas de
   refactor opportuniste pendant un debug.
5. **Preuve forte** — relance la repro et **CONSTATE** le succès, preuve runnable à l'appui
   (`check_interactive` avec une post-condition réelle ; sortie/exit d'une commande). « Ça
   devrait marcher » n'est pas une preuve. Une preuve sans assertion réelle ne compte pas.
6. **Réécris si la base est pourrie** — si le code de départ est incohérent / cumule des bugs
   liés / n'a pas de structure, **repartir de zéro** est souvent plus sûr et plus rapide que
   rapiécer. Le greenfield est une force des modèles ; décide-le explicitement plutôt que
   d'empiler les rustines.

Garde-fous (anti-thrash) :
- Un fix qui ne marche pas → **retour à l'étape 2 (localiser)**, pas un autre patch au hasard.
- Ne change qu'UNE chose entre deux preuves (sinon on ne sait pas ce qui a agi).
- Ne déclare pas « corrigé » sans avoir relancé la repro et vu la preuve.

Style : court, impératif, généraliste (web, CLI, python, script — pas web-only). Pas de
référence à des outils absents de Loom (ni TDD/git worktrees facultatifs).

## 2. Outillage — `check_page` localisant (`loom/tools/browser.py`)

Comportement actuel : sur timeout de `goto`, `check_page` **lève** une `ToolError`
« échec du chargement … Timeout » et **jette** les erreurs console captées avant le hang →
le modèle n'a rien à localiser.

Changements (dans `make_check_page` → `run`) :
- **Ne jamais jeter les preuves.** Les handlers `console`/`pageerror` accumulent dans des
  listes qui survivent à l'exception. Sur timeout/échec de chargement, **retourner** (via
  `untrusted`, comme le cas nominal) un rapport structuré incluant ces erreurs/warnings + un
  diagnostic — au lieu de lever (sauf l'absence de navigateur, qui reste une `ToolError` de
  setup).
- **Distinguer** le timeout de chargement du timeout de `wait_selector`, et joindre un
  **indice de localisation** sur timeout de chargement : « chargement non terminé en 15s →
  script bloquant probable (ex. boucle infinie à l'init) ; bisecte en désactivant les scripts ;
  les erreurs console ci-dessus pointent souvent la cause ».
- **Ne pas re-hang après un timeout.** En cas de timeout de chargement, NE PAS tenter
  `title()`/`inner_text()`/`query_selector_all()` (le thread JS peut être gelé → nouveaux
  timeouts de 30 s). On ferme et on renvoie le diagnostic + ce qui a été capté. `set_default_timeout`
  court (≈5 s) pour borner les lectures du cas nominal partiellement bloqué.
- **Honnêteté** : sur une boucle synchrone qui gèle le thread, Playwright ne peut pas tout
  introspecter ; on renvoie l'indice + les preuves captées, jamais un timeout muet.

`check_interactive`/`run_interactive` : déjà structurés (ne lèvent pas) ; hors scope de ce
changement, sauf à réutiliser le même message d'indice si utile.

## Tests (pas de pytest ; smokes `uv run python -c`)

1. **Page qui hang** (script avec `while(true){}` à l'init) → `check_page` **retourne** un
   rapport contenant l'indice « script bloquant » et ne lève pas ; l'appel termine en < ~20 s.
2. **Page avec erreur console** (JS qui `throw`/référence indéfinie) → l'erreur est listée
   dans le rapport (cas nominal, déjà couvert, à re-vérifier après refactor).
3. **Page saine** → 0 erreur, comptes corrects (non-régression du cas nominal).
4. **Skill** : `collect_skills('loom/skills', None)` liste `debugging` ; sa `description`
   contient les mots déclencheurs ; `load_skill_body(..., 'debugging')` renvoie le corps.

## Décisions actées

- Livraison = **skill Loom natif** (pas la consommation de superpowers), adapté aux outils de
  Loom et aux modèles 24B+.
- Périmètre = **skill + outillage de localisation**, livrés ensemble.
- Généraliste (le démineur n'est qu'un litmus).
- Le **banc d'éval** (différé) mesurera l'effet de ce skill. [[loom-banc-eval]]
