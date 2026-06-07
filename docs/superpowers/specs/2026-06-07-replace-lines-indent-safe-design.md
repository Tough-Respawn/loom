# Spec — `replace_lines` / `insert_lines` indent-safe (auto-snap + validation différentielle)

Date : 2026-06-07
Statut : design validé, prêt pour plan d'implémentation
Périmètre : `loom/tools/fs.py` (édition par numéro de ligne)

## 1. Problème

`replace_lines` (fs.py:296) et `insert_lines` (fs.py:374) écrivent le `content`
fourni par le modèle **verbatim** entre les lignes voisines. Un petit modèle fournit
souvent un bloc **mal indenté** → en Python (où l'indentation EST la syntaxe), le
fichier casse (`IndentationError`).

Observé en live (run démineur, 2026-06-06/07) :
- Qwen3.5-4B a remplacé une ligne par `if self.game...:` **en colonne 0** à
  l'intérieur d'une méthode → `IndentationError`, et n'a pas su se rattraper.
- Gemma 4B casse de la même façon en patchant un fichier de ~200 lignes.

C'est **le** point faible d'édition identifié. Les autres langages (JS/C…) utilisent
des séparateurs (`;`, `{}`) → l'indentation y est cosmétique, pas porteuse de syntaxe.

## 2. Objectif

1. **Garantie** : un edit par numéro de ligne ne doit **jamais transformer un fichier
   qui compilait en fichier cassé pour cause d'indentation**.
2. **Proactif** : corriger automatiquement l'erreur la plus courante (bloc à l'indent
   de base faux, typiquement col-0) en collant le bloc au contexte.
3. **Ne pas bloquer l'édition incrémentale** d'un fichier déjà en cours de construction
   (donc déjà non compilable) — la validation est **différentielle**.

## 3. Non-objectifs (specs séparés)

- Guillemets typographiques `’`/`‘` émis par Qwen → **profils par modèle** (spec à part).
- Refonte AST / édition structurelle par symbole.
- Auto-format systématique à chaque écriture (le modèle a déjà `format_code` à la demande).
- Toucher `write_file` / `append_file` / `edit_file` (contenu autonome ou string exacte).

## 4. Design

### 4.1 `_snap_indent(content, target_indent) -> str` (pure)

- Normalise `content` en `\n`.
- `textwrap.dedent(content)` : retire l'indent **commun** → **préserve l'indentation
  relative interne** du bloc.
- Re-préfixe chaque ligne **non vide** par `target_indent` ; les lignes vides restent
  vides (pas de whitespace résiduel).
- Renvoie joint en `\n` (le style de fin de ligne réel du fichier est appliqué ensuite
  par `_new_block`).
- **Idempotent** : si le modèle indentait déjà juste, dedent+reprefix redonne l'identique.

### 4.2 `_indent_unit(lines) -> str` + `_target_indent(lines, anchor, suffix) -> str` (pure)

- `_indent_unit` : déduit l'unité d'indentation du fichier (1er saut d'indentation
  rencontré : un `\t`, ou N espaces). Défaut : 4 espaces. **Tab-aware** : on renvoie le
  vrai caractère du fichier.
- `_target_indent` : indentation (string whitespace réelle) à appliquer au bloc :
  - `cur` = ligne d'ancrage (replace : `lines[start-1]` ; insert : la ligne *suivant*
    l'insertion, sinon la précédente).
  - `prev` = dernière ligne **non vide** au-dessus de l'ancre.
  - Si le fichier est Python (`.py/.pyi/.pyw`) **et** `prev` strippée se termine par `:`
    (elle ouvre un bloc) → `target = indent(prev) + _indent_unit`.
  - Sinon → `target = indent(cur)` si `cur` non vide, sinon `indent(prev)`, sinon `""`.

### 4.3 Validation différentielle (fichiers à indentation : `.py/.pyi/.pyw`)

Helper `_py_compiles(text) -> bool` : `compile(text, "<edit>", "exec")` dans un
try/except ; `True` si OK, `False` sinon.

Helper `_is_indent_error(text) -> bool` : `compile(...)` ; `True` **seulement** si
l'exception est `IndentationError`/`TabError` (sous-classes ciblées), `False` sinon.

Flux dans `run()` (replace_lines et insert_lines), **uniquement si le fichier est
Python** ; sinon on saute toute validation :

1. `before_ok = _py_compiles(before_text)`.
2. Calculer `target_indent`, `snap`, et `new_text` **en mémoire** (rien d'écrit encore).
3. Si `before_ok` **et** `_is_indent_error(new_text)` :
   → **NE PAS écrire** (rollback implicite). Renvoyer une erreur actionnable :
   « ton bloc casse l'indentation (IndentationError ligne N : <msg>) — le fichier
   n'a pas été modifié. Réémets avec la bonne indentation. » + le `_context_after_edit`
   de l'état **AVANT** (inchangé, numéros corrects) pour que le modèle corrige.
4. Sinon → écrire (atomic). Si le résultat ne compile pas (pour une raison NON
   d'indentation, ou parce que `before` était déjà cassé) → écrire quand même et
   **annexer un warning** : « note : le fichier ne compile pas encore (… ligne N) —
   poursuis tes edits ». (On ne bloque que la **régression d'indentation d'un fichier
   valide**, rien d'autre.)

Pour les fichiers **non-Python** : `_snap_indent` est appliqué (tab-aware), **pas** de
validation (séparateurs `;`/`{}` → indentation non porteuse).

### 4.4 Transparence

- Quand `_snap_indent` a **changé** l'indentation du bloc → ajouter au message :
  « (bloc ré-indenté pour coller au contexte) ».
- `_context_after_edit` (existant) continue de renvoyer l'état re-numéroté à jour.

## 5. Intégration

- Helpers purs ajoutés à `fs.py` (ou un petit module `loom/tools/indent.py` si `fs.py`
  grossit trop — décision au plan).
- `make_replace_lines.run` : entre la validation de plage et `_new_block`, insérer
  `target = _target_indent(...)`, `content = _snap_indent(content, target)`, puis la
  validation différentielle autour de l'écriture.
- `make_insert_lines.run` : même traitement (ancre = ligne suivant l'insertion).

## 6. Limites assumées

- Éditer par n° de ligne **à l'intérieur d'une chaîne multi-lignes** (docstring,
  heredoc) ré-indente ce contenu. Rare ; documenté (choix « tous fichiers » assumé).
- `_snap_indent` ne corrige pas une indentation **relative interne** fausse — mais la
  **validation différentielle l'attrape** (rollback) si ça casse un fichier Python valide.
- Tabs/espaces mixtes dans le bloc du modèle : `dedent` peut ne rien retirer ; la
  validation différentielle attrape la casse en Python.

## 7. Vérification (pas de suite de tests — règle projet)

Smoke manuels + run live :
1. `.py` valide, `replace_lines` d'une ligne par un bloc `if …:` en col-0 →
   le bloc colle à l'indent de la méthode + `py_compile` OK.
2. `.py` valide, bloc qui casse vraiment l'indent relatif → **rollback** (fichier
   inchangé) + message avec n° de ligne.
3. `.py` **déjà cassé** (mid-construction), `replace_lines` → l'edit passe (pas de
   blocage) + warning annexé.
4. `.js`, `replace_lines` → snap appliqué, pas de validation, écrit normalement.
5. Run live : re-tester le démineur avec Gemma.

## 8. Fichiers touchés

- `loom/tools/fs.py` (helpers + intégration `make_replace_lines`, `make_insert_lines`).
- Éventuel `loom/tools/indent.py` (si extraction préférable — à trancher au plan).
