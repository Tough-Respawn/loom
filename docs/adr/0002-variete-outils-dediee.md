# ADR 0002 — Variété d'outils dédiés plutôt qu'outils généraux

- Statut : Accepté
- Date : 2026-06-06

## Contexte
Loom expose 18 outils, dont 5 rien que pour l'édition de fichiers
(write_file, append_file, edit_file, replace_lines, insert_lines). La question se
pose régulièrement : n'est-ce pas trop ? Faut-il consolider pour réduire la charge
de sélection du petit modèle (Gemma 4B) ?

La tentation de consolidation vient d'un raisonnement « moins d'outils = moins de
confusion ». Elle est trompeuse : elle déplace la complexité du nombre d'outils vers
la difficulté d'usage de chaque outil restant — et c'est exactement ce qu'un 4B gère
le moins bien.

## Décision
On garde la variété. Principe directeur (partagé par les gros harnais agentic) :

> Un outil DÉDIÉ, au schéma serré et aux messages d'erreur sur mesure, bat un outil
> GÉNÉRAL que le modèle doit piloter correctement.

Chaque outil dédié retire un degré de liberté au modèle, donc un mode d'échec. La
complexité quitte le modèle (peu fiable) pour aller dans du Python déterministe
(fiable). Exemple : `run_shell` pourrait théoriquement chercher/lister/lire, mais il
faudrait alors que le 4B écrive du PowerShell correct, parse la sortie, gère les
spécificités Windows. `find_files` / `search_text` / `list_dir` / `read_file`
suppriment tout cela.

### Pourquoi 5 outils d'édition : chacun neutralise une contrainte distincte
- **write_file** — baseline : créer un nouveau fichier ou réécrire un petit fichier.
- **append_file** — contrainte d'**overflow** : un gros fichier ne tient pas dans une
  seule réponse (plafond `max_tokens`) sans être tronqué → JSON d'appel cassé. On
  écrit le début puis on complète par morceaux successifs.
- **replace_lines** — contrainte de **recopie exacte** : un 4B ne recopie pas un bloc
  existant au caractère près (indentation, espaces). On adresse par NUMÉROS de ligne
  (vus dans read_file), sans recopier l'ancien texte. Outil d'édition principal.
- **insert_lines** — ajouter du code au MILIEU sans rien remplacer ni recopier le
  contexte autour (adressage par ligne, `after_line`).
- **edit_file** — le petit remplacement UNIQUE (un nom, un token) où matcher une
  string exacte est plus naturel que compter des lignes.

La matrice est orthogonale à l'INTENTION :
nouveau fichier → write_file · ajouter à la fin → append_file · ajouter au milieu →
insert_lines · remplacer un bloc par sa position → replace_lines · remplacer une
courte string unique → edit_file.

### La seule vraie ambiguïté de sélection : edit_file vs replace_lines
Tous deux « remplacent quelque chose au milieu ». Règle de départage, inscrite dans
la description de `edit_file` et le prompt : si tu connais les numéros de ligne (tu
viens de lire le fichier) ou si le bloc est long/indenté → `replace_lines` (pas de
recopie exacte à risque) ; si c'est un petit texte unique recopiable → `edit_file`.

## Conséquences
- + Chaque outil supprime un mode d'échec concret du 4B ; descriptions et erreurs
  taillées pour son cas réduisent les boucles d'essai-erreur.
- + La frontière outil centralisée (loom/tools/base.py, `validate_and_coerce` +
  récupération d'outil inconnu, cf. ADR de durcissement) absorbe les fautes de type
  et de nommage : ajouter des outils ne multiplie pas les chemins d'erreur.
- − Charge de sélection réelle : on l'adresse par des descriptions à niche claire
  (« quand l'utiliser / quand NE PAS ») dans le schéma vu par le modèle, et un guide
  « QUEL OUTIL, QUAND » dans le prompt — PAS par la réduction du nombre d'outils.
- Règle pour l'avenir : on ajoute un outil quand il neutralise un mode d'échec qu'un
  outil existant ne couvre pas proprement ; on ne consolide pas par simple goût du
  minimalisme.
