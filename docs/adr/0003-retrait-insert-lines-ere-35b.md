# ADR 0003 — Retrait de l'édition par numéro de ligne (insert_lines + replace_lines)

- Statut : Accepté
- Date : 2026-06-18
- Révise : ADR 0002 (Variété d'outils dédiés) — partiellement

## Contexte
ADR 0002 (2026-06-06) justifiait CINQ outils d'édition pour un petit modèle (Gemma 4B),
dont deux fondés sur les NUMÉROS de ligne (replace_lines, insert_lines) : un 4B ne recopiait
pas un bloc au caractère près, donc on l'adressait par numéros. Loom ne cible plus que des
MoE 24B+ : un 35B fait très bien l'édition par texte exact (paradigme Claude Code), tandis que
l'adressage par ligne provoque du THRASH (les numéros se périment après chaque edit → le modèle
réémet la même plage → garde-fou repeat_limit → arrêt « sans raison » observé).

## Décision
On retire insert_lines ET replace_lines. Le paradigme d'édition par numéro de ligne disparaît.
Outils d'écriture/édition = TROIS :
- write_file — créer / réécrire (petit fichier, ou grosse portion refaite) ;
- append_file — compléter un gros fichier par morceaux (unité logique par appel) ;
- edit_file — édition chirurgicale par texte exact (old_string copié du fichier, indentation
  comprise) : l'éditeur des blocs existants.

Le module loom/tools/indent.py (aides d'indentation pour l'édition par ligne) devient mort et
est supprimé.

## Ce qui tient toujours (ADR 0002)
- « outil DÉDIÉ > outil GÉNÉRAL » reste valide : on retire des outils dont le mode d'échec
  propre (recopie impossible d'un 4B) n'existe plus à l'échelle du 35B, pas par minimalisme.
- Plus d'ambiguïté edit_file vs replace_lines : edit_file est l'unique éditeur chirurgical.

## Conséquences
- 5 → 3 outils d'écriture ; le thrash de numéros de ligne disparaît.
- edit_file exige un old_string exact : à VALIDER en runtime. Si un modèle donné recopie mal
  (mis-indentation), ré-évaluer (réintroduire un éditeur par ligne, ou un modèle plus fort).
- Si un petit modèle (4B) était réintroduit, l'argumentaire 0002 redeviendrait pertinent.
