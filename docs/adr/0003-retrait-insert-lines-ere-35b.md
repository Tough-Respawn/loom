# ADR 0003 — Retrait d'insert_lines (révision d'ADR 0002 pour l'ère 35B MoE)

- Statut : Accepté
- Date : 2026-06-18
- Révise : ADR 0002 (Variété d'outils dédiés) — partiellement

## Contexte
ADR 0002 (2026-06-06) justifiait CINQ outils d'édition pour un petit modèle (Gemma 4B).
insert_lines couvrait l'insertion au milieu sans recopie. Loom ne cible plus que des MoE 24B+ :
sur un 35B la prémisse « incapable de recopier / sélection fragile » est faible, alors que la
grappe de cinq outils d'écriture reste une charge de sélection réelle.

## Décision
On retire insert_lines. Outils d'écriture = QUATRE : write_file, append_file, replace_lines,
edit_file. L'insertion au milieu passe par replace_lines (remplacer la ligne d'ancrage par
elle-même + le nouveau bloc) ou edit_file.

## Ce qui tient toujours (ADR 0002)
- « outil DÉDIÉ > outil GÉNÉRAL » reste valide ; on retire un outil dont le mode d'échec propre
  n'existe plus à l'échelle du 35B, pas par minimalisme.
- Seule ambiguïté restante : edit_file vs replace_lines, tranchée par le prompt.

## Conséquences
- 5 → 4 outils d'écriture. Si un 4B était réintroduit, ré-évaluer.
