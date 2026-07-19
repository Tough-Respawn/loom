# Boutons de confirmation des wizards (/add-model, /remove-model, /rebench)

Date : 2026-07-19 · Statut : validé (design approuvé dans le chat)

## Problème

Les confirmations des wizards se répondent en tapant « oui »/« non ». Confort
demandé : des BOUTONS cliquables, comme les autorisations d'outils (PermAsk).

## Principe

Les boutons sont de purs raccourcis de frappe : le clic envoie le libellé du
bouton comme message normal (fil, journal, wizard : identiques à la frappe).
Le déterminisme du wizard ne change pas ; taper reste toujours possible.

## Mécanique

1. `WizardResult` gagne `choices: list[str] | None = None`. Renseigné sur :
   - menu de type `/add-model` sans argument : ["local", "distant", "image", "vidéo"] ;
   - confirmation `/remove-model` (d_confirm) : ["oui", "annuler"] ;
   - confirmation `/rebench` (b_confirm) : ["oui", "annuler"].
2. routes : le flux SSE du wizard émet `{"type": "choices", "options": [...]}`
   après le texte quand `res.choices` est présent. Le verdict différé de
   /rebench (worker) porte aussi ses boutons (`job.choices` = ["oui",
   "annuler"] quand l'état b_apply est posé), émis dans la branche rb_job.
3. Front (app.js) : événement `choices` → item timeline `{kind: "choices"}` ;
   composant `WizChoices` (style des boutons PermAsk) ; clic → item marqué
   `decided` + libellé envoyé via `submitChat()`. `submitChat` retire
   (decided) tout bloc de choix en attente — une réponse TAPÉE fait donc
   disparaître les boutons aussi.
4. Limite assumée : les boutons ne sont pas persistés — après un rechargement
   de page, le journal rejoue le texte seul ; taper « oui » marche toujours
   (l'état wizard est intact côté serveur).

## Tests

- Unit wizard : `choices` présent avec les bons libellés sur les 3 étapes,
  None ailleurs (ex. i_id).
- Routes : l'événement SSE `choices` apparaît dans le flux de d_confirm et de
  b_confirm ; le flux du job /rebench (stub) l'émet avec le verdict.
- E2E réel : /rebench <id> → boutons visibles → clic « annuler » → wizard
  annulé proprement ; /add-model → clic « distant » → étape id atteinte →
  /cancel. (Pas de clic « oui » destructif en E2E.)
