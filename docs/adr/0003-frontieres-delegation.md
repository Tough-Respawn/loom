# ADR 0003 — Frontières de délégation (dispatch_agent)

- Statut : Accepté
- Date : 2026-06-06

## Contexte
Loom expose `dispatch_agent` : un sous-agent isolé qui exécute une tâche autonome et
ne renvoie qu'une synthèse. Mais « quand déléguer ? » restait un « ça dépend » flou.
Sur un petit modèle (4B) le flou coûte cher dans les deux sens : sur-délégation (un
simple read confié à un sous-agent = plus lent, et le sous-agent est aussi un 4B donc
surface d'échec en plus) ou sous-délégation (une exploration de 20 outils saturant le
contexte principal qu'un sous-agent aurait absorbée).

La doctrine s'inspire du prompt coordinateur de Claude Code, qui ne laisse jamais la
délégation à l'intuition : il définit des frontières par (a) ce que chaque rôle peut
VOIR, (b) ce qu'il POSSÈDE, (c) un critère mesurable. Adaptée à Loom, qui est
mono-flux : un seul sous-agent à la fois, synchrone, isolé, pas de reprise ni de
parallélisme.

## Décision
Quatre frontières explicites, gravées dans loom/prompts/chat.system.md
(section « FRONTIÈRES DE DÉLÉGATION ») :

1. **Granularité** — déléguer un SOUS-CHANTIER autonome dont seul le résultat importe.
   Un read/search/edit ponctuel se fait soi-même (plus rapide, et on a besoin du
   résultat). Critère opérationnel : « ai-je besoin du détail des outils dans MON
   contexte ? » Non → déléguer ; oui → faire soi-même.
2. **Information** — le sous-agent ne voit PAS la conversation. Son prompt doit être
   autonome (objectif + chemins/contraintes + critère de « fini »). Un ordre
   référentiel (« corrige le bug dont on a parlé ») échoue par construction.
3. **Propriété** — la COMPRÉHENSION ne se délègue jamais. Pas de « d'après tes
   trouvailles, fais X » : l'agent principal lit la synthèse, décide, répond lui-même.
   (CC : « You never hand off understanding to another worker. »)
4. **Regard neuf (vérifier)** — pour s'assurer qu'un travail marche, on confie la
   VÉRIFICATION à un sous-agent : il n'a pas fait le travail, il lance la preuve
   (tests, run_shell) sans préjugé. Frontière faire/juger : un contexte qui a produit
   le code n'est pas le bon pour le valider. Côté sous-agent, le prompt
   subagent.system.md impose alors de PROUVER (lancer les commandes, éprouver les cas
   limites, creuser les échecs), pas de tamponner.

## Conséquences
- + La décision de déléguer devient un critère, pas une intuition ; moins de
  sur/sous-délégation sur le 4B.
- + La vérification à regard neuf répond au besoin central « le LLM parle/affirme mais
  n'exécute pas réellement » : on déporte la preuve vers un contexte qui n'a pas
  d'intérêt à se valider lui-même. Complète l'anti « parle sans agir » de la boucle.
- − Une seule chose à la fois (mono-flux) : pas de fan-out parallèle ni de reprise de
  worker comme chez CC ; on garde volontairement le modèle simple.
- Limite assumée : dispatch_agent reste synchrone et isolé. Si un besoin de continuité
  (reprendre un sous-agent avec son contexte) apparaît, ce sera une décision séparée.
