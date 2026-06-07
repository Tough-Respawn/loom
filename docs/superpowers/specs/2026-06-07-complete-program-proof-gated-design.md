# Rail `complete_program` — proof-gated, non déterministe — Design (palier 2)

> Suite du harnais de réflexion (tranche mince, livrée). Principe FONDATEUR réaffirmé : le
> LLM écrit le programme COMPLET ; le harnais ne le remplace JAMAIS par une fabrique
> déterministe. Il augmente sa fiabilité — contexte, outils, PREUVES observables, relectures
> fraîches, boucles de réparation — et **refuse une sortie non prouvée**. Le déterminisme est
> dans la VÉRIFICATION (le code lit la preuve), jamais dans la GÉNÉRATION (le modèle décide
> et écrit tout le contenu).

## Problème (constat des runs live)

Le rail actuel fonctionne mécaniquement mais : (a) `check_page` ne prouve que le chargement
statique (0 erreur console + compte d'éléments) — « 0 erreur ≠ jouable » ; (b) le modèle se
perd dans les logs bruts et relance/réécrit au lieu de corriger la cause ; (c) rien
n'empêche, au niveau du RUN ENTIER, de conclure « terminé » sans preuve de tous les
comportements demandés ; (d) une mauvaise correction peut empirer un fichier sain.

## Le rail, en une phrase

Un mode `complete_program` (toggle explicite) où le modèle **planifie → écrit → est mis à
l'épreuve par des preuves TYPÉES et INTERACTIVES → corrige cause par cause**, et où le
harnais **tient un registre de preuves** qui interdit de conclure tant que les comportements
déclarés ne sont pas observés.

## Frontière de déterminisme (non négociable)

| Le CODE (harnais) décide | Le MODÈLE décide |
|---|---|
| Quand une preuve a tourné et si elle PASSE (lecture de sa sortie) | Quoi écrire, quelle architecture, quel code |
| La séquence (spec → écrire → prouver → réparer) | Le contenu de la spec, des comportements, des tests |
| Quand « terminé » est autorisé (registre de preuves) | Comment corriger une cause signalée |
| Classer un échec, condenser les logs, snapshot/rollback | — |

Le modèle n'est JAMAIS juge unique de son succès. Mais il reste l'unique AUTEUR.

## Composants

### 1. `submit_spec` — contrat de succès AVANT d'écrire (remplace `submit_plan`)
Le modèle déclare un contrat structuré, refusé s'il est vague :
```python
@dataclass
class Behavior:
    desc: str                 # "cliquer une cellule la révèle"
    action: dict | None       # étape interactive exécutable (voir §3) — None si non-UI
    expect: str               # post-condition observable ("la cellule gagne la classe revealed")

@dataclass
class Spec:
    program_type: str         # html_game | web_page | cli | python_lib | api | script
    files: list[str]          # manifest des fichiers attendus
    launch: str               # comment lancer (URL de page / commande)
    behaviors: list[Behavior] # comportements à PROUVER (≥ 1, concrets)
    checks: list[str]         # preuves runnable additionnelles (pytest, commande golden…)
```
**Gate (dur)** : rejet si un `behavior` est vague (« le jeu marche »), si aucun comporte­ment
n'est testable, si `program_type` est inconnu. Réutilise et étend le gate anti-vague existant
(`validate_plan`). Exemple refusé : « le jeu marche ». Accepté : « check_page trouve 81
cellules, 0 erreur console ; cliquer une cellule change son état ; poser un drapeau marche ;
restart réinitialise ».

### 2. Registre de preuves (proof ledger) — anti-bluff au niveau du RUN
Mémoire FACTUELLE tenue par le harnais (pas une pipeline) : à chaque action réelle observée
dans la boucle, on enregistre `{type, cible, sortie réelle, ok}` — fichiers créés/modifiés,
commandes lancées + leur sortie, pages testées + erreurs console, comportements validés.
**Le modèle ne peut conclure « terminé » que si, pour CHAQUE `behavior` de la spec, le
registre porte une preuve verte.** Sinon le harnais rejette la conclusion et relance sur les
comportements non prouvés. C'est l'anti-confabulation, élevé du tour unique au run entier.

### 3. Preuve INTERACTIVE (le gros unlock) — `check_page` musclé
`check_page` ne fait aujourd'hui que charger + compter. On ajoute un canal d'ACTIONS jouées
dans la page (Playwright headless), piloté par les `behaviors` de la spec :
- `click` / `type` / `hover` / `rightclick` sur un sélecteur ;
- relire le DOM APRÈS interaction (classe/texte/attribut d'un élément) ;
- compter des états visuels (`.cell.revealed ×N`) ;
- screenshot optionnel ;
- enchaîner plusieurs actions utilisateur.

Chaque `behavior.action` se **compile en étapes Playwright** et `behavior.expect` en
**assertion lue par le code**. Pour un démineur : « cliquer `.cell:nth-child(1)` → l'élément
gagne la classe `revealed` » ; « clic droit `.cell` → un `.flag` apparaît » ; « cliquer
`#restart` → toutes les cellules repassent à l'état initial ». Le harnais joue, lit, tranche.
C'est ça qui transforme « 0 erreur » en « jouable PROUVÉ ».

> Nouvel outil interne probable `check_interactive(url, steps)` (ou extension de
> `check_page` avec un paramètre `actions`), réservé au runner de preuve du rail.

### 4. Runner de preuve TYPÉ
Selon `program_type`, le harnais lance la preuve adéquate et lit sa sortie déterministi­quement :
- `html_game` / `web_page` → check_page (chargement + compte) **+ check_interactive** (les behaviors) ;
- `python_lib` → `pytest` (ou assertions générées) ;
- `cli` → commandes golden (`python app.py …`), assert exit 0 + sous-chaîne attendue en stdout ;
- `api` → démarrage serveur BORNÉ (timeout-kill déjà en place) + requêtes HTTP locales, assert statut/JSON ;
- `script` → exécution + sortie attendue.
Aucun verdict émis par un modèle : le code lit le résultat structuré.

### 5. Classifieur d'échec + diagnostic condensé
La sortie brute d'une preuve échouée est transformée (déterministe : regex/heuristiques) en
signal court : `{fichier, erreur exacte, cause probable, prochaine action demandée}`. Catégories :
syntax error · import manquant · assertion de test · élément UI absent · timeout · mauvaise
commande · fichier attendu absent. Le harnais réinjecte un nudge ciblé : **« corrige
UNIQUEMENT cette cause, ne réécris pas tout »**. Le harnais condense ; il ne décide PAS la
solution. (Les petits modèles se noient dans 200 lignes de logs et réécrivent tout.)

### 6. Snapshots + rollback
Avant une grosse correction, snapshot des fichiers touchés. Si, après correction, la preuve
RÉGRESSE (nouvelle erreur, moins d'éléments, behavior qui passait casse), **rollback** à
l'état d'avant et nouvelle tentative sur base propre. Empêche le 4B de « creuser le trou ».

### 7. Relecture fraîche — chercheuse de MANQUES (pas juge-vert)
UNE fois (à la fin, après que les preuves par-tâche passent), un passage à contexte NEUF
reçoit `{objectif utilisateur, fichiers touchés, sorties de tests, diff/contenu final}` avec
la consigne : **« trouve ce qui MANQUE par rapport à l'objectif, ne corrige pas »**. Son
rapport va au modèle principal qui corrige. Recrée l'auto-jugement implicite des gros modèles
API. **Distinction clé** : la porte VERTE reste déterministe (les preuves) ; ce reviewer ne
déclare jamais « vert », il déclare « incomplet » (un behavior non couvert, un cas oublié).
Un seul passage → coût borné (≠ vérificateur par tâche, supprimé à juste titre en palier 1).

### 8. Toggle `complete_program`
Mode explicite (à côté du toggle « réflexion » actuel). Activé : plan + manifest obligatoires,
preuve typée + interactive par behavior, registre de preuves, relecture fraîche finale, boucle
de réparation ciblée bornée. Le modèle écrit TOUT ; le harnais **refuse une sortie non
prouvée**.

## Réconciliation avec la tranche mince déjà livrée

- **Garde** : décomposition + porte déterministe par tâche (`evaluate_executor_proof`) +
  fusion exécution/vérification (pas de model-verifier par tâche) + run_shell durci.
- **Étend** : `submit_plan` → `submit_spec` ; `evaluate_executor_proof` → runner typé +
  registre de preuves ; `check_page` → `check_interactive`.
- **Ajoute (net-neuf)** : classifieur d'échec, snapshots/rollback, relecture fraîche finale,
  toggle `complete_program`.
- **Ne réintroduit PAS** : agent vérificateur model-juge par tâche (coût ×2 + faux-positifs,
  prouvé en live). La rigueur vient du CODE qui lit des preuves observables.

## MVP / ordre de construction (par ROI)

1. **`check_interactive` (actions Playwright réelles) + `submit_spec` portant des behaviors
   structurés** — le cœur de la rigueur, et le cas qu'on teste (démineur jouable). Sans ça,
   tout le reste prouve du vide.
2. **Classifieur d'échec + diagnostic condensé + repair ciblé** — le gros gain de convergence
   (stoppe la réécriture-tout / la relance en boucle).
3. **Registre de preuves** comme porte finale « terminé » (anti-bluff au niveau run).
4. **Snapshots/rollback**, puis **relecture fraîche finale**.

## Hors périmètre (YAGNI)

- Pas d'agent model-juge décidant le « vert » (déterminisme de vérification uniquement).
- Pas de génération de squelette/template par le harnais (le modèle écrit tout).
- Pas de parallélisme entre behaviors (séquentiel + checkpoint).
- Pas de tests pytest persistés sur Loom lui-même (cf. décision « pas de tests ») — les
  pytest/golden/HTTP concernent les programmes PRODUITS, pas le code de Loom.

## Critères de réussite

1. Sur « démineur jouable », le rail refuse une spec vague et exige des behaviors testables.
2. La preuve interactive joue de VRAIS clics (révéler, drapeau, restart) et lit le DOM après
   — « jouable » est prouvé, pas supposé.
3. Un échec produit un diagnostic court ciblé ; le modèle corrige la cause sans tout réécrire ;
   une correction qui régresse est rollback.
4. « Terminé » n'est émis que si le registre porte une preuve verte pour CHAQUE behavior.
5. Le modèle écrit 100 % du code ; le harnais ne génère aucun contenu, il prouve.
