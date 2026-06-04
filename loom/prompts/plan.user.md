Tu produis le PLAN D'IMPLÉMENTATION d'une petite application web. Les fichiers seront ensuite générés SÉPARÉMENT (un appel isolé par fichier), donc le plan est un CONTRAT assez précis pour qu'un développeur déroule sans rien deviner.

Réponds EXACTEMENT dans ce format (rien d'autre) :
===DESIGN===
<le contrat détaillé ci-dessous ; tu PEUX inclure des snippets de code JS>
===FILES===
index.html | rôle de ce fichier
style.css | rôle
app.js | rôle
(un fichier par ligne, format 'chemin | rôle')

Le DESIGN doit contenir ces sections :
1. ÉTAT PARTAGÉ : les variables d'état EXACTES, le type/structure de chacune, et leurs valeurs initiales.
2. MODULES : pour chaque JS, sa responsabilité et la SIGNATURE EXACTE des fonctions/objets globaux exposés. RÈGLE D'INITIALISATION (cruciale pour la cohérence entre fichiers générés séparément) : chaque fichier JS s'AUTO-INITIALISE — il attache son PROPRE écouteur DOMContentLoaded et se garde en vérifiant la présence de SES éléments DOM (`if (!document.getElementById('...')) return;`). AUCUN fichier n'appelle l'initialisation d'un autre, et aucune page HTML n'appelle de fonction JS inline : chaque module est autonome et ne s'active que sur la page où ses éléments existent.
   IMPORTANT : SCRIPTS CLASSIQUES uniquement, PAS d'ES modules (interdit import/export, pas de type=module). Chaque JS peut exposer des globaux sur window si besoin ; chaque page HTML charge les scripts via <script src> classiques (dépendances d'abord), SANS bloc <script> inline qui appelle une fonction par son nom. Toute l'initialisation passe par l'auto-init DOMContentLoaded des fichiers JS.
3. RENDU (DOM) : les éléments visibles = ÉLÉMENTS DOM (PAS un canvas, pour rester vérifiable). id/classes EXACTS des conteneurs et éléments, TYPE de balise, le sélecteur (ex: querySelectorAll('.item') sous #liste), et QUI crée ces éléments (HTML en dur ou JS).
4. INTERACTIONS : événements EXACTS. Clics (quel élément -> quel effet) ET/OU clavier (keydown sur document, quelle touche -> quel effet). Si l'app est TEMPORELLE, décris la BOUCLE : setInterval(fn, N ms), démarrée au chargement.
5. RÈGLES + SNIPPETS : la logique clé de la demande, avec des snippets de code JS pour les parties délicates.

Dans ===FILES===, liste TOUS les fichiers (chaque page HTML, le CSS, le(s) JS). RÈGLE ABSOLUE : tout fichier référencé par une page (chaque `<link rel=stylesheet href>`, chaque `<script src>`, chaque image locale) DOIT figurer dans ===FILES===. Site multi-pages : prévois UNE SEULE feuille de style partagée, avec UN NOM UNIQUE (ex. style.css), et référence-la À L'IDENTIQUE (même chemin exact) depuis TOUTES les pages — jamais 'style.css' sur une page et 'styles.css' sur une autre.

Tâche : __TASK__
