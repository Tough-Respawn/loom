from __future__ import annotations

from __future__ import annotations



# Écritures à GROS contenu intégral : sérialisées (1/tour) pour qu'un batch de N ne
# sature pas max_tokens et ne tronque pas les derniers (P1.1). On NE sérialise QUE
# celles-ci : les éditions par bloc (edit_file) écrivent peu
# -> pas de risque d'overflow, et les laisser passer ensemble réduit le nombre de tours
# d'un refactor multi-fichiers (cf. plafond max_iters).
_SERIAL_WRITE = frozenset({"write_file", "append_file"})

# Outils d'EXÉCUTION / VÉRIFICATION : relancer LE MÊME appel N fois est légitime (« relance
# jusqu'à 3 runs verts », re-tester après un fix, confirmer une stabilité). Le détecteur de
# non-progrès les EXCLUT donc de sa signature : sinon il coupe un modèle qui fait exactement
# ce qu'on lui demande (observé sur le test LRU). Les vraies boucles à attraper sont les
# re-edit_file / re-write_file / re-read_file à l'identique — elles, restent comptées.
_VERIFY_TOOLS = frozenset({"run_shell", "check_page", "serve_and_check"})

# Sur-vérification compulsive (observée en éval : 14 check_interactive verts d'affilée
# sur une page qui marchait -> 20 tours, 230k tokens, arrêt max_iters). Les checks sont
# volontairement EXCLUS du détecteur de non-progrès (re-prouver est légitime, cf.
# _VERIFY_TOOLS) : le remède n'est donc PAS une coupe, c'est un SIGNAL dans le résultat.
_BROWSER_CHECKS = frozenset({"check_page", "serve_and_check"})
_STATE_CHANGERS = frozenset(
    {"write_file", "append_file", "edit_file", "run_shell", "format_code"}
)


# --- Audit de claim déterministe (anti-confabulation, couches A et B) -----------------
# Le modèle décide TOUT ; on l'empêche seulement de PRÉTENDRE un résultat qu'il n'a pas
# produit. Garde de vérité, pas orchestrateur. Deux vérifs déterministes au stop :
#   A. il revendique un FICHIER (créé/contient) qui n'existe pas -> artefact inventé ;
#   B. il rapporte un RÉSULTAT D'EXÉCUTION sans avoir lancé run_shell ni dispatch_agent.
_WRITE_TOOLS = frozenset({"write_file", "append_file", "edit_file"})
# Outils dont un ECHEC = signal de BUG (execution / verification), par opposition aux erreurs
# d'usage d'outil (ligne hors limite, etc.). Une cascade ici impose la methode debug.
_BUG_SIGNAL_TOOLS = frozenset({"run_shell", "check_page", "format_code"})
# Outils PARALLEL-SAFE : lecture seule / indépendants, sans effet de bord, sans confirmation,
# sans ordre entre eux. Pour un modèle DISTANT, un tour n'appelant QUE ceux-ci s'exécute en
# CONCURRENCE (règle Loom : local = inline/1 slot ; distant = on exploite le parallélisme).
# Exclus : écritures, run_shell, todos, notes-écriture, vérifs (check_*), use_skill, install,
# et read_image (accusé + message user multimodal différé -> ordre sensible, on le sérialise).
_PARALLEL_SAFE = frozenset(
    {
        "dispatch_agent",
        "find_files",
        "search_text",
        "list_dir",
        "read_file",
        "web_search",
        "fetch_url",
        "recall",
        "read_note",
        "list_plugins",
    }
)
_DEBUG_FORCE = (
    "STOP — plusieurs erreurs s'enchainent et corriger au coup par coup ne regle pas la "
    "cause. Methode debug OBLIGATOIRE maintenant, ne patche plus au hasard :\n"
    "1. REPRODUIRE : relance la commande/page qui echoue, lis l'erreur EN ENTIER (fichier, ligne, code).\n"
    "2. LOCALISER avec les outils : read_file l'etat reel, search_text la definition, run_shell/check_page "
    "pour VOIR — remonte jusqu'a la SOURCE de la mauvaise valeur, pas la ou elle explose.\n"
    "3. CAUSE RACINE unique : formule UNE hypothese verifiable (X echoue PARCE QUE...).\n"
    "4. UN seul changement minimal a la cause, puis RELANCE la repro et CONSTATE la preuve.\n"
    "Un fix qui ne regle pas -> retour a LOCALISER, jamais un autre patch au hasard. Si chaque "
    "fix en revele un autre, la base est pourrie : reecris proprement le fichier en cause."
)
