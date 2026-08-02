from __future__ import annotations

from __future__ import annotations



# Sérialiser les écritures intégrales volumineuses pour ne pas tronquer un batch ;
# les petites éditions par bloc peuvent rester groupées.
_SERIAL_WRITE = frozenset({"write_file", "append_file"})

# Répéter une exécution ou une vérification peut prouver la stabilité ; ces appels ne
# participent donc pas à la signature de non-progrès.
_VERIFY_TOOLS = frozenset({"run_shell", "check_page", "serve_and_check"})

# Comme les checks sont exclus du non-progrès, leur résultat doit signaler les
# vérifications vertes compulsives sans les bloquer.
_BROWSER_CHECKS = frozenset({"check_page", "serve_and_check"})
_STATE_CHANGERS = frozenset(
    {"write_file", "append_file", "edit_file", "run_shell", "format_code"}
)


# L'audit empêche seulement de revendiquer un fichier absent ou une exécution jamais lancée.
_WRITE_TOOLS = frozenset({"write_file", "append_file", "edit_file"})
# Seuls les échecs d'exécution/vérification déclenchent la méthode de debug.
_BUG_SIGNAL_TOOLS = frozenset({"run_shell", "check_page", "format_code"})
# Ces outils distants sont indépendants, sans effet de bord ni ordre requis. `read_image`
# reste séquentiel car il injecte ensuite un message multimodal.
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
