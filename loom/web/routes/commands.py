# loom/web/routes/commands.py — helper de chat sorti de chat.py (comportement constant).
from __future__ import annotations

from pathlib import Path

from flask import Response

from loom.web.app import (
    _GOAL_CLEAR_WORDS,
    _init_message,
    _sse,
)


def _handle_goal_command(S, message, conv, save, chat_lock):
    """Traite la commande /goal : pose/statut/efface l'objectif de session.

    Retourne (message, response) : response non None = ack immédiat (return direct
    dans chat()) ; response None = continuer le flux normal (message éventuellement
    réécrit en consigne de démarrage)."""

    if message == "/goal" or message.startswith("/goal "):
        arg = message[len("/goal") :].strip()
        if arg and arg.lower() not in _GOAL_CLEAR_WORDS:
            # Pose l'objectif et AMORCE le travail : on remplace le message par une consigne
            # de démarrage et on laisse le flux normal tourner, objectif désormais actif.
            conv.set_goal(arg)
            save()
            message = (
                f"Objectif à atteindre : {arg}\n"
                "Commence MAINTENANT à agir pour l'atteindre, et PROUVE-le (exécute, montre "
                "la sortie réelle). Ne t'arrête pas tant qu'il n'est pas démontré atteint."
            )
            # (pas de return : on tombe dans la génération normale ci-dessous)
        else:
            if not arg:
                ack = (
                    f"Objectif courant : « {conv.goal} » (actif jusqu'à preuve d'atteinte, "
                    "/goal clear pour l'effacer)."
                    if conv.goal
                    else "Aucun objectif actif. Pose-en un : /goal <condition vérifiable>."
                )
            else:
                conv.set_goal("")
                save()
                ack = "Objectif effacé - retour au mode normal (arrêt au stop naturel)."
            chat_lock.release()

            def _goal_ack():
                yield _sse("text", text=ack)
                yield _sse("done")

            return message, Response(_goal_ack(), mimetype="text/event-stream")
    return message, None


def _handle_init_command(S, message, sess):
    """Traite /init : adopte un dossier cible si fourni, et réécrit le message en consigne
    de génération de fiche projet. Retourne le message (éventuellement réécrit).

    `sess` = la session CIBLE capturée par /chat (jamais _session(S)/focus : sinon /init
    adopterait le dossier d'un autre onglet activé pendant la génération)."""
    if message == "/init" or message.startswith("/init "):
        arg = message[len("/init") :].strip()
        _sess = sess
        target_dir = _sess.workspace
        if arg:
            cand = Path(arg).expanduser()
            if cand.is_dir():
                target_dir = str(cand.resolve())
                if target_dir != _sess.workspace:
                    _sess.workspace = target_dir
                    S.session_store.save(_sess)
        target_display = str(Path(target_dir)).replace("\\", "/")
        message = _init_message(target_display)
        # (pas de return : le flux normal ci-dessous exécute la consigne)
    return message
