from __future__ import annotations

from __future__ import annotations
import os
import re
import unicodedata
from collections.abc import Iterator

from loom.agent.compaction import _inject_notes
from loom.agent.debuglog import _debug, log_event
from loom.agent.toolsets import _BROWSER_CHECKS, _STATE_CHANGERS, _VERIFY_TOOLS


_VERIFY_STREAK_NOTE = 3  # nb de checks verts consécutifs avant d'annoter le résultat


def _verify_streak_update(name: str, ok: bool, streak: int) -> int:
    """Nouveau compteur de checks navigateur VERTS consécutifs : un outil qui change
    l'état (écriture, shell) remet à zéro (la preuve précédente est périmée), un check
    raté aussi (échec = information nouvelle) ; les lectures n'y touchent pas."""
    if name in _STATE_CHANGERS:
        return 0
    if name in _BROWSER_CHECKS:
        return streak + 1 if ok else 0
    return streak


# Note de RECENTRAGE après un force-fit : un historique tronqué mais encore répétitif
# induit l'IMITATION (observé en éval : le modèle a « continué » la série de vieux tours
# archivés au lieu d'exécuter la tâche). La note casse le motif et repointe la demande.
# Préfixe '[harnais' = reconnue par _force_fit (jamais prise pour la tâche courante).
# Reformulée le 2026-07-10 : l'ancienne version ordonnait « reprends la DERNIÈRE
# demande telle quelle et exécute-la » — injectée EN PLEIN TOUR (le force-fit
# préventif tourne avant chaque appel), elle pouvait faire repartir de zéro un
# modèle qui avançait bien, et insinuait une dérive là où la troncature n'est
# qu'une opération de routine. On garde UNIQUEMENT l'anti-imitation (le cœur
# validé en éval, cas context_squeeze) : informatif, jamais directif.
_REFOCUS_NOTE = (
    "[harnais : des tours anciens ci-dessus ont été TRONQUÉS pour tenir dans la "
    "fenêtre — opération de routine, rien d'anormal. Ce contenu tronqué est du "
    "contexte ARCHIVÉ : ne l'imite pas, ne le continue pas. Ta tâche en cours et "
    "la dernière demande utilisateur restent INCHANGÉES : poursuis ton travail "
    "normalement, sans repartir de zéro.]"
)


# --- Anti « parle sans agir » --------------------------------------------------------
# Échec central d'un petit modèle : il ÉCRIT l'intention (« je vais lire X ») ou AFFIRME
# le résultat (« j'ai créé le fichier ») SANS émettre l'appel d'outil. Comme un tour sans
# tool_call termine la boucle, la tâche s'arrête inachevée (ou sur une affirmation fausse).
# On détecte ce cas au stop et on RELANCE le modèle pour qu'il exécute réellement (nudge
# borné). Ce n'est pas un orchestrateur : on ne décide pas QUOI faire, on force juste le
# passage de la parole à l'acte. dispatch_agent reste l'autre garde-fou (exécution réelle
# par un sous-agent). On exclut les verbes de PAROLE (résumer/expliquer) qui ne sont pas
# des actions outillées, pour ne pas harceler un vrai message final. Marqueurs SANS accents
# ni apostrophe courbe : la comparaison normalise le texte (un modèle quantifié laisse
# parfois tomber les accents -> on veut quand même détecter).
_ACT_INTENT = (
    "je vais ",
    "laisse-moi ",
    "permets-moi ",
    "je commence par ",
    "je dois d'abord ",
    "il faut que je ",
    "je m'occupe ",
    "commencons par ",
    "je vais maintenant ",
    "je vais d'abord ",
)
_ACT_CLAIM = (
    "j'ai cree",
    "j'ai ecrit",
    "j'ai modifie",
    "j'ai ajoute",
    "j'ai lance",
    "j'ai execute",
    "j'ai teste",
    "j'ai corrige",
    "j'ai supprime",
    "le test passe",
)
_TALK_VERBS = ("resum", "expliqu", "montr", "repond", "decri", "te dire", "vous dire")


def _norm(text: str) -> str:
    """Minuscule, sans accents, apostrophe courbe -> droite : pour comparer aux marqueurs
    quel que soit l'encodage exact produit par le modèle."""
    text = text.lower().replace("’", "'")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _intends_to_act(text: str, executed: bool) -> bool:
    """Vrai si `text` ANNONCE une prochaine action outillée non appelée, ou AFFIRME avoir
    agi alors qu'AUCUNE exécution réelle n'a eu lieu ce tour-ci (confabulation)."""
    low = _norm(text.strip())
    if not low:
        return False
    for marker in _ACT_INTENT:
        pos = low.find(marker)
        if pos != -1:
            tail = low[pos : pos + 60]  # « je vais RESUMER » = parole, pas action outil
            if not any(v in tail for v in _TALK_VERBS):
                return True
    if not executed and any(m in low for m in _ACT_CLAIM):
        return True
    return False
_EXEC_CLAIM = (
    "a affiche",
    "a retourne",
    "resultat :",
    "sortie :",
    "exit=",
    "j'ai lance",
    "j'ai execute",
    "j'ai teste",
    "le test passe",
    "-> success",
    "preuve :",
    "preuve de sortie",
    "j'ai simule",  # aveu typique de confabulation
)
# Chemin ABSOLU de fichier (avec extension), Windows ou POSIX, cité dans le texte.
_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|/)[\w./\\-]+\.[A-Za-z0-9]{1,6}")
# Marqueurs d'ACCOMPLISSEMENT (1re personne passé / état de fait) : ils distinguent une
# vraie revendication (« j'ai créé X », « X a été créé », accusé d'outil « créé : X ») d'une
# simple mention illustrative (« tu pourrais créer X », « voici X, par exemple »). L'audit
# ne se déclenche QUE sur un accomplissement affirmé — sinon il fabriquait l'action à partir
# d'un exemple pédagogique (incident loom-amd 2026-07-23 : un chemin montré en exemple a été
# matérialisé par le nudge). En forme _norm (minuscule, sans accents).
_DONE_MARKERS = (
    "j'ai cree",
    "j'ai ecrit",
    "j'ai genere",
    "j'ai produit",
    "j'ai ajoute",
    "j'ai modifie",
    "j'ai mis a jour",
    "a ete cree",
    "a ete ecrit",
    "a ete genere",
    "a ete produit",
    "a ete ajoute",
    "est cree",
    "est ecrit",
    "est pret",
    "sont crees",
    "cree :",
    "ecrit :",
    "genere :",
)
# Tournures ILLUSTRATIVES : si l'une est proche du chemin, c'est un exemple, pas une
# revendication -> on n'audite pas (double filet, en plus du retrait des blocs de code).
_ILLUSTRATIVE = (
    "par exemple",
    "pourrais",
    "comme si",
    "un exemple",
    "e.g",
    "tu peux mettre",
)


def _strip_code_blocks(text: str) -> str:
    """Retire les blocs de code balisés (``` … ```) : un chemin qui n'apparaît QUE dans un
    exemple de code est une ILLUSTRATION, jamais une revendication d'artefact produit."""
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def _claims_execution(text: str) -> bool:
    """Vrai si `text` rapporte une sortie / un résultat d'exécution."""
    low = _norm(text)
    return any(m in low for m in _EXEC_CLAIM)


def _claims_missing_artifact(text: str, files_written: set) -> str | None:
    """Renvoie le 1er chemin ABSOLU revendiqué comme ACCOMPLI (« j'ai créé X ») qui
    n'existe PAS et n'a pas été écrit ce tour — artefact confabulé. Sinon None.

    Resserré (2026-07-23) pour ne PLUS mordre sur une mention illustrative : (1) les
    blocs de code sont retirés (un chemin montré en exemple n'est pas un artefact),
    (2) il faut une vraie revendication d'accomplissement à proximité (_DONE_MARKERS),
    pas un simple verbe de création (« tu pourrais créer X » ne déclenche plus),
    (3) une tournure illustrative proche (_ILLUSTRATIVE) désamorce. Chemins absolus
    uniquement (vérif fiable sans connaître le workspace)."""
    prose = _strip_code_blocks(text)
    low = _norm(prose)
    for match in _PATH_RE.finditer(prose):
        path = match.group(0).strip("`\"'")
        idx = low.find(_norm(path))
        if idx == -1:
            continue
        window = low[max(0, idx - 80) : idx + len(path) + 80]
        if not any(m in window for m in _DONE_MARKERS):
            continue
        if any(m in window for m in _ILLUSTRATIVE):
            continue
        if path in files_written:
            continue
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
        except OSError:
            pass
        return path
    return None


# Le HARNAIS Loom (garde-fous, relances automatiques) parle au modèle via des
# messages role:user — le standard OpenAI n'a pas de rôle « framework » et on n'y
# touche pas. Mais ce n'est PAS l'utilisateur : le modèle doit le savoir (sinon il
# répond « l'utilisateur a dit… » à un garde-fou — vécu 2026-07-23, incident loom-amd).
#
# On marque chaque nudge d'un tag MINIMAL `[LOOM]` (3-4 tokens par injection, pas un
# pavé re-consommé à chaque relance) ; le SENS du tag est expliqué UNE fois dans le
# system prompt (stable, en tête -> absorbé par le prefix-cache, coût réel ~0).
# Émis AUSSI en event ('harness', …) pour que l'UI le rende comme une 3e voix
# distincte (ni toi ni le modèle). La note en vol de l'utilisateur, elle, vient
# VRAIMENT de lui -> voix user, pas [LOOM] (cf. _inject_notes).
_LOOM_TAG = "[LOOM]"


def _loom_nudge(convo: list[dict], kind: str, text: str) -> tuple[str, object]:
    """Injecte une intervention du HARNAIS dans la conversation (role:user marqué
    [LOOM] pour le modèle) ET renvoie l'event ('harness', {kind, text}) à yielder
    (3e voix, visible en UI). `kind` = étiquette courte (claim_audit, boucle…)."""
    convo.append({"role": "user", "content": f"{_LOOM_TAG}\n{text}"})
    return ("harness", {"kind": kind, "text": text})


def _dispatch_no_tool_calls(
    collector: dict,
    text: str,
    convo: list[dict],
    strong: bool,
    max_loop_breaks: int,
    max_length_continues: int,
    max_empty_retries: int,
    max_act_nudges: int,
    st: dict,
    notes_provider=None,
) -> Iterator[tuple[str, object]]:
    """Fin de tour SANS appel d'outil : boucle dégénérée -> continuation 'length' ->
    réponse vide -> audit de claim / act-nudge -> stop naturel. Issue via
    st["action"] : "continue" (relancer le tour) ou "done" (l'event terminal
    ('done', …) a déjà été yieldé)."""
    # BOUCLE DE DÉGÉNÉRESCENCE (détectée au streaming) : le modèle a répété la
    # même phrase sans agir. À traiter AVANT la continuation 'length' : lui dire
    # « continue où tu t'es arrêté » ne ferait qu'alimenter le cycle. On coupe et
    # on relance avec un ordre FERME d'émettre un appel d'outil, borné.
    if collector.get("looped"):
        if st["loop_breaks"] >= max_loop_breaks:
            yield (
                "content",
                "\n[génération interrompue : le modèle tournait en boucle (même "
                "phrase répétée) sans agir. Reformule ou découpe la demande.]",
            )
            yield ("done", {"reason": "loop_degenerate"})
            st["action"] = "done"
            return
        st["loop_breaks"] += 1
        nudge = (
            "Tu répètes la même phrase en boucle sans rien faire. ARRÊTE de "
            "planifier en prose. Émets MAINTENANT un seul appel d'outil — "
            "manage_todos pour poser le plan, OU directement le premier write_file "
            "— sans aucun texte avant. Un seul outil, tout de suite."
        )
        _debug(
            "LOOP_BREAK",
            f"boucle détectée ({collector.get('looped')!r}), "
            f"relance {st['loop_breaks']}/{max_loop_breaks}",
        )
        yield _loom_nudge(convo, "boucle", nudge)
        st["action"] = "continue"
        return
    # CONTINUATION sur troncature : la réponse texte/raisonnement a été coupée
    # par la limite de tokens (finish_reason == "length") sans appel d'outil.
    # Plutôt que de rendre une réponse tronquée, on relance le modèle pour qu'il
    # POURSUIVE là où il s'est arrêté. Autant de fois que nécessaire (cap dur
    # max_length_continues, anti-runaway). Le texte continue d'être streamé à
    # l'UI tour après tour (le web app concatène). Cas des tool_calls tronqués
    # NON concerné (géré par 'arguments tronqués' / overflow).
    if (
        collector["finish_reason"] == "length"
        and st["length_continues"] < max_length_continues
    ):
        st["length_continues"] += 1
        if text:
            convo.append({"role": "assistant", "content": text})
            nudge = (
                "Ta réponse a été coupée par la limite de tokens. CONTINUE "
                "exactement là où tu t'es arrêté, sans répéter ce qui précède."
            )
        else:
            nudge = (
                "Ta réflexion a été coupée par la limite de tokens. Termine et "
                "DONNE ta réponse (ou émets l'appel d'outil) MAINTENANT, plus "
                "direct."
            )
        yield _loom_nudge(convo, "continuation", nudge)
        _debug(
            "CONTINUATION(length)",
            f"relance {st['length_continues']}/{max_length_continues}",
        )
        st["action"] = "continue"
        return
    # RÉPONSE VIDE (EOS immédiat : 0 texte, 0 tool call — vécu en éval,
    # cas context_squeeze) : le stop naturel serait un SILENCE total pour
    # l'utilisateur. On relance, borné. Filet de FONCTIONNEMENT (pas une
    # garde de comportement) -> actif aussi pour un modèle fort (strong).
    if not text.strip():
        if st["empty_retries"] < max_empty_retries:
            st["empty_retries"] += 1
            nudge = (
                "Ta réponse est arrivée VIDE (aucun texte, aucun appel "
                "d'outil). Réponds MAINTENANT : donne le résultat demandé, "
                "ou émets l'appel d'outil nécessaire."
            )
            log_event(
                "guard",
                level="WARN",
                kind="empty_response",
                retry=st["empty_retries"],
            )
            yield _loom_nudge(convo, "réponse vide", nudge)
            st["action"] = "continue"
            return
        yield (
            "content",
            "\n[génération interrompue : le modèle a rendu une réponse "
            "vide malgré les relances.]",
        )
        yield ("done", {"reason": "empty_response"})
        st["action"] = "done"
        return
    # Audit de claim au stop : le modèle prétend-il un résultat qu'il n'a pas
    # produit ? (A) artefact fichier inventé, (B) résultat d'exécution sans
    # run_shell/dispatch, ou intention/affirmation sans exécution réelle. On le
    # relance pour qu'il FASSE vraiment (borné). Garde de vérité, pas orchestrateur.
    # COUPÉ pour un modèle FORT (distant) : ces relances de comportement, utiles à
    # un petit modèle qui confabule, ne font que sur-piloter un modèle qui se vérifie
    # déjà seul (cf. GLM qui doutait de sa propre preuve correcte).
    missing = _claims_missing_artifact(text, st["files_written"])
    exec_confab = not st["executed"] and _claims_execution(text)
    if (
        not strong
        and st["act_nudges"] < max_act_nudges
        and (missing or exec_confab or _intends_to_act(text, st["executed"]))
    ):
        st["act_nudges"] += 1
        convo.append({"role": "assistant", "content": text or "..."})
        if missing:
            nudge = (
                f"Tu affirmes avoir produit « {missing} » mais ce fichier "
                "n'existe pas (ou est vide). Crée-le RÉELLEMENT avec un outil "
                "puis vérifie-le — n'invente pas d'artefact ni de preuve."
            )
            label = "CLAIM_AUDIT(artefact)"
        elif exec_confab:
            nudge = (
                "Tu rapportes un résultat d'exécution (sortie, « ça marche », "
                "preuve) mais tu n'as lancé AUCUNE commande ce tour (ni run_shell "
                "ni dispatch_agent). Lance-la RÉELLEMENT et rapporte la VRAIE "
                "sortie — n'invente pas de résultat."
            )
            label = "CLAIM_AUDIT(exécution)"
        else:
            nudge = (
                "Tu as annoncé/affirmé une action mais tu n'as rien exécuté : "
                "rien n'a été réellement fait. Émets MAINTENANT l'appel d'outil "
                "directement (aucune phrase avant). Si la tâche est vraiment "
                "terminée ET vérifiée, dis seulement le résultat constaté."
            )
            label = "ACT_NUDGE"
        _debug(label, nudge)
        log_event("guard", kind=label)
        yield _loom_nudge(convo, label, nudge)
        st["action"] = "continue"
        return
    # Note en vol arrivée PENDANT la génération finale (donc après le dernier
    # drain d'avant-appel) : sans ce re-drain, le stop naturel la laisserait en
    # file jusqu'au prochain message manuel — l'utilisateur devait relancer par
    # un « ? » pour qu'elle parte (bug 2026-07-23). On la draine ICI et on
    # reboucle pour y répondre tout de suite, au lieu de clore le tour.
    injected = yield from _inject_notes(notes_provider, convo)
    if injected:
        _debug("NOTE_AU_STOP", f"{injected} note(s) au stop naturel -> relance")
        st["action"] = "continue"
        return
    yield ("done", {"reason": "natural"})
    st["action"] = "done"  # réponse finale déjà streamée (stop naturel du modèle)


def _check_no_progress(
    tool_calls: list[dict], strong: bool, repeat_limit: int, st: dict
) -> Iterator[tuple[str, object]]:
    """Détecteur de non-progrès (mêmes appels que le tour précédent). Issue via
    st["action"] : "proceed" (on exécute les outils) ou "done" (repeat_stop,
    l'event terminal a déjà été yieldé). Met à jour st["repeat_streak"] /
    st["prev_sig_set"]."""
    st["action"] = "proceed"
    # Non-progrès : même jeu d'appels (outils+args) que le tour précédent ? On EXCLUT
    # les outils d'exécution/vérification (_VERIFY_TOOLS) : re-lancer la même preuve est
    # légitime. Un tour PUREMENT de vérif -> signature vide -> compté comme progrès (on
    # ne coupe pas, on remet le compteur à zéro). Backstop ultime contre le vrai runaway :
    # max_iters. Les boucles dégénérées (re-edit/re-write/re-read identiques) restent prises.
    sig_set = frozenset(
        f"{tc['name']}\x00{tc['arguments']}"
        for tc in tool_calls
        if tc["name"] not in _VERIFY_TOOLS
    )
    if not sig_set:
        # Tour purement exécution/vérif : progrès légitime, on ne coupe pas et on
        # laisse passer vers l'exécution des outils (surtout PAS de continue ici).
        st["repeat_streak"] = 0
        st["prev_sig_set"] = None
        return
    st["repeat_streak"] = (
        st["repeat_streak"] + 1 if sig_set == st["prev_sig_set"] else 0
    )
    st["prev_sig_set"] = sig_set
    # COUPÉ pour un modèle FORT (distant) : le seul backstop reste max_iters. Sur un
    # petit modèle, la répétition = dégénérescence ; sur un fort, c'est presque
    # toujours du légitime (le juger « bloqué » l'interrompt à tort).
    if not strong and st["repeat_streak"] >= repeat_limit - 1:
        log_event("guard", level="WARN", kind="repeat_stop")
        yield (
            "content",
            "\n(arrêt : le modèle réémet les mêmes appels sans progresser).",
        )
        yield ("done", {"reason": "repeat_stop"})
        st["action"] = "done"
