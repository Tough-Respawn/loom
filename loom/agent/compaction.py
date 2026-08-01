from __future__ import annotations

from __future__ import annotations
from collections.abc import Iterator

from loom.agent.debuglog import _debug


# --- Microcompact INTERNE à la boucle d'outils ---------------------------------------
# Sur une chaîne longue (refactor multi-fichiers, exploration), `convo` accumule TOUS les
# messages role:tool et finit par approcher la fenêtre du modèle -> overflow. CC règle ça
# par un "microcompact" SANS LLM : on vide le CONTENU des plus vieux résultats d'outils
# (gros stdout, gros read périmés) en gardant les N derniers + toute la STRUCTURE (chaque
# tool_call garde son tool_result). Non destructif pour le raisonnement (user/assistant
# intacts) et bien moins risqué qu'un résumé généré par un 4B.
# Le stub NE dit PAS « relis le fichier » : ça poussait le modèle à re-lire en boucle ce
# qui venait d'être purgé (thrash observé en session). Il l'oriente vers write_note —
# consigner l'essentiel AVANT que le résultat ne soit effacé, puis relire sa note (durable)
# au lieu du fichier entier.
_CLEARED_TOOL = (
    "[résultat d'outil ancien allégé pour tenir dans le contexte — son contenu n'est plus "
    "ici. NE refais PAS le travail déjà fait (ne re-liste pas, ne re-lis pas en boucle) : "
    "reprends depuis tes NOTES (read_note) et ton PLAN (manage_todos). Et désormais, dès "
    "qu'un résultat te servira à une étape ULTÉRIEURE, consigne-le avec write_note PENDANT "
    "que tu l'as encore.]"
)


def _msg_chars(content) -> int:
    """Taille approx. d'un contenu de message (str ou liste de parts multimodales)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
    return 0


def _microcompact_tools(
    convo: list[dict], keep_recent_tools: int, min_clear_chars: int = 400
) -> int:
    """Vide le CONTENU des plus vieux messages role:tool (garde les `keep_recent_tools`
    derniers intacts), en place. Renvoie le nb de messages allégés.

    SÉLECTIF : un petit résultat (accusé « modifié : x.py », code retour, message
    d'erreur court) est une PREUVE dense — le vider ne libère presque rien (le
    placeholder fait ~370 car.) et détruit de l'information que le modèle re-paierait
    en re-lecture. On ne vide que les GROS résultats (dumps de fichiers, stdouts
    longs) : le volumineux part, les preuves techniques restent."""
    idx = [i for i, m in enumerate(convo) if m.get("role") == "tool"]
    older = idx[:-keep_recent_tools] if keep_recent_tools else idx
    n = 0
    for i in older:
        content = convo[i].get("content")
        if content == _CLEARED_TOOL:
            continue
        if isinstance(content, str) and len(content) <= min_clear_chars:
            continue
        convo[i] = {**convo[i], "content": _CLEARED_TOOL}
        n += 1
    return n


# --- Compaction par RÉSUMÉ (dernier étage, avec LLM) ---------------------------------
# Le microcompact ne touche QUE les résultats d'outils. Quand ce sont les TOURS du modèle
# (assistant/reasoning, contenu écrit inline) qui saturent, vider les tool results ne
# suffit plus -> autrefois on abandonnait. Ici on RÉSUME les vieux tours en un bloc dense
# et on poursuit. Le résumé est en ANGLAIS TÉLÉGRAPHIQUE : le plus dense en tokens à
# fidélité égale (le français coûte ~15-20 % de tokens en plus), et il colle aux
# identifiants de code déjà anglais. On préserve les littéraux (chemins, noms, valeurs).
_SUMMARY_MARKER = "[SESSION SUMMARY — older turns compacted to fit the context window]"

_SUMMARY_SYSTEM = (
    "You compact a coding agent's own conversation so it fits the model's context "
    "window. Output a DENSE summary in terse English bullet points — no prose, no "
    "preamble, never restate these instructions. Preserve VERBATIM every file path, "
    "identifier, function/variable name, shell command, URL and numeric value. Capture, "
    "in order: GOAL, what was DONE, what was LEARNED/DECIDED — including approaches "
    "REJECTED and why (so they are not retried) — ERRORS hit with their exact messages, "
    "current STATE of the code, and what remains TODO. Stay faithful; when unsure, keep "
    "the literal token."
)


def _flatten_msg(content) -> str:
    """Texte brut d'un contenu de message (str ou liste de parts multimodales)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for p in content:
            if isinstance(p, dict):
                out.append(str(p.get("text") or p.get("content") or ""))
            else:
                out.append(str(p))
        return "\n".join(x for x in out if x)
    return "" if content is None else str(content)


def _flatten_for_summary(old: list[dict], budget_chars: int) -> str:
    """Aplatit les vieux tours en UN texte borné à envoyer au résumeur. Le convo déborde
    déjà la fenêtre -> on ne peut pas tout renvoyer : on garde le 1er message (le BUT) puis
    on remplit depuis la FIN (le plus récent = le plus utile pour l'état courant) jusqu'au
    budget ; le milieu ancien saute. '' si rien à aplatir."""
    if not old:
        return ""
    head = f"[{old[0].get('role', '?')}] {_flatten_msg(old[0].get('content'))}"
    tail_parts: list[str] = []
    used = len(head)
    for m in reversed(old[1:]):
        seg = f"[{m.get('role', '?')}] {_flatten_msg(m.get('content'))}"
        if used + len(seg) + 40 > budget_chars:
            tail_parts.append("[...older turns elided...]")
            break
        tail_parts.append(seg)
        used += len(seg) + 2
    return head + "\n\n" + "\n\n".join(reversed(tail_parts))


def _drop_orphan_tools(convo: list[dict]) -> None:
    """Retire tout message role:tool ORPHELIN — dont le plus proche message non-tool qui
    précède n'est pas un assistant porteur de tool_calls. Un tool orphelin (son appel a
    été droppé par le force-fit) fait échouer le rendu du chat template (400 llama.cpp) ;
    même règle que summarize_old_turns (« ne pas orpheliner un résultat d'outil »)."""
    i = 0
    while i < len(convo):
        if convo[i].get("role") == "tool":
            j = i - 1
            while j >= 0 and convo[j].get("role") == "tool":
                j -= 1
            anchored = (
                j >= 0
                and convo[j].get("role") == "assistant"
                and convo[j].get("tool_calls")
            )
            if not anchored:
                convo.pop(i)
                continue
        i += 1


def _force_fit(convo: list[dict], system_prompt: str, budget_chars: int) -> bool:
    """Réduction DÉTERMINISTE de dernier recours (AUCUN LLM) : clippe les contenus les plus
    gros — et, à défaut, drope les messages les plus anciens (garde toujours les 2 derniers)
    — jusqu'à passer sous `budget_chars`. GARANTIT un fit tant que `system_prompt` seul tient
    dans le budget. Sert à ne JAMAIS s'arrêter pour saturation : on tronque plutôt qu'on
    abandonne. Mute `convo` en place ; renvoie True si on tient le budget après réduction."""

    def _total() -> int:
        return len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)

    _CLIP_FLOOR = 200

    def _longest(skip: int) -> tuple[int, int]:
        """(index, taille) du message au contenu le plus long, hors `skip`."""
        idx, longest = -1, 0
        for i, m in enumerate(convo):
            if i == skip:
                continue
            c = m.get("content")
            n = len(c) if isinstance(c, str) else _msg_chars(c)
            if n > longest:
                longest, idx = n, i
        return idx, longest

    guard = 0
    while _total() > budget_chars and guard < 5000:
        guard += 1
        # SÉLECTIF : la TÂCHE COURANTE (dernier message user) est exemptée du clip tant
        # qu'il reste autre chose à réduire — c'est elle que le modèle doit exécuter
        # après la coupe (une grosse spec collée par l'utilisateur EST la tâche).
        task_idx = max(
            (
                i
                for i, m in enumerate(convo)
                if m.get("role") == "user"
                and not str(m.get("content", "")).startswith("[harnais")
            ),
            default=-1,
        )
        # Un message est « épuisé » sous _CLIP_FLOOR : le clip tête+queue produit
        # ~120 car. + marqueur (~50) — re-clipper sous ce plancher ne décroît PLUS
        # (boucle sans progrès, vue au self-test). > _CLIP_FLOOR garantit la
        # décroissance stricte : len/2 + marqueur < len dès que len > 2×marqueur.
        idx, longest = _longest(skip=task_idx)
        if longest <= _CLIP_FLOOR:  # plus rien d'autre : la tâche en dernier recours
            idx, longest = _longest(skip=-1)
        if idx >= 0 and longest > _CLIP_FLOOR:
            c = convo[idx].get("content")
            if isinstance(c, str):
                # Clip TÊTE + QUEUE (pas tête seule) : la fin d'un long contenu porte
                # souvent la conclusion/l'erreur — la preuve — plus que son milieu.
                keep = max(120, len(c) // 2)
                head = keep * 2 // 3
                tail = keep - head
                convo[idx] = {
                    **convo[idx],
                    "content": c[:head]
                    + " …[milieu tronqué pour tenir dans le contexte]… "
                    + c[len(c) - tail :],
                }
            elif isinstance(c, list):
                parts = []
                for p in c:
                    t = p.get("text") if isinstance(p, dict) else None
                    if isinstance(t, str) and len(t) > 120:
                        parts.append(
                            {**p, "text": t[: max(120, len(t) // 2)] + " …[tronqué]"}
                        )
                    else:
                        parts.append(p)
                convo[idx] = {**convo[idx], "content": parts}
            else:
                convo[idx] = {**convo[idx], "content": "…[tronqué]"}
        elif len(convo) > 2:
            # Plus rien de clippable mais encore trop : on drope le plus ancien, SAUF la
            # tâche courante (vécu en éval : le pop l'emportait -> conversation réduite à
            # [assistant, tool] SANS message user -> 400 « Unable to generate parser for
            # this template » côté llama.cpp, et un modèle sans but même sinon).
            victim = 1 if task_idx == 0 else 0
            convo.pop(victim)
            _drop_orphan_tools(convo)
        else:
            break
    return _total() <= budget_chars


def _ctx_estimate(system_prompt: str, convo: list[dict]) -> int:
    # Estimation ~3 car./token du contexte VIVANT (prompt + convo courant). Sert à
    # rafraîchir la jauge IMMÉDIATEMENT après une compaction, sans attendre l'usage
    # réel du prochain appel (sinon la jauge reste au pic pendant tout l'appel suivant).
    return (len(system_prompt) + sum(_msg_chars(m.get("content")) for m in convo)) // 3


def _inject_notes(notes_provider, convo: list[dict]) -> Iterator[tuple[str, object]]:
    """Draine les notes en vol : chacune est injectée dans `convo` (role user,
    préfixe explicite) et ré-émise en event ('note', texte injecté)."""
    # Notes en vol : les remarques utilisateur arrivées pendant le tour sont
    # injectées MAINTENANT (juste avant l'appel modèle = le point d'arrêt),
    # sans interrompre quoi que ce soit. L'appelant reçoit l'event 'note'
    # pour persister/afficher exactement ce qui a été injecté.
    if notes_provider is None:
        return 0
    count = 0
    try:
        for _raw_note in notes_provider() or []:
            # Les transferts inter-sessions voyagent dans la même file que les notes,
            # mais gardent leurs métadonnées pour que la timeline cible affiche la
            # provenance et que la réponse puisse la chaîner au prochain transfert.
            # Le modèle, lui, ne reçoit toujours qu'un texte role:user.
            _meta = _raw_note if isinstance(_raw_note, dict) else None
            _note_text = (
                str(_meta.get("text", "")) if _meta is not None else str(_raw_note)
            )
            _wrapped = (
                "[User note received mid-turn — take it into account "
                f"and continue the task] {_note_text}"
            )
            convo.append({"role": "user", "content": _wrapped})
            if _meta is None:
                yield ("note", _wrapped)
            else:
                yield (
                    "note",
                    {
                        **_meta,
                        "text": _wrapped,
                    },
                )
            count += 1
    except Exception as _e:  # noqa: BLE001 - notes best-effort
        _debug("NOTES_ERR", str(_e))
    return count
