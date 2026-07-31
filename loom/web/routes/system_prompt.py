# loom/web/routes/system_prompt.py — helper de chat sorti de chat.py (comportement constant).
from __future__ import annotations
from loom.extend.skills import (
    effective_skills,
    render_catalog,
)
from loom.prompts import CHAT_SYSTEM_STRONG
from loom.runtime.platform_info import detect as platform_detect
from loom.web.routes.helpers import (
    _session,
)
from loom.web.routes.skills import _all_skills




# ---- System prompt --------------------------------------------------------------------


def _build_system_prompt(S, conv):
    """Construit le system prompt complet : identité always-on + base (strong/local) +
    catalogue des skills + déclaration du moteur + conventions OS + dossier de travail +
    objectif de session. Retourne (system_prompt, strong)."""

    skills = effective_skills(
        _all_skills(S),
        overrides=conv.skill_overrides,
        disabled=conv.disabled_skills,
    )

    catalog = render_catalog(skills)

    # Identité always-on (SOUL/USER/MEMORY) EN TÊTE : c'est la définition qui FAIT FOI
    # de qui est Loom (rôle, persona, style). Le mode d'emploi opérationnel (outils,
    # règles) de chat.system.md vient APRÈS et s'y conforme - on ne plante plus un
    # cadrage générique d'abord pour le corriger 12k caractères plus loin. Always-on =>
    # survit toujours à la microcompaction/summarization (qui ne touchent que
    # l'historique). Bornée par identity_max_tokens. Cf. design §5.6.
    _idblk = ""

    if S.identity_paths:
        from loom.memory.identity import identity_block

        _idblk = identity_block(
            S.identity_paths["soul_path"],
            S.identity_paths["user_path"],
            S.identity_paths["memory_md_path"],
            max_tokens=S.settings["identity_max_tokens"],
        )

    # TIER du harnais : un modèle DISTANT (API, non quantifié) se pilote seul -> prompt
    # ALLÉGÉ (identité + outils + mémoire + sécurité), sans le scaffolding de comportement
    # de chat.system.md qui ne sert qu'à un petit modèle local. Le flag `strong` sert
    # aussi (plus bas) à couper les gardes de comportement dans la boucle d'outils.
    strong = bool(
        conv.model
        and conv.model in S.remote_model_ids
        and conv.model not in S.remote_weak_ids
    )

    base_prompt = CHAT_SYSTEM_STRONG if strong else conv.system_prompt

    system_prompt = f"{_idblk}\n\n{base_prompt}" if _idblk else base_prompt

    # Distant (strong) : la machine du provider encaisse le parallélisme -> on incite à
    # GROUPER les sous-agents indépendants dans un même tour (ils tournent en parallèle).
    if strong:
        system_prompt += (
            "\n\nParallélisme : quand plusieurs sous-tâches sont INDÉPENDANTES (auditer/"
            "explorer des pans distincts), émets PLUSIEURS dispatch_agent dans le MÊME "
            "tour - ils s'exécutent EN PARALLÈLE, bien plus vite qu'un par tour. Un pan = "
            "un agent, lance-les ensemble."
        )

    if catalog:
        system_prompt += f"\n\n{catalog}"

    # Le modèle ignore par défaut sous quel backend il tourne (le prompt dit
    # "Tu es Loom") -> il baratine quand on lui demande "quel modèle ?". On lui
    # injecte son modèle courant pour qu'il réponde honnêtement. DISTANT vs LOCAL :
    # sans ça un modèle servi par une API répétait « je tourne en local/offline sur
    # llama.cpp » (la persona de Loom est « agent local ») -> confabulation d'infra.
    if conv.model:
        if conv.model in S.remote_model_ids:
            _pm = S.remote_model_names.get(conv.model)

            _label = (
                f"« {_pm} » (route « {conv.model} »)" if _pm else f"« {conv.model} »"
            )

            system_prompt += (
                f"\n\n# Ton moteur\nTon raisonnement est servi par le modèle DISTANT "
                f"{_label}, via une API externe - PAS en local. Tes OUTILS, eux, "
                "s'exécutent bien sur la machine de l'utilisateur, mais toi (le cerveau) "
                "non. Ne prétends donc JAMAIS être offline, ni tourner sur llama.cpp / "
                "llama-swap / une carte graphique locale : ce serait faux. Si on te "
                "demande quel modèle/moteur tu utilises, donne ce nom honnêtement, sans "
                "inventer de détails d'infrastructure."
            )

        else:
            system_prompt += (
                f"\n\n# Ton moteur\nTu tournes sur le modèle local « {conv.model} ». "
                "Si on te demande quel modèle/moteur tu utilises, réponds-le "
                "honnêtement et directement (ce nom), sans esquiver."
            )

    # Système : Loom détecte SEUL l'OS et injecte ses conventions (shell, commandes,
    # chemins) -> le modèle produit du PowerShell sous Windows, du bash/unix sous
    # macOS/Linux, sans qu'on code l'OS en dur dans le prompt. Source unique partagée
    # avec run_shell (loom.runtime.platform_info) : jamais de divergence.
    system_prompt += "\n\n" + platform_detect().prompt_block()

    # Dossier de travail courant : le modèle l'IGNORE sinon et le devine en sondant
    # (git rev-parse à l'aveugle, list_dir…) -> tours gaspillés. On le lui dit, avec
    # le réflexe anti-tâtonnement quand ce dossier n'est pas un repo git. Reste EN BAS
    # (contexte volatil, près de l'action).
    _ws = _session(S).workspace

    system_prompt += (
        f"\n\n# Dossier de travail courant\nTes commandes (run_shell) tournent dans "
        f"`{_ws}` et les chemins relatifs s'y résolvent - n'y répète pas le nom de ce "
        "dossier dans tes chemins. Si une commande git échoue par « not a git "
        "repository », c'est que CE dossier n'est pas un repo : fais UN list_dir pour "
        "repérer le bon sous-dossier (puis `git -C <sous-dossier>`), ne relance pas la "
        "même commande à l'identique."
    )

    # Fiche projet auto-injectée : si `<workspace>/loom.md` existe (générée par /init),
    # le modèle la reçoit d'office au lieu de re-sonder le projet à chaque session.
    # Cache mtime (read_md) -> préfixe stable en session = prompt caching préservé ;
    # suit le workspace courant (adoption/changement en cours de session). Les DEUX
    # tiers la reçoivent (c'est de la mémoire, pas du scaffolding de comportement).
    # L'en-tête la cadre en CONTEXTE (pas instructions, possiblement périmée) : une
    # fiche est écrite en lisant le projet, un repo piégé ne doit pas pouvoir élever
    # ses consignes au rang de system prompt.
    from loom.memory.identity import project_block

    _pm_blk = project_block(_ws, max_tokens=S.settings["project_memory_max_tokens"])

    if _pm_blk:
        system_prompt += f"\n\n{_pm_blk}"

    # Objectif de session (/goal), en DIRECTIVE DOUCE : pas de juge externe qui te
    # contredit (retiré - il recalait des preuves correctes). Tu restes seul maître de
    # ta propre vérification : ne te déclare pas fini tant que l'objectif n'est pas
    # ATTEINT ET PROUVÉ par tes exécutions (montre la sortie réelle) ; une fois prouvé,
    # dis-le et arrête-toi. L'utilisateur l'efface avec « /goal clear ».
    if conv.goal:
        system_prompt += (
            f"\n\n# Objectif de session\nTant qu'il est actif, oriente ton travail vers "
            f"cet objectif et ne le déclare atteint qu'une fois PROUVÉ par tes propres "
            f"exécutions (sortie réelle affichée) :\n{conv.goal}"
        )

    return system_prompt, strong
