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



def _build_system_prompt(S, conv, workspace=None):
    """Construit le system prompt complet : identité always-on + base (strong/local) +
    catalogue des skills + déclaration du moteur + conventions OS + dossier de travail +
    objectif de session. Retourne (system_prompt, strong)."""

    skills = effective_skills(
        _all_skills(S),
        overrides=conv.skill_overrides,
        disabled=conv.disabled_skills,
    )

    catalog = render_catalog(skills)

    # Placer l'identité en tête la rend prioritaire et insensible à la compaction d'historique.
    _idblk = ""

    if S.identity_paths:
        from loom.memory.identity import identity_block

        _idblk = identity_block(
            S.identity_paths["soul_path"],
            S.identity_paths["user_path"],
            S.identity_paths["memory_md_path"],
            max_tokens=S.settings["identity_max_tokens"],
        )

    # Un modèle distant fort garde identité, outils, mémoire et sécurité sans scaffolding local.
    strong = bool(
        conv.model
        and conv.model in S.remote_model_ids
        and conv.model not in S.remote_weak_ids
    )

    base_prompt = CHAT_SYSTEM_STRONG if strong else conv.system_prompt

    system_prompt = f"{_idblk}\n\n{base_prompt}" if _idblk else base_prompt

    # Les sous-agents distants indépendants gagnent à être groupés dans un même tour.
    if strong:
        system_prompt += (
            "\n\nParallélisme : quand plusieurs sous-tâches sont INDÉPENDANTES (auditer/"
            "explorer des pans distincts), émets PLUSIEURS dispatch_agent dans le MÊME "
            "tour - ils s'exécutent EN PARALLÈLE, bien plus vite qu'un par tour. Un pan = "
            "un agent, lance-les ensemble."
        )

    if catalog:
        system_prompt += f"\n\n{catalog}"

    # Injecter le backend courant évite les affirmations inventées sur l'infrastructure.
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

    # Partager la détection d'OS avec run_shell garde les conventions cohérentes.
    system_prompt += "\n\n" + platform_detect().prompt_block()

    # Garder le workspace volatil en fin de prompt et lié à la session cible.
    _ws = workspace if workspace is not None else _session(S).workspace

    system_prompt += (
        f"\n\n# Dossier de travail courant\nTes commandes (run_shell) tournent dans "
        f"`{_ws}` et les chemins relatifs s'y résolvent - n'y répète pas le nom de ce "
        "dossier dans tes chemins. Si une commande git échoue par « not a git "
        "repository », c'est que CE dossier n'est pas un repo : fais UN list_dir pour "
        "repérer le bon sous-dossier (puis `git -C <sous-dossier>`), ne relance pas la "
        "même commande à l'identique."
    )

    # Injecter loom.md comme contexte non fiable, avec cache mtime pour stabiliser le préfixe.
    from loom.memory.identity import project_block

    _pm_blk = project_block(_ws, max_tokens=S.settings["project_memory_max_tokens"])

    if _pm_blk:
        system_prompt += f"\n\n{_pm_blk}"

    # L'objectif guide la vérification sans ajouter un juge externe contradictoire.
    if conv.goal:
        system_prompt += (
            f"\n\n# Objectif de session\nTant qu'il est actif, oriente ton travail vers "
            f"cet objectif et ne le déclare atteint qu'une fois PROUVÉ par tes propres "
            f"exécutions (sortie réelle affichée) :\n{conv.goal}"
        )

    return system_prompt, strong
