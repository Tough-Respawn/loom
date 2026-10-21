from __future__ import annotations

import hashlib
import json
import re

from loom.agent.debuglog import log_event
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


def _workspace_text(S, ws: str) -> str:
    """Consigne du dossier de travail + loom.md du projet (contexte non fiable)."""
    text = (
        f"Tes commandes (run_shell) tournent dans "
        f"`{ws}` et les chemins relatifs s'y résolvent - n'y répète pas le nom de ce "
        "dossier dans tes chemins. Si une commande git échoue par « not a git "
        "repository », c'est que CE dossier n'est pas un repo : fais UN list_dir pour "
        "repérer le bon sous-dossier (puis `git -C <sous-dossier>`), ne relance pas la "
        "même commande à l'identique."
    )
    from loom.memory.identity import project_block

    _pm_blk = project_block(ws, max_tokens=S.settings["project_memory_max_tokens"])
    return f"{text}\n\n{_pm_blk}" if _pm_blk else text


def _scratch_dir(S, sid: str) -> str:
    """Chemin du bac à sable d'une session, en barres obliques comme le dossier de travail :
    avec des antislashs, un petit modèle a réécrit `C:\\...\\var\\sessions\\<id>\\scratch`
    en `/var/sessions/<id>/scratch` (fichiers partis dans C:\\var, vécu 2026-10-19)."""
    return str(S.session_store.session_dir(sid) / "scratch").replace("\\", "/")


_SCRATCH_PYPROJECT = (
    '[project]\nname = "loom-scratch"\nversion = "0"\nrequires-python = ">=3.11"\n'
    "dependencies = []\n"
)


def _scratch_init(scratch: str) -> None:
    """Crée le bac à sable et en fait un projet uv nu (`pyproject.toml` sans build) :
    `uv add` y installe dans un `.venv` propre à la session, `uv run` l'utilise. Rien
    n'est écrasé si le fichier existe (dépendances déjà ajoutées par le modèle)."""
    from pathlib import Path

    root = Path(scratch)
    root.mkdir(parents=True, exist_ok=True)
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text(_SCRATCH_PYPROJECT, encoding="utf-8")


def _scratch_text(scratch: str) -> str:
    """Consigne du bac à sable de la session : dossier jetable pour les essais, et
    environnement Python propre à la session (projet uv) quand une bibliothèque manque —
    le modèle n'installe rien sur la machine et n'abandonne pas « faute de matplotlib »."""
    return (
        f"`{scratch}` est le bac à sable de CETTE session (scratchpad), supprimé avec "
        "elle. Tout ce qui n'est pas un livrable demandé va là : scripts d'essai, fichiers "
        "temporaires, sorties intermédiaires, images que tu produis pour les montrer. "
        "Jamais dans le dossier de travail sans qu'on te l'ait demandé.\n"
        "C'est aussi un projet Python (uv) avec son propre environnement, vide au départ. "
        "Bibliothèque manquante (ModuleNotFoundError) : n'installe rien sur la machine et "
        f"n'abandonne pas. Ajoute-la au projet, une fois : `uv --directory {scratch} add "
        '<paquet>` (plusieurs : `add a b` ; version : `add "matplotlib==3.9"`), puis '
        f"exécute tes scripts avec `uv --directory {scratch} run python <script.py>` : "
        "l'environnement du bac à sable sert, les dépendances ajoutées y restent pour toute "
        "la session, rien à activer. Ces commandes marchent depuis n'importe quel dossier. "
        "Pour un essai unique sans garder le paquet : `uv run --with <paquet> python "
        "<script.py>`. La première installation demande le réseau.\n"
        "Figure, schéma, graphique demandé : écris un SVG ou une page HTML dans le bac à "
        "sable et vérifie-la avec check_page, ou produis un PNG (`uv --directory "
        f"{scratch} add matplotlib` puis `uv --directory {scratch} run python "
        "<script.py>`) et relis-le avec read_image ; donne ensuite le chemin du fichier. "
        "Une image que tu produis n'est vérifiée QUE par read_image (ou check_page pour "
        "un SVG/HTML) : ne décris jamais le contenu d'une image que tu n'as pas regardée, "
        "une taille de fichier ou un `Image.open` ne prouvent rien. Les sorties de tes "
        "scripts (save, open en écriture) vont dans ce même bac à sable, jamais dans /tmp."
    )


PRIVATE_TEXT = (
    "Rien ne doit sortir de cette machine. Les outils web, vidéo, plugins et MCP sont "
    "coupés. N'essaie jamais d'atteindre internet depuis tes commandes (run_shell) ou "
    "check_page : `uv add`, pip, curl, git vers internet échoueront, et contourner ce "
    "blocage trahirait la demande de l'utilisateur. Travaille avec ce qui est déjà "
    "installé et sur le disque. S'il manque quelque chose (paquet, page web, document "
    "en ligne), dis-le à l'utilisateur au lieu de chercher un détour. "
    "La mémoire partagée est en lecture seule (recall oui, remember non) : pour retenir "
    "quelque chose dans cette session, utilise write_note."
)


def _prompt_build() -> str:
    """Empreinte du HARNAIS lui-même : gabarits de prompt livrés avec Loom et consigne
    du bac à sable. Une mise à jour de Loom qui change l'un d'eux doit atteindre les
    sessions existantes (vécu 2026-10-19 : la règle « bibliothèque manquante → uv run
    --with » livrée, le modèle d'une session ouverte ne l'a jamais vue, prompt figé)."""
    from loom.prompts import CHAT_SYSTEM

    # Le bloc Système (conventions OS) compte aussi : il est livré avec Loom.
    raw = (
        CHAT_SYSTEM
        + CHAT_SYSTEM_STRONG
        + _scratch_text("@")
        + PRIVATE_TEXT
        + platform_detect().prompt_block()
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def _explicit_key(S, conv) -> str:
    """Empreinte des choix EXPLICITES de l'utilisateur et de la version du harnais.
    Seuls eux refigent le prompt ; mémoire, skills appris, loom.md et dossier
    auto-adopté ne le réécrivent jamais."""
    strong = bool(
        conv.model
        and conv.model in S.remote_model_ids
        and conv.model not in S.remote_weak_ids
    )
    raw = json.dumps(
        [
            conv.model,
            strong,
            conv.system_prompt,
            conv.goal,
            sorted(conv.disabled_skills),
            sorted(conv.skill_overrides.items()),
            _prompt_build(),
            bool(getattr(conv, "local_only", False)),
        ],
        ensure_ascii=False,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _build_system_prompt(S, conv, workspace=None, scratch=None):
    """System prompt FIGÉ pour la session : rendu une fois, puis réutilisé tel quel
    (append-only, préfixe KV stable). Rerendu si un choix explicite change (modèle,
    prompt de base, objectif, skills de la session) ou si le harnais a été mis à jour
    (gabarits de prompt) : dans ce cas le prompt de base de la session, instantané pris à
    sa création, est rafraîchi depuis le gabarit courant. `scratch` : bac à sable de la
    session (créé ici, projet uv inclus, annoncé dans le prompt). Retourne (texte, strong)."""
    frozen = conv.frozen_prompt
    build = _prompt_build()
    if frozen and frozen.get("build") not in (None, build):
        # Mise à jour du harnais : la session reprend le gabarit livré (aucune édition par
        # session n'existe, l'instantané ne portait que la stabilité du cache).
        default = getattr(
            getattr(S, "session_store", None), "default_system_prompt", ""
        )
        if default:
            conv.system_prompt = default
    key = _explicit_key(S, conv)
    if frozen and frozen.get("key") == key and frozen.get("text"):
        return frozen["text"], bool(frozen.get("strong"))
    _ws = workspace if workspace is not None else _session(S).workspace
    if scratch:
        _scratch_init(scratch)
    text, strong = _render_system_prompt(
        S, conv, workspace=_ws, **({"scratch": scratch} if scratch else {})
    )
    log_event(
        "prompt.fige",
        raison=(
            "premier"
            if not frozen
            else "maj_harnais"
            if frozen.get("build") != build
            else "choix_explicite"
        ),
        chars=len(text),
    )
    conv.frozen_prompt = {
        "key": key,
        "text": text,
        "strong": strong,
        "workspace": _ws,
        "build": build,
    }
    return text, strong


_NOTE_RE = re.compile(r"^\[Changement de dossier de travail : (.+?)\]$", re.MULTILINE)
_NOTE_END = "\n[Fin du changement de dossier]\n\n"


def _known_workspace(conv) -> str:
    """Dossier que le modèle croit courant : la dernière note encore PRÉSENTE dans le
    fil, sinon celui figé dans le system prompt. Relu dans le fil (et non mémorisé à
    part) pour rester juste après un /fork ou une compaction qui retire la note."""
    for m in reversed(conv.messages):
        content = m.get("content")
        if m.get("role") != "user":
            continue
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        found = _NOTE_RE.findall(str(content or ""))
        if found:
            return found[-1]
    return (conv.frozen_prompt or {}).get("workspace", "")


def strip_workspace_note(text: str) -> str:
    """Texte saisi par l'utilisateur, sans la note de dossier ajoutée par Loom."""
    if not _NOTE_RE.match(text or "") or _NOTE_END not in text:
        return text
    return text.split(_NOTE_END, 1)[1]


def _workspace_note(S, conv, workspace: str) -> str:
    """Note à placer EN TÊTE du prochain message utilisateur quand le dossier de travail
    a changé depuis ce que le modèle connaît : le changement s'ajoute au fil au lieu de
    réécrire le system prompt. Vide si rien à annoncer."""
    if not conv.frozen_prompt or not workspace:
        return ""
    known = _known_workspace(conv)
    if known == workspace:
        return ""
    log_event("prompt.note_dossier", avant=known, apres=workspace)
    return (
        f"[Changement de dossier de travail : {workspace}]\n"
        f"{_workspace_text(S, workspace)}{_NOTE_END}"
    )


def _render_system_prompt(S, conv, workspace=None, scratch=None):
    """Construit le system prompt complet : identité always-on + base (strong/local) +
    catalogue des skills + déclaration du moteur + conventions OS + dossier de travail +
    bac à sable + objectif de session. Retourne (system_prompt, strong)."""

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

    system_prompt += "\n\n# Dossier de travail courant\n" + _workspace_text(S, _ws)

    if scratch:
        system_prompt += "\n\n# Bac à sable de la session\n" + _scratch_text(scratch)

    if getattr(conv, "local_only", False):
        system_prompt += "\n\n# Session privée\n" + PRIVATE_TEXT

    # L'objectif guide la vérification sans ajouter un juge externe contradictoire.
    if conv.goal:
        system_prompt += (
            f"\n\n# Objectif de session\nTant qu'il est actif, oriente ton travail vers "
            f"cet objectif et ne le déclare atteint qu'une fois PROUVÉ par tes propres "
            f"exécutions (sortie réelle affichée) :\n{conv.goal}"
        )

    return system_prompt, strong
