"""Étape `reflect` : capitalisation post-tour, HORS de la loop d'action (design §6).

Un seul appel modèle borné sur la trajectoire du tour -> JSON strict -> validation pure
-> écritures internes (épisodes via provider, faits vers SOUL/USER/MEMORY, skills appris).
Toute défaillance est NON bloquante : la réponse à l'utilisateur est déjà rendue.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loom.memory import identity as _id
from loom.prompts import REFLECT_SYSTEM

_KEYS = (
    "new_skills",
    "improved_skills",
    "episodes",
    "memory_updates",
    "user_updates",
    "soul_updates",
)
_MIN_SKILL_BODY = 80  # anti-trivial : un skill doit dire quelque chose
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")


@dataclass
class ReflectResult:
    new_skills: list = field(default_factory=list)
    improved_skills: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    memory_updates: list = field(default_factory=list)
    user_updates: list = field(default_factory=list)
    soul_updates: list = field(default_factory=list)


def validate_reflect_json(obj) -> ReflectResult | None:
    """Filtre le JSON de reflect. Renvoie un ReflectResult propre, ou None si inexploitable.

    PURE (sans IO, sans modèle) -> testable. Rejette le hors-schéma en silence ; anti-trivial
    sur les skills (nom kebab-case, corps assez long) ; déduplication des lignes texte.
    """
    if not isinstance(obj, dict):
        return None
    res = ReflectResult()
    for s in obj.get("new_skills") or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip().lower()
        body = str(s.get("body", "")).strip()
        desc = str(s.get("description", "")).strip()
        if _NAME_RE.match(name) and len(body) >= _MIN_SKILL_BODY:
            res.new_skills.append({"name": name, "description": desc, "body": body})
    for s in obj.get("improved_skills") or []:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", "")).strip()
        body = str(s.get("body", "")).strip()
        base = name.split(":", 1)[1] if ":" in name else ""
        # MÊME validation kebab-case que new_skills sur la base : sans elle, un nom
        # « learned:../../x » s'échapperait du dossier à l'écriture (traversée de chemin).
        if (
            name.startswith("learned:")
            and _NAME_RE.match(base)
            and len(body) >= _MIN_SKILL_BODY
        ):
            res.improved_skills.append({"name": name, "body": body})
    for e in obj.get("episodes") or []:
        text = (e.get("text", "") if isinstance(e, dict) else str(e)).strip()
        if text:
            res.episodes.append({"text": text})
    for key in ("memory_updates", "user_updates", "soul_updates"):
        seen = set()
        for line in obj.get(key) or []:
            line = str(line).strip()
            if line and line not in seen:
                seen.add(line)
                getattr(res, key).append(line)
    if not any(getattr(res, k) for k in _KEYS):
        return None
    return res


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_learned_skill(
    learned_dir: str, name: str, description: str, body: str, *, improve: bool
) -> None:
    base = name.split(":", 1)[1] if name.startswith("learned:") else name
    # Défense en profondeur (traversée de chemin) : nom borné kebab-case ET dossier résolu
    # confiné sous learned_dir. Le JSON de reflect vient du modèle (trajectoire potentiellement
    # influencée par du contenu ingéré) -> on ne fait JAMAIS confiance au nom pour un chemin.
    if not _NAME_RE.match(base):
        return
    root = Path(learned_dir).resolve()
    d = (root / base).resolve()
    if d != root and root not in d.parents:
        return
    d.mkdir(parents=True, exist_ok=True)
    md = d / "SKILL.md"
    uses, created = 0, _now()
    if md.exists():
        from loom.extend.skills import _parse_skill_md

        _n, old_desc, _b, meta = _parse_skill_md(md.read_text("utf-8"), base)
        uses = meta.get("uses", 0) + (1 if improve else 0)
        created = meta.get("created_at") or created
        description = description or old_desc
    front = (
        f"---\nname: {base}\ndescription: {description}\nlearned: true\n"
        f"created_at: {created}\nupdated_at: {_now()}\nuses: {uses}\n---\n"
    )
    md.write_text(front + body.strip() + "\n", encoding="utf-8")


def apply_reflect(
    res: ReflectResult, *, provider, paths: dict, learned_dir: str
) -> None:
    """Écrit le résultat validé (skills appris, épisodes, identité). Appelé sous le
    try/except global de l'appelant : best-effort, jamais bloquant."""
    for s in res.new_skills:
        _write_learned_skill(
            learned_dir, s["name"], s["description"], s["body"], improve=False
        )
    for s in res.improved_skills:
        _write_learned_skill(learned_dir, s["name"], "", s["body"], improve=True)
    for e in res.episodes:
        provider.remember(e["text"], kind="episodic", source="reflect")
    for line in res.memory_updates:
        _id.append_unique(paths["memory_md_path"], line)
    for line in res.user_updates:
        _id.append_unique(paths["user_path"], line)
    for line in res.soul_updates:
        _id.append_unique(paths["soul_path"], line)


def _trajectory_summary(messages: list, actions: list, answer: str) -> str:
    """Condense le tour pour le prompt de reflect (borné)."""
    last_user = next(
        (
            m["content"]
            for m in reversed(messages)
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ),
        "",
    )
    acts = "\n".join(f"- {a}" for a in actions) or "(aucune action)"
    return (
        f"DEMANDE :\n{str(last_user)[:1000]}\n\n"
        f"ACTIONS DU TOUR :\n{acts}\n\n"
        f"RÉPONSE FINALE :\n{(answer or '')[:1500]}"
    )


def reflect(
    messages: list,
    actions: list,
    answer: str,
    *,
    client,
    model,
    provider,
    paths: dict,
    learned_dir: str,
) -> ReflectResult | None:
    """Un tour de réflexion complet : appel modèle borné -> validation -> écritures.

    Renvoie le ReflectResult appliqué, ou None si rien à capitaliser. NON bloquant :
    l'appelant enveloppe dans un try/except (design §11) — une erreur ici ne doit jamais
    remonter à la réponse utilisateur.
    """
    user = _trajectory_summary(messages, actions, answer)
    txt = ""
    for kind, chunk in client.stream_chat(
        [{"role": "user", "content": user}],
        REFLECT_SYSTEM,
        max_tokens=800,
        model=model,
        thinking=False,
    ):
        if kind == "content":
            txt += chunk
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    res = validate_reflect_json(obj)
    if res is None:
        return None
    apply_reflect(res, provider=provider, paths=paths, learned_dir=learned_dir)
    return res
