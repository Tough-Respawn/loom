# loom/agent/session.py
"""Session : le fil de travail persistant d'un projet (un chat par session).

Une session vit sous `root/<id>/session.json` : conversation (historique + outils
actifs), métadonnées (titre, workspace cible, horodatage). Un pointeur `root/active`
retient la session courante (survit au redémarrage du serveur)."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loom.agent.conversation import Conversation


@dataclass
class SessionMeta:
    """Vue légère pour lister les sessions sans charger toute la conversation."""

    id: str
    title: str
    workspace: str
    updated_at: str


@dataclass
class Session:
    id: str
    title: str
    workspace: str
    conversation: Conversation
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "conversation": self.conversation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, default_system_prompt: str) -> "Session":
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            workspace=data.get("workspace", "."),
            conversation=Conversation.from_dict(
                data.get("conversation", {}), default_system_prompt
            ),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    """Persiste les sessions sous root/<id>/session.json + un pointeur `active`."""

    def __init__(
        self,
        root,
        default_system_prompt: str,
        default_tools: list[str] | None = None,
        default_model: str = "",
    ) -> None:
        self.root = Path(root)
        self.default_system_prompt = default_system_prompt
        # Outils armés sur CHAQUE session neuve. Sans ça, la conversation d'une session
        # part avec active_tools=[] -> le chat tourne sans `tools=` -> le modèle, sommé
        # d'agir, crache ses appels d'outil en texte (`<|tool_call|>...`) faute d'interface.
        self.default_tools = list(default_tools or [])
        # Modèle armé sur chaque session neuve. Sans ça, model="" -> llama-swap 404.
        self.default_model = default_model
        self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, sid: str) -> Path:
        return self.root / sid / "session.json"

    def session_dir(self, sid: str) -> Path:
        """Dossier d'une session (root/<id>) : porte session.json ET les logs runtime
        (debug.log) propres à cette session."""
        return self.root / sid

    def create(self, *, workspace: str = ".", title: str = "") -> Session:
        sid = uuid.uuid4().hex[:12]
        now = _now_iso()
        session = Session(
            id=sid,
            title=title or "Nouvelle session",
            workspace=str(workspace),
            conversation=Conversation(system_prompt=self.default_system_prompt),
            created_at=now,
            updated_at=now,
        )
        if self.default_tools:
            session.conversation.set_tools(self.default_tools)
        if self.default_model:
            session.conversation.set_model(self.default_model)
        self.save(session)
        self.set_active(sid)  # créer une session la focalise
        return session

    def save(self, session: Session) -> None:
        session.updated_at = _now_iso()
        f = self._file(session.id)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, f)

    def load(self, sid: str) -> Session | None:
        f = self._file(sid)
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            return Session.from_dict(data, self.default_system_prompt)
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def list(self) -> list[SessionMeta]:
        metas: list[SessionMeta] = []
        for d in self.root.iterdir():
            f = d / "session.json"
            if not d.is_dir() or not f.exists():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            metas.append(
                SessionMeta(
                    id=data.get("id", d.name),
                    title=data.get("title", ""),
                    workspace=data.get("workspace", "."),
                    updated_at=data.get("updated_at", ""),
                )
            )
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas

    def delete(self, sid: str) -> None:
        shutil.rmtree(self.root / sid, ignore_errors=True)
        if self._active_id() == sid:
            (self.root / "active").unlink(missing_ok=True)

    def set_active(self, sid: str) -> None:
        (self.root / "active").write_text(sid, encoding="utf-8")

    def _active_id(self) -> str | None:
        f = self.root / "active"
        if not f.exists():
            return None
        return f.read_text(encoding="utf-8").strip() or None

    def active(self) -> Session | None:
        sid = self._active_id()
        return self.load(sid) if sid else None
