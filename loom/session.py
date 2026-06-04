# loom/session.py
"""Session : le fil de travail persistant qui unifie chat et runs agentic.

Une session vit sous `root/<id>/session.json` : conversation unifiée (chat ET
résumés des runs), métadonnées (titre, workspace cible, horodatage) et journal
des runs. C'est le substrat qui permet au modèle de reprendre où il s'est arrêté :
un run agentic écrit son résumé DANS la conversation via `add_run`, donc le tour
suivant (chat ou build) le voit. Un pointeur `root/active` retient la session
courante (survit au redémarrage du serveur)."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loom.conversation import Conversation


@dataclass
class RunRecord:
    """Trace persistée d'un run agentic (plan/dev/verif), réinjectable au modèle."""

    task: str
    summary: str = ""
    files: list[str] = field(default_factory=list)
    ok: bool = False

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "summary": self.summary,
            "files": list(self.files),
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(
            task=d.get("task", ""),
            summary=d.get("summary", ""),
            files=list(d.get("files", [])),
            ok=bool(d.get("ok", False)),
        )


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
    runs: list[RunRecord] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def add_run(self, record: RunRecord) -> None:
        """Journalise un run ET en laisse une trace dans la conversation unifiée, pour
        que le modèle reprenne le fil au tour suivant (le fondamental de la session)."""
        self.runs.append(record)
        verdict = "vérifié OK" if record.ok else "défauts restants / non vérifié"
        files = ", ".join(record.files) if record.files else "aucun fichier"
        self.conversation.add(
            "assistant",
            f"(Run « {record.task} ») {record.summary} "
            f"[fichiers : {files} ; {verdict}]",
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "conversation": self.conversation.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
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
            runs=[RunRecord.from_dict(r) for r in data.get("runs", [])],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    """Persiste les sessions sous root/<id>/session.json + un pointeur `active`."""

    def __init__(self, root, default_system_prompt: str) -> None:
        self.root = Path(root)
        self.default_system_prompt = default_system_prompt
        self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, sid: str) -> Path:
        return self.root / sid / "session.json"

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
