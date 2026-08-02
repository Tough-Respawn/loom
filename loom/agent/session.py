# loom/agent/session.py
"""Session : le fil de travail persistant d'un projet (un chat par session).

Une session vit sous `root/<id>/session.json` : conversation (historique + outils
actifs), métadonnées (titre, workspace cible, horodatage). Un pointeur `root/active`
retient la session courante (survit au redémarrage du serveur)."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from loom.agent.conversation import Conversation
from loom.utils import now_iso as _now_iso

# Format d'un id de session : 12 hex minuscules (uuid4().hex[:12]). Tout id qui ne
# matche PAS est refusé avant de toucher au disque -> aucune opération (suppression,
# lecture, export) ne peut sortir de root : un id vide (root / "" == root), un
# traversal (../voisin) ou un chemin absolu (pathlib remplace la base) échapperaient
# sinon à SessionStore.root (revue sécu : rmtree/lecture hors racine).
_SID_RE = re.compile(r"[0-9a-f]{12}")


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

    def __post_init__(self) -> None:
        # Identifiant runtime non sérialisé sur Conversation : les outils liés à
        # un processus (monitor) retrouvent leur file/session sans dupliquer l'id
        # dans session.json.
        self.conversation.runtime_session_id = self.id

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
    def from_dict(cls, data: dict, default_system_prompt: str) -> Session:
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


class SessionStore:
    """Persiste les sessions sous root/<id>/session.json + un pointeur `active`."""

    def __init__(
        self,
        root,
        default_system_prompt: str,
        default_tools: list[str] | None = None,
        default_model: str = "",
        known_models: list[str] | None = None,
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
        # Mémoire du dernier modèle choisi : devient le défaut effectif des sessions
        # neuves (prioritaire sur le config, suit l'utilisateur). Machine-local, gitignoré.
        self._known_models = set(known_models or [])
        self._last_model_file = self.root.parent / "last_model"
        self._load_last_model()

    def _load_last_model(self) -> None:
        """Restaure le dernier modèle sélectionné comme défaut effectif, s'il est encore
        connu. Prioritaire sur le default_model du config : on suit le choix utilisateur."""
        try:
            saved = self._last_model_file.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if saved and (not self._known_models or saved in self._known_models):
            self.default_model = saved

    def set_default_model(self, model_id: str) -> None:
        """Mémorise le modèle choisi : défaut des sessions neuves ET du prochain lancement.
        Persistance best-effort (ne casse jamais un switch si l'écriture échoue)."""
        model_id = (model_id or "").strip()
        if not model_id:
            return
        self.default_model = model_id
        try:
            self._last_model_file.write_text(model_id, encoding="utf-8")
        except OSError:
            pass

    def _safe_dir(self, sid: str) -> Path | None:
        """root/<sid> SEULEMENT si sid est un id de session valide (_SID_RE) ; None
        sinon. Choke-point de confinement : toute opération FS par id non fiable
        (delete, load, read_timeline, export_zip) passe par ici avant de toucher root."""
        if not (isinstance(sid, str) and _SID_RE.fullmatch(sid)):
            return None
        return self.root / sid

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
        # Jamais d'écriture hors racine : un id invalide (traversal, absolu, vide)
        # est refusé net plutôt que de laisser _file() sortir de root.
        if not _SID_RE.fullmatch(session.id or ""):
            raise ValueError(f"id de session invalide : {session.id!r}")
        session.conversation.runtime_session_id = session.id
        session.updated_at = _now_iso()
        f = self._file(session.id)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + ".tmp")
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, f)

    # --- Journal d'affichage TEMPS RÉEL (timeline.jsonl) ---------------------------------
    # Append-only, UNE ligne par événement écrite À L'INSTANT où il sort (raisonnement, texte,
    # appel/résultat d'outil…). Distinct de session.json (contexte lean du modèle) : ce journal
    # sert à REJOUER l'UI au rechargement -> on retrouve exactement ce qui s'affichait en direct,
    # cartes d'outils comprises. Vrai temps réel, pas de batch : append + flush par événement.
    def _timeline_file(self, sid: str):
        return self.session_dir(sid) / "timeline.jsonl"

    def append_event(self, sid: str, event: str, data: dict) -> None:
        """Écrit un événement d'affichage dans le journal, tout de suite. Best-effort."""
        try:
            p = self._timeline_file(sid)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"event": event, "data": data}, ensure_ascii=False)
                    + "\n"
                )
                fh.flush()
        except OSError:
            pass

    def read_timeline(self, sid: str) -> list[dict]:
        """Relit le journal (liste ordonnée d'événements). Vide si absent/illisible."""
        if self._safe_dir(sid) is None:
            return []
        p = self._timeline_file(sid)
        if not p.exists():
            return []
        out: list[dict] = []
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            return []
        return out

    def clear_timeline(self, sid: str) -> None:
        """Efface le journal (reset de conversation)."""
        try:
            self._timeline_file(sid).unlink(missing_ok=True)
        except OSError:
            pass

    # --- Export / import (.zip clair) ----------------------------------------------------
    # Une session se donne : session.json + timeline.jsonl + manifeste, dans un zip
    # SANS chiffrement (le chiffré, c'est l'Âme — identité globale). debug.log exclu
    # (log runtime machine). À l'import : id NEUF (jamais d'écrasement), modèle inconnu
    # sur cette machine replié sur le défaut local (sinon 404 llama-swap au 1er tour).

    EXPORT_FORMAT = "loom-session"
    EXPORT_VERSION = 1
    # Refus des archives démesurées (déclaré OU décompressé) : une session légitime
    # pèse quelques Ko à quelques Mo — 200 Mo, c'est déjà une anomalie (ou une bombe zip).
    MAX_IMPORT_BYTES = 200 * 1024 * 1024

    def export_zip(self, sid: str) -> bytes | None:
        """Archive portable d'une session (None si inconnue)."""
        import io
        import zipfile

        if self._safe_dir(sid) is None:
            return None
        if not self._file(sid).exists():
            return None
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(
                "loom-session.json",
                json.dumps(
                    {
                        "format": self.EXPORT_FORMAT,
                        "version": self.EXPORT_VERSION,
                        "exported_at": _now_iso(),
                    },
                    ensure_ascii=False,
                ),
            )
            z.write(self._file(sid), "session.json")
            tl = self._timeline_file(sid)
            if tl.exists():
                z.write(tl, "timeline.jsonl")
        return buf.getvalue()

    def import_zip(self, data: bytes) -> Session:
        """Recrée une session depuis un export, sous un id NEUF, et l'active.

        ValueError ACTIONNABLE (affichée telle quelle à l'utilisateur) si l'archive
        n'est pas un export de session Loom. Seuls les membres CONNUS sont lus —
        aucun chemin du zip n'est suivi sur disque (pas de traversal possible)."""
        import io
        import zipfile

        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ValueError(
                "archive illisible — un export .zip de Loom est attendu"
            ) from exc
        if sum(i.file_size for i in z.infolist()) > self.MAX_IMPORT_BYTES:
            raise ValueError("archive anormalement grosse — import refusé")
        names = set(z.namelist())
        if "session.json" not in names:
            raise ValueError(
                "archive sans session.json — ce n'est pas un export de session Loom"
            )
        try:
            payload = json.loads(z.read("session.json").decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError
            sess = Session.from_dict(payload, self.default_system_prompt)
        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            UnicodeDecodeError,
        ) as exc:
            # TypeError/AttributeError : JSON valide mais mal typé (conversation/messages
            # null) -> rejet PROPRE (400) au lieu d'une 500 non maîtrisée.
            raise ValueError("session.json illisible dans l'archive") from exc
        sess.id = uuid.uuid4().hex[:12]
        if self._known_models and sess.conversation.model not in self._known_models:
            # Machine différente : le modèle d'origine n'existe pas ici.
            sess.conversation.set_model(self.default_model or "")
        self.save(sess)
        if "timeline.jsonl" in names:
            self.session_dir(sess.id).joinpath("timeline.jsonl").write_bytes(
                z.read("timeline.jsonl")
            )
        self.set_active(sess.id)
        return sess

    def load(self, sid: str) -> Session | None:
        if self._safe_dir(sid) is None:
            return None  # id invalide : ne lit jamais un session.json hors racine
        f = self._file(sid)
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            sess = Session.from_dict(data, self.default_system_prompt)
            # Le NOM DE DOSSIER (sid, déjà validé par _safe_dir) fait foi, PAS le champ
            # id du fichier : un session.json au contenu forgé ("id": "../victim") ne
            # doit pas contaminer sess.id (sinon save/session_dir ressortiraient de root).
            sess.id = sid
            return sess
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
        d = self._safe_dir(sid)
        if d is None:
            return  # id invalide : aucune suppression (jamais rmtree hors racine)
        shutil.rmtree(d, ignore_errors=True)
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
