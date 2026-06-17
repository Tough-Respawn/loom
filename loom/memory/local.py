"""Provider mémoire `local` : store épisodique en SQLite + FTS5 (full-text, offline).

Une table `episodes` (vérité) + une table virtuelle FTS5 `episodes_fts` synchronisée par
triggers. `remember` insère, `recall` fait un MATCH FTS5 trié par pertinence. Aucune
dépendance externe, aucun embedding, un seul fichier `.db`. Store PUR : pas de Flask, pas
de modèle — testable sur `:memory:`. Best-effort sur lock (log + skip), cf. design §11.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from loom.memory import Snippet

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id     INTEGER PRIMARY KEY,
    ts     TEXT NOT NULL,
    kind   TEXT NOT NULL DEFAULT 'episodic',
    source TEXT NOT NULL DEFAULT '',
    text   TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
    USING fts5(text, content='episodes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO episodes_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

# Garde-fou : un épisode = une leçon dense, pas un dump (design §6.5).
_MAX_TEXT = 4000


def _fts_query(query: str) -> str:
    """Transforme une requête libre en requête FTS5 sûre : chaque token alphanum est mis
    entre guillemets (littéral) puis joint par OR. Le guillemetage NEUTRALISE les mots-clés
    FTS5 (AND/OR/NOT/NEAR) qui, en bareword, feraient planter le MATCH (OperationalError)."""
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '""'


class LocalMemory:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def remember(self, text: str, *, kind: str = "episodic", source: str = "") -> None:
        text = (text or "").strip()[:_MAX_TEXT]
        if not text:
            return
        ts = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO episodes(ts, kind, source, text) VALUES (?, ?, ?, ?)",
                (ts, kind, source, text),
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            # lock/contention : best-effort, on n'interrompt jamais le tour (design §11).
            pass

    def recall(self, query: str, *, k: int = 5) -> list[Snippet]:
        q = _fts_query(query)
        try:
            rows = self._conn.execute(
                "SELECT e.text, e.kind, e.source, bm25(episodes_fts) AS score "
                "FROM episodes_fts JOIN episodes e ON e.id = episodes_fts.rowid "
                "WHERE episodes_fts MATCH ? ORDER BY score LIMIT ?",
                (q, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            Snippet(text=r[0], kind=r[1], source=r[2], score=float(r[3])) for r in rows
        ]
