# loom/lessons.py
"""Auto-amélioration : Loom retient ses erreurs en LEÇONS persistées et les réinjecte.

Le vérificateur déterministe est un prof fiable : il dit « cassé, et pourquoi ». Au lieu
de jeter ce signal à la fin de chaque run, on en distille UNE leçon générale (via le
modèle) et on la conserve globalement ; les runs suivants la reçoivent dans le prompt de
génération. C'est le mécanisme qui rend le petit modèle plus fiable dans le temps SANS
qu'on code chaque cas en dur."""

from __future__ import annotations

import json
import os
from pathlib import Path

from loom.prompts import LESSON_SYSTEM, lesson_prompt


class LessonStore:
    """Liste de leçons persistée (JSON), dédupliquée (insensible à la casse) et bornée."""

    def __init__(self, path, *, cap: int = 50) -> None:
        self.path = Path(path)
        self.cap = cap
        self._lessons: list[str] = self._load()

    def _load(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [str(x) for x in data] if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def add(self, lesson: str) -> None:
        lesson = (lesson or "").strip()
        if not lesson:
            return
        low = lesson.lower()
        if any(low == x.lower() for x in self._lessons):
            return  # dédup exact (insensible à la casse)
        self._lessons.append(lesson)
        self._lessons = self._lessons[-self.cap :]
        self._save()

    def recent(self, n: int = 8) -> list[str]:
        return self._lessons[-n:]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(self._lessons, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self.path)


def distill_lesson(client, defects_text: str, task: str, *, model: str | None) -> str:
    """Distille UNE leçon générale à partir des défauts rencontrés (vide si pas de défaut
    ou si le modèle ne dégage rien de généralisable). Ne lève jamais."""
    if not (defects_text or "").strip():
        return ""
    try:
        raw = client.complete(
            [{"role": "user", "content": lesson_prompt(defects_text, task)}],
            LESSON_SYSTEM,
            max_tokens=256,
            model=model,
            thinking=False,
        )
    except Exception:  # noqa: BLE001 - apprendre ne doit jamais casser le run
        return ""
    return (raw or "").strip()
