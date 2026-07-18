# loom/setup/report.py
"""Bilan de l'installeur : accumulation des résultats d'étapes et rendu final.

Statuts : ok (déjà en place), fait (installé ce run), ignore (refusé/reporté),
echec (tenté mais raté), manuel (guidage donné, action à faire à la main)."""

from __future__ import annotations

from loom.setup.steps import StepOutcome

_ICONS = {"ok": "✅", "fait": "✅", "ignore": "⏭️", "echec": "❌", "manuel": "🔧"}
_LABELS = {
    "detection": "Détection",
    "binaire": "Binaire",
    "modele": "Modèle",
    "bench": "Réglages",
}


class SetupReport:
    def __init__(self) -> None:
        self.outcomes: list[StepOutcome] = []

    def add(self, name: str, status: str, detail: str) -> StepOutcome:
        out = StepOutcome(name, status, detail)
        self.outcomes.append(out)
        return out

    @property
    def failed(self) -> bool:
        return any(o.status == "echec" for o in self.outcomes)

    def render(self) -> str:
        lines = ["", "── Bilan ─────────────────────────────────────────────"]
        for o in self.outcomes:
            icon = _ICONS.get(o.status, "•")
            label = _LABELS.get(o.name, o.name).ljust(10)
            lines.append(f"  {icon} {label} {o.detail}")
        return "\n".join(lines)
