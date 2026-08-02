"""Outil current_date : la date/heure RÉELLE, calculée à la demande.

Un modèle ne connaît pas la date (coupure d'entraînement) et la confabule si on la
lui demande — « on est quel jour ? », « hier », « d'ici fin juillet » (constaté
2026-07-10). Décision : PAS d'injection dans le system prompt (préfixe stable =
prompt caching, et le déterministe se CALCULE) — un outil, appelé quand la question
se pose, dont le résultat entre naturellement dans le contexte. L'arithmétique de
dates (jours entre deux dates, +N jours à cheval sur les mois) est elle aussi
error-prone de tête -> les deux paramètres optionnels la rendent exacte.
"""

from __future__ import annotations

import datetime

from loom.tools.base import ToolError, ToolSpec


def current_date(plus_days: int | None = None, until: str = "") -> str:
    now = datetime.datetime.now()
    today = now.date()
    lines = [
        f"Now: {today.strftime('%A')} {today.isoformat()}, {now.strftime('%H:%M')} "
        f"(local time, ISO week {today.isocalendar().week})",
        f"Yesterday: {(today - datetime.timedelta(days=1)).isoformat()} | "
        f"Tomorrow: {(today + datetime.timedelta(days=1)).isoformat()}",
    ]
    if plus_days is not None:
        try:
            target = today + datetime.timedelta(days=int(plus_days))
        except (OverflowError, ValueError) as exc:
            raise ToolError(f"plus_days invalide : {exc}") from exc
        lines.append(
            f"Today {int(plus_days):+d} day(s) = {target.strftime('%A')} "
            f"{target.isoformat()}"
        )
    if until:
        try:
            target = datetime.date.fromisoformat(until.strip())
        except ValueError as exc:
            raise ToolError(
                f"`until` invalide : {until!r} (format attendu YYYY-MM-DD)"
            ) from exc
        delta = (target - today).days
        lines.append(
            f"From today until {target.strftime('%A')} {target.isoformat()} = "
            f"{delta} day(s)"
        )
    return "\n".join(lines)


def make_current_date() -> ToolSpec:
    return ToolSpec(
        name="current_date",
        description=(
            "Returns the REAL current date and local time — call it before answering "
            "anything involving today or a relative date; NEVER guess dates from "
            "training data. `plus_days` gives the date N days away; `until` counts "
            "days to a YYYY-MM-DD."
        ),
        parameters={
            "type": "object",
            "properties": {
                "plus_days": {
                    "type": "integer",
                    "description": "Date N days from today (negative = past).",
                },
                "until": {
                    "type": "string",
                    "description": "Count days from today to this YYYY-MM-DD.",
                },
            },
        },
        run=lambda args: current_date(
            plus_days=args.get("plus_days"),
            until=args.get("until", "") or "",
        ),
    )
