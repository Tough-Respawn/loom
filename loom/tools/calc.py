# loom/tools/calc.py
"""Outil calculate : arithmétique EXACTE, parce qu'un LLM calcule « de tête » par
prédiction de tokens et se trompe (vécu 2026-07-10 : somme fausse dans une analyse
financière). Même logique que format_code : plutôt que d'espérer un `python -c`
correct via run_shell (quoting PowerShell, garde de permission pour une simple
addition), un outil DÉDIÉ, sûr et nommé — le nom est l'affordance.

Évaluateur AST STRICT : nombres, opérateurs arithmétiques, parenthèses et une
poignée de fonctions math whitelistées. Aucun accès aux noms/attributs/appels hors
liste -> rien d'autre n'est exécutable, d'où : pas de garde de permission.
"""

from __future__ import annotations

import ast
import math

from loom.tools.base import ToolError, ToolSpec

# Fonctions autorisées (whitelist STRICTE — tout le reste est refusé).
_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
}

_CONSTS = {"pi": math.pi, "e": math.e}

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_UNARY = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError(f"constante non numérique : {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ToolError(
            f"nom inconnu : {node.id} (constantes admises : {', '.join(_CONSTS)})"
        )
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
        name = getattr(node.func, "id", "?")
        raise ToolError(
            f"fonction non autorisée : {name} (admises : {', '.join(sorted(_FUNCS))})"
        )
    raise ToolError(f"syntaxe non supportée : {type(node).__name__}")


def _fmt(value) -> str:
    """Résultat exact d'abord ; arrondi lisible en plus quand le float est verbeux."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = repr(value)
    if isinstance(value, float) and len(text) > 12:
        # ASCII uniquement (consoles Windows cp1252 + préférence style du projet).
        return f"{text} (~ {value:,.4f})".replace(",", " ")
    return text


def calculate(expression: str) -> str:
    expression = (expression or "").strip()
    if not expression:
        raise ToolError("argument 'expression' manquant")
    if len(expression) > 2000:
        raise ToolError("expression trop longue (2000 caractères max)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"expression invalide : {exc.msg}") from exc
    try:
        result = _eval_node(tree)
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("division par zéro") from exc
    except (OverflowError, ValueError) as exc:
        raise ToolError(f"calcul impossible : {exc}") from exc
    return f"{expression} = {_fmt(result)}"


def make_calculate() -> ToolSpec:
    return ToolSpec(
        name="calculate",
        description=(
            "Evaluates an arithmetic expression EXACTLY and deterministically. ALWAYS "
            "use it for ANY calculation beyond a single trivial operation — especially "
            "money, percentages, totals, VAT — and NEVER compute multi-step arithmetic "
            "in your head: language models routinely get sums wrong. Supports + - * / "
            "// % ** parentheses, and abs, round, min, max, sqrt, floor, ceil, log, "
            "log10, log2, exp, pi, e. Example: (55000-600)*0.2"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression to evaluate, e.g. (55000-600)*0.2",
                }
            },
            "required": ["expression"],
        },
        run=lambda args: calculate(args.get("expression", "")),
    )
