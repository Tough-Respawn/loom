# loom/tools/base.py
"""Cœur des outils : erreur métier, spec, registre, résolution de chemin bornée.

Un outil = un `ToolSpec` (nom, description, schéma JSON des arguments, fonction
`run`). Le `ToolRegistry` expose les schémas au format OpenAI `tools=[...]` et
exécute un appel par nom en transformant toute erreur en message exploitable par
le modèle (jamais d'exception qui casserait la boucle de streaming).

C'est le socle commun : read/document/image (read.py), localisation (search.py),
write/edit (fs.py), run_shell (shell.py), web (web.py), todos (todo.py) et
dispatch_agent (agent.py) s'enregistrent dessus sans toucher au transport.
"""

from __future__ import annotations

import difflib
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loom.runtime.models_profile import Profile


class ToolError(Exception):
    """Erreur métier d'un outil, message en français montrable au modèle."""


# --- Frontière d'entrée des outils : coercition tolérante + erreurs actionnables ------
# Un petit modèle (4B) génère souvent un input mal typé ou mal nommé. Plutôt que de
# laisser CHAQUE outil revérifier à la main (et rendre une erreur sèche que le modèle
# réémet à l'identique → boucle), on normalise ICI, une fois, au niveau du registre :
#   - coercition des fautes de type courantes ("5"→5, "true"→true) d'après le JSON Schema ;
#   - sur un champ requis ABSENT, une erreur qui NOMME le champ et, si un champ inconnu
#     proche a été fourni, suggère le renommage (file→path).
# On ne touche PAS aux strings ni aux champs présents-mais-vides : les outils gardent
# leurs propres vérifs sémantiques (path vide, plage de lignes, etc.) en défense en
# profondeur. La coercition ne fait que RAPPROCHER l'input du schéma, jamais l'inverser.
_TRUE_TOKENS = frozenset({"true", "1", "yes", "oui", "vrai", "on"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "non", "faux", "off"})


def _coerce_scalar(value: object, jtype: str) -> object:
    """Rapproche `value` du type JSON Schema attendu. Renvoie value inchangé si la
    coercition n'est pas évidente (l'outil ou le check requis tranchera ensuite)."""
    if jtype == "integer":
        if isinstance(value, bool) or isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                try:
                    f = float(value.strip())
                    return int(f) if f.is_integer() else value
                except ValueError:
                    return value
        return value
    if jtype == "number":
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
        return value
    if jtype == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in _TRUE_TOKENS:
                return True
            if s in _FALSE_TOKENS:
                return False
        return value
    if jtype == "string":
        # Un scalaire isolé (nombre/booléen) là où une string est attendue : on
        # stringifie (ex. path renvoyé comme nombre).
        if isinstance(value, (int, float, bool)):
            return str(value)
        # Liste de scalaires là où une string est attendue (ex. command=["git","status"]
        # au lieu de "git status") : on joint par espace, la faute argv->ligne de commande.
        if isinstance(value, list) and all(
            isinstance(x, (str, int, float, bool)) for x in value
        ):
            return " ".join(str(x) for x in value)
        return value
    if jtype == "array":
        # Liste sérialisée en JSON-string ("[1,2,3]") : un petit modèle le fait souvent.
        # On PARSE avant d'envelopper — sinon "[1,2,3]" devenait ['[1,2,3]'] (corruption
        # silencieuse). On n'enveloppe qu'un vrai scalaire seul.
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, list):
                    return parsed
        if value is not None and not isinstance(value, list):
            return [value]
        return value
    if jtype == "object":
        # Objet sérialisé en JSON-string ('{"a":1}') là où un dict est attendu (ex.
        # calculate `where`) : on parse, sinon l'outil recevait une str et plantait sur
        # `.get` avec un message opaque.
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("{"):
                try:
                    parsed = json.loads(s)
                except (ValueError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
        return value
    return value


def coerce_enum(value: object, allowed: list, aliases: dict | None = None) -> object:
    """Rapproche une valeur d'enum du jeu autorisé SANS jamais deviner une intention :
    on ne renvoie QUE `value` inchangé ou une valeur RÉELLEMENT présente dans `allowed`.
    Normalisation mécanique (sûre) : casse, tiret/espace -> underscore. Puis table
    d'alias explicite (synonymes métier : 'completed'->'done'). Si rien ne matche, on
    laisse `value` : la vérif sémantique de l'outil rendra son erreur actionnable.
    C'est la généralisation de la leçon `^`->`**` : accepter les écritures équivalentes."""
    if not isinstance(value, str) or value in allowed:
        return value
    norm = value.strip().lower().replace("-", "_").replace(" ", "_")
    for a in allowed:
        if isinstance(a, str) and a.lower() == norm:
            return a
    if aliases:
        cand = (
            aliases.get(value)
            or aliases.get(value.strip().lower())
            or aliases.get(norm)
        )
        if cand in allowed:
            return cand
    return value


def validate_and_coerce(name: str, schema: dict, args: dict) -> dict:
    """Normalise les arguments d'un appel d'outil avant exécution. Lève `ToolError`
    avec un message ACTIONNABLE si un argument requis est absent."""
    if not isinstance(args, dict):
        raise ToolError(
            f"arguments de '{name}' invalides (objet JSON attendu). Réémets l'appel."
        )
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []
    coerced: dict = {}
    for key, value in args.items():
        spec = props.get(key)
        jtype = spec.get("type") if isinstance(spec, dict) else None
        v = _coerce_scalar(value, jtype) if jtype else value
        # Enum top-level : normalise casse/tiret + alias déclarés dans le schéma
        # (`x_aliases`). Un enum imbriqué (items d'un array, ex. todo.status) est
        # traité par l'outil via coerce_enum — le schéma top-level ne le voit pas.
        if isinstance(spec, dict) and spec.get("enum"):
            v = coerce_enum(v, spec["enum"], spec.get("x_aliases"))
        coerced[key] = v
    # Requis ABSENT (clé manquante ou null) : on nomme le champ, on suggère un renommage
    # si une clé inconnue PROCHE a été fournie (typo), et on liste les champs non reconnus
    # restants (cas fréquent du 4B : 'file' au lieu de 'path', clés inventées).
    missing = [r for r in required if args.get(r) is None]
    if missing:
        unknown = [k for k in args if k not in props]
        matched: set[str] = set()
        hints: list[str] = []
        for r in missing:
            pool = [u for u in unknown if u not in matched]
            near = difflib.get_close_matches(r, pool, n=1, cutoff=0.6)
            if near:
                matched.add(near[0])
                hints.append(f"'{r}' manquant (tu as envoyé '{near[0]}' ? renomme-le)")
            else:
                jtype = props.get(r, {}).get("type", "valeur")
                hints.append(f"'{r}' manquant ({jtype} attendu)")
        leftover = [u for u in unknown if u not in matched]
        extra = (
            f" Champs non reconnus fournis : {', '.join(leftover)}." if leftover else ""
        )
        raise ToolError(
            f"arguments de '{name}' invalides : {' ; '.join(hints)}.{extra} "
            "Réémets l'appel corrigé (ne répète pas l'appel à l'identique)."
        )
    return coerced


# Univers des outils proposés dans l'UI (activables par conversation). `danger`
# marque ceux qui modifient le système (gardés par le mode permission).
AVAILABLE_TOOLS = [
    {"name": "find_files", "label": "find_files", "danger": False},
    {"name": "search_text", "label": "search_text", "danger": False},
    {"name": "list_dir", "label": "list_dir", "danger": False},
    {"name": "read_file", "label": "read_file", "danger": False},
    {"name": "read_image", "label": "read_image", "danger": False},
    {"name": "calculate", "label": "calculate", "danger": False},
    {"name": "current_date", "label": "current_date", "danger": False},
    {"name": "web_search", "label": "web_search", "danger": False},
    {"name": "fetch_url", "label": "fetch_url", "danger": False},
    {"name": "check_page", "label": "check_page", "danger": False},
    {"name": "serve_and_check", "label": "serve_and_check", "danger": True},
    {"name": "dispatch_agent", "label": "dispatch_agent", "danger": False},
    {"name": "run_workflow", "label": "run_workflow", "danger": True},
    {"name": "manage_todos", "label": "manage_todos", "danger": False},
    {"name": "write_note", "label": "write_note", "danger": False},
    {"name": "read_note", "label": "read_note", "danger": False},
    {"name": "recall", "label": "recall", "danger": False},
    {"name": "remember", "label": "remember", "danger": False},
    {"name": "use_skill", "label": "use_skill", "danger": False},
    {"name": "list_plugins", "label": "list_plugins", "danger": False},
    {"name": "add_marketplace", "label": "add_marketplace", "danger": True},
    {"name": "install_plugin", "label": "install_plugin", "danger": True},
    {"name": "write_file", "label": "write_file", "danger": True},
    {"name": "append_file", "label": "append_file", "danger": True},
    {"name": "edit_file", "label": "edit_file", "danger": True},
    {"name": "format_code", "label": "format_code", "danger": True},
    {"name": "run_shell", "label": "run_shell", "danger": True},
]


def _strip_x_keys(obj):
    """Copie d'un fragment de schéma sans les clés x_* (métadonnées internes,
    jamais envoyées au modèle). Récursif, ne mute pas l'original."""
    if isinstance(obj, dict):
        return {k: _strip_x_keys(v) for k, v in obj.items() if not k.startswith("x_")}
    if isinstance(obj, list):
        return [_strip_x_keys(v) for v in obj]
    return obj


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema des arguments
    run: Callable[[dict], str]
    # Outil STREAMANT (optionnel) : au lieu de rendre une str d'un bloc, il yield les
    # events de sa propre activité (mêmes tuples que stream_chat_tools). La boucle les
    # relaie à l'UI EN DIRECT et reconstruit le résultat final. Sert à `dispatch_agent`
    # pour qu'on VOIE ce que fait le sous-agent. `run` reste le repli (1 bloc).
    run_stream: Callable[[dict], Iterator[tuple[str, object]]] | None = None

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                # Les clés x_* (métadonnées internes, ex. x_aliases de coercition)
                # ne partent pas au modèle : bruit de schéma sans valeur d'usage.
                "parameters": _strip_x_keys(self.parameters),
            },
        }


class ToolRegistry:
    """Collection d'outils : expose les schémas et exécute par nom (sans lever)."""

    def __init__(self, specs: list[ToolSpec], profile: "Profile | None" = None) -> None:
        self._specs = {s.name: s for s in specs}
        self._profile = (
            profile  # loom.runtime.models_profile.Profile | None (duck-typed)
        )

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def add(self, spec: ToolSpec) -> None:
        """Ajoute/remplace un outil APRÈS construction. Sert à `submit_result`, dont le
        schéma dépend de l'appel (sortie structurée d'un workflow) et ne peut donc pas
        être déclaré dans build_registry. Le registre d'un sous-agent est refabriqué à
        chaque délégation -> pas d'état partagé entre appels."""
        self._specs[spec.name] = spec

    def openai_tools(self) -> list[dict]:
        return [s.to_openai() for s in self._specs.values()]

    def _unknown_tool(self, name: str) -> str:
        """Message d'outil inconnu RÉCUPÉRABLE : liste les outils réels et suggère le
        plus proche (le modèle hallucine parfois un nom — `grep`, `read`, `bash`…)."""
        available = list(self._specs)
        near = difflib.get_close_matches(name, available, n=1, cutoff=0.5)
        sugg = f" Tu voulais dire '{near[0]}' ?" if near else ""
        return (
            f"erreur: outil inconnu '{name}'.{sugg} "
            f"Outils disponibles : {', '.join(available)}."
        )

    def run(self, name: str, args: dict) -> str:
        spec = self._specs.get(name)
        if spec is None:
            return self._unknown_tool(name)
        try:
            args = validate_and_coerce(name, spec.parameters, args)
            if self._profile is not None:
                p = args.get("path")
                suffix = Path(p).suffix if isinstance(p, str) else ""
                args = self._profile.apply(name, args, suffix)
            return spec.run(args)
        except ToolError as exc:
            return f"erreur: {exc}"
        except Exception as exc:  # noqa: BLE001 - on ne casse jamais la boucle
            return f"erreur inattendue: {exc}"

    def is_streaming(self, name: str) -> bool:
        """Vrai si l'outil expose une exécution STREAMANTE (run_stream)."""
        spec = self._specs.get(name)
        return bool(spec and spec.run_stream)

    def run_stream(self, name: str, args: dict) -> Iterator[tuple[str, object]]:
        """Exécute un outil streamant en relayant ses events ; ne lève jamais (toute
        erreur devient un event ('content', 'erreur: …') que la boucle traite comme
        un résultat d'outil en échec)."""
        spec = self._specs.get(name)
        if spec is None or spec.run_stream is None:
            yield ("content", f"erreur: outil non streamant '{name}'")
            return
        try:
            args = validate_and_coerce(name, spec.parameters, args)
            yield from spec.run_stream(args)
        except ToolError as exc:
            yield ("content", f"erreur: {exc}")
        except Exception as exc:  # noqa: BLE001 - on ne casse jamais la boucle
            yield ("content", f"erreur inattendue: {exc}")


def _resolve_in_root(root: Path, rel: str) -> Path:
    """Résout un chemin d'outil. PLUS DE CONFINEMENT (Loom agit sur tout le système,
    comme un agent généraliste) :
    - chemin ABSOLU (ex. `C:/Users/.../x`, `/home/.../x`) -> utilisé tel quel ;
    - chemin RELATIF -> résolu sous `root`, qui n'est qu'un DOSSIER DE TRAVAIL par défaut.

    Le garde-fou n'est plus le périmètre mais la deny-list dure de loom.permissions
    (rm -rf, format, …), incontournable même ici.
    """
    # Nettoyage des écritures fréquentes d'un modèle : guillemets englobants (copiés
    # d'une commande shell) et `~` (home). Sans ça, `"C:/x"` et `~/x` étaient résolus
    # littéralement sous root -> « fichier introuvable » alors que le chemin était valide.
    rel = rel.strip()
    if len(rel) >= 2 and rel[0] == rel[-1] and rel[0] in ("'", '"'):
        rel = rel[1:-1].strip()
    rel = os.path.expanduser(rel)
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    return (Path(root).resolve() / p).resolve()
