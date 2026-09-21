"""Décision fermée par lecture des logits (mode « Jev » NATIF, llama-server stock).

Principe (Harsha Gondala puis Codacus, 2026-09) : pour une question fermée, on ne GÉNÈRE
pas la réponse, on LIT la distribution du prochain token et on ne garde que les options du
schéma. Ici sans fork : une requête `/completion` par nœud (n_predict=1, n_probs) sur le
slot demandé, préfixe (instructions + catalogue + contexte) en cache. On lit les
probabilités PRÉ-sampling (softmax sur tout le vocabulaire : indépendantes de la grammaire
et des samplers, cf. piège vérifié dans sampling.cpp b9442 avec grammar_first=false) et on
les renormalise sur les options.

Garanties : la valeur est TOUJOURS une option du schéma. La probabilité n'est PAS calibrée
(aucun entraînement RLCD) : signal relatif, à valider sur cas annotés. Les champs ne se
voient pas entre eux : questions indépendantes seulement.

La coupe entre préfixe et options se fait là où les TOKENS des options divergent (jamais
« après le deux-points » : `": ` + `true` ne se tokenise pas comme `":` + ` true`). Deux
options qui partagent leurs premiers tokens (« very bad » / « very good ») coûtent une
lecture de plus, à l'embranchement suivant.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

Tokenize = Callable[[str], list[int]]


@dataclass(frozen=True)
class Field:
    """Une question fermée : `choices` sont les textes EXACTS scorés après le préfixe."""

    name: str
    type: str  # enum | boolean | integer
    choices: list[str]
    description: str = ""

    @staticmethod
    def enum(name: str, choices: list[str], description: str = "") -> Field:
        return Field(name, "enum", list(choices), description)

    @staticmethod
    def boolean(name: str, description: str = "") -> Field:
        return Field(name, "boolean", ["true", "false"], description)

    @staticmethod
    def integer(name: str, lo: int, hi: int, description: str = "") -> Field:
        return Field(name, "integer", [str(i) for i in range(lo, hi + 1)], description)

    def renderings(self, choice: str) -> list[str]:
        """Textes qu'un modèle peut écrire pour cette option dans le JSON, sommés par
        valeur : les enums sont cités (le guillemet fermant sert de terminateur : « ok »
        vs « okay » divergent alors sûrement) ; booléens et entiers existent nus ET
        cités (mesuré sur Gemma 4 E4B : après `":` il pose ` "` dans 15 à 99 % des cas
        selon le champ, la forme nue seule perdait cette masse)."""
        quoted = json.dumps(choice)
        return [quoted] if self.type == "enum" else [choice, quoted]

    def cast(self, choice: str):
        if self.type == "boolean":
            return choice == "true"
        if self.type == "integer":
            return int(choice)
        return choice


@dataclass
class OptionTree:
    """Nœud : `prefix` = tokens à envoyer ; `children` = token suivant -> sous-arbre ;
    `leaf` = l'option (texte de `choices`) quand une seule survit."""

    prefix: list[int]
    children: dict[int, OptionTree] = field(default_factory=dict)
    leaf: str | None = None


def _common_prefix(paths: list[list[int]]) -> int:
    n = min(len(p) for p in paths)
    i = 0
    while i < n and all(p[i] == paths[0][i] for p in paths):
        i += 1
    return i


def _build(paths: dict[str, list[int]]) -> OptionTree:
    if len(paths) == 1:
        ((choice, toks),) = paths.items()
        return OptionTree(prefix=toks, leaf=choice)
    cut = _common_prefix(list(paths.values()))
    node = OptionTree(prefix=paths[next(iter(paths))][:cut])
    groups: dict[int, dict[str, list[int]]] = {}
    for choice, toks in paths.items():
        if len(toks) <= cut:
            raise ValueError(
                f"option {choice!r} est un préfixe (en tokens) d'une autre option : "
                "ajoute un terminateur (guillemet fermant) pour les distinguer"
            )
        groups.setdefault(toks[cut], {})[choice] = toks
    for tok, sub in groups.items():
        node.children[tok] = _build(sub)
    return node


def build_option_tree(tokenize: Tokenize, prompt: str, texts: list[str]) -> OptionTree:
    """Arbre de divergence des `texts` (rendus d'options) concaténés à `prompt`."""
    return _build({t: tokenize(prompt + t) for t in texts})


def normalize_probs(raw: dict[str, float]) -> tuple[dict[str, float], float]:
    """Renormalise sur les options. `coverage` = masse brute que les options couvraient
    (1.0 = le modèle ne voulait rien dire d'autre ; proche de 0 = il voulait répondre
    hors schéma : la valeur reste dans le schéma mais le signal est faible)."""
    total = sum(raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}, 0.0
    return {k: v / total for k, v in raw.items()}, total


@dataclass
class Decision:
    values: dict[str, object]
    probs: dict[str, dict[str, float]]  # champ -> option -> p renormalisée
    coverage: dict[str, float]  # champ -> masse brute couverte à la racine
    requests: int = 0
    ms: float = 0.0  # mur, tout compris
    server_ms: float = 0.0  # prompt_ms + predicted_ms rapportés par llama-server
    tokenize_ms: float = 0.0  # appels /tokenize (construction des arbres)
    prompt_tokens: int = 0


_CATALOGUE = (
    "{instructions}\n\n"
    "Answer each question below about the CONTEXT. Reply with a JSON object whose keys "
    "are the question names; each value must be exactly one of the allowed values.\n\n"
    "Questions:\n{questions}\n\nCONTEXT:\n{context}"
)


def render_prompt(instructions: str, schema: list[Field], context: str) -> str:
    qs = "\n".join(
        f"- {f.name}: {f.description or f.name} [{' | '.join(f.choices)}]"
        for f in schema
    )
    return _CATALOGUE.format(instructions=instructions, questions=qs, context=context)


class Decider:
    """Client du mode décision sur UN llama-server. Réutilise une connexion httpx."""

    def __init__(
        self,
        base_url: str,
        id_slot: int | None = None,
        n_probs: int = 64,
        timeout: float = 600.0,
        min_branch_p: float = 0.05,
    ):
        self.base = base_url.rstrip("/")
        self.id_slot = id_slot
        self.n_probs = n_probs
        # ponytail: une branche interne (ex. forme citée `"true"`) sous ce seuil n'est
        # pas descendue (ses feuilles restent à 0) : 1 requête de moins par champ quand
        # le modèle a une préférence nette ; la masse ignorée se voit dans `coverage`.
        self.min_branch_p = min_branch_p
        self.http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.http.close()

    # -- primitives serveur ---------------------------------------------------------
    def tokenize(self, text: str) -> list[int]:
        r = self.http.post(
            f"{self.base}/tokenize",
            json={"content": text, "add_special": False, "parse_special": True},
        )
        r.raise_for_status()
        return r.json()["tokens"]

    def apply_template(self, user: str) -> str:
        """Le texte du tour utilisateur passé au template du modèle (BOS, balises),
        prêt pour la continuation assistant : obligatoire pour un modèle instruct."""
        r = self.http.post(
            f"{self.base}/apply-template",
            json={"messages": [{"role": "user", "content": user}]},
        )
        r.raise_for_status()
        return r.json()["prompt"]

    def next_token_probs(
        self, prefix: list[int]
    ) -> tuple[dict[int, float], int, float]:
        """Probabilités PRÉ-sampling des `n_probs` tokens les plus probables après
        `prefix`, nombre de tokens de prompt réellement traités (cache déduit) et temps
        serveur (ms)."""
        body = {
            "prompt": prefix,
            "n_predict": 1,
            "n_probs": self.n_probs,
            "post_sampling_probs": False,
            "cache_prompt": True,
            "temperature": 0.0,
        }
        if self.id_slot is not None:
            body["id_slot"] = self.id_slot
        r = self.http.post(f"{self.base}/completion", json=body)
        r.raise_for_status()
        d = r.json()
        # Pré-sampling, llama-server nomme la liste `top_logprobs` (et `top_probs`
        # en post-sampling) : accepter les deux, la valeur est prob ou logprob.
        first = (d.get("completion_probabilities") or [{}])[0]
        top = first.get("top_logprobs") or first.get("top_probs") or []
        probs = {
            t["id"]: (t["prob"] if "prob" in t else math.exp(t["logprob"])) for t in top
        }
        timings = d.get("timings") or {}
        server_ms = float(timings.get("prompt_ms") or 0) + float(
            timings.get("predicted_ms") or 0
        )
        return probs, int(timings.get("prompt_n") or 0), server_ms

    # -- scoring ---------------------------------------------------------------------
    def _score(
        self, node: OptionTree, out: dict[str, float], scale: float, stats: dict
    ):
        probs, n_prompt, server_ms = self.next_token_probs(node.prefix)
        stats["requests"] += 1
        stats["prompt_tokens"] += n_prompt
        stats["server_ms"] += server_ms
        if stats["requests"] == 1:
            stats["coverage"] = sum(probs.get(t, 0.0) for t in node.children)
        for tok, child in node.children.items():
            p = probs.get(tok, 0.0) * scale
            if child.leaf is not None:
                out[child.leaf] = p
            elif p >= self.min_branch_p:
                self._score(child, out, p, stats)
            else:
                # Branche morte ou négligeable : ses feuilles restent dans le schéma, à 0.
                for leaf in _leaves(child):
                    out[leaf] = 0.0

    def decide(self, instructions: str, schema: list[Field], context: str) -> Decision:
        t0 = time.perf_counter()
        prefix_text = self.apply_template(render_prompt(instructions, schema, context))
        res = Decision(values={}, probs={}, coverage={})
        for fld in schema:
            stub = prefix_text + "{" + json.dumps(fld.name) + ": "
            by_text = {t: c for c in fld.choices for t in fld.renderings(c)}
            t1 = time.perf_counter()
            tree = build_option_tree(self.tokenize, stub, list(by_text))
            res.tokenize_ms += (time.perf_counter() - t1) * 1000
            raw_text: dict[str, float] = {}
            stats = {
                "requests": 0,
                "prompt_tokens": 0,
                "coverage": 0.0,
                "server_ms": 0.0,
            }
            if tree.leaf is not None:  # une seule option : rien à lire
                raw_text[tree.leaf] = 1.0
            else:
                self._score(tree, raw_text, 1.0, stats)
            raw: dict[str, float] = {c: 0.0 for c in fld.choices}
            for t, p in raw_text.items():
                raw[by_text[t]] += p
            probs, _ = normalize_probs(raw)
            best = max(probs, key=probs.get)
            res.values[fld.name] = fld.cast(best)
            res.probs[fld.name] = probs
            res.coverage[fld.name] = stats["coverage"]
            res.requests += stats["requests"]
            res.prompt_tokens += stats["prompt_tokens"]
            res.server_ms += stats["server_ms"]
        res.ms = (time.perf_counter() - t0) * 1000
        return res


def _leaves(node: OptionTree) -> list[str]:
    if node.leaf is not None:
        return [node.leaf]
    return [leaf for c in node.children.values() for leaf in _leaves(c)]
