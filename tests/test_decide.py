"""Décision fermée par lecture des logits (loom/agent/decide.py) : parties PURES.

Un faux tokeniseur (caractère = token, sauf digrammes fusionnés déclarés) reproduit le
piège réel : `": ` + `true` ne se tokenise pas comme `":` + ` true`. La coupe doit se
faire là où les OPTIONS divergent, jamais là où l'humain couperait.
"""

from __future__ import annotations

from loom.agent.decide import (
    Field,
    OptionTree,
    build_option_tree,
    normalize_probs,
)


def _tok(merges: dict[str, str]):
    """Tokeniseur jouet : chaque caractère est un token, sauf les paires listées."""

    def tokenize(text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            pair = text[i : i + 2]
            if pair in merges:
                out.append(hash(merges[pair]) & 0xFFFF)
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    return tokenize


def test_tree_cuts_where_options_diverge_not_at_the_colon():
    # Piège Codacus : `: t` fusionne. La coupe est APRÈS le préfixe commun des options.
    tok = _tok({": ": "colon-space"})
    f = Field("urgent", "boolean", ["true", "false"])
    tree = build_option_tree(tok, "x: ", f.choices)
    # Préfixe commun = tokens de 'x' + 'colon-space' ; puis 't' vs 'f' divergent.
    assert tree.prefix == tok("x: ")
    assert set(tree.children) == {ord("t"), ord("f")}
    assert tree.children[ord("t")].leaf == "true"
    assert tree.children[ord("f")].leaf == "false"


def test_tree_recurses_when_two_options_share_first_tokens():
    tok = _tok({})
    f = Field("mood", "enum", ["very bad", "very good", "ok"])
    tree = build_option_tree(tok, "m=", f.choices)
    assert set(tree.children) == {ord("v"), ord("o")}
    assert tree.children[ord("o")].leaf == "ok"
    sub: OptionTree = tree.children[ord("v")]
    assert sub.leaf is None  # nœud interne : il faudra une 2e lecture
    # Descente : 'ery ' commun, puis 'b' vs 'g'.
    node = sub
    while node.leaf is None and len(node.children) == 1:
        node = next(iter(node.children.values()))
    assert set(node.children) == {ord("b"), ord("g")}


def test_normalize_probs_renormalizes_over_options_and_reports_coverage():
    raw = {"a": 0.3, "b": 0.1}  # le reste (0.6) est parti ailleurs
    probs, coverage = normalize_probs(raw)
    assert abs(probs["a"] - 0.75) < 1e-9
    assert abs(probs["b"] - 0.25) < 1e-9
    assert abs(coverage - 0.4) < 1e-9


def test_normalize_probs_all_zero_is_uniform_with_zero_coverage():
    probs, coverage = normalize_probs({"a": 0.0, "b": 0.0})
    assert probs == {"a": 0.5, "b": 0.5}
    assert coverage == 0.0


def test_boolean_renderings_cover_bare_and_quoted_forms():
    f = Field.boolean("urgent")
    assert f.renderings("true") == ["true", '"true"']
    assert Field.enum("c", ["a"]).renderings("a") == ['"a"']
