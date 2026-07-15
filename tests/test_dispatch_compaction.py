# Régression session 2026-07-14 (3b630bc667ac) : les sous-agents dispatch_agent
# tournaient SANS compaction -> saturation de la fenêtre (usage prompt=24447
# completion=129 total=24576), tool calls coupés en génération, 52 × « arguments
# tronqués » en boucle avec re-prefill 24k à chaque retry. Deux gardes :
# (1) dispatch_agent transmet le seuil de compaction à sa sous-boucle ;
# (2) des troncatures d'arguments CONSÉCUTIVES déclenchent une compaction forcée
#     avant l'appel suivant, même sans seuil configuré.
from __future__ import annotations

from .fakes import FakeRegistry, collect, make_client, only, turn_text, turn_tools

USER = [{"role": "user", "content": "fais le travail"}]
SYSTEM = "tu es un agent de test"


class _RecClient:
    """Client duck-typé minimal pour make_dispatch_agent : enregistre les kwargs
    que la sous-boucle passe à stream_chat_tools."""

    def __init__(self):
        self.kwargs = None

    def is_remote(self, model):
        return False

    def save_slot(self, model, name):
        return False

    def restore_slot(self, model, name):
        pass

    def stream_chat_tools(self, messages, system_prompt, max_tokens, **kw):
        self.kwargs = {"max_tokens": max_tokens, **kw}
        yield ("content", "synthèse")


# ---------- (1) le seuil de compaction atteint la sous-boucle ----------


def test_dispatch_transmet_compact_after_tokens():
    from loom.tools.agent import make_dispatch_agent

    client = _RecClient()
    spec = make_dispatch_agent(
        client,
        lambda: FakeRegistry(),
        system_prompt="s",
        compact_after_tokens=999,
    )
    assert spec.run({"task": "tâche autonome"}) == "synthèse"
    assert client.kwargs["compact_after_tokens"] == 999


def test_build_registry_cable_le_seuil_sous_agent():
    from loom.tools import build_registry

    client = _RecClient()
    reg = build_registry(
        workspace_dir=".",
        max_bytes=1000,
        enabled=["dispatch_agent"],
        client=client,
        sub_compact_after_tokens=777,
    )
    out = reg.run("dispatch_agent", {"task": "tâche autonome"})
    assert out == "synthèse"
    assert client.kwargs["compact_after_tokens"] == 777


# ---------- (2) troncatures consécutives -> compaction forcée ----------


def test_troncatures_consecutives_forcent_compaction():
    # 2 tours de suite avec des arguments d'outil TRONQUÉS (JSON coupé) : au tour
    # suivant, compaction forcée AVANT l'appel modèle, signalée par un tool_result
    # « (compaction sur troncature) » — même sans compact_after_tokens configuré
    # (cas réel : la sous-boucle d'avant le fix n'avait aucun seuil).
    client, fake = make_client(
        [
            turn_tools([("id1", "read_file", '{"path": "C:/tr')]),
            turn_tools([("id2", "read_file", '{"path": "C:/tr')]),
            turn_text("Terminé."),
        ]
    )
    registry = FakeRegistry({"read_file": lambda a: "contenu"})
    events, done = collect(client.stream_chat_tools(USER, SYSTEM, registry=registry))
    assert done["reason"] == "natural"
    assert len(fake.calls) == 3
    names = [p.get("name") for p in only(events, "tool_result")]
    assert "(compaction sur troncature)" in names


def test_troncature_isolee_ne_compacte_pas():
    # UNE troncature suivie d'un appel valide : pas de compaction forcée (le
    # compteur se réinitialise sur un parse réussi).
    client, fake = make_client(
        [
            turn_tools([("id1", "read_file", '{"path": "C:/tr')]),
            turn_tools([("id2", "read_file", '{"path": "ok.txt"}')]),
            turn_tools([("id3", "read_file", '{"path": "C:/tr')]),
            turn_text("Terminé."),
        ]
    )
    registry = FakeRegistry({"read_file": lambda a: "contenu"})
    events, done = collect(client.stream_chat_tools(USER, SYSTEM, registry=registry))
    assert done["reason"] == "natural"
    names = [p.get("name") for p in only(events, "tool_result")]
    assert "(compaction sur troncature)" not in names


def test_sous_agents_sans_outils_plugins():
    # Les outils plugins (setup, danger) ne font pas partie du kit d'un ouvrier :
    # sortis du défaut le 2026-07-15, ils ne doivent pas non plus arriver aux
    # sous-agents via _SUBAGENT_TOOLS (dérivé du catalogue complet).
    from loom.tools import _SUBAGENT_TOOLS

    for name in ("list_plugins", "add_marketplace", "install_plugin"):
        assert name not in _SUBAGENT_TOOLS, name
