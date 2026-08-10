# tests/test_subagent_cancel.py
"""ST-03 : annulation CIBLÉE d'un sous-agent — (session_id, agent_id).

Coopérative (Event observé entre deux événements de la sous-boucle, jamais un
thread tué), idempotente, un SEUL subagent_end même en course avec la fin
naturelle, parent et frères intacts, arbre de processus d'un run_shell tué via
le mécanisme existant. Les tests sont PULL-BASED ou à barrières : aucun timing
probabiliste.
"""

from __future__ import annotations

import threading

from loom.runtime.platform_info import detect
from loom.tools.agent import CANCELLATIONS, SubAgentRunner, make_dispatch_agent
from loom.tools.base import ToolRegistry, ToolSpec
from loom.tools.shell import make_run_shell
from loom.tools.workflow import make_run_workflow

from .fakes import make_client, turn_text, turn_tools
from .test_dispatch_routing import FakeRegistry
from .test_mcp_client import _pid_alive


class _MultiTurnClient:
    """Sous-boucle scriptée multi-événements (2 « tours » puis stop naturel)."""

    def is_remote(self, model):
        return False

    def save_slot(self, *a, **k):
        return False

    def restore_slot(self, *a, **k):
        pass

    def stream_chat_tools(self, *a, **kw):
        yield ("content", "tour 1. ")
        yield ("usage", {"prompt_tokens": 10, "completion_tokens": 1})
        yield ("content", "tour 2. ")
        yield ("usage", {"prompt_tokens": 20, "completion_tokens": 2})
        yield ("done", {"reason": "natural"})


def _runner(client, sid="s1"):
    return SubAgentRunner(
        client, lambda: FakeRegistry(), system_prompt="s", model="m", session_id=sid
    )


def _subevents(events):
    return [(k, p) for k, p in events if k.startswith("subagent_")]


# --- annulation ENTRE deux tours ------------------------------------------------


def test_annulation_entre_deux_tours_pull_based():
    gen = _runner(_MultiTurnClient()).stream("t")
    events = []
    aid = None
    for kind, payload in gen:
        events.append((kind, payload))
        if kind == "subagent_start":
            aid = payload["agent_id"]
        # Annuler APRÈS le premier tour (premier usage vu) : le point
        # d'observation suivant doit stopper la sous-boucle.
        if kind == "subagent_usage" and payload["prompt_tokens"] == 10:
            assert CANCELLATIONS.request("s1", aid) == "cancelling"
    subs = _subevents(events)
    ends = [p for k, p in subs if k == "subagent_end"]
    assert len(ends) == 1  # UN SEUL subagent_end
    assert ends[0]["status"] == "cancelled"
    assert ends[0]["stop_reason"] == "cancelled"
    # Le tour 2 n'a jamais été tiré ; le marqueur d'annulation est explicite.
    contents = "".join(p for k, p in events if k == "content")
    assert "tour 2" not in contents
    assert "[annulé" in contents
    # Plus d'entrée zombie : l'ouvrier terminé s'est désenregistré.
    assert CANCELLATIONS.state("s1", aid) is None


def test_course_fin_naturelle_contre_annulation():
    """L'annulation qui arrive APRÈS le stop naturel perd la course : le résultat
    est conservé (completed) et il n'y a toujours qu'UN subagent_end."""
    gen = _runner(_MultiTurnClient()).stream("t")
    events = []
    aid = None
    for kind, payload in gen:
        events.append((kind, payload))
        if kind == "subagent_start":
            aid = payload["agent_id"]
        if kind == "done":  # fin naturelle déjà émise -> annulation tardive
            CANCELLATIONS.request("s1", aid)
    ends = [p for k, p in _subevents(events) if k == "subagent_end"]
    assert len(ends) == 1
    assert ends[0]["status"] == "completed"
    assert ends[0]["stop_reason"] == "natural"


# --- idempotence ----------------------------------------------------------------


def test_idempotence_et_cibles_inconnues():
    gen = _runner(_MultiTurnClient()).stream("t")
    aid = None
    for kind, payload in gen:
        if kind == "subagent_start":
            aid = payload["agent_id"]
            # Deux demandes -> même réponse, jamais d'erreur.
            assert CANCELLATIONS.request("s1", aid) == "cancelling"
            assert CANCELLATIONS.request("s1", aid) == "cancelling"
    # Terminé -> désenregistré : re-demander reste sans erreur.
    assert CANCELLATIONS.request("s1", aid) == "unknown"
    assert CANCELLATIONS.request("s1", "inexistant") == "unknown"
    assert CANCELLATIONS.request("autre-session", aid) == "unknown"


# --- run_shell long : arbre tué, aucun orphelin ---------------------------------


def test_shell_long_interrompu_sans_orphelin(tmp_path):
    ev = threading.Event()
    spec = make_run_shell(str(tmp_path), timeout=60, cancel_event=ev)
    pid_file = tmp_path / "p.txt"
    if detect().is_windows:
        cmd = f'$PID | Out-File -Encoding ascii "{pid_file}"; Start-Sleep -Seconds 60'
    else:
        cmd = f'echo $$ > "{pid_file}"; sleep 60'
    result: dict = {}

    def _work():
        result["out"] = spec.run({"command": cmd})

    th = threading.Thread(target=_work, daemon=True)
    th.start()
    # Barrière déterministe : la commande signale son démarrage par le pid file.
    deadline = threading.Event()
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text(errors="replace").strip():
            break
        deadline.wait(0.05)
    pid = int(pid_file.read_text(errors="replace").strip())
    assert _pid_alive(pid)  # la commande longue tourne vraiment
    ev.set()  # annulation ciblée
    th.join(timeout=15)
    assert not th.is_alive(), "run_shell n'a pas rendu la main après l'annulation"
    assert "annulé" in result["out"] or "annulation" in result["out"]
    # Aucun orphelin : l'arbre de processus est mort (mécanisme _kill_tree).
    for _ in range(100):
        if not _pid_alive(pid):
            break
        deadline.wait(0.05)
    assert not _pid_alive(pid)


# --- deux ouvriers concurrents : l'un annulé, l'autre et le parent intacts ------


class _PairClient:
    """Deux ouvriers : « tache A » BLOQUE sur une barrière (annulable au moment
    choisi par le test), « tache B » finit naturellement."""

    def __init__(self):
        self.go_a = threading.Event()

    def is_remote(self, model):
        return True  # parallélisme réel côté workflow

    def save_slot(self, *a, **k):
        return False

    def restore_slot(self, *a, **k):
        pass

    def stream_chat_tools(self, messages, *a, **kw):
        prompt = str(messages[0]["content"])
        if "tache A" in prompt:
            yield ("content", "A commence")
            self.go_a.wait(timeout=10)  # barrière : reprend quand le test l'ouvre
            yield ("content", "A suite jamais utile")
            yield ("done", {"reason": "natural"})
        else:
            yield ("content", "B fini")
            yield ("done", {"reason": "natural"})


def test_deux_ouvriers_annulation_ciblee_ne_touche_pas_le_frere(tmp_path):
    (tmp_path / "wf.py").write_text(
        "meta = {'name': 'x'}\n"
        "return parallel([lambda: agent('tache A'), lambda: agent('tache B')])\n",
        encoding="utf-8",
    )
    client = _PairClient()
    runner = SubAgentRunner(
        client,
        lambda: FakeRegistry(),
        system_prompt="s",
        model="m",
        session_id="s-wf",
    )
    reg = ToolRegistry([make_run_workflow(runner, str(tmp_path), is_remote=True)])
    events = []
    for kind, payload in reg.run_stream("run_workflow", {"path": "wf.py"}):
        events.append((kind, payload))
        if kind == "subagent_start" and payload.get("label") == "tache A":
            # Cibler A puis le débloquer : son prochain point d'observation annule.
            assert CANCELLATIONS.request("s-wf", payload["agent_id"]) == "cancelling"
            client.go_a.set()
    ends = {p["label"]: p for k, p in _subevents(events) if k == "subagent_end"}
    assert ends["tache A"]["status"] == "cancelled"  # l'ouvrier ciblé
    assert ends["tache B"]["status"] == "completed"  # son frère, intact
    # Contrat workflow : agent() annulé -> None ; le parent (script) continue et
    # rend quand même son résultat avec la synthèse de B.
    final = "".join(p for k, p in events if k == "content")
    assert "B fini" in final and "null" in final


# --- dispatch_agent : résultat explicite, parent intact -------------------------


def _sub_registry():
    return ToolRegistry(
        [
            ToolSpec(
                name="list_dir",
                description="liste",
                parameters={"type": "object", "properties": {}},
                run=lambda a: "docs/ src/",
            )
        ]
    )


def test_dispatch_annule_rend_un_resultat_explicite_et_le_parent_continue():
    scripts = [
        turn_tools([("p1", "dispatch_agent", '{"task": "explore le projet"}')]),
        turn_tools([("s1", "list_dir", "{}")]),  # 1er tour du sous-agent
        turn_text("Le parent conclut malgré l'annulation de l'ouvrier."),
    ]
    client, fake = make_client(scripts)
    client.save_slot = lambda *a, **k: False
    client.restore_slot = lambda *a, **k: None
    runner = SubAgentRunner(
        client,
        _sub_registry,
        system_prompt="sub-sys",
        model=None,
        session_id="s-disp",
    )
    parent_reg = ToolRegistry(
        [
            make_dispatch_agent(
                client, _sub_registry, system_prompt="sub-sys", runner=runner
            )
        ]
    )
    events = []
    for kind, payload in client.stream_chat_tools(
        [{"role": "user", "content": "vas-y"}],
        "parent-sys",
        1024,
        model=None,
        registry=parent_reg,
        thinking=False,
        max_iters=6,
    ):
        events.append((kind, payload))
        # Annuler dès que l'ouvrier a produit son premier résultat d'outil.
        if kind == "subagent_tool_result":
            CANCELLATIONS.request("s-disp", payload["agent_id"])
    subs = _subevents(events)
    end = [p for k, p in subs if k == "subagent_end"]
    assert len(end) == 1 and end[0]["status"] == "cancelled"
    # dispatch_agent rend un résultat EXPLICITE côté parent…
    results = [p for k, p in events if k == "tool_result"]
    assert len(results) == 1 and "annulé" in results[0]["out_full"]
    # …et le parent continue : son tour final consomme le 3e script.
    assert len(fake.calls) == 3
    final = "".join(p for k, p in events if k == "content")
    assert "parent conclut" in final
    done = [p for k, p in events if k == "done"][-1]
    assert done.get("reason") == "natural"  # le PARENT n'est pas annulé


# --- arrêt GLOBAL d'une session : ses ouvriers, et seulement les siens ----------


def test_cancel_session_registre_cible_une_seule_session():
    ev_a1 = CANCELLATIONS.register("sess-A", "a1")
    ev_a2 = CANCELLATIONS.register("sess-A", "a2")
    ev_b = CANCELLATIONS.register("sess-B", "b1")
    try:
        assert CANCELLATIONS.cancel_session("sess-A") == 2
        assert ev_a1.is_set() and ev_a2.is_set()
        assert not ev_b.is_set()  # l'autre session n'est JAMAIS touchée
        assert CANCELLATIONS.state("sess-B", "b1") == "running"
        assert CANCELLATIONS.cancel_session("sess-A") == 2  # idempotent
        assert CANCELLATIONS.cancel_session("session-inconnue") == 0
    finally:
        for sid, aid in (("sess-A", "a1"), ("sess-A", "a2"), ("sess-B", "b1")):
            CANCELLATIONS.unregister(sid, aid)


def test_route_cancel_global_arrete_le_parent_et_nettoie_ses_ouvriers(app):
    """L'ANCIEN /cancel garde son comportement (204, event de session posé pour
    stopper le parent) et annule EN PLUS les ouvriers de cette session — sans
    toucher à ceux des autres sessions."""
    c = app.test_client()
    sid1 = c.post("/session/new", data={"title": "s1"}).get_json()["id"]
    sid2 = c.post("/session/new", data={"title": "s2"}).get_json()["id"]
    ev1 = CANCELLATIONS.register(sid1, "w1")
    ev2 = CANCELLATIONS.register(sid2, "w2")
    try:
        r = c.post("/cancel", data={"session_id": sid1})
        assert r.status_code == 204  # contrat historique inchangé
        assert ev1.is_set()  # l'ouvrier du parent arrêté est annulé
        assert not ev2.is_set()  # l'autre session continue
        assert CANCELLATIONS.state(sid2, "w2") == "running"
    finally:
        CANCELLATIONS.unregister(sid1, "w1")
        CANCELLATIONS.unregister(sid2, "w2")


def test_cancel_global_debloque_un_run_shell_actif(tmp_path):
    """/cancel pendant qu'un ouvrier est RÉELLEMENT bloqué dans run_shell :
    retour borné, processus et descendance morts, registre vide après la fin."""
    import json as _json

    pid_file = tmp_path / "p.txt"
    if detect().is_windows:
        cmd = f'$PID | Out-File -Encoding ascii "{pid_file}"; Start-Sleep -Seconds 60'
    else:
        cmd = f'echo $$ > "{pid_file}"; sleep 60'

    def build_sub(cancel_event=None):
        return ToolRegistry(
            [make_run_shell(str(tmp_path), timeout=60, cancel_event=cancel_event)]
        )

    client, _ = make_client(
        [
            turn_tools([("s1", "run_shell", _json.dumps({"command": cmd}))]),
            turn_text("jamais atteint"),
        ]
    )
    client.save_slot = lambda *a, **k: False
    client.restore_slot = lambda *a, **k: None
    runner = SubAgentRunner(
        client, build_sub, system_prompt="s", model=None, session_id="s-shell"
    )
    events: list = []

    def _consume():
        for kind, payload in runner.stream("bosse longtemps"):
            events.append((kind, payload))

    th = threading.Thread(target=_consume, daemon=True)
    th.start()
    waiter = threading.Event()
    for _ in range(200):  # barrière : la commande a signalé son pid
        if pid_file.exists() and pid_file.read_text(errors="replace").strip():
            break
        waiter.wait(0.05)
    pid = int(pid_file.read_text(errors="replace").strip())
    assert _pid_alive(pid)
    # Le chemin de l'endpoint /cancel : annulation de TOUTE la session.
    assert CANCELLATIONS.cancel_session("s-shell") == 1
    th.join(timeout=15)
    assert not th.is_alive(), "le run_shell bloqué n'a pas rendu la main"
    ends = [p for k, p in events if k == "subagent_end"]
    assert len(ends) == 1 and ends[0]["status"] == "cancelled"
    for _ in range(100):  # l'arbre de processus est mort (aucun orphelin)
        if not _pid_alive(pid):
            break
        waiter.wait(0.05)
    assert not _pid_alive(pid)
    # Registre vide après terminaison : plus rien d'annulable dans la session.
    assert CANCELLATIONS.cancel_session("s-shell") == 0


def test_abandon_du_generateur_desenregistre_l_ouvrier():
    """Fermeture/abandon du générateur (fermeture de session, stop du flux) :
    le finally du runner désenregistre l'ouvrier — jamais d'entrée zombie."""
    gen = _runner(_MultiTurnClient(), sid="s-close").stream("t")
    kind, payload = next(gen)
    assert kind == "subagent_start"
    aid = payload["agent_id"]
    assert CANCELLATIONS.state("s-close", aid) == "running"
    gen.close()  # abandon en plein vol, sans consommer la fin
    assert CANCELLATIONS.state("s-close", aid) is None
    assert CANCELLATIONS.cancel_session("s-close") == 0


# --- route HTTP : ciblage (session_id, agent_id), idempotente -------------------


def test_route_cancel_idempotente_et_sans_erreur(app):
    c = app.test_client()
    # Cible inconnue/terminée : état en 200, jamais une erreur.
    r = c.post("/session/abc/subagent/defg/cancel")
    assert r.status_code == 200
    assert r.get_json()["state"] == "unknown"
    # Rejouer la même annulation ne crée pas d'erreur non plus.
    assert c.post("/session/abc/subagent/defg/cancel").status_code == 200
    # Ouvrier actif : cancelling, deux fois de suite (idempotence).
    ev = CANCELLATIONS.register("abc", "defg")
    try:
        assert c.post("/session/abc/subagent/defg/cancel").get_json()["state"] == (
            "cancelling"
        )
        assert c.post("/session/abc/subagent/defg/cancel").get_json()["state"] == (
            "cancelling"
        )
        assert ev.is_set()
    finally:
        CANCELLATIONS.unregister("abc", "defg")


# --- persistance / replay de l'état terminal ------------------------------------


def test_replay_conserve_l_etat_cancelled(tmp_path):
    from loom.agent.session import SessionStore

    store = SessionStore(tmp_path / "sessions", default_system_prompt="p")
    sess = store.create(workspace=".")
    gen = _runner(_MultiTurnClient(), sid="s-replay").stream("t")
    events = []
    for kind, payload in gen:
        events.append((kind, payload))
        if kind == "subagent_start":
            CANCELLATIONS.request("s-replay", payload["agent_id"])
    for kind, payload in _subevents(events):
        store.append_event(sess.id, kind, payload)
    replayed = [
        e for e in store.read_timeline(sess.id) if e["event"].startswith("subagent_")
    ]
    assert replayed[-1]["event"] == "subagent_end"
    assert replayed[-1]["data"]["status"] == "cancelled"
    assert replayed[-1]["data"]["stop_reason"] == "cancelled"
