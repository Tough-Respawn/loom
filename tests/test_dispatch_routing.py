# Routage DÉTERMINISTE des sous-agents (design validé user 2026-07-15) :
# chaîne de modèles configurée (ex. gratuit -> payant), repli final = le modèle du
# fil parent (local). Un tier qui meurt en api_error (429 free tier, 5xx, timeout)
# passe la main au suivant. `local_only` (session) court-circuite TOUTE la chaîne :
# données privées/sensibles -> rien ne part vers une API. Le modèle ne choisit rien.
from __future__ import annotations

from .fakes import FakeRegistry


class ChainClient:
    """Client scripté par modèle : behaviors[id] = 'ok' | 'api_error'. Enregistre
    l'ordre des tiers essayés, les kwargs de chaque sous-boucle et les slots KV."""

    def __init__(self, behaviors, remote_ids):
        self.behaviors = behaviors
        self.remote = set(remote_ids)
        self.tried = []
        self.kwargs = []
        self.slots = []

    def is_remote(self, model):
        return model in self.remote

    def save_slot(self, model, name):
        self.slots.append(("save", model))
        return True

    def restore_slot(self, model, name):
        self.slots.append(("restore", model))

    def stream_chat_tools(self, messages, system_prompt, max_tokens, **kw):
        m = kw.get("model")
        self.tried.append(m)
        self.kwargs.append({"max_tokens": max_tokens, **kw})
        if self.behaviors.get(m) == "api_error":
            yield ("done", {"reason": "api_error"})
        else:
            yield ("content", f"synthèse de {m}")
            yield ("done", {"reason": "natural"})


def _dispatch(client, **kw):
    from loom.tools.agent import make_dispatch_agent

    return make_dispatch_agent(client, lambda: FakeRegistry(), system_prompt="s", **kw)


def test_chaine_utilise_le_premier_tier():
    client = ChainClient({"flash": "ok"}, remote_ids={"flash", "zai"})
    spec = _dispatch(client, model="local-x", model_chain=["flash", "zai"])
    out = spec.run({"task": "cherche"})
    assert client.tried == ["flash"]
    assert "synthèse de flash" in out


def test_fallback_api_error_escalade_puis_local():
    # flash meurt (429 free tier), zai meurt aussi -> repli final sur le local.
    client = ChainClient(
        {"flash": "api_error", "zai": "api_error", "local-x": "ok"},
        remote_ids={"flash", "zai"},
    )
    spec = _dispatch(client, model="local-x", model_chain=["flash", "zai"])
    out = spec.run({"task": "cherche"})
    assert client.tried == ["flash", "zai", "local-x"]
    assert "synthèse de local-x" in out


def test_local_only_ignore_la_chaine():
    # Session privée : AUCUN appel distant, même avec une chaîne configurée.
    client = ChainClient({"local-x": "ok"}, remote_ids={"flash", "zai"})
    spec = _dispatch(
        client, model="local-x", model_chain=["flash", "zai"], local_only=True
    )
    out = spec.run({"task": "dossier perso"})
    assert client.tried == ["local-x"]
    assert "synthèse de local-x" in out


def test_kv_sauve_seulement_pour_tier_local():
    # Tier distant : le slot KV local du parent n'est PAS touché (pas de save/restore,
    # pas de re-prefill après le dispatch). Repli local : save/restore comme avant.
    client = ChainClient({"flash": "ok"}, remote_ids={"flash"})
    _dispatch(client, model="local-x", model_chain=["flash"]).run({"task": "t"})
    assert client.slots == []

    client2 = ChainClient({"flash": "api_error", "local-x": "ok"}, remote_ids={"flash"})
    _dispatch(client2, model="local-x", model_chain=["flash"]).run({"task": "t"})
    assert ("save", "local-x") in client2.slots
    assert ("restore", "local-x") in client2.slots


def test_limites_resolues_par_tier():
    # max_iters et seuil de compaction suivent le TIER (fenêtre du modèle qui bosse),
    # pas le modèle parent : flash 200k ne compacte pas comme un local 24k.
    seuils = {"flash": 190_000, "local-x": 15_000}
    client = ChainClient({"flash": "api_error", "local-x": "ok"}, remote_ids={"flash"})
    spec = _dispatch(
        client,
        model="local-x",
        model_chain=["flash"],
        compact_for=lambda mid: seuils[mid],
    )
    spec.run({"task": "t"})
    kw_flash, kw_local = client.kwargs
    assert kw_flash["compact_after_tokens"] == 190_000
    assert kw_flash["max_iters"] == 500  # distant
    assert kw_local["compact_after_tokens"] == 15_000
    assert kw_local["max_iters"] == 30  # local bridé


# --- Épinglage par appel (agent(model=...) des workflows) -----------------------


def _runner(client, **kw):
    from loom.tools.agent import SubAgentRunner

    return SubAgentRunner(client, lambda: FakeRegistry(), system_prompt="s", **kw)


def _drain(runner, task="t", model=None):
    return [p for k, p in runner.stream(task, model=model) if k == "content"]


def test_override_epingle_le_modele_avec_repli_session():
    # Vérificateur épinglé sur zai : zai d'abord, repli sur le modèle de session.
    client = ChainClient({"zai": "ok"}, remote_ids={"flash", "zai"})
    r = _runner(client, model="local-x", model_chain=["flash", "zai"])
    out = "".join(_drain(r, model="zai"))
    assert client.tried == ["zai"]  # PAS flash : l'épinglage remplace la chaîne
    assert "synthèse de zai" in out


def test_override_mort_replie_sur_le_modele_session():
    client = ChainClient(
        {"zai": "api_error", "local-x": "ok"}, remote_ids={"flash", "zai"}
    )
    r = _runner(client, model="local-x", model_chain=["flash", "zai"])
    out = "".join(_drain(r, model="zai"))
    assert client.tried == ["zai", "local-x"]
    assert "relève" in out


def test_override_ignore_en_session_privee():
    """CARDINAL : la confidentialité prime sur le routage — un script qui épingle un
    modèle distant dans une session privée n'exfiltre RIEN, la demande est ignorée."""
    client = ChainClient({"local-x": "ok"}, remote_ids={"flash", "zai"})
    r = _runner(client, model="local-x", model_chain=["flash", "zai"], local_only=True)
    _drain(r, model="zai")
    assert client.tried == ["local-x"]


def test_override_inconnu_ignore():
    # Modèle non routé (is_remote False) : la chaîne normale s'applique, pas d'erreur.
    client = ChainClient({"flash": "ok"}, remote_ids={"flash", "zai"})
    r = _runner(client, model="local-x", model_chain=["flash", "zai"])
    _drain(r, model="inconnu-42")
    assert client.tried == ["flash"]


def test_override_egal_au_modele_session_sans_doublon():
    client = ChainClient({"zai": "ok"}, remote_ids={"zai"})
    r = _runner(client, model="zai", model_chain=["flash"])
    _drain(r, model="zai")
    assert client.tried == ["zai"]


def test_roles_cheap_strong_resolus_depuis_la_config():
    """Un script épingle des RÔLES ("cheap"/"strong"), pas des ids machine : la
    résolution suit la config (flags strong + ordre de chaîne) — un renommage de
    modèle ne dégrade pas le routage en silence."""
    roles = {"cheap": "flash", "strong": "zai"}
    client = ChainClient({"zai": "ok"}, remote_ids={"flash", "zai"})
    r = _runner(
        client, model="local-x", model_chain=["flash", "zai"], model_roles=roles
    )
    _drain(r, model="strong")
    assert client.tried == ["zai"]

    client2 = ChainClient({"flash": "ok"}, remote_ids={"flash", "zai"})
    r2 = _runner(
        client2, model="local-x", model_chain=["flash", "zai"], model_roles=roles
    )
    _drain(r2, model="cheap")
    assert client2.tried == ["flash"]


def test_role_non_resolu_retombe_sur_la_chaine():
    # Pas de modèle strong dans la config -> "strong" inconnu -> chaîne normale.
    client = ChainClient({"flash": "ok"}, remote_ids={"flash"})
    r = _runner(
        client, model="local-x", model_chain=["flash"], model_roles={"cheap": "flash"}
    )
    _drain(r, model="strong")
    assert client.tried == ["flash"]


def test_role_ignore_en_session_privee():
    client = ChainClient({"local-x": "ok"}, remote_ids={"flash", "zai"})
    r = _runner(
        client,
        model="local-x",
        model_chain=["flash", "zai"],
        local_only=True,
        model_roles={"strong": "zai"},
    )
    _drain(r, model="strong")
    assert client.tried == ["local-x"]
