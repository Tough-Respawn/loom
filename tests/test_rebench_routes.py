# /rebench côté routes : job de calibration STUBBÉ (routes._run_calibration),
# verdict persisté, état b_apply, application au model.toml, verrou anti-double.
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from loom.agent.session import SessionStore
from loom.web.app import create_app

FAKE_MODEL = "fake-model"
CALIB = {
    "context": 8192,
    "mode": "capacite",
    "mecanisme": "pente 9.0 Ko/token mesurée, vitesse validée",
    "slope_kb_tok": 9.0,
    "valide_jusqua": 7000,
}


def _sse_texts(body: bytes) -> str:
    out = []
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data: "):
            evt = json.loads(line[6:])
            if evt["type"] == "text":
                out.append(evt.get("text", ""))
    return "\n".join(out)


@pytest.fixture()
def env(tmp_path):
    for d in ("skills", "skills_user", "workspace"):
        (tmp_path / d).mkdir()
    mdir = tmp_path / "models" / "loc-test"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'filename = "fake.gguf"\ncontext = 4096\n', encoding="utf-8"
    )
    (mdir / "fake.gguf").write_bytes(b"GGUF")
    store = SessionStore(
        tmp_path / "sessions",
        default_system_prompt="prompt de test",
        default_model=FAKE_MODEL,
        known_models=[FAKE_MODEL, "loc-test"],
    )
    app = create_app(
        client=None,
        skills_dir=str(tmp_path / "skills"),
        session_store=store,
        models=[FAKE_MODEL, "loc-test"],
        keepwarm_enabled=False,
        workspace_dir=str(tmp_path / "workspace"),
        user_skills_dir=str(tmp_path / "skills_user"),
        plugins_dir=str(tmp_path / "plugins"),
        remote_store_path=str(tmp_path / "remote_models.json"),
        models_dir=str(tmp_path / "models"),
        local_models=[
            {"id": "loc-test", "dir": str(mdir), "size_mb": 1024, "context": 4096}
        ],
    )
    web = app.test_client()
    assert web.post("/session/new", data={}).status_code == 200
    return SimpleNamespace(web=web, tmp=tmp_path, mdir=mdir)


def _wait_verdict(env, timeout=10.0):
    """Attend que le worker ait posté le verdict dans le journal (thread réel)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for p in (env.tmp / "sessions").rglob("timeline.jsonl"):
            txt = p.read_text(encoding="utf-8")
            if (
                "Verdict" in txt
                or "déjà au top" in txt
                or "mesures disponibles" in txt
                or "échouée" in txt
            ):
                return txt
        time.sleep(0.1)
    raise AssertionError("verdict jamais posté")


def _launch(env, monkeypatch, calib=CALIB, error=None):
    from loom.web import routes

    monkeypatch.setitem(routes._REBENCH, "job", None)

    def fake_run(S, spec, progress):
        progress("sonde 4096")
        if error is not None:
            raise error
        return calib, env.mdir / "fake.gguf"

    monkeypatch.setattr(routes.rebench, "_run_calibration", fake_run)
    env.web.post("/chat", data={"message": "/rebench loc-test"})
    return env.web.post("/chat", data={"message": "oui"})


def _sse_events(body: bytes) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in body.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


def test_rebench_confirmation_et_verdict_portent_des_boutons(env, monkeypatch):
    from loom.web import routes

    monkeypatch.setitem(routes._REBENCH, "job", None)
    monkeypatch.setattr(
        routes.rebench, "_run_calibration", lambda S, spec, progress: (CALIB, None)
    )
    r = env.web.post("/chat", data={"message": "/rebench loc-test"})
    assert {"type": "choices", "options": ["oui", "annuler"]} in _sse_events(r.data)
    # le flux du lancement attend le job (stub instantané) et émet le verdict + boutons
    r = env.web.post("/chat", data={"message": "oui"})
    events = _sse_events(r.data)
    assert any(
        e["type"] == "choices" and e["options"] == ["oui", "annuler"] for e in events
    )
    assert any("Verdict" in e.get("text", "") for e in events if e["type"] == "text")


def test_rebench_job_poste_verdict_et_etat_apply(env, monkeypatch):
    r = _launch(env, monkeypatch)
    assert "lancée" in _sse_texts(r.data)
    txt = _wait_verdict(env)
    assert "4096 → 8192" in txt and "oui" in txt
    # l'état b_apply est actif : « oui » applique
    r = env.web.post("/chat", data={"message": "oui"})
    assert "Application" in _sse_texts(r.data)
    toml = (env.mdir / "model.toml").read_text(encoding="utf-8")
    assert "context = 8192" in toml
    # visible aussi par l'endpoint disque (onglet Modèles locaux)
    payload = env.web.get("/models/local").get_json()
    m = next(x for x in payload["models"] if x["id"] == "loc-test")
    assert m["context"] == 8192


def test_rebench_rien_a_changer_sans_comparaison_ne_dit_pas_deja_au_top(
    env, monkeypatch
):
    # Contexte inchangé mais AUCUNE comparaison de placement exploitable : le verdict
    # correspond aux preuves disponibles, il ne proclame pas « déjà au top ».
    r = _launch(env, monkeypatch, calib=dict(CALIB, context=4096))
    assert "lancée" in _sse_texts(r.data)
    txt = _wait_verdict(env)
    assert "déjà au top" not in txt
    assert "mesures disponibles" in txt and "placement" in txt
    # PAS d'état wizard b_apply persisté : rien à « appliquer »
    sessions = "".join(
        p.read_text(encoding="utf-8")
        for p in (env.tmp / "sessions").rglob("session.json")
    )
    assert "b_apply" not in sessions
    assert "context = 4096" in (env.mdir / "model.toml").read_text(encoding="utf-8")


def test_rebench_applique_le_verdict_isolation(env, monkeypatch):
    # Contexte inchangé MAIS la sonde d'isolation vient de mesurer que le cache ne
    # survit pas -> il y a bien quelque chose à appliquer (cache_isolation = true),
    # écrit dans le model.toml en même temps que le contexte (couple mesuré ensemble).
    calib = dict(
        CALIB,
        context=4096,
        isolation=True,
        isolation_detail="retour 590/600 tokens retraités",
        isolation_avant=False,
    )
    r = _launch(env, monkeypatch, calib=calib)
    # Drainer le flux SSE du lancement = attendre la fin du job (comme le stream réel).
    assert "lancée" in _sse_texts(r.data)
    txt = _wait_verdict(env)
    assert "cache_isolation → true" in txt and "cache PERDU" in txt
    r = env.web.post("/chat", data={"message": "oui"})
    assert "Application" in _sse_texts(r.data)
    toml_txt = (env.mdir / "model.toml").read_text(encoding="utf-8")
    assert "cache_isolation = true" in toml_txt
    assert "context = 4096" in toml_txt


def test_rebench_isolation_imposee_par_la_recurrence_ne_dit_pas_perdu(env, monkeypatch):
    calib = dict(
        CALIB,
        context=4096,
        isolation=True,
        isolation_first=721,
        isolation_back=6,
        isolation_detail="retour 6/721 tokens retraités, mémoire récurrente",
        isolation_avant=False,
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "cache_isolation → true" in txt
    assert "PERDU" not in txt and "survit" in txt and "imposés" in txt


def test_rebench_applique_le_placement_mesure(env, monkeypatch):
    # Contexte inchangé, mais la sonde de placement a mesuré tout-GPU +19 % contre
    # experts-CPU (Ornith, 2026-10-09) : verdict nommé, puis écrit dans le model.toml.
    calib = dict(
        CALIB,
        context=4096,
        placement={
            "label": "gpu_total",
            "ngl": 999,
            "cpu_moe": False,
            "n_cpu_moe": None,
            "tg_ts": 14.4,
            "pp_ts": 262.0,
            "gain_pct": 19.0,
            "mecanisme": "gpu_total adopté : génération 14.4 t/s contre 12.1 (experts_cpu), +19 %",
        },
        placement_avant={"cpu_moe": True, "n_cpu_moe": None, "n_gpu_layers": None},
    )
    r = _launch(env, monkeypatch, calib=calib)
    assert "lancée" in _sse_texts(r.data)
    txt = _wait_verdict(env)
    assert "placement → gpu_total" in txt and "+19" in txt
    r = env.web.post("/chat", data={"message": "oui"})
    assert "Application" in _sse_texts(r.data)
    toml_txt = (env.mdir / "model.toml").read_text(encoding="utf-8")
    assert "cpu_moe = false" in toml_txt and "n_gpu_layers = 999" in toml_txt
    assert "context = 4096" in toml_txt


def test_rebench_placement_identique_ne_change_rien(env, monkeypatch):
    calib = dict(
        CALIB,
        context=4096,
        placement={
            "label": "experts_cpu",
            "ngl": 999,
            "cpu_moe": True,
            "n_cpu_moe": None,
            "tg_ts": 12.1,
            "pp_ts": 217.0,
            "gain_pct": None,
            "compare": True,
            "mecanisme": "experts_cpu conservé : gpu_total à +3 % de tg, sous la marge de 5 %",
        },
        placement_avant={"cpu_moe": True, "n_cpu_moe": None, "n_gpu_layers": None},
        cache_verifie={
            "first": 600,
            "back": 4,
            "annex_slot": 1,
            "slots": 2,
            "reused": True,
        },
        final=dict(
            _FINAL,
            placement="experts_cpu",
            reference_tg=12.1,
            tg_ts=12.0,
            ecart_pct=-0.8,
        ),
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    # Preuves complètes (contexte validé, placement comparé, réglage final validé, cache
    # vérifié) : « déjà au top ».
    assert "déjà au top" in txt and "sous la marge" in txt
    assert "cache réutilisé" in txt and "4/600" in txt


_FINAL = {
    "tg_ts": 11.7,
    "pp_ts": 280.0,
    "n": 2,
    "tg_disp_pct": 1.0,
    "ctx": 65536,
    "depth": 16384,
    "slots": 2,
    "ubatch": 512,
    "batch": 2048,
    "placement": "gpu_total",
    "reference_tg": 11.9,
    "ecart_pct": -1.7,
    "coherent": True,
}


def test_rebench_verdict_porte_la_validation_du_reglage_final(env, monkeypatch):
    calib = dict(CALIB, context=8192, final=_FINAL)
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "réglage final" in txt and "11.7" in txt and "280" in txt
    assert "ctx 65536" in txt and "2 slots" in txt and "ub 512" in txt
    assert "ne reproduit pas" not in txt


def test_rebench_verdict_previent_quand_le_reglage_final_ne_reproduit_pas(
    env, monkeypatch
):
    calib = dict(
        CALIB,
        context=8192,
        final=dict(_FINAL, tg_ts=9.0, ecart_pct=-24.4, coherent=False),
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "ne reproduit pas" in txt and "-24" in txt


def test_rebench_deja_au_top_exige_un_reglage_final_valide(env, monkeypatch):
    # Preuves complètes sauf la validation finale : le verdict dit ce qui manque.
    calib = dict(
        CALIB,
        context=4096,
        placement={
            "label": "experts_cpu",
            "ngl": 999,
            "cpu_moe": True,
            "n_cpu_moe": None,
            "tg_ts": 12.1,
            "pp_ts": 217.0,
            "gain_pct": None,
            "compare": True,
            "mecanisme": "experts_cpu conservé : gpu_total à +3 % de tg, sous la marge de 5 %",
        },
        placement_avant={"cpu_moe": True, "n_cpu_moe": None, "n_gpu_layers": None},
        cache_verifie={
            "first": 600,
            "back": 4,
            "annex_slot": 1,
            "slots": 2,
            "reused": True,
        },
        final={"echec": "RuntimeError: health timeout", "ctx": 4096, "depth": 2048},
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "déjà au top" not in txt and "réglage final" in txt


def test_rebench_verdict_dit_quand_le_cache_n_est_pas_reutilise(env, monkeypatch):
    calib = dict(
        CALIB,
        context=4096,
        isolation=True,
        isolation_detail="retour 590/600 tokens retraités",
        isolation_avant=False,
        cache_verifie={
            "first": 600,
            "back": 580,
            "annex_slot": 1,
            "slots": 2,
            "reused": False,
        },
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "NON réutilisé" in txt and "580/600" in txt


def test_rebench_apply_ecrit_le_build_et_garde_les_echantillons(env, monkeypatch):
    calib = dict(
        CALIB,
        context=4096,
        build="b7000-abc1234",
        placement={
            "label": "gpu_total",
            "ngl": 999,
            "cpu_moe": False,
            "n_cpu_moe": None,
            "tg_ts": 14.4,
            "pp_ts": 262.0,
            "gain_pct": 19.0,
            "compare": True,
            "mecanisme": "gpu_total adopté : génération 14.4 t/s contre 12.1 (experts_cpu), +19 %",
            "mesures": {
                "gpu_total": {
                    "tg_ts": 14.4,
                    "echantillons": [{"tg_ts": 14.0}, {"tg_ts": 14.8}],
                }
            },
        },
        placement_avant={"cpu_moe": True, "n_cpu_moe": None, "n_gpu_layers": None},
    )
    _launch(env, monkeypatch, calib=calib)
    _wait_verdict(env)
    # L'état d'application conserve les mesures détaillées (échantillons compris).
    # Le worker poste le verdict PUIS sauvegarde la session : attendre l'état persisté.
    # Attendre l'état COMPLET (sous la charge de la suite, la sauvegarde peut suivre
    # le verdict de plusieurs secondes), puis seulement asserter.
    deadline = time.time() + 20.0
    sessions = ""
    attendus = ("b_apply", "echantillons", "b7000-abc1234")
    while time.time() < deadline and not all(a in sessions for a in attendus):
        sessions = "".join(
            p.read_text(encoding="utf-8")
            for p in (env.tmp / "sessions").rglob("session.json")
        )
        time.sleep(0.05)
    assert "echantillons" in sessions and "b7000-abc1234" in sessions
    r = env.web.post("/chat", data={"message": "oui"})
    assert "Application" in _sse_texts(r.data)
    toml_txt = (env.mdir / "model.toml").read_text(encoding="utf-8")
    assert "n_gpu_layers = 999" in toml_txt
    assert "b7000-abc1234" in toml_txt  # le build du moteur dans le commentaire


def test_rebench_echec_calibration_message_persiste(env, monkeypatch):
    _launch(env, monkeypatch, error=RuntimeError("binaire llama-server introuvable"))
    txt = _wait_verdict(env)
    assert "échouée" in txt and "introuvable" in txt
    assert "context = 4096" in (env.mdir / "model.toml").read_text(encoding="utf-8")


def test_rebench_refus_types_et_inconnu(env, monkeypatch):
    r = env.web.post("/chat", data={"message": "/rebench nexiste-pas"})
    assert "inconnu" in _sse_texts(r.data)


def test_rebench_verrou_un_seul_job(env, monkeypatch):
    from loom.web import routes

    monkeypatch.setitem(
        routes._REBENCH, "job", SimpleNamespace(done=False, label="", final=None)
    )
    env.web.post("/chat", data={"message": "/rebench loc-test"})
    r = env.web.post("/chat", data={"message": "oui"})
    assert "déjà en cours" in _sse_texts(r.data)
