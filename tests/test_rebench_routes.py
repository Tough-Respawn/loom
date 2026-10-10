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
                or "aucun placement faisable" in txt
            ):
                return txt
        time.sleep(0.1)
    raise AssertionError("verdict jamais posté")


def _sessions_text(env, needles=(), timeout=20.0) -> str:
    """Contenu concaténé des session.json, lu avec TOLÉRANCE : le worker remplace le
    fichier atomiquement (os.replace) et Windows refuse alors la lecture un instant
    (PermissionError). Attend jusqu'à `timeout` que toutes les `needles` soient là."""
    deadline = time.time() + timeout
    txt = ""
    while True:
        try:
            txt = "".join(
                p.read_text(encoding="utf-8")
                for p in (env.tmp / "sessions").rglob("session.json")
            )
        except OSError:
            txt = ""
        if all(n in txt for n in needles) and txt:
            return txt
        if time.time() >= deadline:
            return txt
        time.sleep(0.05)


# Trace progressive que la vraie _run_calibration remplit étape par étape (compte
# rendu commun loom-setup / rebench) : le stub la rejoue.
TRACE = {
    "etape": "fin",
    "gguf": "C:/models/fake.gguf",
    "server_bin": "C:/rt/llama-server.exe",
    "materiel": {"gpu_name": "GPU test", "vram_total_mb": 24_000},
    "flags": {"threads": 8, "gpu_tuning": True, "unified_memory": False, "slots": 2},
    "profil": ["architecture test, 32 couches (déclaré)"],
    "plan": {"candidates": [{"key": "gpu_total"}], "non_explores": []},
}


def _launch(env, monkeypatch, calib=CALIB, error=None, trace=TRACE, before_error=None):
    from loom.web import routes

    monkeypatch.setitem(routes._REBENCH, "job", None)

    def fake_run(S, spec, progress, trace_out=None):
        progress("sonde 4096")
        if trace_out is not None:
            trace_out.update(trace)
        if error is not None:
            if before_error and trace_out is not None:
                trace_out.update(before_error)
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
        routes.rebench,
        "_run_calibration",
        lambda S, spec, progress, trace_out=None: (CALIB, None),
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
    sessions = _sessions_text(env, timeout=1.0)
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


_VIOL = {
    "new_tokens": 2000,
    "max_seconds": 10.0,
    "secondes": 13.3,
    "respectee": False,
}


def test_rebench_prefill_final_viole_sans_reference_pas_de_keyerror(env, monkeypatch):
    """Revue #14 P2, reproduction 1 : sans référence de génération, l'affichage
    supposait un `ecart_pct` -> KeyError, verdict « échouée »."""
    fin = {k: v for k, v in _FINAL.items() if k not in ("reference_tg", "ecart_pct")}
    fin.update(
        coherent=None,
        prefill_contrainte=_VIOL,
        bloquant="contrainte prefill non respectée : 2000 tokens en 13.3 s > 10 s",
    )
    _launch(env, monkeypatch, calib=dict(CALIB, context=8192, final=fin))
    txt = _wait_verdict(env)
    assert "échouée" not in txt and "KeyError" not in txt
    assert "contrainte prefill NON respectée" in txt and "13.3 s > 10 s" in txt
    assert "non applicable" in txt
    assert "b_apply" not in _sessions_text(env, timeout=1.0)


def test_rebench_prefill_final_viole_reference_identique_non_applicable(
    env, monkeypatch
):
    """Reproduction 2 : référence identique -> « ne reproduit pas… (+0,0 %) », config
    applicable, sans la contrainte violée. Désormais : génération cohérente, prefill
    NON respecté chiffré, rien à appliquer."""
    fin = dict(
        _FINAL,
        tg_ts=11.9,
        ecart_pct=0.0,
        coherent=True,
        prefill_contrainte=_VIOL,
        bloquant="contrainte prefill non respectée : 2000 tokens en 13.3 s > 10 s",
    )
    _launch(env, monkeypatch, calib=dict(CALIB, context=8192, final=fin))
    txt = _wait_verdict(env)
    assert "ne reproduit pas" not in txt and "cohérente" in txt
    assert "NON respectée" in txt and "13.3 s > 10 s" in txt and "non applicable" in txt
    assert "b_apply" not in _sessions_text(env, timeout=1.0)


def test_rebench_prefill_insatisfiable_applicable_avec_avertissement(env, monkeypatch):
    fin = dict(_FINAL, prefill_contrainte=dict(_VIOL, insatisfiable=True))
    _launch(env, monkeypatch, calib=dict(CALIB, context=8192, final=fin))
    txt = _wait_verdict(env)
    assert "insatisfiable" in txt and "Tape « oui »" in txt
    assert "b_apply" in _sessions_text(env, needles=("b_apply",))


def test_rebench_verdict_porte_la_validation_du_reglage_final(env, monkeypatch):
    calib = dict(CALIB, context=8192, final=_FINAL)
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "réglage final" in txt and "11.7" in txt and "280" in txt
    assert "ctx 65536" in txt and "2 slots" in txt and "ub 512" in txt
    assert "ne reproduit pas" not in txt


def test_rebench_verdict_dit_les_checkpoints_effectifs_du_reglage_final(
    env, monkeypatch
):
    """Revue 2026-10-10 : 32 est un plafond, pas le nombre créé pendant la mesure. Le
    verdict dit le compte EFFECTIF lu dans le journal serveur, ou « non mesuré »."""
    calib = dict(
        CALIB,
        context=8192,
        final=dict(
            _FINAL, checkpoints_effectifs=3, checkpoints_plafond=32, checkpoint_mb=149.6
        ),
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "checkpoints effectifs 3" in txt and "plafond 32 par slot" in txt
    assert "149.6" in txt


def test_rebench_verdict_dit_checkpoints_non_mesures(env, monkeypatch):
    calib = dict(
        CALIB,
        context=8192,
        final=dict(_FINAL, checkpoints_detail="non mesuré : journal serveur vide"),
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "checkpoints : non mesuré" in txt


def test_rebench_aucun_placement_faisable_verdict_explicite_et_archive(
    env, monkeypatch
):
    """Revue #14 P1 : rien ne tient d'après l'estimation -> verdict dédié, rien à
    appliquer, archive en échec à l'étape placement."""
    from loom.setup import archive as _archive
    from loom.setup.placement import AucunPlacementFaisable

    monkeypatch.setattr(_archive, "BENCH_DIR", env.tmp / "var" / "bench")
    raison = (
        "aucun placement faisable d'après l'estimation mémoire — cpu : non exploré : "
        "ne tient pas — proportion : ~50000 Mo hôte pour 12928 Mo de RAM"
    )
    _launch(
        env,
        monkeypatch,
        error=AucunPlacementFaisable(raison),
        before_error={"etape": "placement"},
    )
    txt = _wait_verdict(env)
    assert "aucun placement faisable" in txt and "50000" in txt
    assert "Recalibration" not in txt or "échouée" not in txt  # pas un plantage
    # Revue #15 : la sonde d'isolation a DÉJÀ chargé le modèle avant le contrôle
    # mémoire — « Rien n'a été mesuré » était faux. Ce qui est vrai :
    assert (
        "Aucun placement comparé, calibration non lancée, configuration inchangée."
        in txt
    )
    assert "Rien n'a été mesuré" not in txt
    assert "b_apply" not in _sessions_text(env, timeout=1.0)
    archives = list((env.tmp / "var" / "bench" / "loc-test").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "placement"
    assert "aucun placement faisable" in arch["echec"]["erreur"]


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


def test_rebench_reglage_final_en_echec_n_est_pas_applicable(env, monkeypatch):
    """P1 (revue 2026-10-10) : la configuration finale n'a pas FONCTIONNÉ
    (ErrorOutOfDeviceMemory) -> aucune proposition d'application, l'existant est
    conservé. Une baisse de vitesse avertit ; un échec de fonctionnement empêche."""
    calib = dict(
        CALIB,
        context=8192,  # un changement que l'on proposerait d'appliquer…
        final={
            "echec": "RuntimeError: ErrorOutOfDeviceMemory",
            "ctx": 8192,
            "depth": 4096,
        },
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "non applicable" in txt and "ErrorOutOfDeviceMemory" in txt
    # (la confirmation initiale dit « Tape « oui » pour lancer » ; le verdict, lui, ne
    # propose plus d'appliquer)
    assert "Tape « oui » pour appliquer" not in txt
    # …mais rien à appliquer : pas d'état b_apply, et « oui » ne touche à rien.
    sessions = _sessions_text(env, timeout=1.0)
    assert "b_apply" not in sessions
    env.web.post("/chat", data={"message": "oui"})
    assert "context = 4096" in (env.mdir / "model.toml").read_text(encoding="utf-8")


def test_rebench_archive_le_verdict_puis_l_application(env, monkeypatch):
    """Archive durable : le verdict écrit un JSON horodaté sous var/bench/<modèle>/ (hors
    état de session, qui peut être consommé ou annulé) ; « oui » y note l'application."""
    from loom.setup import archive as _archive

    monkeypatch.setattr(_archive, "BENCH_DIR", env.tmp / "var" / "bench")
    calib = dict(
        CALIB,
        context=8192,
        final=_FINAL,
        build="b7000-abc1234",
        rungs_detail=[
            {"ctx": 8192, "mem_mb": 3000, "checkpoints": {"effectifs": 0, "pic": 0}}
        ],
    )
    _launch(env, monkeypatch, calib=calib)
    _wait_verdict(env)
    archives = list((env.tmp / "var" / "bench" / "loc-test").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["model_id"] == "loc-test" and arch["build"] == "b7000-abc1234"
    assert arch["calibration"]["context"] == 8192 and arch["final"]["tg_ts"] == 11.7
    # Les points de pente gardent leurs checkpoints effectifs (revue #14).
    assert arch["calibration"]["rungs_detail"][0]["checkpoints"]["effectifs"] == 0
    assert "Verdict" in arch["verdict_texte"] and "application" not in arch
    # Compte rendu COMMUN : matériel, binaire, flags, profil, plan — reproductible.
    assert arch["materiel"]["gpu_name"] == "GPU test"
    assert arch["server_bin"].endswith("llama-server.exe")
    assert arch["flags"]["threads"] == 8 and arch["flags"]["slots"] == 2
    assert arch["profil"] and arch["plan"]["candidates"]
    r = env.web.post("/chat", data={"message": "oui"})
    assert "Application" in _sse_texts(r.data)
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["application"]["context"] == 8192


def test_rebench_echec_archive_la_trace_partielle(env, monkeypatch):
    """P3 : une exception pendant la calibration ne perd plus les mesures déjà faites —
    la trace progressive est archivée avec l'étape et l'erreur."""
    from loom.setup import archive as _archive

    monkeypatch.setattr(_archive, "BENCH_DIR", env.tmp / "var" / "bench")
    _launch(
        env,
        monkeypatch,
        error=RuntimeError("health timeout à ctx=65536"),
        before_error={
            "etape": "calibration",
            "isolation": {"necessaire": True, "first": 721, "back": 6},
            "placement": {"key": "gpu_total@ub512@b2048", "tg_ts": 11.3},
        },
    )
    txt = _wait_verdict(env)
    assert "échouée" in txt
    archives = list((env.tmp / "var" / "bench" / "loc-test").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "calibration"
    assert "health timeout" in arch["echec"]["erreur"]
    assert arch["isolation"]["first"] == 721 and arch["placement"]["tg_ts"] == 11.3
    assert arch["materiel"]["gpu_name"] == "GPU test" and "application" not in arch


def test_rebench_erreur_d_entree_sortie_finalise_le_job_et_archive(env, monkeypatch):
    """Revue 2026-10-10 : seules RuntimeError / ValueError étaient rattrapées ; une
    FileNotFoundError ou PermissionError pendant la calibration laissait le job sans fin
    et sans archive."""
    from loom.setup import archive as _archive

    monkeypatch.setattr(_archive, "BENCH_DIR", env.tmp / "var" / "bench")
    _launch(
        env,
        monkeypatch,
        error=PermissionError("accès refusé au GGUF"),
        before_error={"etape": "calibration"},
    )
    txt = _wait_verdict(env)
    assert "échouée" in txt and "accès refusé" in txt
    archives = list((env.tmp / "var" / "bench" / "loc-test").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert (
        arch["echec"]["etape"] == "calibration"
        and "accès refusé" in arch["echec"]["erreur"]
    )
    # Le job est bien terminé : un nouveau /rebench n'est pas « déjà en cours ».
    r = env.web.post("/chat", data={"message": "/rebench loc-test"})
    assert "déjà en cours" not in _sse_texts(r.data)


def test_rebench_apply_dit_quand_l_annotation_de_l_archive_echoue(env, monkeypatch):
    from loom.setup import archive as _archive
    from loom.web.routes import models as models_routes

    monkeypatch.setattr(_archive, "BENCH_DIR", env.tmp / "var" / "bench")
    calib = dict(CALIB, context=8192, final=_FINAL)
    _launch(env, monkeypatch, calib=calib)
    _wait_verdict(env)
    assert "b_apply" in _sessions_text(env, needles=("b_apply",))  # état persisté
    monkeypatch.setattr(
        models_routes, "note_application", lambda *a, **k: False, raising=False
    )
    monkeypatch.setattr(_archive, "note_application", lambda *a, **k: False)
    r = env.web.post("/chat", data={"message": "oui"})
    body = _sse_texts(r.data)
    assert "Application" in body
    assert "appliquée" in body and "annotation de l'archive échouée" in body
    assert "context = 8192" in (env.mdir / "model.toml").read_text(encoding="utf-8")


def test_rebench_signale_une_archive_non_ecrite(env, monkeypatch):
    from loom.setup import archive as _archive

    def _boom(*a, **k):
        raise OSError("disque plein")

    monkeypatch.setattr(_archive, "archive_bench", _boom)
    _launch(env, monkeypatch, calib=dict(CALIB, context=8192))
    txt = _wait_verdict(env)
    assert "archive non écrite" in txt and "disque plein" in txt


def test_rebench_applique_les_threads_mesures(env, monkeypatch):
    calib = dict(
        CALIB,
        context=4096,
        threads_probe={
            "threads": 5,
            "baseline": 8,
            "tg_ts": 6.0,
            "pp_ts": 40.0,
            "gain_pct": 20.0,
            "compare": True,
            "mesures": {"t8": {"tg_ts": 5.0}, "t5": {"tg_ts": 6.0}},
            "mecanisme": "t5 adopté : génération 6.0 t/s contre 5.0 (t8), +20 %",
            "placement": "gpu_partiel_ngl11",
        },
        threads_avant=None,
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "threads" in txt and "→ 5" in txt and "+20" in txt
    assert "b_apply" in _sessions_text(env, needles=("b_apply",))
    r = env.web.post("/chat", data={"message": "oui"})
    assert "Application" in _sse_texts(r.data)
    toml_txt = (env.mdir / "model.toml").read_text(encoding="utf-8")
    assert "threads = 5" in toml_txt


def test_rebench_threads_non_explores_rien_a_appliquer(env, monkeypatch):
    calib = dict(
        CALIB,
        context=4096,
        threads_probe={
            "non_explore": "non exploré : tout GPU, aucun calcul CPU attendu"
        },
        threads_avant=None,
    )
    _launch(env, monkeypatch, calib=calib)
    txt = _wait_verdict(env)
    assert "threads" in txt and "non exploré" in txt
    assert "b_apply" not in _sessions_text(env, timeout=1.0)


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
    sessions = _sessions_text(env, needles=("b_apply", "echantillons", "b7000-abc1234"))
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
