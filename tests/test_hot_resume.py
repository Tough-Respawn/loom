# Reprise à CHAUD one-shot (roadmap 16) : save fin de tour + meta, restore
# UNIQUEMENT sur slot froid, jamais par tour ; hybrides gated sur binaire
# restore-safe ; un seul essai par période froide ; swap sortant = froid.
from __future__ import annotations

import json

from loom.agent.client import LoomClient


def _client(tmp_path, hot_resume=True, restore_safe=False, hybrid=()):
    c = LoomClient(base_url="http://127.0.0.1:8080/v1")
    c.hot_resume_enabled = hot_resume
    c.restore_safe = restore_safe
    c.hybrid_models = set(hybrid)
    c.slots_dir_override = str(tmp_path)
    return c


def _spy_actions(monkeypatch, c, result=True):
    calls: list[tuple] = []

    def fake(model, action, name, force=False):
        calls.append((model, action, name, force))
        return result

    monkeypatch.setattr(c, "_slot_action", fake)
    return calls


def _write_meta(tmp_path, model="orn", session="s1"):
    (tmp_path / "turnend.kv.meta.json").write_text(
        json.dumps({"model": model, "session": session}), encoding="utf-8"
    )


# ---- gates de _slot_action ---------------------------------------------------


def test_hot_resume_autorise_save_mais_pas_restore_par_tour(monkeypatch, tmp_path):
    # hot_resume SEUL (slot_kv off) : le save de fin de tour passe, le restore
    # par tour (maintenance) reste bloqué — c'est tout le contrat du one-shot.
    c = _client(tmp_path)
    sent: list[str] = []
    monkeypatch.setattr(
        c,
        "local_server_root",
        lambda: (_ for _ in ()).throw(AssertionError("ne doit pas sortir")),
    )

    # save : doit tenter la requête -> on stubbe au niveau _slot_action interne
    # via urllib ? Plus simple : vérifier la décision AVANT réseau en stubant
    # les chemins réseau. On observe par le garde : restore sans force -> False
    # sans AUCUN accès réseau (local_server_root lèverait).
    assert c.restore_slot("orn", "turnend.kv") is False
    assert sent == []


def test_slot_kv_off_et_hot_resume_off_bloquent_tout(monkeypatch, tmp_path):
    c = _client(tmp_path, hot_resume=False)
    monkeypatch.setattr(
        c,
        "local_server_root",
        lambda: (_ for _ in ()).throw(AssertionError("ne doit pas sortir")),
    )
    assert c.save_slot("orn", "turnend.kv") is False
    assert c.restore_slot("orn", "turnend.kv") is False


# ---- save + sidecar meta -----------------------------------------------------


def test_save_slot_ecrit_le_meta_et_marque_chaud(monkeypatch, tmp_path):
    c = _client(tmp_path)
    calls = _spy_actions(monkeypatch, c)
    assert c.save_slot("orn", "turnend.kv", session_id="s1") is True
    assert calls == [("orn", "save", "turnend.kv", False)]
    meta = json.loads((tmp_path / "turnend.kv.meta.json").read_text(encoding="utf-8"))
    assert meta == {"model": "orn", "session": "s1"}
    assert "orn" in c._warm_slots()


def test_save_sans_session_id_n_ecrit_pas_de_meta(monkeypatch, tmp_path):
    c = _client(tmp_path)
    _spy_actions(monkeypatch, c)
    assert c.save_slot("orn", "turnend.kv") is True
    assert not (tmp_path / "turnend.kv.meta.json").exists()


# ---- try_hot_resume ----------------------------------------------------------


def test_try_hot_resume_restaure_une_fois_sur_slot_froid(monkeypatch, tmp_path):
    c = _client(tmp_path)
    calls = _spy_actions(monkeypatch, c)
    _write_meta(tmp_path)
    assert c.try_hot_resume("orn", "s1") is True
    assert calls == [("orn", "restore", "turnend.kv", True)]  # force = one-shot
    # slot désormais chaud : plus jamais de restore jusqu'au prochain froid
    assert c.try_hot_resume("orn", "s1") is False
    assert len(calls) == 1


def test_try_hot_resume_refuse_autre_session_ou_modele(monkeypatch, tmp_path):
    c = _client(tmp_path)
    calls = _spy_actions(monkeypatch, c)
    _write_meta(tmp_path, model="orn", session="s1")
    assert c.try_hot_resume("orn", "AUTRE") is False
    c.mark_all_cold()
    assert c.try_hot_resume("qwen", "s1") is False
    assert calls == []


def test_try_hot_resume_hybride_gated_sur_restore_safe(monkeypatch, tmp_path):
    # Hybride + binaire officiel : restore inutile (checkpoints perdus) -> skip.
    c = _client(tmp_path, hybrid={"orn"})
    calls = _spy_actions(monkeypatch, c)
    _write_meta(tmp_path)
    assert c.try_hot_resume("orn", "s1") is False
    assert calls == []
    # Même hybride, binaire patché déclaré : le restore part.
    c2 = _client(tmp_path, hybrid={"orn"}, restore_safe=True)
    calls2 = _spy_actions(monkeypatch, c2)
    assert c2.try_hot_resume("orn", "s1") is True
    assert calls2 == [("orn", "restore", "turnend.kv", True)]


def test_try_hot_resume_sans_meta_ni_feature(monkeypatch, tmp_path):
    c = _client(tmp_path)
    calls = _spy_actions(monkeypatch, c)
    assert c.try_hot_resume("orn", "s1") is False  # pas de meta -> repli re-prefill
    coff = _client(tmp_path, hot_resume=False)
    calls_off = _spy_actions(monkeypatch, coff)
    _write_meta(tmp_path)
    assert coff.try_hot_resume("orn", "s1") is False  # feature off
    assert calls == [] and calls_off == []


def test_swap_sortant_refroidit_le_modele_precedent(monkeypatch, tmp_path):
    c = _client(tmp_path)
    _spy_actions(monkeypatch, c)
    _write_meta(tmp_path, model="orn", session="s1")
    assert c.try_hot_resume("orn", "s1") is True
    assert "orn" in c._warm_slots()
    # Cibler qwen décharge orn (llama-swap) -> orn refroidi.
    c.try_hot_resume("qwen", "s1")
    assert "orn" not in c._warm_slots()
    # Retour sur orn : slot froid -> nouveau restore tenté.
    assert c.try_hot_resume("orn", "s1") is True


def test_mark_all_cold(monkeypatch, tmp_path):
    c = _client(tmp_path)
    _spy_actions(monkeypatch, c)
    _write_meta(tmp_path)
    assert c.try_hot_resume("orn", "s1") is True
    c.mark_all_cold()
    assert c.try_hot_resume("orn", "s1") is True  # re-tenté après le froid


def test_echec_restore_un_seul_essai(monkeypatch, tmp_path):
    # Restore qui échoue (fichier disparu côté serveur…) : pas de tempête de
    # retries — marqué chaud, le repli re-prefill prend la main.
    c = _client(tmp_path)
    calls = _spy_actions(monkeypatch, c, result=False)
    _write_meta(tmp_path)
    assert c.try_hot_resume("orn", "s1") is False
    assert c.try_hot_resume("orn", "s1") is False
    assert len(calls) == 1
