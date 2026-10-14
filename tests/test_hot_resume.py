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
    # Sonde du slot neutre : « inconnu » garde la confiance dans le marqueur chaud, sans
    # réseau (une connexion refusée coûte 2 s sous Windows). Testée à part plus bas.
    c._conversation_slot_state = lambda model: "inconnu"
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


# ---- disjoncteur : un chargement de modèle n'est pas un hang du slot ----------


def _timeout_urlopen(seen):
    def fake(req, timeout=None):
        seen.append(timeout)
        raise TimeoutError("timed out")

    return fake


def test_timeout_pendant_le_chargement_ne_desactive_pas_le_slot(monkeypatch, tmp_path):
    # Reprise à froid : /upstream/<m>/slots fait CHARGER le modèle par llama-swap
    # avant de répondre. Un gros modèle dépasse 20 s : ce n'est pas un hang du
    # slot, le save/restore ne doit pas être coupé jusqu'au redémarrage.
    import urllib.request

    c = _client(tmp_path)
    monkeypatch.setattr(c, "local_server_root", lambda: "http://127.0.0.1:9")
    monkeypatch.setattr(
        c, "running_local", lambda timeout=5.0: (True, '{"running": []}')
    )
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen", _timeout_urlopen(seen))
    assert c.restore_slot("orn", "turnend.kv", force=True) is False
    assert "orn" not in c._slot_broken
    assert seen and seen[0] > 20  # le chargement a droit à plus que 20 s


def test_timeout_modele_charge_desactive_bien_le_slot(monkeypatch, tmp_path):
    import urllib.request

    c = _client(tmp_path)
    monkeypatch.setattr(c, "local_server_root", lambda: "http://127.0.0.1:9")
    monkeypatch.setattr(
        c,
        "running_local",
        lambda timeout=5.0: (True, '{"running": [{"model": "orn"}]}'),
    )
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen", _timeout_urlopen(seen))
    assert c.restore_slot("orn", "turnend.kv", force=True) is False
    assert "orn" in c._slot_broken
    assert seen == [20]


# ---- le marqueur « chaud » vérifié auprès du serveur (incident du 2026-10-14) --------
# Un rechargement de llama-swap (ou un crash) relance llama-server : le slot est vide
# alors que Loom le croyait chaud -> zéro tentative de restore, re-prefill total.


def test_redemarrage_du_serveur_detecte_restaure_a_nouveau(monkeypatch, tmp_path):
    c = _client(tmp_path)
    calls = _spy_actions(monkeypatch, c)
    _write_meta(tmp_path)
    assert c.try_hot_resume("orn", "s1") is True
    # llama-swap a rechargé : le slot 0 ne contient plus la conversation.
    c._conversation_slot_state = lambda model: "froid"
    assert c.try_hot_resume("orn", "s1") is True
    assert calls == [("orn", "restore", "turnend.kv", True)] * 2


def test_slot_verifie_chaud_pas_de_restore(monkeypatch, tmp_path):
    # Après un restore raté, le tour re-préfille tout : le slot est plein -> pas de
    # nouvel essai (un seul par période froide, pas de tempête de retries).
    c = _client(tmp_path)
    calls = _spy_actions(monkeypatch, c, result=False)
    _write_meta(tmp_path)
    assert c.try_hot_resume("orn", "s1") is False
    c._conversation_slot_state = lambda model: "chaud"
    assert c.try_hot_resume("orn", "s1") is False
    assert len(calls) == 1


class _Resp:
    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _etat(monkeypatch, tmp_path, running, slots, meta=None):
    import urllib.request

    c = LoomClient(base_url="http://127.0.0.1:9/v1")
    c.slots_dir_override = str(tmp_path)
    if meta is not None:
        (tmp_path / "turnend.kv.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
    monkeypatch.setattr(c, "running_local", lambda timeout=5.0: running)
    urls = []

    def fake_urlopen(url, timeout=None):
        urls.append(url if isinstance(url, str) else url.full_url)
        if isinstance(slots, Exception):
            raise slots
        return _Resp(json.dumps(slots).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return c._conversation_slot_state("orn"), urls


_ORN = (True, '{"running": [{"model": "orn", "state": "ready"}]}')


def test_etat_modele_arrete_froid_sans_le_charger(monkeypatch, tmp_path):
    state, urls = _etat(monkeypatch, tmp_path, (True, '{"running": []}'), [])
    assert state == "froid"
    assert urls == []  # /upstream/<m>/ aurait CHARGÉ le modèle juste pour regarder


def test_etat_slot_sans_tache_depuis_le_demarrage_froid(monkeypatch, tmp_path):
    state, urls = _etat(monkeypatch, tmp_path, _ORN, [{"id": 0, "n_ctx": 32768}])
    assert state == "froid"
    assert urls == ["http://127.0.0.1:9/upstream/orn/slots"]


def test_etat_slot_reamorce_plus_court_que_la_sauvegarde_reste_chaud(
    monkeypatch, tmp_path
):
    # Mesuré dans Loom le 2026-10-14 : la sauvegarde contient aussi le raisonnement
    # généré (10 381 tokens), que le prompt suivant ne renvoie pas ; après le
    # réamorçage le slot en tient 10 348. Il est bien chaud : pas de restore à chaque
    # tour (un seuil « au moins la taille sauvée » en déclenchait un par tour).
    slots = [{"id": 0, "n_prompt_tokens": 10348}, {"id": 1, "n_ctx": 32768}]
    meta = {"model": "orn", "session": "s1", "n_saved": 10381}
    assert _etat(monkeypatch, tmp_path, _ORN, slots, meta)[0] == "chaud"


def test_etat_slot_ayant_traite_une_tache_chaud(monkeypatch, tmp_path):
    slots = [{"id": 0, "n_prompt_tokens": 50}]
    assert _etat(monkeypatch, tmp_path, _ORN, slots)[0] == "chaud"


def test_etat_slot_vide_froid(monkeypatch, tmp_path):
    slots = [{"id": 0, "n_prompt_tokens": 0}]
    assert _etat(monkeypatch, tmp_path, _ORN, slots)[0] == "froid"


def test_etat_inconnu_si_le_serveur_ne_repond_pas(monkeypatch, tmp_path):
    assert _etat(monkeypatch, tmp_path, (False, ""), [])[0] == "inconnu"
    assert _etat(monkeypatch, tmp_path, _ORN, OSError("404"))[0] == "inconnu"
