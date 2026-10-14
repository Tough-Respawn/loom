# llama-swap recharge sa config de façon GLOBALE (il remplace son serveur et arrête
# TOUS les modèles, même ceux dont l'entrée n'a pas changé — vérifié sur v241 et dans
# le code de v264). Une écriture du yaml qui ne concerne pas le modèle actif doit donc
# être DIFFÉRÉE jusqu'à une transition qui l'arrête de toute façon (incident du
# 2026-10-14 : supprimer un autre modèle a coupé Bonsai et perdu 10 258 tokens de cache).
from __future__ import annotations

import yaml

from loom.runtime import swap_sync


def _yaml(models: dict) -> str:
    return yaml.safe_dump(
        {"healthCheckTimeout": 600, "models": models}, sort_keys=False
    )


A = {"cmd": "llama-server -m a.gguf"}
B = {"cmd": "llama-server -m b.gguf"}
C = {"cmd": "llama-server -m c.gguf"}


# ---- décision -------------------------------------------------------------------


def test_contenu_identique_rien_a_faire():
    cur = _yaml({"A": A})
    assert swap_sync.decide(cur, cur, {"A"}) == "unchanged"


def test_rien_ne_tourne_ecriture_immediate():
    assert swap_sync.decide(_yaml({"A": A, "B": B}), _yaml({"A": A}), set()) == "write"


def test_pas_de_fichier_ecriture_immediate():
    assert swap_sync.decide(None, _yaml({"A": A}), {"A"}) == "write"


def test_suppression_d_un_autre_modele_differee():
    # Le cas de l'incident : A tourne, on retire B -> A ne doit pas être rechargé.
    cur, new = _yaml({"A": A, "B": B}), _yaml({"A": A})
    assert swap_sync.decide(cur, new, {"A"}) == "defer"


def test_ajout_d_un_modele_differe():
    cur, new = _yaml({"A": A}), _yaml({"A": A, "C": C})
    assert swap_sync.decide(cur, new, {"A"}) == "defer"


def test_reglage_d_un_autre_modele_differe():
    cur = _yaml({"A": A, "B": B})
    new = _yaml({"A": A, "B": {"cmd": B["cmd"] + " -c 8192"}})
    assert swap_sync.decide(cur, new, {"A"}) == "defer"


def test_entree_du_modele_actif_modifiee_ecriture_immediate():
    # Changement voulu pour le modèle qui tourne (paramètre serveur, réglage du modèle
    # actif) : il doit être relancé de toute façon.
    cur = _yaml({"A": A, "B": B})
    new = _yaml({"A": {"cmd": A["cmd"] + " -c 8192"}, "B": B})
    assert swap_sync.decide(cur, new, {"A"}) == "write"


def test_modele_actif_retire_ecriture_immediate():
    cur, new = _yaml({"A": A, "B": B}), _yaml({"B": B})
    assert swap_sync.decide(cur, new, {"A"}) == "write"


def test_etat_inconnu_differe_par_prudence():
    # /running illisible : impossible de savoir ce qui tourne -> ne rien interrompre.
    assert swap_sync.decide(_yaml({"A": A, "B": B}), _yaml({"A": A}), None) == "defer"


# ---- lecture de /running ------------------------------------------------------------


def test_running_models_lit_le_json_de_llama_swap():
    txt = '{"running": [{"model": "A", "state": "ready"}, {"model": "B", "state": "starting"}]}'
    assert swap_sync.running_models(txt) == {"A", "B"}
    assert swap_sync.running_models('{"running": []}') == set()


def test_running_models_illisible_renvoie_none():
    # Pas d'égalité par sous-chaîne : « bonsai-2-27b » n'est pas « bonsai-2-27b-q2_0 ».
    assert swap_sync.running_models("") is None
    assert swap_sync.running_models("pas du json") is None
    assert swap_sync.running_models('{"data": [{"id": "x"}]}') is None  # /v1/models


def test_running_models_sans_egalite_par_sous_chaine():
    txt = '{"running": [{"model": "ternary-bonsai-2-27b-q2_0", "state": "ready"}]}'
    assert "ternary-bonsai-2-27b" not in swap_sync.running_models(txt)


# ---- écriture ou mise en attente ----------------------------------------------------


def test_write_or_defer_met_en_attente_sans_toucher_au_yaml(tmp_path):
    y = tmp_path / "llama-swap.yaml"
    cur = _yaml({"A": A, "B": B})
    y.write_text(cur, encoding="utf-8")
    new = _yaml({"A": A})
    assert swap_sync.write_or_defer(new, y, {"A"}) == "deferred"
    assert y.read_text(encoding="utf-8") == cur  # llama-swap ne voit RIEN changer
    assert swap_sync.pending_path(y).read_text(encoding="utf-8") == new


def test_write_or_defer_ecrit_et_supprime_l_attente(tmp_path):
    y = tmp_path / "llama-swap.yaml"
    y.write_text(_yaml({"A": A, "B": B}), encoding="utf-8")
    swap_sync.pending_path(y).write_text("ancienne attente", encoding="utf-8")
    new = _yaml({"A": A})
    assert swap_sync.write_or_defer(new, y, set()) == "written"
    assert y.read_text(encoding="utf-8") == new
    assert not swap_sync.pending_path(y).exists()  # l'attente est dépassée


def test_retour_a_la_config_appliquee_annule_l_attente(tmp_path):
    y = tmp_path / "llama-swap.yaml"
    cur = _yaml({"A": A, "B": B})
    y.write_text(cur, encoding="utf-8")
    swap_sync.pending_path(y).write_text(_yaml({"A": A}), encoding="utf-8")
    assert swap_sync.write_or_defer(cur, y, {"A"}) == "unchanged"
    assert not swap_sync.pending_path(y).exists()


# ---- application à la transition --------------------------------------------------


def test_apply_pending_pose_la_version_en_attente(tmp_path):
    y = tmp_path / "llama-swap.yaml"
    y.write_text("ancien", encoding="utf-8")
    assert swap_sync.apply_pending(y) is False  # rien n'attendait
    swap_sync.pending_path(y).write_text("nouveau", encoding="utf-8")
    assert swap_sync.apply_pending(y) is True
    assert y.read_text(encoding="utf-8") == "nouveau"
    assert not swap_sync.pending_path(y).exists()


# ---- « configuration écrite » != « configuration chargée » ------------------------
# Toute écriture du yaml surveillé (immédiate OU à la transition) ouvre un état
# « rechargement attendu » : la requête suivante attend que llama-swap l'ait chargé,
# sinon elle part vers l'ancien catalogue (404 sur un modèle ajouté, reproduit avec
# le vrai llama-swap en revue).

OLD = _yaml({"A": A, "B": B})
NEW = _yaml({"A": A, "C": C})


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += max(dt, 0.01)


def test_watch_inactif_ne_coute_rien():
    w = swap_sync.ReloadWatch()
    assert not w.active
    called = []
    assert w.wait("r", fetch_logs=lambda r: called.append(r)) is True
    assert called == []


def test_watch_confirme_par_le_journal():
    w = swap_sync.ReloadWatch()
    before = "[INFO] llama-swap listening\n"
    w.mark(before, OLD, NEW)
    logs = iter(
        [
            before,
            before + "[INFO] reloading configuration\n",
            before + "[INFO] reloading configuration\n[INFO] configuration reloaded\n",
        ]
    )
    clock = _Clock()
    assert (
        w.wait("r", fetch_logs=lambda r: next(logs), clock=clock, sleep=clock.sleep)
        is True
    )
    assert not w.active


def test_watch_ignore_un_rechargement_anterieur():
    w = swap_sync.ReloadWatch()
    old_logs = (
        "[INFO] reloading configuration\n[INFO] configuration reloaded\n[INFO] x\n"
    )
    w.mark(old_logs, OLD, NEW)
    logs = iter([old_logs, old_logs + "[INFO] configuration reloaded\n"])
    clock = _Clock()
    assert (
        w.wait("r", fetch_logs=lambda r: next(logs), clock=clock, sleep=clock.sleep)
        is True
    )


def test_watch_sans_journal_le_catalogue_confirme():
    # /logs indisponible : le nouveau catalogue servi par /v1/models fait foi (B retiré,
    # C ajouté -> la liste servie ne correspond qu'après rechargement).
    w = swap_sync.ReloadWatch()
    w.mark(None, OLD, NEW)
    served = iter([{"A", "B"}, {"A", "B"}, {"A", "C"}])
    clock = _Clock()
    ok = w.wait(
        "r",
        fetch_logs=lambda r: None,
        fetch_models=lambda r: next(served),
        clock=clock,
        sleep=clock.sleep,
    )
    assert ok is True
    assert not w.active


def test_watch_sans_journal_catalogue_inchange_ne_confirme_rien():
    # Même liste de modèles avant/après (réglage d'un modèle) et pas de journal : rien
    # n'atteste le rechargement. Un délai écoulé n'est PAS une preuve (revue post-
    # fusion du 2026-10-14) : l'état reste en attente, l'appelant refusera.
    w = swap_sync.ReloadWatch()
    same = _yaml({"A": {"cmd": A["cmd"] + " -c 4096"}, "B": B})
    clock = _Clock()
    w.mark(None, OLD, same)
    ok = w.wait(
        "r",
        timeout=30,
        fetch_logs=lambda r: None,
        fetch_models=lambda r: {"A", "B"},
        clock=clock,
        sleep=clock.sleep,
    )
    assert ok is False
    assert w.active
    assert clock.t >= 30


def test_watch_delai_depasse_reste_actif():
    w = swap_sync.ReloadWatch()
    w.mark("x", OLD, NEW)
    clock = _Clock()
    ok = w.wait(
        "r",
        timeout=1.0,
        fetch_logs=lambda r: "x",
        fetch_models=lambda r: {"A", "B"},
        clock=clock,
        sleep=clock.sleep,
    )
    assert ok is False
    assert (
        w.active
    )  # la requête suivante attendra encore : rien ne part vers l'ancien catalogue


def test_fetch_models_lit_v1_models():
    body = '{"object": "list", "data": [{"id": "A"}, {"id": "C"}]}'
    assert swap_sync.models_from_v1(body) == {"A", "C"}
    assert swap_sync.models_from_v1("pas du json") is None
