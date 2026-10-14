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


def test_apply_pending_sans_attente_ne_fait_rien(tmp_path):
    y = tmp_path / "llama-swap.yaml"
    y.write_text("x", encoding="utf-8")
    assert swap_sync.apply_pending(y, root=None) is None
    assert y.read_text(encoding="utf-8") == "x"


def test_apply_pending_attend_la_fin_du_rechargement(tmp_path):
    # La fin du rechargement se lit dans /logs : « configuration reloaded » apparaît
    # APRÈS l'arrêt de l'ancien serveur (vérifié sur v241 : 1,4 s après l'écriture).
    y = tmp_path / "llama-swap.yaml"
    y.write_text(_yaml({"A": A, "B": B}), encoding="utf-8")
    new = _yaml({"A": A, "C": C})
    swap_sync.pending_path(y).write_text(new, encoding="utf-8")
    before = "[INFO] llama-swap listening\n[INFO] <A> Health check passed\n"
    logs = iter(
        [
            before,
            before,
            before + "[INFO] reloading configuration\n",
            before + "[INFO] reloading configuration\n[INFO] configuration reloaded\n",
        ]
    )
    seen = []

    def fetch(root):
        seen.append(root)
        return next(logs)

    out = swap_sync.apply_pending(
        y, root="http://swap", fetch_logs=fetch, poll=0, timeout=5
    )
    assert out == "applied"
    assert y.read_text(encoding="utf-8") == new
    assert not swap_sync.pending_path(y).exists()
    assert len(seen) == 4  # 1 ancre avant l'écriture + 3 relevés jusqu'au signal


def test_apply_pending_ignore_un_ancien_rechargement(tmp_path):
    # Un « configuration reloaded » déjà présent AVANT l'écriture ne compte pas.
    y = tmp_path / "llama-swap.yaml"
    y.write_text("ancien", encoding="utf-8")
    swap_sync.pending_path(y).write_text("nouveau", encoding="utf-8")
    old = "[INFO] reloading configuration\n[INFO] configuration reloaded\n[INFO] x\n"
    logs = iter([old, old, old + "[INFO] configuration reloaded\n"])
    out = swap_sync.apply_pending(
        y, root="r", fetch_logs=lambda r: next(logs), poll=0, timeout=5
    )
    assert out == "applied"


def test_apply_pending_delai_depasse_applique_quand_meme(tmp_path):
    y = tmp_path / "llama-swap.yaml"
    y.write_text("ancien", encoding="utf-8")
    swap_sync.pending_path(y).write_text("nouveau", encoding="utf-8")
    out = swap_sync.apply_pending(
        y, root="r", fetch_logs=lambda r: "rien", poll=0, timeout=0.05
    )
    assert out == "applied_unconfirmed"
    assert y.read_text(encoding="utf-8") == "nouveau"


def test_apply_pending_llama_swap_injoignable(tmp_path):
    # Rien n'écoute : pas de rechargement à attendre, le fichier est simplement posé.
    y = tmp_path / "llama-swap.yaml"
    y.write_text("ancien", encoding="utf-8")
    swap_sync.pending_path(y).write_text("nouveau", encoding="utf-8")
    assert swap_sync.apply_pending(y, root=None) == "applied_unconfirmed"
    assert y.read_text(encoding="utf-8") == "nouveau"
