# Un log sans date ne se relit pas. Post-mortem du 2026-08-03 : le serveur modèle
# s'est arrêté pendant un téléchargement, et les blocs porteurs du diagnostic
# (HOT_RESUME, SLOT_RESTORE_ERR, WARM_CTX_ERR) n'avaient AUCUN horodatage — donc
# impossible de les ordonner entre eux ni de les recouper avec les lignes
# `log_event`. Ces tests verrouillent l'invariant : tout ce qui est écrit peut
# être replacé sur une chronologie.
import re

from loom.agent import debuglog
from loom.agent.debuglog import _debug, log_event, set_debug_log_path

# Horodatage ISO 8601 UTC à la milliseconde, EN TÊTE DE LIGNE (c'est ce qui permet
# de fusionner deux fichiers par simple tri).
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z ")


def _lines(path):
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_bloc_de_dump_est_horodate(tmp_path, monkeypatch):
    monkeypatch.delenv("LOOM_DEBUG", raising=False)
    log = tmp_path / "debug.log"
    set_debug_log_path(log)

    _debug("HOT_RESUME", {"model": "ornith", "ok": False}, terminal=False)

    entete = next(ln for ln in _lines(log) if "LOOM_DEBUG" in ln)
    assert _TS.match(entete), f"en-tête non horodatée : {entete!r}"
    # Le label reste greppable tel quel : les outils existants ne cassent pas.
    assert "[LOOM_DEBUG] HOT_RESUME" in entete


def test_blocs_et_evenements_se_trient_ensemble(tmp_path, monkeypatch):
    """Le vrai besoin : reconstituer l'ordre des faits. Blocs et événements
    structurés doivent partager le même format d'horodatage en tête de ligne."""
    monkeypatch.delenv("LOOM_DEBUG", raising=False)
    log = tmp_path / "debug.log"
    set_debug_log_path(log)

    log_event("turn.request", model="ornith", msgs=42)
    _debug("SLOT_RESTORE_ERR", "connexion refusee", terminal=False)
    log_event("api.error", level="WARN", kind="connection")

    dates = [ln[:24] for ln in _lines(log) if _TS.match(ln)]
    assert len(dates) == 3, f"attendu 3 lignes datées, obtenu {len(dates)}"
    assert dates == sorted(dates), "l'ordre chronologique n'est pas préservé"


def test_desactivation_reste_silencieuse(tmp_path, monkeypatch):
    """`LOOM_DEBUG=0` coupe tout : l'horodatage ne doit pas réintroduire d'écriture."""
    monkeypatch.setenv("LOOM_DEBUG", "0")
    log = tmp_path / "debug.log"
    set_debug_log_path(log)

    _debug("REQUETE", "payload", terminal=False)
    log_event("turn.request", model="ornith")

    assert not log.exists() or _lines(log) == []


def test_horodatage_ne_leve_jamais(tmp_path, monkeypatch):
    """Le debug est best-effort : une horloge qui casse ne doit pas tuer un tour."""
    monkeypatch.delenv("LOOM_DEBUG", raising=False)
    set_debug_log_path(tmp_path / "debug.log")
    monkeypatch.setattr(debuglog, "_ts", lambda: 1 / 0)

    _debug("REQUETE", "payload", terminal=False)  # ne doit pas lever
