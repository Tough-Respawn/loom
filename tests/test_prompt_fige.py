"""System prompt figé par session : seules les décisions explicites le refigent, un
changement de dossier s'annonce en fin de fil (append-only, préfixe KV stable)."""

from types import SimpleNamespace as NS

import pytest

from loom.agent.conversation import Conversation
from loom.web.routes import system_prompt as sp


@pytest.fixture
def S(monkeypatch):
    """Rendu simulé : il reflète une « mémoire » qui change pendant la session."""
    state = NS(memory="note 1", renders=0)

    def fake_render(S, conv, workspace=None):
        state.renders += 1
        return f"MEM:{state.memory}|GOAL:{conv.goal}|WS:{workspace}", False

    monkeypatch.setattr(sp, "_render_system_prompt", fake_render)
    monkeypatch.setattr(sp, "_workspace_text", lambda S, ws: f"consigne {ws}")
    monkeypatch.setattr(sp, "log_event", lambda *a, **k: None)
    return NS(state=state, remote_model_ids=set(), remote_weak_ids=set())


def _conv():
    return Conversation(system_prompt="base", model="bonsai")


def test_memoire_modifiee_ne_reecrit_pas_le_prompt(S):
    conv = _conv()
    first, _ = sp._build_system_prompt(S, conv, workspace="C:/a")
    S.state.memory = "note 2 (écrite par reflect)"
    again, _ = sp._build_system_prompt(S, conv, workspace="C:/b")
    assert again == first and "note 1" in again and "C:/a" in again
    assert S.state.renders == 1


def test_choix_explicite_refige(S):
    conv = _conv()
    sp._build_system_prompt(S, conv, workspace="C:/a")
    S.state.memory = "note 2"
    conv.set_goal("livrer le correctif")
    text, _ = sp._build_system_prompt(S, conv, workspace="C:/a")
    assert "livrer le correctif" in text and "note 2" in text
    assert S.state.renders == 2


def test_changement_de_dossier_annonce_une_seule_fois(S):
    conv = _conv()
    sp._build_system_prompt(S, conv, workspace="C:/a")
    assert sp._workspace_note(S, conv, "C:/a") == ""
    note = sp._workspace_note(S, conv, "C:/b")
    assert note.startswith("[Changement de dossier de travail : C:/b]")
    conv.add("user", note + "salut")
    assert sp._workspace_note(S, conv, "C:/b") == ""  # déjà dans le fil
    # retour au dossier figé : il faut le réannoncer (le fil dit C:/b)
    assert sp._workspace_note(S, conv, "C:/a").startswith(
        "[Changement de dossier de travail : C:/a]"
    )


def test_note_retiree_du_fil_est_reannoncee(S):
    # /fork ou compaction retire le message portant la note
    conv = _conv()
    sp._build_system_prompt(S, conv, workspace="C:/a")
    conv.add("user", sp._workspace_note(S, conv, "C:/b") + "salut")
    conv.messages = []
    assert sp._workspace_note(S, conv, "C:/b") != ""


def test_texte_utilisateur_sans_la_note(S):
    conv = _conv()
    sp._build_system_prompt(S, conv, workspace="C:/a")
    msg = "ligne 1\n\nligne 2"
    assert sp.strip_workspace_note(sp._workspace_note(S, conv, "C:/b") + msg) == msg
    assert sp.strip_workspace_note(msg) == msg


def test_instantane_persiste_et_efface_au_reset():
    conv = _conv()
    conv.frozen_prompt = {"key": "k", "text": "T", "strong": False, "workspace": "w"}
    back = Conversation.from_dict(conv.to_dict(), "defaut")
    assert back.frozen_prompt == conv.frozen_prompt
    back.reset()
    assert back.frozen_prompt is None
