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

    def fake_render(S, conv, workspace=None, scratch=None):
        state.renders += 1
        return (
            f"MEM:{state.memory}|BASE:{conv.system_prompt}|GOAL:{conv.goal}"
            f"|WS:{workspace}|SCR:{scratch}",
            False,
        )

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


def test_bac_a_sable_cree_et_annonce(S, tmp_path):
    """Le bac à sable de la session est créé et passé au rendu ; sans lui, rien ne
    change (sous-agents, anciens appelants)."""
    scratch = tmp_path / "sessions" / "abc" / "scratch"
    conv = _conv()
    text, _ = sp._build_system_prompt(S, conv, workspace="C:/a", scratch=str(scratch))
    assert scratch.is_dir() and str(scratch) in text
    assert "SCR:None" in sp._build_system_prompt(S, _conv(), workspace="C:/a")[0]
    # Le texte réel dit le chemin, l'environnement jetable uv et la voie « figure ».
    section = sp._scratch_text(str(scratch))
    assert str(scratch) in section
    assert "uv run --with" in section and "check_page" in section
    assert "ModuleNotFoundError" in section


def test_mise_a_jour_du_harnais_refige_et_rafraichit_le_prompt_de_base(S, monkeypatch):
    """Une session ouverte avant une mise à jour de Loom reçoit le nouveau gabarit au
    tour suivant (vécu : règle « uv run --with » invisible dans une session figée)."""
    S.session_store = NS(default_system_prompt="base v2 (gabarit livré)")
    conv = _conv()
    first, _ = sp._build_system_prompt(S, conv, workspace="C:/a")
    assert conv.frozen_prompt["build"] == sp._prompt_build()
    # Même harnais : rien ne bouge, même avec un gabarit par défaut différent.
    assert sp._build_system_prompt(S, conv, workspace="C:/a")[0] == first
    assert conv.system_prompt == "base"
    # Harnais mis à jour (empreinte différente) : refige + prompt de base rafraîchi.
    monkeypatch.setattr(sp, "_prompt_build", lambda: "nouveau1")
    text, _ = sp._build_system_prompt(S, conv, workspace="C:/a")
    assert text != first and conv.system_prompt == "base v2 (gabarit livré)"
    assert conv.frozen_prompt["build"] == "nouveau1" and S.state.renders == 2
    # Un instantané d'avant cette version (sans `build`) n'est pas refigé pour autant.
    conv2 = _conv()
    sp._build_system_prompt(S, conv2, workspace="C:/a")
    conv2.frozen_prompt.pop("build")
    assert S.state.renders == 3
    sp._build_system_prompt(S, conv2, workspace="C:/a")
    assert S.state.renders == 3 and conv2.system_prompt == "base"


def test_chemin_du_bac_a_sable_en_barres_obliques(S, tmp_path):
    """Avec des antislashs, Bonsai a réécrit le chemin en /var/sessions/<id>/scratch."""
    S.session_store = NS(session_dir=lambda sid: tmp_path / "sessions" / sid)
    path = sp._scratch_dir(S, "abc")
    assert "\\" not in path and path.endswith("/sessions/abc/scratch")


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
