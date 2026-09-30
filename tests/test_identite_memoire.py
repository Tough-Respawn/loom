"""Mémoire d'identité : notes de reflect au format objet, troncature par lignes et
journalisée (elle était silencieuse et coupait les notes récentes en plein milieu)."""

from loom.agent import reflect
from loom.memory import identity


def test_reflect_accepte_les_notes_au_format_objet():
    res = reflect.validate_reflect_json(
        {"user_updates": [{"text": "Aime les réponses courtes."}, "Sur Windows."]}
    )
    assert res.user_updates == ["Aime les réponses courtes.", "Sur Windows."]


def test_troncature_par_lignes_et_journalisee(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "loom.agent.debuglog.log_event", lambda ev, **f: seen.append((ev, f))
    )
    mem = tmp_path / "MEMORY.md"
    mem.write_text("\n".join(f"note {i} " + "x" * 30 for i in range(20)), "utf-8")
    block = identity.identity_block(
        str(tmp_path / "SOUL.md"), str(tmp_path / "USER.md"), str(mem), max_tokens=100
    )
    lines = block.splitlines()
    assert all(ln.endswith("x" * 30) for ln in lines[1:-1])  # aucune note coupée
    assert lines[-1].startswith("[…") and "ligne(s) tronquée(s)" in lines[-1]
    ev, fields = seen[0]
    assert ev == "identity.tronquee" and fields["lignes_perdues"] > 0


def test_sans_depassement_rien_ne_change(tmp_path):
    mem = tmp_path / "MEMORY.md"
    mem.write_text("note courte", "utf-8")
    block = identity.identity_block(
        str(tmp_path / "S.md"), str(tmp_path / "U.md"), str(mem), max_tokens=100
    )
    assert block == "# Mémoire durable (MEMORY)\nnote courte"
