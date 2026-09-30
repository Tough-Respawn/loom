"""Outil de trace : recalage des horloges client/serveur, alertes, rapport."""

from datetime import datetime, timedelta, timezone

from loom.diag import trace

ORIGIN = datetime(2026, 9, 30, 9, 15, 41, 150000, tzinfo=timezone.utc)


def _ts(seconds: float) -> str:
    t = ORIGIN + timedelta(seconds=seconds)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _rel(seconds: float) -> str:
    us = round(seconds * 1e6)
    return f"{us // 60_000_000}.{us // 1_000_000 % 60:02d}.{us // 1000 % 1000:03d}.{us % 1000:03d}"


def _session(tmp_path):
    sdir = tmp_path / "sessions" / "abc"
    sdir.mkdir(parents=True)
    client = [
        (10.0, "call.request purpose=turn model=bonsai slot=0 msgs=2 prefixe=premier"),
        (
            250.0,
            "call.end purpose=turn model=bonsai slot=0 issue=ok duree_s=240.0 cache_tok=0 prefill_tok=8844 prefill_s=213.2 generation_tok=111",
        ),
        (
            251.0,
            "maint.start model=bonsai local=true reflect=true kv_saved=true titre=false",
        ),
        (
            252.0,
            'call.request purpose=reflect model=bonsai slot=0 msgs=1 prefixe=diverge diverge_a="system « (début) »"',
        ),
        (
            320.0,
            "call.end purpose=reflect model=bonsai slot=0 issue=ok duree_s=68.0 cache_tok=0 prefill_tok=1356 prefill_s=27.9 generation_tok=230",
        ),
        (
            321.0,
            "slot.action action=restore fichier=turnend.kv model=bonsai force=false resultat=refuse_slot_kv_off duree_s=0.0",
        ),
        (
            400.0,
            'call.request purpose=turn model=bonsai slot=0 msgs=4 prefixe=diverge diverge_a="system « Mémoire durable »"',
        ),
        (
            400.0,
            'prefix.diff purpose=turn model=bonsai slot=0 element="system « Mémoire durable »" index=3 avant=100:aaaa apres=140:bbbb elements_communs=3',
        ),
        (
            600.0,
            "call.end purpose=turn model=bonsai slot=0 issue=ok duree_s=200.0 cache_tok=0 prefill_tok=9352 prefill_s=186.0 generation_tok=91",
        ),
        (601.0, "maint.end duree_s=0.5 interrompue=true"),
    ]
    (sdir / "debug.log").write_text(
        "\n".join(f"{_ts(s)} [DEBUG] {txt}" for s, txt in client) + "\n",
        encoding="utf-8",
    )
    server = [
        (5.5, "I srv  llama_server: model loaded"),
        (10.07, "I slot launch_slot_: id  0 | task 0 | processing task, is_child = 0"),
        (
            250.0,
            "I slot      release: id  0 | task 0 | stop processing: n_tokens = 8954, truncated = 0",
        ),
        (
            252.1,
            "I slot launch_slot_: id  0 | task 117 | processing task, is_child = 0",
        ),
        (
            320.0,
            "I slot      release: id  0 | task 117 | stop processing: n_tokens = 1585, truncated = 0",
        ),
        (
            400.1,
            "I slot launch_slot_: id  0 | task 120 | processing task, is_child = 0",
        ),
        (
            600.0,
            "I slot      release: id  0 | task 120 | stop processing: n_tokens = 9442, truncated = 0",
        ),
    ]
    slog = tmp_path / "bonsai.log"
    slog.write_text(
        "\n".join(f"{_rel(s)} {txt}" for s, txt in server) + "\n", encoding="utf-8"
    )
    return tmp_path / "sessions", slog


def test_parse_client_valeurs_typees(tmp_path):
    root, _ = _session(tmp_path)
    events = trace.parse_client_log(root / "abc" / "debug.log")
    end = next(e for e in events if e.name == "call.end")
    assert end.fields["prefill_tok"] == 8844 and end.fields["prefill_s"] == 213.2
    diff = next(e for e in events if e.name == "prefix.diff")
    assert diff.fields["element"] == "system « Mémoire durable »"


def test_recalage_retrouve_le_demarrage_serveur(tmp_path):
    root, slog = _session(tmp_path)
    client = trace.parse_client_log(root / "abc" / "debug.log")
    path, n, total, events = trace.load_server_events(client, "bonsai", slog)
    assert (n, total) == (3, 3)
    launch = next(e for e in events if "task 0 " in e.text and "launch" in e.text)
    assert abs((launch.t - (ORIGIN + timedelta(seconds=10.07))).total_seconds()) < 0.01


def test_alertes(tmp_path):
    root, _ = _session(tmp_path)
    found = trace.alerts(trace.parse_client_log(root / "abc" / "debug.log"))
    text = "\n".join(found)
    assert "Mémoire durable" in text  # la section modifiée est nommée
    assert "pris par un appel reflect" in text  # slot de la conversation occupé
    assert "refusée : refuse_slot_kv_off (1 fois" in text  # regroupé
    assert "recalcul important (turn) : 9352" in text
    assert "maintenance interrompue" in text


def test_rapport_complet(tmp_path):
    root, slog = _session(tmp_path)
    report = trace.build_report("abc", slog, sessions_root=root)
    assert "recalé sur 3/3" in report
    assert "| reflect | 0 | ok |" in report
    assert "[tâche 117]" in report  # lignes serveur fusionnées
