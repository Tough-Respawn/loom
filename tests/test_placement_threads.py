# tests/test_placement_threads.py
"""Threads mesurés sur le placement ÉLU (option « par modèle », 2026-10-10).

Les threads venaient d'un llama-bench de 128 tokens sur la configuration « doctrine »
(experts en RAM pour un MoE), une répétition, puis servaient à tous les placements. Un
placement experts-CPU et un placement tout-GPU n'ont pas la même charge CPU. Après le
2x2, les candidats de threads (physiques, physiques/2, logiques) sont mesurés sur la
configuration élue, au même contexte, à la même profondeur, aux mêmes slots, tours
alternés ; la génération décide, le prefill départage, l'actuel reste en cas
d'indécision. Tout-GPU : non exploré, et la trace le dit. Le résultat va dans
model.toml (`threads`), prioritaire sur l'override machine.
"""

from __future__ import annotations

import tomllib

from loom.setup.placement import (
    Placement,
    needs_cpu_compute,
    probe_threads,
    thread_options,
)
from loom.setup.topology import ProbeResult


def test_options_de_threads_actuel_puis_candidats_du_parc():
    opts = thread_options(current=8, logical=16, physical=8)
    assert [o.threads for o in opts] == [8, 4, 16]
    assert opts[0].actuel is True and opts[1].actuel is False
    assert opts[0].key == "t8" and "8 threads" in opts[0].describe()
    # Machine hybride 10 physiques / 12 logiques : 5, 10, 12 — l'actuel d'abord.
    assert [o.threads for o in thread_options(current=10, logical=12, physical=10)] == [
        10,
        5,
        12,
    ]
    # Un actuel hors du parc reste la base ; physique inconnu -> logique.
    assert [o.threads for o in thread_options(current=6, logical=8, physical=None)] == [
        6,
        4,
        8,
    ]


def test_tout_gpu_n_a_pas_de_calcul_cpu_a_regler():
    assert needs_cpu_compute(Placement("gpu_total", 999)) is False
    assert needs_cpu_compute(Placement("experts_cpu", 999, cpu_moe=True)) is True
    assert needs_cpu_compute(Placement("experts_partiel", 999, n_cpu_moe=20)) is True
    assert needs_cpu_compute(Placement("gpu_partiel", 36)) is True
    assert needs_cpu_compute(Placement("cpu", 0)) is True


class _Sonde:
    """Rejoue (tg, pp) par nombre de threads ; journalise (threads, ctx, depth)."""

    def __init__(self, threads, table, journal):
        self.threads, self.table, self.journal = threads, table, journal
        self.ubatch, self.batch = 512, 2048

    def run(self, ctx, depth):
        self.journal.append((self.threads, ctx, depth))
        tg, pp = self.table[self.threads]
        if tg == "boom":
            raise RuntimeError("serveur KO")
        return ProbeResult(ctx=ctx, mem_mb=100, tg_ts=tg, pp_ts=pp)


def _usine(table):
    journal = []

    def make(option):
        return _Sonde(option.threads, table, journal)

    make.journal = journal
    return make


OPTS = thread_options(current=8, logical=16, physical=8)


def test_sonde_de_threads_alternee_decision_a_la_generation():
    make = _usine({8: (10.0, 200.0), 4: (9.0, 190.0), 16: (11.0, 210.0)})
    r = probe_threads(make, OPTS, ctx=32_768, depth=16_384, reps=2)
    # Tours alternés sur la configuration élue, même contexte, même profondeur.
    assert [j[0] for j in make.journal] == [8, 4, 16, 8, 4, 16]
    assert all(j[1:] == (32_768, 16_384) for j in make.journal)
    assert r["threads"] == 16 and r["baseline"] == 8 and r["gain_pct"] == 10.0
    assert r["compare"] is True and "t16" in r["mecanisme"]
    assert r["mesures"]["t4"]["tg_ts"] == 9.0


def test_sonde_de_threads_indecise_garde_l_actuel():
    make = _usine({8: (10.0, 200.0), 4: (10.2, 205.0), 16: (10.3, 203.0)})
    r = probe_threads(make, OPTS, ctx=32_768, depth=16_384, reps=1)
    assert r["threads"] == 8 and r["gain_pct"] is None
    assert "conservé" in r["mecanisme"]


def test_sonde_de_threads_prefill_departage_a_generation_egale():
    make = _usine({8: (10.0, 200.0), 4: (10.1, 260.0), 16: (10.0, 150.0)})
    r = probe_threads(make, OPTS, ctx=32_768, depth=16_384, reps=1)
    assert r["threads"] == 4 and "prefill" in r["mecanisme"]


def test_sonde_de_threads_option_en_echec_ecartee():
    make = _usine({8: (10.0, 200.0), 4: ("boom", 0), 16: (12.0, 210.0)})
    r = probe_threads(make, OPTS, ctx=32_768, depth=16_384, reps=1)
    assert r["threads"] == 16 and r["mesures"]["t4"]["echec"].startswith("RuntimeError")


def test_sonde_de_threads_respecte_la_contrainte_de_prefill():
    """Revue P3 (2026-10-10) : la sonde de threads ignorait les contraintes de prefill.
    « 1 000 tokens en 10 s » : 50 t/s = 20 s -> écarté même s'il génère plus vite."""
    from loom.setup.placement import PrefillConstraint

    make = _usine({8: (10.0, 200.0), 4: (12.0, 50.0), 16: (10.5, 210.0)})
    r = probe_threads(
        make,
        OPTS,
        ctx=32_768,
        depth=16_384,
        reps=1,
        prefill=PrefillConstraint(new_tokens=1_000, max_seconds=10.0),
    )
    assert r["threads"] != 4 and "contrainte prefill" in r["mecanisme"]


def test_validation_finale_verifie_la_contrainte_de_prefill():
    from loom.setup.placement import PrefillConstraint, validate_final

    class _Sonde:
        ngl, cpu_moe, n_cpu_moe, n_parallel, ubatch, batch = (
            999,
            False,
            None,
            2,
            512,
            2048,
        )

        def run(self, ctx, depth):
            return ProbeResult(
                ctx=ctx, mem_mb=1, tg_ts=11.0, pp_ts=50.0, prompt_n=depth
            )

    f = validate_final(
        _Sonde(),
        ctx=65_536,
        depth=16_384,
        n_layers=41,
        prefill=PrefillConstraint(new_tokens=1_000, max_seconds=10.0),
    )
    assert f["prefill_contrainte"]["respectee"] is False
    assert f["prefill_contrainte"]["secondes"] == 20.0 and f["coherent"] is False
    ok = validate_final(
        _Sonde(),
        ctx=65_536,
        depth=16_384,
        n_layers=41,
        prefill=PrefillConstraint(new_tokens=1_000, max_seconds=30.0),
    )
    assert ok["prefill_contrainte"]["respectee"] is True and ok["coherent"] is None


def test_sonde_de_threads_muette_sans_mesure():
    def _boom(option):
        raise OSError("serveur KO")

    assert probe_threads(_boom, OPTS, ctx=32_768, depth=16_384, reps=1) is None


def test_set_model_threads_ecrit_et_remplace(tmp_path):
    from loom.setup.cli import _set_model_threads

    p = tmp_path / "model.toml"
    p.write_text('filename = "m.gguf"\ncontext = 8192\n', encoding="utf-8")
    gguf = tmp_path / "m.gguf"
    _set_model_threads(gguf, 5, "6,0 contre 5,0 t/s (+20 %) sur gpu_partiel_ngl11")
    d = tomllib.loads(p.read_text(encoding="utf-8"))
    assert d["threads"] == 5 and d["context"] == 8192
    txt = p.read_text(encoding="utf-8")
    assert "threads élus par la sonde" in txt and "+20" in txt
    _set_model_threads(gguf, 10, "nouvelle mesure")
    txt = p.read_text(encoding="utf-8")
    assert txt.count("threads =") == 1 and txt.count("threads élus par la sonde") == 1
    assert tomllib.loads(txt)["threads"] == 10
