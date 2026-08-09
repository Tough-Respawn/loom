# Sonde d'ubatch : élit (ubatch, batch) au prefill MESURÉ, via le VRAI llama-server
# (topology.ServerProbe) — pas llama-bench, absent des builds maison qui ne compilent
# que la cible serveur (vécu 2026-08-03 : build-vulkan sans llama-bench, la sonde ne
# tournait jamais). Sans elle, un modèle ajouté par /add-model tombait sur l'ubatch
# 512 par défaut — 61 % de prefill perdus, mesuré le 2026-07-21.
#
# Le point critique reste la LONGUEUR DU PROMPT : à 128 tokens tout tient dans un
# seul micro-batch et le levier n'a aucun effet.
from dataclasses import replace

from loom.setup.bench import (
    UBATCH_CANDIDATES,
    UBATCH_PROBE_CTX,
    UBATCH_PROBE_PROMPT,
    probe_ubatch,
)
from loom.setup.topology import ProbeResult, ServerProbe


class _Probe:
    """Sonde factice : rejoue un débit de prefill selon l'ubatch demandé."""

    def __init__(self, ub, pp_ts, journal):
        self.ub = ub
        self.pp_ts = pp_ts
        self.journal = journal

    def run(self, ctx, depth):
        self.journal.append((self.ub, ctx, depth))
        return ProbeResult(ctx=ctx, mem_mb=0, pp_ts=self.pp_ts.get(self.ub))


def _factory(par_ubatch):
    journal = []

    def make(ub, b):
        return _Probe(ub, par_ubatch, journal)

    make.journal = journal
    return make


def test_le_prompt_de_sonde_depasse_le_plus_grand_ubatch():
    """Garde-fou de conception : à prompt trop court, la sonde mesurerait du vent."""
    assert UBATCH_PROBE_PROMPT >= 2 * max(u for u, _ in UBATCH_CANDIDATES)
    assert UBATCH_PROBE_CTX > UBATCH_PROBE_PROMPT


def test_elit_l_ubatch_le_plus_rapide_et_chiffre_le_gain():
    r = probe_ubatch(_factory({512: 116.0, 2048: 187.0}))

    assert r["ubatch"] == 2048
    assert r["batch"] == 4096
    assert r["pp_ts"] == 187.0
    assert r["gain_pct"] == 61.2, "gain rapporté à la ligne de base (ubatch 512)"


def test_garde_la_ligne_de_base_si_elle_gagne():
    """Sur une petite machine, un gros ubatch peut régresser : la sonde doit le voir."""
    r = probe_ubatch(_factory({512: 90.0, 2048: 71.0}))

    assert r["ubatch"] == 512
    assert r["gain_pct"] is None, "pas de gain à annoncer quand la base gagne"


def test_toutes_les_variantes_sont_sondees_au_bon_contexte():
    make = _factory({512: 100.0, 2048: 120.0})
    probe_ubatch(make)

    assert [
        (u, UBATCH_PROBE_CTX, UBATCH_PROBE_PROMPT) for u, _ in UBATCH_CANDIDATES
    ] == (make.journal)


def test_sonde_muette_plutot_que_valeur_inventee():
    """Une mesure impossible ne doit RIEN écrire : mieux vaut le défaut historique
    qu'une valeur inventée présentée comme mesurée."""

    def _boom(ub, b):
        raise OSError("serveur KO")

    assert probe_ubatch(_boom) is None


def test_debit_illisible_est_ignore():
    assert probe_ubatch(_factory({})) is None


def test_server_probe_transmet_les_batchs_au_serveur():
    """Le clonage par dataclasses.replace (voie réelle du setup et du rebench) doit
    produire une sonde qui passe -ub/-b à llama-server."""
    captured = {}

    def _popen(args, **kw):
        captured["args"] = [str(a) for a in args]
        raise RuntimeError("stop avant le vrai lancement")

    base = ServerProbe(
        server_bin="llama-server",
        model_path="m.gguf",
        threads=8,
        ngl=99,
        topology="ram",
        popen=_popen,
        kill=lambda p: None,
    )
    probe = replace(base, ubatch=2048, batch=4096)
    try:
        probe._start(UBATCH_PROBE_CTX)
    except RuntimeError:
        pass
    args = captured["args"]
    assert (
        "2048" in args[args.index("-ub") + 1] or "2048" == args[args.index("-ub") + 1]
    )
    assert args[args.index("-b") + 1] == "4096"


def test_set_model_ubatch_ecrit_et_remplace(tmp_path):
    """L'écriture dans model.toml doit remplacer les lignes existantes sans toucher
    au reste (commentaires compris), comme _set_model_context."""
    from loom.setup.cli import _set_model_ubatch

    mt = tmp_path / "model.toml"
    mt.write_text(
        '# en-tete conserve\nfilename = "m.gguf"\nubatch = 512\nbatch = 1024\n',
        encoding="utf-8",
    )
    gguf = tmp_path / "m.gguf"

    _set_model_ubatch(gguf, 2048, 4096, "187 t/s sur 4096 tokens")
    txt = mt.read_text(encoding="utf-8")
    assert "ubatch = 2048" in txt and "batch = 4096" in txt
    assert "ubatch = 512" not in txt
    assert "# en-tete conserve" in txt
    assert "187 t/s" in txt

    # Idempotent : une seconde écriture ne duplique ni lignes ni tampon.
    _set_model_ubatch(gguf, 1024, 2048, "nouvelle mesure")
    txt = mt.read_text(encoding="utf-8")
    lignes = [l.split("=")[0].strip() for l in txt.splitlines() if "=" in l]
    assert lignes.count("ubatch") == 1 and lignes.count("batch") == 1
    assert txt.count("élus par la sonde") == 1


def test_set_model_ubatch_ajoute_si_absent(tmp_path):
    from loom.setup.cli import _set_model_ubatch

    mt = tmp_path / "model.toml"
    mt.write_text('filename = "m.gguf"\n', encoding="utf-8")

    _set_model_ubatch(tmp_path / "m.gguf", 2048, 4096, "mesure")
    txt = mt.read_text(encoding="utf-8")
    assert "ubatch = 2048" in txt and "batch = 4096" in txt
