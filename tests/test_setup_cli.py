# Installeur loom-setup : parcours console complets avec Console scriptée et
# effets de bord fakés (release GitHub figée, download no-op) — SANS réseau.
import tomllib
from types import SimpleNamespace

from loom.runtime.hardware import HardwareProfile
from loom.setup import cli
from loom.setup.cli import Console, Deps, run
from loom.setup.llama_release import AssetPlan


def _console(answers=None, assume_yes=False):
    answers = list(answers or [])
    printed = []

    def fake_input(prompt):
        printed.append(prompt)
        return answers.pop(0) if answers else ""

    con = Console(
        log_path=None,
        assume_yes=assume_yes,
        input_fn=fake_input,
        print_fn=lambda *a, **k: printed.append(a[0] if a else ""),
    )
    return con, printed


_PLAT = SimpleNamespace(key="windows", label="Windows 11")
_HW = HardwareProfile(True, "RTX 2060", 6000, 16)

_RELEASE = {
    "tag_name": "b5321",
    "html_url": "https://github.com/ggml-org/llama.cpp/releases/b5321",
    "assets": [
        {
            "name": "llama-b5321-bin-win-cuda-x64.zip",
            "browser_download_url": "https://x/cuda.zip",
            "size": 250 * 1024 * 1024,
        },
        {
            "name": "cudart-llama-bin-win-cuda-x64.zip",
            "browser_download_url": "https://x/cudart.zip",
            "size": 400 * 1024 * 1024,
        },
    ],
}

_FILES = [
    {
        "filename": "m.Q4_K_M.gguf",
        "part_files": ["m.Q4_K_M.gguf"],
        "size_mb": 15_000,
        "is_mmproj": False,
    },
    {
        "filename": "mmproj-F16.gguf",
        "part_files": ["mmproj-F16.gguf"],
        "size_mb": 800,
        "is_mmproj": True,
    },
]


class _FakeJob:
    done = True
    error = None

    def progress_mb(self):
        return 0


def _patch_paths(monkeypatch, tmp_path):
    """Repointe tous les chemins du module cli vers tmp_path (repo factice)."""
    defaults = tmp_path / "config" / "defaults.toml"
    defaults.parent.mkdir(parents=True)
    defaults.write_text('[server]\nbin = "llama-server"\n', encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_PATH", defaults)
    monkeypatch.setattr(cli, "PERSONAL_CONFIG_PATH", tmp_path / "config" / "local.toml")
    monkeypatch.setattr(cli, "PACKAGE_MODELS", tmp_path / "models")
    monkeypatch.setattr(cli, "RUNTIME_DIR", tmp_path / "var" / "runtime" / "llama")
    monkeypatch.setattr(cli, "SETUP_LOG", tmp_path / "var" / "logs" / "setup.log")


def _deps(tmp_path, **over):
    def fake_download(plan, dest_root, progress_cb):
        assert isinstance(plan, AssetPlan)
        dest = dest_root / plan.tag
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "llama-server.exe").write_bytes(b"")
        progress_cb(plan.assets[0]["name"], 1, 2)
        return dest

    base = dict(
        detect_platform=lambda: _PLAT,
        detect_hardware=lambda server_bin=None: _HW,
        ram_available_mb=lambda: 24_000,
        fetch_release=lambda: _RELEASE,
        download_and_extract=fake_download,
        verify_binary=lambda p: "b5321" if p else None,
        probe_repo=lambda repo: list(_FILES),
        # résolution live des entrées du catalogue : un repo 35B (fit large)
        search_models=lambda q: [
            {"repo_id": "org/Qwen3.6-35B-A3B-GGUF", "downloads": 9}
        ],
        start_download=lambda repo, filenames, dest, total_mb: _FakeJob(),
        top_ram_processes=lambda limit=8: [
            {"name": "chrome.exe", "mb": 2310, "count": 14}
        ],
        sleep=lambda s: None,
    )
    base.update(over)
    return Deps(**base)


def test_parcours_complet_machine_vierge(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    con, printed = _console(assume_yes=True)  # accepte tout, choix par défaut
    code = run(con, _deps(tmp_path))
    assert code == 0
    # binaire : installé + config écrite
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert local["server"]["bin"].endswith("llama-server.exe")
    # le premier modèle installé devient le défaut de CETTE machine
    assert local["chat"]["default_model"] == "qwen3.6-35b-a3b"
    # modèle : model.toml écrit dans <package_models>/local/text/<id>/ avec mmproj.
    # Le recommandé n°1 = l'entrée la plus gourmande qui tient (budget 25 904 Mo)
    # -> famille 35B (min_budget 18 000), résolue en live sur le repo faké.
    mdir = tmp_path / "models" / "local" / "text" / "qwen3.6-35b-a3b"
    raw = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
    assert raw["filename"] == "m.Q4_K_M.gguf"
    assert raw["mmproj_filename"] == "mmproj-F16.gguf"
    out = "\n".join(printed)
    assert "── Bilan ──" in out and "[échec]" not in out


def test_relance_idempotente(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    # binaire déjà en place (chemin absolu existant dans local.toml)
    exe = tmp_path / "bin" / "llama-server.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\n', encoding="utf-8"
    )
    # modèle déjà branché
    mdir = tmp_path / "models" / "local" / "text" / "deja-la"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 1\nsize_mb = 1\n',
        encoding="utf-8",
    )

    def boom():
        raise AssertionError("aucun réseau ne doit être touché en relance")

    con, printed = _console(assume_yes=True)
    code = run(con, _deps(tmp_path, fetch_release=boom))
    assert code == 0
    out = "\n".join(printed)
    assert "rien à faire" in out and "deja-la" in out


def test_refus_utilisateur(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    # réponses : "n" (refus binaire), "n" (pas de ménage RAM), "0" (passer le modèle)
    con, printed = _console(answers=["n", "n", "0"])
    code = run(con, _deps(tmp_path))
    assert code == 0  # un refus n'est PAS un échec
    assert not (tmp_path / "config" / "local.toml").exists()
    out = "\n".join(printed)
    assert "[passé]" in out


def test_hors_ligne(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    def offline():
        raise RuntimeError("GitHub injoignable (ConnectError) — vérifie la connexion.")

    con, printed = _console(assume_yes=True)
    code = run(
        con, _deps(tmp_path, fetch_release=offline, probe_repo=lambda repo: None)
    )
    assert code == 1  # échec binaire -> code 1
    out = "\n".join(printed)
    assert "GitHub injoignable" in out  # binaire : echec
    assert "injoignable (hors-ligne, renommé ?)" in out  # modèle : dégradé, pas planté


def test_recherche_filtree_par_budget_petite_machine(monkeypatch, tmp_path):
    """Machine sans GPU et RAM serrée (le cas Iris Xe) : la recherche libre ne
    doit PAS proposer un 397B — masqué, et les jouables annotés."""
    _patch_paths(monkeypatch, tmp_path)
    petite = HardwareProfile(False, None, 0, 8)
    hits = [
        {"repo_id": "meshllm/Qwen3.5-397B-A17B-UD-Q4_K_XL-layers", "downloads": 66234},
        {
            "repo_id": "Joshua65535/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "downloads": 52499,
        },
    ]
    small_files = [
        {
            "filename": "q.Q4_K_M.gguf",
            "part_files": ["q.Q4_K_M.gguf"],
            "size_mb": 900,
            "is_mmproj": False,
        }
    ]
    # réponses : binaire déjà réglé -> rien ; modèle : budget 1604 -> seule
    # l'entrée ~1.5B du catalogue tient (n°1), recherche libre = n°2 ; puis
    # requête, repo 1 (le 1.5b), quant oui
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\n', encoding="utf-8"
    )
    con, printed = _console(answers=["n", "2", "qwen q4", "1", "o"])
    deps = _deps(
        tmp_path,
        detect_hardware=lambda server_bin=None: petite,
        ram_available_mb=lambda: 5_700,
        search_models=lambda q: hits,
        probe_repo=lambda repo: small_files,
    )
    code = run(con, deps)
    assert code == 0
    out = "\n".join(printed)
    assert "1 résultat(s) masqué(s)" in out  # le 397B a disparu
    assert "397B" not in out.split("masqué")[1].split("Quel repo")[0]
    assert "~675 Mo mini" in out  # l'annotation d'estimation
    # le 1.5b a bien été installé
    mdir = tmp_path / "models" / "local" / "text" / "qwen2.5-1.5b-instruct-q4_k_m"
    assert (mdir / "model.toml").exists()


def test_liberer_ram_avant_le_choix(monkeypatch, tmp_path):
    """Machine serrée : la boucle « libère ta RAM » liste les gourmands, laisse
    fermer, re-mesure — et la shortlist profite du nouveau budget."""
    _patch_paths(monkeypatch, tmp_path)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\n', encoding="utf-8"
    )
    # RAM : 5700 à la détection, puis 12000 après fermeture des applis
    values = iter([5_700, 12_000, 12_000])

    def fake_ram():
        return next(values)

    # réponses : Entrée (=oui, machine serrée -> défaut O), Entrée (re-mesurer),
    # "c" (continuer), "0" (passer — on teste la boucle, pas l'install)
    con, printed = _console(answers=["", "", "c", "0"])
    deps = _deps(
        tmp_path,
        detect_hardware=lambda server_bin=None: HardwareProfile(False, None, 0, 8),
        ram_available_mb=fake_ram,
    )
    code = run(con, deps)
    assert code == 0
    out = "\n".join(printed)
    assert "chrome.exe" in out and "2310" in out  # les gourmands listés
    assert "budget : 7904 Mo" in out  # 12000 - 4096 après re-mesure
    assert "~8B instruct" in out  # la shortlist a profité du nouveau budget


def test_etape_bench_ecrit_les_reglages(monkeypatch, tmp_path):
    """Binaire + modèle en place -> le bench mesure et écrit threads/ngl/context
    dans local.toml (+ table [bench] pour l'idempotence)."""
    _patch_paths(monkeypatch, tmp_path)
    # binaire réel sur disque (résolvable) + modèle avec GGUF téléchargé
    exe = tmp_path / "rt" / "llama-server.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\n', encoding="utf-8"
    )
    mdir = tmp_path / "models" / "local" / "text" / "m1"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 1\nsize_mb = 5600\n',
        encoding="utf-8",
    )
    (mdir / "m.gguf").write_bytes(b"pas-un-vrai-gguf")  # meta illisible -> repli

    rows = [
        {"threads": 10, "ngl": 99, "kind": "tg", "ts": 3.4},
        {"threads": 10, "ngl": 99, "kind": "pp", "ts": 25.0},
        {"threads": 12, "ngl": 0, "kind": "tg", "ts": 2.0},
        {"threads": 12, "ngl": 0, "kind": "pp", "ts": 22.0},
    ]

    # Sonde topologique bouchonnée : mémoire linéaire (pente ~10,5 Ko/token),
    # débits constants — la calibration doit valider chaque barreau <= capacité.
    class _FakeProbe:
        def __init__(self, **kw):
            self.kw = kw

        def run(self, ctx, depth):
            from loom.setup.topology import ProbeResult

            r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
            if depth:
                r.tg_ts, r.pp_ts = 5.0, 20.0
            return r

    con, printed = _console(assume_yes=True)
    deps = _deps(
        tmp_path,
        ram_available_mb=lambda: 10_240,
        run_bench=lambda b, m, t, g: rows,
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: True,
        cpu_physical=lambda: 10,
        gpu_vram_total_mb=lambda: 6_144,
        make_probe=_FakeProbe,
    )
    code = run(con, deps)
    assert code == 0
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert local["override"]["threads"] == 10
    assert local["override"]["n_gpu_layers"] == 99
    # meta GGUF illisible -> limite modèle par défaut 32768 ; budget 6144-640 et
    # pente ~10,5 Ko/tok portent bien au-delà -> borné par le modèle, vitesse
    # validée à chaque barreau. La DÉCISION porte son mécanisme dans [bench].
    assert local["server"]["context"] == 32_768
    assert local["bench"]["context_mode"] == "gpu_dense"
    assert local["bench"]["context_valide_jusqua"] == 32_768
    assert (
        "pente" in local["bench"]["context_mecanisme"]
        or "capacité" in local["bench"]["context_mecanisme"]
    )
    assert local["bench"]["tg_ts"] == 3.4
    out = "\n".join(printed)
    assert "3.4 t/s" in out.replace(",", ".") or "3,4 t/s" in out

    # relance : la table [bench] existe -> déjà calibré, rien ne tourne
    def no_bench(*a, **k):
        raise AssertionError("le bench ne doit pas re-tourner")

    con2, printed2 = _console(assume_yes=True)
    code = run(con2, _deps(tmp_path, run_bench=no_bench))
    assert code == 0
    assert "Déjà calibré" in "\n".join(printed2)


def test_aucun_asset_compatible(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    release = {"tag_name": "b1", "html_url": "https://gh/r", "assets": []}
    con, printed = _console(answers=["n", "0"])  # pas de ménage RAM, passer le modèle
    code = run(con, _deps(tmp_path, fetch_release=lambda: release))
    assert code == 0  # guidage manuel n'est pas un échec
    out = "\n".join(printed)
    assert "[manuel]" in out and "config/local.toml" in out


# ── Modèle incomplet (Ctrl+C pendant le download) : honnêteté + reprise ──────


def _modele_incomplet(tmp_path, monkeypatch):
    """model.toml écrit mais GGUF absent (download interrompu)."""
    _patch_paths(monkeypatch, tmp_path)
    mdir = tmp_path / "models" / "local" / "text" / "ornith-35b"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'repo = "org/Ornith-35B-GGUF"\nfilename = "ornith.Q8_0.gguf"\nsize_mb = 35193\n',
        encoding="utf-8",
    )
    return mdir


def test_modele_incomplet_propose_la_reprise(monkeypatch, tmp_path):
    mdir = _modele_incomplet(tmp_path, monkeypatch)
    seen = {}

    def fake_start(repo, filenames, dest, total_mb):
        seen.update(repo=repo, filenames=filenames, dest=dest)
        return _FakeJob()

    con, printed = _console(answers=["o"])
    report = cli.SetupReport()
    cli.step_model(
        con, report, _deps(tmp_path, start_download=fake_start), _HW, 24_000, {}
    )
    out = "\n".join(printed)
    assert "[attention]" in out and "GGUF absent" in out
    assert seen["repo"] == "org/Ornith-35B-GGUF"
    assert seen["filenames"] == ["ornith.Q8_0.gguf"]
    assert seen["dest"] == mdir
    assert report.outcomes[-1].status == "fait"


def test_modele_incomplet_reprise_refusee(monkeypatch, tmp_path):
    _modele_incomplet(tmp_path, monkeypatch)
    con, printed = _console(answers=["n"])
    report = cli.SetupReport()
    cli.step_model(con, report, _deps(tmp_path), _HW, 24_000, {})
    out = "\n".join(printed)
    assert "[attention]" in out and "[passé]" in out
    assert report.outcomes[-1].status == "ignore"


def test_bench_saute_dit_ce_qui_manque(monkeypatch, tmp_path):
    # Binaire présent, GGUF absent : le message doit nommer le GGUF, pas
    # l'ambigu « binaire ou modèle ».
    _modele_incomplet(tmp_path, monkeypatch)
    binp = tmp_path / "llama-server.exe"
    binp.write_bytes(b"")
    con, printed = _console()
    report = cli.SetupReport()
    cli.step_bench(con, report, _deps(tmp_path), {"server": {"bin": str(binp)}})
    out = "\n".join(printed)
    assert "GGUF" in out and "binaire" not in out


def test_say_colorise_chaque_ligne():
    # Le bilan arrive en UN bloc multi-lignes : chaque ligne doit être colorée
    # (avant, seule la 1re ligne passait par les règles -> bilan tout blanc).
    from loom.runtime.term import DIM, GREEN

    printed = []
    con = Console(
        print_fn=lambda *a, **k: printed.append(a[0] if a else ""), color=True
    )
    con.say("── Bilan ──\n  [ok] Modèle x\n  [passé] Réglages y")
    out = printed[-1]
    assert GREEN + "  [ok] Modèle x" in out
    assert DIM + "  [passé] Réglages y" in out
