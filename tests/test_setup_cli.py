# Parcours console complets avec effets externes simulés, sans réseau.
import json
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

_SWAP_RELEASE = {
    "tag_name": "v241",
    "assets": [
        {
            "name": "llama-swap_241_linux_amd64.tar.gz",
            "browser_download_url": "https://x/l.tar.gz",
            "size": 8 * 1024 * 1024,
        },
        {
            "name": "llama-swap_241_windows_amd64.zip",
            "browser_download_url": "https://x/w.zip",
            "size": 8 * 1024 * 1024,
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
    from loom.setup import archive as _archive

    monkeypatch.setattr(_archive, "BENCH_DIR", tmp_path / "var" / "bench")


def _deps(tmp_path, **over):
    def fake_download(plan, dest_root, progress_cb):
        assert isinstance(plan, AssetPlan)
        dest = dest_root / plan.tag
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "llama-server.exe").write_bytes(b"")
        (dest / "llama-swap.exe").write_bytes(b"")
        progress_cb(plan.assets[0]["name"], 1, 2)
        return dest

    base = dict(
        detect_platform=lambda: _PLAT,
        detect_hardware=lambda server_bin=None: _HW,
        ram_available_mb=lambda: 24_000,
        fetch_release=lambda: _RELEASE,
        fetch_swap_release=lambda: _SWAP_RELEASE,
        download_and_extract=fake_download,
        verify_binary=lambda p: "b5321" if p else None,
        probe_repo=lambda repo: list(_FILES),
        search_models=lambda q: [
            {"repo_id": "org/Qwen3.6-35B-A3B-GGUF", "downloads": 9}
        ],
        start_download=lambda repo, filenames, dest, total_mb: _FakeJob(),
        top_ram_processes=lambda limit=8: [
            {"name": "chrome.exe", "mb": 2310, "count": 14}
        ],
        # Outillage agent SIMULÉ complet : sans ces fakes, step_tooling sonde la
        # VRAIE machine (et install_playwright le réseau) -> tests machine-dépendants,
        # constaté 2026-07-26 (3 échecs sur un poste sans l'outillage de l'auteur).
        tool_checks=lambda: [],
        install_playwright=lambda: (True, ""),
        sleep=lambda s: None,
    )
    base.update(over)
    return Deps(**base)


def test_parcours_complet_machine_vierge(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    con, printed = _console(assume_yes=True)  # accepte tout, choix par défaut
    code = run(con, _deps(tmp_path))
    assert code == 0
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert local["server"]["bin"].endswith("llama-server.exe")
    assert local["chat"]["default_model"] == "qwen3.6-35b-a3b"
    # Le premier choix doit être la famille la plus gourmande compatible avec le budget.
    mdir = tmp_path / "models" / "local" / "text" / "qwen3.6-35b-a3b"
    raw = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
    assert raw["filename"] == "m.Q4_K_M.gguf"
    assert raw["mmproj_filename"] == "mmproj-F16.gguf"
    out = "\n".join(printed)
    assert "── Bilan ──" in out and "[échec]" not in out


def test_relance_idempotente(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    exe = tmp_path / "bin" / "llama-server.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\n', encoding="utf-8"
    )
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
    # Avec ce budget, seule la famille 1.5B doit être proposée avant la recherche libre.
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
    values = iter([5_700, 12_000, 12_000])

    def fake_ram():
        return next(values)

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

    # La sonde linéaire doit valider chaque barreau sous la capacité. Dataclass :
    # la sonde de placement la clone par dataclasses.replace, comme la vraie.
    from dataclasses import dataclass as _dc

    @_dc
    class _FakeProbe:
        server_bin: str = ""
        model_path: str = ""
        threads: int = 0
        ngl: int = 0
        topology: str = ""
        mmproj_path: object = None
        cpu_moe: bool = False
        n_cpu_moe: object = None
        n_parallel: int = 1
        ubatch: object = None
        batch: object = None
        checkpoint_min_step: object = None
        ctx_checkpoints: object = None
        profile: object = None

        def probe_isolation(self, ctx=4096):
            return 600, 4

        def verify_cache(self, ctx=4096):
            # Séquence réelle avec les slots finaux : le cache est réutilisé.
            return {
                "first": 600,
                "back": 4,
                "annex_slot": 0,
                "slots": self.n_parallel,
                "reused": True,
            }

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
        run_bench=lambda b, m, t, g, n_cpu_moe=0, progress=None: rows,
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: True,
        cpu_physical=lambda: 10,
        gpu_vram_total_mb=lambda: 6_144,
        make_probe=_FakeProbe,
        # Une VRAM suffisante doit faire élire l'offload total.
        detect_hardware=lambda server_bin=None: HardwareProfile(
            True, "GPU 20Go", 20_000, 16, vram_is_discrete=True
        ),
    )
    code = run(con, deps)
    assert code == 0
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert local["override"]["threads"] == 10
    assert local["override"]["n_gpu_layers"] == 99
    # Sans métadonnée, la limite par défaut doit borner une capacité pourtant supérieure.
    assert local["server"]["context"] == 32_768
    assert local["bench"]["context_mode"] == "gpu_dense"
    assert local["bench"]["context_valide_jusqua"] == 32_768
    assert (
        "pente" in local["bench"]["context_mecanisme"]
        or "capacité" in local["bench"]["context_mecanisme"]
    )
    assert local["bench"]["tg_ts"] == 3.4
    # Persister le verdict mesuré d'isolation dans le profil, avec la vérification du
    # cache sur la configuration FINALE (séquence réelle, slots finaux).
    mt_txt = (mdir / "model.toml").read_text(encoding="utf-8")
    mt = tomllib.loads(mt_txt)
    assert mt["cache_isolation"] is False
    assert "cache réutilisé" in mt_txt and "4/600" in mt_txt
    assert local["bench"]["cache_verifie"] is True
    assert "4/600" in local["bench"]["cache_verifie_detail"]
    # La trace du placement dit avec quoi il a été mesuré : moteur, slots, flags machine.
    assert local["bench"]["placement_build"] == "b5321"
    assert local["bench"]["placement_slots"] == 1
    assert local["bench"]["placement_flags"]["threads"] == 10
    assert local["bench"]["placement_flags"]["gpu_tuning"] is True
    # Validation du RÉGLAGE FINAL complet au contexte calibré (même fausse sonde : 5 t/s).
    assert local["bench"]["final_ctx"] == 32_768 and local["bench"]["final_slots"] == 1
    assert (
        local["bench"]["final_tg_ts"] == 5.0
        and local["bench"]["final_coherent"] is True
    )
    # Archive DURABLE du bench : un JSON horodaté par modèle, mesures et verdict compris.
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["model_id"] == "m1" and arch["build"] == "b5321"
    assert arch["calibration"]["context"] == 32_768
    assert arch["final"]["tg_ts"] == 5.0 and arch["cache"]["reused"] is True
    assert arch["placement"]["placement"]["key"].startswith("gpu_total")
    assert (
        arch["application"]["context"] == 32_768
    )  # loom-setup applique dans la foulée
    out = "\n".join(printed)
    assert "archive :" in out
    assert "cache survit à la pollution" in out
    assert "cache réutilisé" in out
    assert "réglage final" in out
    assert "3.4 t/s" in out.replace(",", ".") or "3,4 t/s" in out

    def no_bench(*a, **k):
        raise AssertionError("le bench ne doit pas re-tourner")

    con2, printed2 = _console(assume_yes=True)
    code = run(con2, _deps(tmp_path, run_bench=no_bench))
    assert code == 0
    assert "Déjà calibré" in "\n".join(printed2)


def test_etape_bench_reglage_final_en_echec_n_ecrit_rien(monkeypatch, tmp_path):
    """P1 (revue 2026-10-10) : si la configuration FINALE complète ne fonctionne pas,
    loom-setup n'écrit aucun réglage (ni local.toml ni model.toml) et le dit."""
    from dataclasses import dataclass as _dc

    from loom.setup.placement import final_depth

    _patch_paths(monkeypatch, tmp_path)
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
    (mdir / "m.gguf").write_bytes(b"pas-un-vrai-gguf")
    rows = [
        {"threads": 10, "ngl": 99, "kind": "tg", "ts": 3.4},
        {"threads": 10, "ngl": 99, "kind": "pp", "ts": 25.0},
    ]

    @_dc
    class _FakeProbe:
        server_bin: str = ""
        model_path: str = ""
        threads: int = 0
        ngl: int = 0
        topology: str = ""
        mmproj_path: object = None
        cpu_moe: bool = False
        n_cpu_moe: object = None
        n_parallel: int = 1
        ubatch: object = None
        batch: object = None
        checkpoint_min_step: object = None
        ctx_checkpoints: object = None
        profile: object = None

        def probe_isolation(self, ctx=4096):
            return 600, 4

        def verify_cache(self, ctx=4096):
            return {
                "first": 600,
                "back": 4,
                "annex_slot": 0,
                "slots": 1,
                "reused": True,
            }

        def run(self, ctx, depth):
            from loom.setup.topology import ProbeResult

            # Seule la validation FINALE tourne à (contexte calibré 32 768, profondeur de
            # la comparaison = final_depth du contexte utile 8 192) : elle échoue, tout
            # le reste (barreaux à 27 852, ubatch à 8 192) fonctionne.
            if ctx == 32_768 and depth == final_depth(8_192):
                raise RuntimeError("ErrorOutOfDeviceMemory")
            r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
            if depth:
                r.tg_ts, r.pp_ts = 5.0, 20.0
            return r

    con, printed = _console(assume_yes=True)
    deps = _deps(
        tmp_path,
        ram_available_mb=lambda: 10_240,
        run_bench=lambda b, m, t, g, n_cpu_moe=0, progress=None: rows,
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: True,
        cpu_physical=lambda: 10,
        gpu_vram_total_mb=lambda: 6_144,
        make_probe=_FakeProbe,
        detect_hardware=lambda server_bin=None: HardwareProfile(
            True, "GPU 20Go", 20_000, 16, vram_is_discrete=True
        ),
    )
    # Le pas « bench » est en échec : code de sortie non nul, comme une calibration ratée.
    assert run(con, deps) != 0
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert "bench" not in local and "context" not in (local.get("server") or {})
    mt = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
    assert "context" not in mt and "cache_isolation" not in mt
    out = "\n".join(printed)
    assert "NON écrits" in out and "ErrorOutOfDeviceMemory" in out
    # P3 : l'échec est archivé avec l'étape, l'erreur et les mesures déjà faites.
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "réglage final"
    assert "ErrorOutOfDeviceMemory" in arch["echec"]["erreur"]
    assert arch["calibration"]["context"] == 32_768 and arch["placement"]
    assert arch["materiel"]["gpu_name"] == "GPU 20Go" and "application" not in arch


def _harnais_bench(monkeypatch, tmp_path, fake_probe_cls, assume_yes=True):
    """Harnais commun des scénarios de bench : binaire, model.toml minimal, GGUF factice,
    lignes llama-bench, deps. Renvoie (con, printed, deps, mdir)."""
    _patch_paths(monkeypatch, tmp_path)
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
    (mdir / "m.gguf").write_bytes(b"pas-un-vrai-gguf")
    rows = [
        {"threads": 10, "ngl": 99, "kind": "tg", "ts": 3.4},
        {"threads": 10, "ngl": 99, "kind": "pp", "ts": 25.0},
    ]
    con, printed = _console(assume_yes=assume_yes)
    deps = _deps(
        tmp_path,
        ram_available_mb=lambda: 10_240,
        run_bench=lambda b, m, t, g, n_cpu_moe=0, progress=None: rows,
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: True,
        cpu_physical=lambda: 10,
        gpu_vram_total_mb=lambda: 6_144,
        make_probe=fake_probe_cls,
        detect_hardware=lambda server_bin=None: HardwareProfile(
            True, "GPU 20Go", 20_000, 16, vram_is_discrete=True
        ),
    )
    return con, printed, deps, mdir


def _fake_probe_cls(run_impl, isolation=(600, 4)):
    from dataclasses import dataclass as _dc

    @_dc
    class _FakeProbe:
        server_bin: str = ""
        model_path: str = ""
        threads: int = 0
        ngl: int = 0
        topology: str = ""
        mmproj_path: object = None
        cpu_moe: bool = False
        n_cpu_moe: object = None
        n_parallel: int = 1
        ubatch: object = None
        batch: object = None
        checkpoint_min_step: object = None
        ctx_checkpoints: object = None
        profile: object = None

        def probe_isolation(self, ctx=4096):
            return isolation

        def verify_cache(self, ctx=4096):
            return {
                "first": 600,
                "back": 4,
                "annex_slot": 0,
                "slots": self.n_parallel,
                "reused": True,
            }

        def run(self, ctx, depth):
            return run_impl(ctx, depth)

    return _FakeProbe


def test_etape_bench_erreur_d_entree_sortie_en_calibration_est_archivee(
    monkeypatch, tmp_path
):
    """Revue 2026-10-10 : une PermissionError pendant la calibration n'était pas
    rattrapée (seules RuntimeError / ValueError l'étaient) : pas d'archive, pas de
    compte rendu."""
    from loom.setup.topology import ProbeResult

    def run_impl(ctx, depth):
        if depth is None:
            # Premier barreau de PENTE de la calibration : erreur d'E/S.
            raise PermissionError("accès refusé au GGUF")
        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch, tmp_path, _fake_probe_cls(run_impl)
    )
    assert run(con, deps) != 0  # le pas bench est en échec, pas un plantage
    out = "\n".join(printed)
    assert "calibration échouée" in out and "accès refusé" in out
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "calibration"
    assert "accès refusé" in arch["echec"]["erreur"]
    assert arch["placement"]  # les mesures déjà faites sont conservées


def test_etape_bench_dit_quand_l_annotation_de_l_archive_echoue(monkeypatch, tmp_path):
    from loom.setup import archive as _archive
    from loom.setup.topology import ProbeResult

    monkeypatch.setattr(_archive, "note_application", lambda *a, **k: False)

    def run_impl(ctx, depth):
        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch, tmp_path, _fake_probe_cls(run_impl)
    )
    assert run(con, deps) == 0  # la configuration EST appliquée
    out = "\n".join(printed)
    assert "appliquée" in out and "annotation de l'archive échouée" in out


def test_etape_bench_part_de_l_isolation_actuelle_si_la_sonde_echoue(
    monkeypatch, tmp_path
):
    """P1 : un modèle déjà en `cache_isolation = true` tournera à 2 slots ; si la sonde
    d'isolation échoue, placement, calibration et réglage final doivent mesurer à 2
    slots, pas à 1. Un nouveau verdict seul remplace l'isolation actuelle."""
    from dataclasses import dataclass as _dc

    _patch_paths(monkeypatch, tmp_path)
    exe = tmp_path / "rt" / "llama-server.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    (tmp_path / "config" / "local.toml").write_text(
        f'[server]\nbin = "{str(exe).replace(chr(92), "/")}"\n', encoding="utf-8"
    )
    mdir = tmp_path / "models" / "local" / "text" / "m1"
    mdir.mkdir(parents=True)
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 1\nsize_mb = 5600\n'
        "cache_isolation = true\n",
        encoding="utf-8",
    )
    (mdir / "m.gguf").write_bytes(b"pas-un-vrai-gguf")
    rows = [
        {"threads": 10, "ngl": 99, "kind": "tg", "ts": 3.4},
        {"threads": 10, "ngl": 99, "kind": "pp", "ts": 25.0},
    ]
    sondes: list = []

    @_dc
    class _FakeProbe:
        server_bin: str = ""
        model_path: str = ""
        threads: int = 0
        ngl: int = 0
        topology: str = ""
        mmproj_path: object = None
        cpu_moe: bool = False
        n_cpu_moe: object = None
        n_parallel: int = 1
        ubatch: object = None
        batch: object = None
        checkpoint_min_step: object = None
        ctx_checkpoints: object = None
        profile: object = None

        def __post_init__(self):
            sondes.append(self)

        def probe_isolation(self, ctx=4096):
            raise RuntimeError("health timeout")

        def verify_cache(self, ctx=4096):
            return {
                "first": 600,
                "back": 4,
                "annex_slot": 1,
                "slots": self.n_parallel,
                "reused": True,
            }

        def run(self, ctx, depth):
            from loom.setup.topology import ProbeResult

            r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
            if depth:
                r.tg_ts, r.pp_ts = 5.0, 20.0
            return r

    con, _printed = _console(assume_yes=True)
    deps = _deps(
        tmp_path,
        ram_available_mb=lambda: 10_240,
        run_bench=lambda b, m, t, g, n_cpu_moe=0, progress=None: rows,
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: True,
        cpu_physical=lambda: 10,
        gpu_vram_total_mb=lambda: 6_144,
        make_probe=_FakeProbe,
        detect_hardware=lambda server_bin=None: HardwareProfile(
            True, "GPU 20Go", 20_000, 16, vram_is_discrete=True
        ),
    )
    assert run(con, deps) == 0
    # Toutes les sondes (initiale et clones) ont mesuré à 2 slots.
    assert sondes and all(s.n_parallel == 2 for s in sondes)
    mt = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
    assert mt["cache_isolation"] is True  # conservé : pas de nouveau verdict


def test_aucun_asset_compatible(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)
    release = {"tag_name": "b1", "html_url": "https://gh/r", "assets": []}
    con, printed = _console(answers=["n", "0"])  # pas de ménage RAM, passer le modèle
    code = run(con, _deps(tmp_path, fetch_release=lambda: release))
    assert code == 0  # guidage manuel n'est pas un échec
    out = "\n".join(printed)
    assert "[manuel]" in out and "config/local.toml" in out


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
    # Le diagnostic doit nommer le GGUF absent malgré un binaire valide.
    _modele_incomplet(tmp_path, monkeypatch)
    binp = tmp_path / "llama-server.exe"
    binp.write_bytes(b"")
    con, printed = _console()
    report = cli.SetupReport()
    cli.step_bench(con, report, _deps(tmp_path), {"server": {"bin": str(binp)}})
    out = "\n".join(printed)
    assert "GGUF" in out and "binaire" not in out


def test_say_colorise_chaque_ligne():
    # Chaque ligne d'un bilan multiligne doit être colorée séparément.
    from loom.runtime.term import DIM, GREEN

    printed = []
    con = Console(
        print_fn=lambda *a, **k: printed.append(a[0] if a else ""), color=True
    )
    con.say("── Bilan ──\n  [ok] Modèle x\n  [passé] Réglages y")
    out = printed[-1]
    assert GREEN + "  [ok] Modèle x" in out
    assert DIM + "  [passé] Réglages y" in out


def test_recherche_accepte_une_url_hf(monkeypatch, tmp_path):
    # Une URL explicite doit ouvrir directement l'inventaire des quants.
    _patch_paths(monkeypatch, tmp_path)
    from loom.setup.catalog import budget_mb, fitting_entries

    libre = len(fitting_entries(budget_mb(_HW.budget_vram_mb, 24_000))) + 1
    probed = []

    def boom(q):
        raise AssertionError("URL collée -> aucune recherche ne doit partir")

    def fake_probe(repo):
        probed.append(repo)
        return list(_FILES)

    con, printed = _console(
        answers=["n", str(libre), "https://huggingface.co/org/Mon-Repo-GGUF", "O"]
    )
    report = cli.SetupReport()
    cli.step_model(
        con,
        report,
        _deps(tmp_path, search_models=boom, probe_repo=fake_probe),
        _HW,
        24_000,
        {},
    )
    assert probed == ["org/Mon-Repo-GGUF"]
    assert report.outcomes[-1].status == "fait"


def test_step_swap_installe_et_configure(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    def fake_dl(plan, dest_root, progress_cb):
        dest = dest_root / plan.tag
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "llama-swap.exe").write_bytes(b"")
        return dest

    exe = tmp_path / "bin" / "llama-server.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    raw = {"server": {"bin": str(exe)}}
    con, printed = _console()
    report = cli.SetupReport()
    cli.step_swap(
        con,
        report,
        _deps(
            tmp_path,
            fetch_swap_release=lambda: _SWAP_RELEASE,
            download_and_extract=fake_dl,
        ),
        _PLAT,
        raw,
    )
    out = "\n".join(printed)
    assert "[ok]" in out and "swap_bin" in out
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert local["server"]["swap_bin"].endswith("llama-swap.exe")
    assert report.outcomes[-1].status == "fait"

    raw = {"server": {"bin": str(exe), "swap_bin": local["server"]["swap_bin"]}}
    con2, printed2 = _console()
    cli.step_swap(con2, report, _deps(tmp_path), _PLAT, raw)
    assert "rien à faire" in "\n".join(printed2)


def test_step_swap_echec_reseau_non_bloquant(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    def offline():
        raise RuntimeError("GitHub injoignable")

    exe = tmp_path / "bin" / "llama-server.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"")
    con, printed = _console()
    report = cli.SetupReport()
    cli.step_swap(
        con,
        report,
        _deps(tmp_path, fetch_swap_release=offline),
        _PLAT,
        {"server": {"bin": str(exe)}},
    )
    out = "\n".join(printed)
    assert "[attention]" in out and "multi-modèles" in out
    assert report.outcomes[-1].status == "manuel"


def test_select_swap_asset_par_plateforme():
    from loom.setup.llama_release import select_swap_asset

    plan = select_swap_asset(_SWAP_RELEASE, "windows", "x64")
    assert plan is not None and plan.assets[0]["name"].endswith("windows_amd64.zip")
    plan_l = select_swap_asset(_SWAP_RELEASE, "linux", "x64")
    assert plan_l.assets[0]["name"].endswith("linux_amd64.tar.gz")
    assert select_swap_asset(_SWAP_RELEASE, "macos", "arm64") is None


def test_step_tooling_constate_et_conseille(tmp_path):
    checks = [
        {
            "name": "rg (ripgrep)",
            "present": False,
            "role": "search_text",
            "hint": "winget install ripgrep",
            "autofix": None,
        },
        {
            "name": "docker",
            "present": True,
            "role": "web_search",
            "hint": "-",
            "autofix": None,
        },
    ]
    con, printed = _console()
    report = cli.SetupReport()
    cli.step_tooling(con, report, _deps(tmp_path, tool_checks=lambda: checks))
    out = "\n".join(printed)
    assert "[attention]" in out and "rg (ripgrep)" in out and "winget" in out
    assert report.outcomes[-1].status == "manuel"


def test_step_tooling_installe_playwright(tmp_path):
    state = {"installed": False}

    def checks():
        return [
            {
                "name": "navigateur Playwright (chromium)",
                "present": state["installed"],
                "role": "check_page",
                "hint": "-",
                "autofix": "playwright",
            }
        ]

    def install():
        state["installed"] = True
        return True, "chromium installé"

    con, printed = _console(answers=["o"])
    report = cli.SetupReport()
    cli.step_tooling(
        con, report, _deps(tmp_path, tool_checks=checks, install_playwright=install)
    )
    out = "\n".join(printed)
    assert "[ok] navigateur Playwright installé" in out
    assert report.outcomes[-1].status == "fait"
