# Parcours console complets avec effets externes simulés, sans réseau.
import json
import tomllib
from types import SimpleNamespace

from loom.runtime.hardware import HardwareProfile
from loom.setup import cli
from loom.setup.cli import Console, Deps, run
from loom.setup.llama_release import AssetPlan


def _gguf_sans_catalogue(path):
    """GGUF VALIDE sans catalogue de tenseurs : métadonnées incomplètes, le précontrôle
    dit « incertain » et le flux continue. (Un en-tête rejeté, lui, arrête tout avant
    le moindre processus : llama-server le refuserait aussi.)"""
    from tests.test_gguf_profile import _gguf

    return _gguf(path, {"general.architecture": "llama"})


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
    _gguf_sans_catalogue(mdir / "m.gguf")  # métadonnées incomplètes -> repli

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
        # Une VRAM suffisante doit faire élire l'offload total. La VRAM TOTALE doit
        # être celle du « GPU 20Go » : depuis la revue #14, la configuration actuelle
        # passe elle aussi l'estimation, et 6 144 Mo ne portaient pas 5,6 Go + KV.
        gpu_vram_total_mb=lambda: 20_480,
        ram_total_mb=lambda: 64_000,
        make_probe=_FakeProbe,
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
    _gguf_sans_catalogue(mdir / "m.gguf")
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


def _harnais_bench(
    monkeypatch,
    tmp_path,
    fake_probe_cls,
    assume_yes=True,
    ram_total_mb=64_000,
    hw=None,
    meta=None,
    run_bench=None,
):
    """Harnais commun des scénarios de bench : binaire, model.toml minimal, GGUF factice,
    lignes llama-bench, deps. Renvoie (con, printed, deps, mdir). La RAM totale est
    FIXÉE (la faisabilité hôte en dépend : pas la RAM de la machine de test). `hw`
    remplace le profil matériel, `meta` les métadonnées GGUF (le GGUF factice donne
    sinon {}), `run_bench` le faux llama-bench."""
    if meta is not None:
        monkeypatch.setattr(cli, "read_gguf_meta", lambda p: meta)
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
    _gguf_sans_catalogue(mdir / "m.gguf")
    rows = [
        {"threads": 10, "ngl": 99, "kind": "tg", "ts": 3.4},
        {"threads": 10, "ngl": 99, "kind": "pp", "ts": 25.0},
    ]
    con, printed = _console(assume_yes=assume_yes)
    deps = _deps(
        tmp_path,
        ram_available_mb=lambda: 10_240,
        run_bench=run_bench or (lambda b, m, t, g, n_cpu_moe=0, progress=None: rows),
        find_llama_bench=lambda sb: sb.parent / "llama-bench.exe",
        has_gpu_backend=lambda sb: True,
        cpu_physical=lambda: 10,
        gpu_vram_total_mb=lambda: 6_144,
        ram_total_mb=lambda: ram_total_mb,
        make_probe=fake_probe_cls,
        detect_hardware=lambda server_bin=None: (
            hw or HardwareProfile(True, "GPU 20Go", 20_000, 16, vram_is_discrete=True)
        ),
    )
    return con, printed, deps, mdir


def _fake_probe_cls(run_impl, isolation=(600, 4), journal=None, avec_sonde=False):
    """Fausse sonde compatible dataclasses.replace. `journal` (liste) enregistre chaque
    CONSTRUCTION et chaque LANCEMENT (probe_isolation, run, verify_cache) avec les flags
    — la preuve « aucun processus modèle » ne doit pas reposer sur une exception, que
    les `except Exception` du bench avaleraient. `avec_sonde` : run_impl reçoit aussi
    la sonde (un échec qui dépend du -ngl)."""
    from dataclasses import dataclass as _dc

    def _note(kind, s, *extra):
        if journal is not None:
            journal.append((kind, s.ngl, s.cpu_moe, s.n_cpu_moe, s.n_parallel, *extra))

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
            _note("construite", self)

        def probe_isolation(self, ctx=4096):
            _note("isolation", self, ctx)
            return isolation

        def verify_cache(self, ctx=4096):
            _note("verify_cache", self, ctx)
            return {
                "first": 600,
                "back": 4,
                "annex_slot": 0,
                "slots": self.n_parallel,
                "reused": True,
            }

        def run(self, ctx, depth):
            _note("run", self, ctx, depth)
            return run_impl(ctx, depth, self) if avec_sonde else run_impl(ctx, depth)

    return _FakeProbe


_MIB = 1024 * 1024


def _meta_complete(n=40, par_mb=300, sortie_mb=300, emb_mb=300):
    """Métadonnées COMPLÈTES d'un dense (catalogue, dimensions KV, clés optionnelles
    absentes comme dans un vrai en-tête) : le précontrôle peut conclure."""
    return {
        "architecture": "llama",
        "n_layers": n,
        "context_length": 32768,
        "expert_count": None,
        "expert_used_count": None,
        "head_count": 32,
        "head_count_kv": 8,
        "embedding_length": 4096,
        "key_length": 128,
        "value_length": 128,
        "sliding_window": None,
        "sliding_window_pattern": None,
        "full_attention_interval": None,
        "recurrent": False,
        "split_count": None,
        "key_length_mla": None,
        "kv_lora_rank": None,
        "shared_kv_layers": None,
        "key_length_swa": None,
        "value_length_swa": None,
        "head_count_kv_array": False,
        "arrays": {},
        "weights": {
            "total": (n * par_mb + sortie_mb + emb_mb) * _MIB,
            "familles": {"embeddings": emb_mb * _MIB, "output": sortie_mb * _MIB},
            "par_couche": [par_mb * _MIB] * n,
            "experts_par_couche": [0] * n,
            "couches_attention": list(range(n)),
            "couches_recurrentes": [],
            "couches_nextn": [],
            "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
        },
    }


def _gpu(vram_total, vram_free=None, *, discret=True, backend="CUDA", count=1):
    return HardwareProfile(
        True,
        f"GPU {vram_total}",
        vram_total if vram_free is None else vram_free,
        16,
        vram_total_mb=vram_total,
        backend=backend,
        vram_is_discrete=discret,
        gpu_count=count,
    )


def _bench_espion(appels):
    rows = [
        {"threads": 10, "ngl": 0, "kind": "tg", "ts": 3.4},
        {"threads": 10, "ngl": 0, "kind": "pp", "ts": 25.0},
    ]

    def run_bench(b, m, t, g, n_cpu_moe=0, progress=None):
        appels.append({"threads": list(t), "ngl": list(g), "ncmoe": n_cpu_moe})
        return rows

    return run_bench


def _rien_ne_doit_tourner(ctx, depth):
    raise AssertionError("aucune mesure ne doit être lancée")


def _scenario_sortie_precontrole(monkeypatch, tmp_path, hw, ram):
    journal: list = []
    appels: list = []
    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_rien_ne_doit_tourner, journal=journal),
        assume_yes=False,
        ram_total_mb=ram,
        hw=hw,
        meta=_meta_complete(),
        run_bench=_bench_espion(appels),
    )
    avant = (mdir / "model.toml").read_text(encoding="utf-8")
    code = run(con, deps)
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    return code, "\n".join(printed), journal, appels, mdir, avant, archives


def test_precontrole_impossibilite_etablie_aucun_processus_modele(
    monkeypatch, tmp_path
):
    """TEST DÉCISIF de la revue n°16 (loom-setup) : impossibilité établie (12 600 Mo de
    poids résidents, GPU discret de 6 144 Mo + 4 000 Mo de RAM) → ni llama-bench, ni
    sonde construite ou lancée ; verdict explicite ; une archive « précontrôle »."""
    code, out, journal, appels, mdir, avant, archives = _scenario_sortie_precontrole(
        monkeypatch, tmp_path, _gpu(6144), 4000
    )
    assert code != 0
    assert appels == [] and journal == []
    assert "démarrage impossible" in out and "aucun processus modèle lancé" in out
    assert "Lancer le bench maintenant" not in out  # sortie AVANT la confirmation
    assert "aucun placement comparé" not in out.lower()  # pas la sortie de l'étape 2
    assert (mdir / "model.toml").read_text(encoding="utf-8") == avant
    local = tomllib.loads(
        (tmp_path / "config" / "local.toml").read_text(encoding="utf-8")
    )
    assert "bench" not in local
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "précontrôle"
    assert arch["precontrole"]["verdict"] == "impossible"
    assert arch["precontrole"]["complet"] is True
    assert arch["llama_bench"] is None and arch["isolation"] is None
    assert arch["materiel"]["gpu_name"] == "GPU 6144" and arch["gguf"]


def test_precontrole_hors_budget_au_plancher_aucun_processus_modele(
    monkeypatch, tmp_path
):
    """Mémoire unifiée seulement présumée (Vulkan) : pas d'« établi », mais aucun
    placement ne tient même à 4096 x 1 → sortie avant tout chargement, dite telle."""
    hw = _gpu(48_789, 46_350, discret=False, backend="Vulkan")
    code, out, journal, appels, mdir, avant, archives = _scenario_sortie_precontrole(
        monkeypatch, tmp_path, hw, 4000
    )
    assert code != 0 and appels == [] and journal == []
    assert "hors budget même au contexte plancher" in out
    assert "aucun processus modèle lancé" in out
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "précontrôle"
    assert arch["precontrole"]["verdict"] == "hors_budget"
    assert arch["precontrole"]["etabli"] is False


def test_precontrole_metadonnees_incompletes_flux_inchange(monkeypatch, tmp_path):
    """Contre-test : même machine, GGUF valide sans catalogue de tenseurs → « incertain »,
    llama-bench et la sonde tournent comme avant ; l'étape 2 tranche."""
    journal: list = []
    appels: list = []

    def run_impl(ctx, depth):
        from loom.setup.topology import ProbeResult

        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl, journal=journal),
        ram_total_mb=4000,
        hw=_gpu(6144),
        run_bench=_bench_espion(appels),
    )
    run(con, deps)
    out = "\n".join(printed)
    assert "précontrôle incertain" in out and "flux inchangé" in out
    assert appels and appels[0]["ngl"]  # llama-bench a tourné, liste non filtrée
    assert any(e[0] == "isolation" for e in journal)
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert arch["precontrole"]["verdict"] == "incertain"


def test_precontrole_retire_les_ngl_impossibles_de_llama_bench(monkeypatch, tmp_path):
    """-ngl 99 (12 300 Mo certains sur le device) ne tient pas dans 6 144 Mo de VRAM : un
    seul -ngl qui échoue faisait échouer tout llama-bench — il est retiré, et dit."""
    appels: list = []

    def run_impl(ctx, depth):
        from loom.setup.topology import ProbeResult

        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    # VRAM libre (20 000) incohérente avec le total (6 144) : ngl_candidates garde 99.
    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl),
        hw=_gpu(6144, 20_000),
        meta=_meta_complete(),
        run_bench=_bench_espion(appels),
    )
    run(con, deps)
    out = "\n".join(printed)
    assert appels and 99 not in appels[0]["ngl"] and 0 in appels[0]["ngl"]
    assert "-ngl 99" in out and "VRAM" in out
    # Revue adverse : la note n'est jamais vide (« llama-bench : . »).
    assert "-ngl 0 seul" in out and "llama-bench : ." not in out


def test_precontrole_metadonnees_incompletes_llama_bench_non_filtre(
    monkeypatch, tmp_path
):
    """Contre-test du filtre : mêmes métadonnées, mais GGUF en 2 parties (catalogue
    partiel) → « incertain » : la liste de llama-bench reste celle d'avant, 99 compris."""
    appels: list = []

    def run_impl(ctx, depth):
        from loom.setup.topology import ProbeResult

        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl),
        hw=_gpu(6144, 20_000),
        meta=dict(_meta_complete(), split_count=2),
        run_bench=_bench_espion(appels),
    )
    run(con, deps)
    out = "\n".join(printed)
    assert "précontrôle incertain" in out and "2 parties" in out
    assert appels and appels[0]["ngl"] == [0, 99]


def _meta_moe(n=8, dense=900, experts=4000):
    meta = _meta_complete(n=n)
    meta.update(
        expert_count=64,
        weights=dict(
            meta["weights"],
            total=(n * (dense + experts) + 600) * _MIB,
            par_couche=[(dense + experts) * _MIB] * n,
            experts_par_couche=[experts * _MIB] * n,
        ),
    )
    return meta


def _mesure_ok(ctx, depth):
    from loom.setup.topology import ProbeResult

    r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
    if depth:
        r.tg_ts, r.pp_ts = 5.0, 20.0
    return r


def test_precontrole_moe_llama_bench_repli_sur_ngl_0_dit(monkeypatch, tmp_path):
    """MoE : llama-bench ne mesure que -ngl 999 -ncmoe n (la configuration du runtime).
    Ses denses (7 500 Mo) dépassent le budget device de 6 Go : repli sur -ngl 0, -ncmoe
    0, et la console le dit (jamais une note vide)."""
    appels: list = []
    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_mesure_ok),
        hw=_gpu(6144),
        meta=_meta_moe(),
        run_bench=_bench_espion(appels),
    )
    run(con, deps)
    out = "\n".join(printed)
    assert appels and appels[0]["ngl"] == [0] and appels[0]["ncmoe"] == 0
    assert "-ngl 999 retiré" in out and "repli sur -ngl 0" in out
    assert "llama-bench : ." not in out


def test_precontrole_vram_de_repli_filtre_llama_bench_et_le_dit(monkeypatch, tmp_path):
    """Revue adverse de L7 : VRAM lue par nvidia-smi seulement (profil de repli de
    detect_hardware, total 0). Le précontrôle et la sonde jugeaient cette VRAM,
    llama-bench non : il recevait -ngl 999 -ncmoe 8, refusé par le même parcours. Et la
    console doit dire que la capacité physique n'est pas établie."""
    appels: list = []
    repli = HardwareProfile(True, "RTX 8G", 7900, 16, vram_is_discrete=True)
    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_mesure_ok),
        hw=repli,
        meta=_meta_moe(),
        run_bench=_bench_espion(appels),
    )
    run(con, deps)
    out = "\n".join(printed)
    assert appels and appels[0]["ngl"] == [0] and appels[0]["ncmoe"] == 0
    assert "[attention] précontrôle : faisable" in out
    assert "capacité physique non établie" in out


def test_precontrole_gguf_a_l_en_tete_rejete_aucun_processus(monkeypatch, tmp_path):
    """Revue adverse : comme /rebench, un en-tête rejeté (ici une page HTML de 404
    enregistrée sous le nom du GGUF) est une impossibilité établie — llama-server le
    refuserait aussi. Ni llama-bench, ni sonde ; verdict et archive « précontrôle »."""
    journal: list = []
    appels: list = []
    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_rien_ne_doit_tourner, journal=journal),
        assume_yes=False,
        run_bench=_bench_espion(appels),
    )
    (mdir / "m.gguf").write_bytes(b"<html>404</html>")
    avant = (mdir / "model.toml").read_text(encoding="utf-8")
    assert run(con, deps) != 0
    out = "\n".join(printed)
    assert appels == [] and journal == []
    assert "GGUF illisible" in out and "aucun processus modèle lancé" in out
    assert "Lancer le bench maintenant" not in out
    assert (mdir / "model.toml").read_text(encoding="utf-8") == avant
    arch = json.loads(
        next((tmp_path / "var" / "bench" / "m1").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert arch["echec"]["etape"] == "précontrôle"
    assert arch["precontrole"]["verdict"] == "impossible"
    assert arch["precontrole"]["etabli"] is True


def test_precontrole_mmproj_absent_aucun_processus(monkeypatch, tmp_path):
    """Un mmproj annoncé par le model.toml mais absent (téléchargement interrompu : il
    est récupéré en dernier) comptait pour 0 Mo — or la sonde passe --mmproj et
    llama-server échoue au chargement, après le modèle principal. Bloquant."""
    journal: list = []
    appels: list = []
    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_rien_ne_doit_tourner, journal=journal),
        assume_yes=False,
        hw=_gpu(24_576),
        meta=_meta_complete(),
        run_bench=_bench_espion(appels),
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 12600\n'
        'mmproj_filename = "mmproj.gguf"\n',
        encoding="utf-8",
    )
    assert run(con, deps) != 0
    out = "\n".join(printed)
    assert appels == [] and journal == []
    assert "mmproj absent" in out and "aucun processus modèle lancé" in out


def test_mmproj_compte_cote_hote_par_la_sonde_et_le_plan_d_etape_2(
    monkeypatch, tmp_path
):
    """Chaque sonde charge le mmproj en RAM (--no-mmproj-offload) : la sonde
    d'isolation et le plan d'étape 2 le comptent côté hôte (vérification adverse)."""
    from tests.test_gguf_profile import _gguf

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_mesure_ok),
        hw=_gpu(24_576),
        meta=_meta_complete(),
    )
    _gguf(
        mdir / "mmproj.gguf",
        {"general.architecture": "clip"},
        [("v.blk.0.attn_k.weight", 2 * _MIB)],
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 12600\n'
        'mmproj_filename = "mmproj.gguf"\n',
        encoding="utf-8",
    )
    run(con, deps)
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert "mmproj 2 Mo" in arch["isolation"]["demarrage"]["raison"]
    assert "mmproj 2 Mo" in json.dumps(arch["plan"], ensure_ascii=False)


def test_precontrole_compte_le_mmproj_du_model_toml(monkeypatch, tmp_path):
    """Chemin `mmproj_filename` jamais exercé (revue adverse) : son catalogue est une
    allocation hôte certaine. 10 200 Mo de poids pour 6 144 + 4 057 Mo : seuls, ils
    tiennent physiquement ; avec 2 Mo de mmproj, non — impossibilité établie."""
    from pathlib import Path

    from loom.runtime.gguf_meta import read_gguf_meta as lire_vraiment
    from tests.test_gguf_profile import _gguf

    journal: list = []
    meta = _meta_complete(par_mb=240)
    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_rien_ne_doit_tourner, journal=journal),
        assume_yes=False,
        ram_total_mb=4057,
        hw=_gpu(6144),
    )
    monkeypatch.setattr(
        cli,
        "read_gguf_meta",
        lambda p: lire_vraiment(p) if Path(p).name == "mmproj.gguf" else meta,
    )
    _gguf(
        mdir / "mmproj.gguf",
        {"general.architecture": "clip"},
        [("v.blk.0.attn_k.weight", 2 * _MIB)],
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 10200\n'
        'mmproj_filename = "mmproj.gguf"\n',
        encoding="utf-8",
    )
    assert run(con, deps) != 0
    out = "\n".join(printed)
    assert journal == [] and "démarrage impossible" in out and "mmproj 2 Mo" in out
    arch = json.loads(
        next((tmp_path / "var" / "bench" / "m1").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert arch["precontrole"]["borne"]["mmproj_mb"] == 2
    assert arch["precontrole"]["verdict"] == "impossible"


def test_isolation_sur_une_copie_a_un_slot_demarrage_modeste(monkeypatch, tmp_path):
    """La sonde d'isolation tourne sur une COPIE à 1 slot. Le démarrage prévu (tout GPU,
    12 300 Mo) ne tient pas dans 8 192 Mo de VRAM : elle part sur un offload partiel qui
    tient. La sonde principale est construite avec la configuration prévue, mais aucun
    processus n'est lancé avec elle : la comparaison part des candidats qui tiennent."""
    journal: list = []

    def run_impl(ctx, depth):
        from loom.setup.topology import ProbeResult

        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl, journal=journal),
        ram_total_mb=32_000,
        hw=_gpu(8192, 8000),
        meta=_meta_complete(),
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 12600\n'
        "n_gpu_layers = 999\n",
        encoding="utf-8",
    )
    run(con, deps)
    construites = [e for e in journal if e[0] == "construite"]
    isolations = [e for e in journal if e[0] == "isolation"]
    lances = [e for e in journal if e[0] in ("run", "verify_cache", "isolation")]
    assert construites[0][1] == 999  # construite avec la configuration prévue…
    assert lances and not any(e[1] == 999 for e in lances)  # … jamais lancée
    assert len(isolations) == 1
    _, ngl, _cpu_moe, _ncmoe, slots, _ctx = isolations[0]
    assert slots == 1 and 0 < ngl < 40
    out = "\n".join(printed)
    assert "sonde d'isolation : le démarrage prévu ne tient pas" in out
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert arch["isolation"]["demarrage"]["modeste"] is True
    assert arch["isolation"]["demarrage"]["prevu_tient"] is False
    assert arch["isolation"]["slots_mesure"] == 1


def test_demarrage_prevu_qui_ne_tient_pas_jamais_repris_en_repli(monkeypatch, tmp_path):
    """Revue adverse : le démarrage prévu (tout GPU) ne tient pas à 4096 x 1. Quand
    aucun placement n'est validé, la suite revenait aux « flags actuels » — la
    calibration chargeait le démarrage condamné. Sortie explicite à l'étape placement,
    aucun processus lancé avec lui, rien d'écrit."""
    journal: list = []

    def run_impl(ctx, depth):
        raise RuntimeError("ErrorOutOfDeviceMemory")

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl, journal=journal),
        ram_total_mb=32_000,
        hw=_gpu(8192, 8000),
        meta=_meta_complete(),
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 12600\n'
        "n_gpu_layers = 999\n",
        encoding="utf-8",
    )
    avant = (mdir / "model.toml").read_text(encoding="utf-8")
    assert run(con, deps) != 0
    lances = [e for e in journal if e[0] in ("run", "verify_cache", "isolation")]
    assert lances and not any(e[1] == 999 for e in lances)
    out = "\n".join(printed)
    assert "aucun placement validé" in out and "calibration non lancée" in out
    assert "flags actuels conservés" not in out and "NON écrits" in out
    assert "ErrorOutOfDeviceMemory" in out  # la vraie erreur des candidats
    assert (mdir / "model.toml").read_text(encoding="utf-8") == avant
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "placement"
    assert "chargements de pente de la calibration" in arch["echec"]["erreur"]
    assert arch["repli_calibration"]["tient"] is False


def test_repli_refuse_aux_chargements_de_pente_de_la_calibration(monkeypatch, tmp_path):
    """Vérifications adverses : le démarrage prévu (tout GPU) tient à 4096 x 1 — la
    sonde d'isolation tourne dessus — et à 8192, mais pas à 16384, le second barreau de
    pente que la calibration charge quoi qu'il arrive ; l'étape 2 (contexte 32768) le
    refuse et ses partiels échouent. La calibration relançait -ngl 999. Sortie à
    l'étape placement, avec les VRAIES erreurs des candidats."""
    journal: list = []

    def run_impl(ctx, depth):
        raise RuntimeError("ErrorOutOfDeviceMemory: vk::Device::allocateMemory")

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl, journal=journal),
        ram_total_mb=32_000,
        hw=_gpu(8192, 8000),
        meta=_meta_complete(par_mb=160, sortie_mb=280, emb_mb=280),
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 6960\n'
        "n_gpu_layers = 999\ncontext = 32768\n",
        encoding="utf-8",
    )
    assert run(con, deps) != 0
    assert any(e[0] == "isolation" and e[1] == 999 for e in journal)  # tient à 4096
    assert not any(e[0] == "run" and e[1] == 999 for e in journal)
    out = "\n".join(printed)
    assert "aucun placement validé (toutes les mesures en échec" in out
    assert "vk::Device::allocateMemory" in out
    assert "chargements de pente de la calibration (16384 x 1 slot)" in out
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "placement"
    assert "allocateMemory" in arch["echec"]["erreur"]


def _meta_hybride(n=64, par_mb=100):
    """qwen35 : 1 couche d'attention sur 4, état récurrent de Bonsai 2 (~150 Mio)."""
    meta = _meta_complete(n=n, par_mb=par_mb)
    meta.update(
        architecture="qwen35",
        context_length=262144,
        head_count_kv=4,
        key_length=256,
        value_length=256,
        full_attention_interval=4,
        recurrent=True,
        ssm_conv_kernel=4,
        ssm_inner_size=6144,
        ssm_state_size=128,
        ssm_group_count=16,
    )
    meta["weights"] = dict(
        meta["weights"],
        couches_attention=[i for i in range(n) if (i + 1) % 4 == 0],
        couches_recurrentes=[i for i in range(n) if (i + 1) % 4 != 0],
    )
    return meta


def test_repli_hybride_calibre_sans_checkpoints_fantomes(monkeypatch, tmp_path):
    """Régression du lot L11 (vérification adverse) : la garde du repli comptait 32
    checkpoints par slot aux barreaux de pente, chargements NUS qui n'en créent aucun.
    Hybride sur 8 Go + 13 000 Mo, -ngl 40 actuel, 2 slots imposés : le seul candidat
    (tout GPU) échoue ; le repli -ngl 40 tient (~3 200 Mo côté hôte) — il était refusé
    pour 9 576 Mo de checkpoints fantômes et la calibration n'avait jamais lieu."""
    journal: list = []

    def run_impl(ctx, depth, sonde):
        if sonde.ngl == 999:
            raise RuntimeError("ErrorOutOfDeviceMemory")
        return _mesure_ok(ctx, depth)

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl, journal=journal, avec_sonde=True),
        ram_total_mb=13_000,
        hw=_gpu(8192, 8000),
        meta=_meta_hybride(),
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 64\nsize_mb = 7000\n'
        "n_gpu_layers = 40\n",
        encoding="utf-8",
    )
    run(con, deps)
    out = "\n".join(printed)
    assert "aucun placement validé" not in out
    pente = [e for e in journal if e[0] == "run" and e[1] == 40 and e[6] is None]
    assert [(e[4], e[5]) for e in pente][:2] == [(2, 8192), (2, 16384)]
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert arch["repli_calibration"]["tient"] is True


def test_isolation_imposee_par_la_memoire_recurrente_sonde_non_lancee(
    monkeypatch, tmp_path
):
    """Mémoire récurrente, démarrage prévu qui ne tient pas à 4096 x 1 : la sonde
    d'isolation n'est pas lancée (chargement condamné, verdict imposé de toute façon).
    La suite mesure à 2 slots et l'isolation est écrite ; la trace le dit."""
    journal: list = []

    def run_impl(ctx, depth):
        from loom.setup.topology import ProbeResult

        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(run_impl, journal=journal),
        ram_total_mb=32_000,
        hw=_gpu(8192, 8000),
        meta=dict(_meta_complete(), recurrent=True),
    )
    (mdir / "model.toml").write_text(
        'repo = "org/r"\nfilename = "m.gguf"\nn_layers = 40\nsize_mb = 12600\n'
        "n_gpu_layers = 999\n",
        encoding="utf-8",
    )
    assert run(con, deps) == 0
    assert not any(e[0] == "isolation" for e in journal)
    mesures = [e for e in journal if e[0] in ("run", "verify_cache")]
    assert mesures and all(e[4] == 2 for e in mesures)
    out = "\n".join(printed)
    assert "isolation imposée, sonde non lancée" in out
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    iso = json.loads(archives[-1].read_text(encoding="utf-8"))["isolation"]
    assert iso["necessaire"] is True and iso["slots_mesure"] == 0
    assert iso["slots_retenus"] == 2 and iso["demarrage"]["lancer"] is False
    mt = tomllib.loads((mdir / "model.toml").read_text(encoding="utf-8"))
    assert mt["cache_isolation"] is True


def test_echec_de_llama_bench_archive(monkeypatch, tmp_path):
    """Le compte rendu existe désormais avant llama-bench : son échec est archivé."""

    def run_bench(b, m, t, g, n_cpu_moe=0, progress=None):
        raise RuntimeError("llama-bench a échoué : vulkan: out of memory")

    con, printed, deps, _mdir = _harnais_bench(
        monkeypatch,
        tmp_path,
        _fake_probe_cls(_rien_ne_doit_tourner),
        run_bench=run_bench,
    )
    assert run(con, deps) != 0
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "llama-bench"
    assert "out of memory" in arch["echec"]["erreur"]


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


def test_etape_bench_aucun_placement_faisable_n_ecrit_rien_et_archive(
    monkeypatch, tmp_path
):
    """Revue #14 P1 : rien ne tient d'après l'estimation (4 Go de RAM pour un modèle de
    5,6 Go) -> pas de mesure de placement ni de calibration, réglages NON écrits,
    échec archivé à l'étape placement. Avant : CPU seul proposé sans vérification."""
    from loom.setup.topology import ProbeResult

    lancés: list = []

    def run_impl(ctx, depth):
        lancés.append((ctx, depth))
        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            r.tg_ts, r.pp_ts = 5.0, 20.0
        return r

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch, tmp_path, _fake_probe_cls(run_impl), ram_total_mb=4_000
    )
    avant = (mdir / "model.toml").read_text(encoding="utf-8")
    assert run(con, deps) != 0
    out = "\n".join(printed)
    assert "aucun placement faisable" in out and "NON écrits" in out
    # Revue #15 : formulation exacte — le contrôle arrive à l'étape placement.
    assert "aucun placement comparé, calibration non lancée" in out
    # Revue n°16 : à l'étape 2, c'est le contexte DEMANDÉ qui ne tient pas — pas
    # « impossible de démarrer » : le message nomme le contexte, les slots et les postes.
    assert "le contexte utile" in out and "ne tient avec aucun placement" in out
    assert "postes à ce contexte" in out
    # GGUF sans catalogue : précontrôle « incertain » — il n'a rien établi au plancher,
    # le message ne prétend pas que le démarrage « passait » (revue adverse).
    assert "passait" not in out and "plancher non établie" in out
    assert lancés == []  # ni comparaison de placement, ni calibration
    assert (mdir / "model.toml").read_text(encoding="utf-8") == avant
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    assert len(archives) == 1
    arch = json.loads(archives[0].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "placement"
    assert "aucun placement faisable" in arch["echec"]["erreur"]
    # … APRÈS llama-bench et la sonde d'isolation, qui ont déjà chargé le modèle.
    assert arch["llama_bench"] and arch["isolation"]["first"] == 600


def test_etape_bench_reglage_final_hors_contrainte_prefill_n_ecrit_rien(
    monkeypatch, tmp_path
):
    """Revue #14 P2 : la contrainte de prefill, tenue pendant la comparaison, est
    violée par le réglage final (2 000 tokens en 20 s > 10 s). Ce n'est pas une
    divergence de génération : réglages NON écrits, durée face à la limite dite,
    échec archivé à l'étape « réglage final »."""
    from loom.setup.placement import final_depth, useful_context
    from loom.setup.topology import ProbeResult

    depth_finale = final_depth(useful_context(None, None, None))
    etat = {"calibration": False}

    def run_impl(ctx, depth):
        if depth is None:
            etat["calibration"] = True  # barreaux de pente : la calibration a commencé
        r = ProbeResult(ctx=ctx, mem_mb=int(1000 + ctx * 0.01))
        if depth:
            final = etat["calibration"] and depth == depth_finale
            r.tg_ts, r.pp_ts = 5.0, (100.0 if final else 500.0)
        return r

    con, printed, deps, mdir = _harnais_bench(
        monkeypatch, tmp_path, _fake_probe_cls(run_impl)
    )
    local_path = tmp_path / "config" / "local.toml"
    local_path.write_text(
        local_path.read_text(encoding="utf-8")
        + "\n[placement]\nprefill_new_tokens = 2000\nprefill_max_s = 10\n",
        encoding="utf-8",
    )
    avant = (mdir / "model.toml").read_text(encoding="utf-8")
    assert run(con, deps) != 0
    out = "\n".join(printed)
    assert "contrainte prefill NON respectée" in out and "20.0 s > 10 s" in out
    assert "NON écrits" in out and "KeyError" not in out
    assert (mdir / "model.toml").read_text(encoding="utf-8") == avant
    archives = list((tmp_path / "var" / "bench" / "m1").glob("*.json"))
    arch = json.loads(archives[-1].read_text(encoding="utf-8"))
    assert arch["echec"]["etape"] == "réglage final"
    assert "contrainte prefill" in arch["echec"]["erreur"]


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
    _gguf_sans_catalogue(mdir / "m.gguf")
    rows = [
        {"threads": 10, "ngl": 99, "kind": "tg", "ts": 3.4},
        {"threads": 10, "ngl": 99, "kind": "pp", "ts": 25.0},
    ]
    sondes: list = []
    isolations: list = []

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
            isolations.append(self)
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
        ram_total_mb=lambda: 64_000,
        make_probe=_FakeProbe,
        detect_hardware=lambda server_bin=None: HardwareProfile(
            True, "GPU 20Go", 20_000, 16, vram_is_discrete=True
        ),
    )
    assert run(con, deps) == 0
    # La sonde d'isolation tourne sur une COPIE à 1 slot (scénario A -> B -> A : à 2
    # slots, B partirait sur le slot libre) ; toutes les AUTRES sondes (initiale et
    # clones) ont mesuré à 2 slots, l'isolation actuelle étant conservée.
    assert len(isolations) == 1 and isolations[0].n_parallel == 1
    autres = [s for s in sondes if s is not isolations[0]]
    assert autres and all(s.n_parallel == 2 for s in autres)
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
