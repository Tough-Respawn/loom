# Installeur loom-setup : sélection d'assets llama.cpp (matrice OS+GPU),
# localisation du binaire extrait — SANS réseau.
from loom.setup.llama_release import find_llama_server, select_assets


def _release(names, tag="b5321"):
    return {
        "tag_name": tag,
        "assets": [
            {
                "name": n,
                "browser_download_url": f"https://example.test/{n}",
                "size_mb_src": 0,
                "size": 250 * 1024 * 1024,
            }
            for n in names
        ],
    }


_MODERN = [
    "llama-b5321-bin-macos-arm64.zip",
    "llama-b5321-bin-ubuntu-x64.zip",
    "llama-b5321-bin-win-cpu-x64.zip",
    "llama-b5321-bin-win-cuda-12.4-x64.zip",
    "llama-b5321-bin-win-vulkan-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
]


def test_windows_nvidia_prend_cuda_et_cudart():
    plan = select_assets(_release(_MODERN), "windows", "x64", has_nvidia=True)
    assert plan is not None and plan.backend == "cuda" and plan.tag == "b5321"
    names = [a["name"] for a in plan.assets]
    assert names == [
        "llama-b5321-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ]
    assert plan.total_mb == 500  # tailles agrégées pour l'écran de confirmation


def test_windows_nvidia_sans_cudart_refuse():
    names = [n for n in _MODERN if not n.startswith("cudart")]
    assert select_assets(_release(names), "windows", "x64", True) is None


def test_windows_sans_gpu_prefere_vulkan_puis_cpu():
    plan = select_assets(_release(_MODERN), "windows", "x64", has_nvidia=False)
    assert plan.assets[0]["name"] == "llama-b5321-bin-win-vulkan-x64.zip"
    sans_vulkan = [n for n in _MODERN if "vulkan" not in n]
    plan = select_assets(_release(sans_vulkan), "windows", "x64", False)
    assert plan.assets[0]["name"] == "llama-b5321-bin-win-cpu-x64.zip"


def test_windows_ancien_nommage_avx2():
    plan = select_assets(
        _release(["llama-b3000-bin-win-avx2-x64.zip"]), "windows", "x64", False
    )
    assert plan.assets[0]["name"] == "llama-b3000-bin-win-avx2-x64.zip"


def test_linux_nvidia_sans_build_cuda_renvoie_none():
    # Les releases ne portent pas toujours de build CUDA Linux -> guidage manuel.
    assert select_assets(_release(_MODERN), "linux", "x64", True) is None


def test_linux_sans_gpu_et_macos():
    plan = select_assets(_release(_MODERN), "linux", "x64", False)
    assert plan.assets[0]["name"] == "llama-b5321-bin-ubuntu-x64.zip"
    plan = select_assets(_release(_MODERN), "macos", "arm64", False)
    assert plan.backend == "metal"


def test_macos_nvidia_impossible_retombe_sans_gpu():
    # (macos, arm64, True) n'existe pas dans la matrice -> repli variante CPU.
    plan = select_assets(_release(_MODERN), "macos", "arm64", True)
    assert plan is not None and plan.backend == "metal"


def test_aucun_asset_ne_matche():
    assert select_assets(_release(["source.tar.gz"]), "windows", "x64", False) is None


def test_find_llama_server_zip_plat_et_arbo_ubuntu(tmp_path):
    # zip Windows : exe à plat
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "llama-server.exe").write_bytes(b"")
    assert find_llama_server(flat).name == "llama-server.exe"
    # tar ubuntu : build/bin/llama-server
    deep = tmp_path / "deep" / "build" / "bin"
    deep.mkdir(parents=True)
    (deep / "llama-server").write_bytes(b"")
    found = find_llama_server(tmp_path / "deep")
    assert found is not None and found.name == "llama-server"
    # ne confond pas avec d'autres binaires llama-*
    other = tmp_path / "other"
    other.mkdir()
    (other / "llama-quantize.exe").write_bytes(b"")
    assert find_llama_server(other) is None
    assert find_llama_server(tmp_path / "absent") is None
