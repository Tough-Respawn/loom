# loom/setup/bench.py
"""Bench du matériel avec le VRAI modèle installé (llama-bench, livré avec la
release llama.cpp) : on MESURE au lieu de deviner.

Deux réglages sortent d'ici :
- threads + offload GPU (-t, -ngl) : mesurés — la génération (tg) tranche, c'est
  elle que l'utilisateur vit ; le prefill (pp) départage les ex æquo.
- context : CALCULÉ (pas benché) — le plus grand cache KV qui tient dans la RAM
  disponible à côté des poids du modèle, borné au défaut du parc. Le KV/token
  vient du header GGUF (gguf_meta), avec un repli conservateur.

Les résultats vont dans config/local.toml : [override] threads / n_gpu_layers,
[server] context, et une table [bench] (trace + idempotence)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from loom.runtime.hardware import recommend_gpu_layers

# Charges courtes : sur une petite machine CPU, chaque token compte — on veut un
# CLASSEMENT fiable (quel -t gagne), pas des chiffres de communiqué de presse.
BENCH_PROMPT = 128
BENCH_GEN = 16
BENCH_REPS = 2

# Marge RAM système (Mo) laissée libre à côté des poids + KV.
_RAM_MARGIN_MB = 2048
# KV/token de repli si le header GGUF ne porte pas les champs d'attention
# (~valeur d'un 8B dense GQA : 36 couches x 1024 dims KV x 2 (K+V) x 2 octets).
_KV_FALLBACK_BYTES = 150_000


def find_llama_bench(server_bin: str | Path) -> Path | None:
    """llama-bench(.exe) vit à côté de llama-server dans la release."""
    d = Path(server_bin).parent
    for name in ("llama-bench.exe", "llama-bench"):
        p = d / name
        if p.is_file():
            return p
    return None


def has_gpu_backend(server_bin: str | Path) -> bool:
    """Vrai si la release embarque un backend GPU (Vulkan/CUDA) : l'offload -ngl
    vaut alors la peine d'être MESURÉ (iGPU Intel/AMD : ça se joue au cas par cas)."""
    d = Path(server_bin).parent
    return any(
        (d / n).is_file()
        for n in (
            "ggml-vulkan.dll",
            "libggml-vulkan.so",
            "ggml-cuda.dll",
            "libggml-cuda.so",
        )
    )


def ngl_candidates(
    gpu_backend: bool,
    vram_free_mb: int,
    model_size_mb: int,
    n_layers: int | None,
) -> list[int]:
    """Candidats -ngl à bencher. Sans backend GPU : [0]. Sinon 0 (CPU pur) et 99
    (offload total) ; si la VRAM libre ne couvre qu'une PARTIE du modèle, la
    recommandation proportionnelle s'ajoute comme candidat intermédiaire — on la
    MESURE au lieu de la supposer. n_layers None (GGUF illisible) : 0/99 seuls."""
    if not gpu_backend:
        return [0]
    cands = {0, 99}
    if n_layers:
        reco = recommend_gpu_layers(vram_free_mb, model_size_mb, n_layers)
        if 0 < reco < n_layers:
            cands.add(reco)
    return sorted(cands)


def thread_candidates(logical: int, physical: int | None) -> list[int]:
    """Valeurs de -t à mesurer : cœurs physiques (référence llama.cpp), tous les
    threads, et physiques/2 (machines hybrides P+E : moins de contention).
    Dédupliquées, croissantes, jamais vides."""
    phys = physical or logical
    cands = {max(1, phys // 2), phys, logical}
    return sorted(cands)


def kv_bytes_per_token(meta: dict) -> int:
    """Octets de cache KV par token depuis le header GGUF (K+V, f16), ou repli
    conservateur si les champs d'attention manquent."""
    layers = meta.get("n_layers")
    kv_heads = meta.get("head_count_kv")
    head_dim = meta.get("key_length")
    if not head_dim and meta.get("embedding_length") and meta.get("head_count"):
        head_dim = meta["embedding_length"] // meta["head_count"]
    if not (layers and kv_heads and head_dim):
        return _KV_FALLBACK_BYTES
    return 2 * layers * kv_heads * head_dim * 2  # K+V x couches x dims x f16


def compute_context(
    ram_avail_mb: int, model_size_mb: int, kv_per_token: int, cap: int = 24576
) -> int:
    """Plus grand contexte dont le KV tient dans la RAM restante (RAM dispo −
    poids du modèle − marge), borné à [4096, cap], arrondi au multiple de 2048
    inférieur. C'est LE réglage qui évite le swap (vécu : 24576 par défaut sur
    16 Go → KV 3,6 Go → swap → génération à 0,8 t/s)."""
    budget_mb = ram_avail_mb - model_size_mb - _RAM_MARGIN_MB
    if budget_mb <= 0:
        return 4096
    tokens = (budget_mb * 1024 * 1024) // max(1, kv_per_token)
    ctx = max(4096, min(cap, (tokens // 2048) * 2048))
    return int(ctx)


def run_llama_bench(
    bench_bin: str | Path,
    model_path: str | Path,
    threads: list[int],
    ngl: list[int],
    runner=subprocess.run,
    timeout: int = 1200,
) -> list[dict]:
    """Lance llama-bench sur toutes les combinaisons (une seule invocation :
    llama-bench croise les listes) et renvoie les lignes JSON normalisées :
    [{threads, ngl, kind: 'pp'|'tg', ts}]. Lève RuntimeError si le bench échoue."""
    cmd = [
        str(bench_bin),
        "-m",
        str(model_path),
        "-p",
        str(BENCH_PROMPT),
        "-n",
        str(BENCH_GEN),
        "-r",
        str(BENCH_REPS),
        "-t",
        ",".join(str(t) for t in threads),
        "-ngl",
        ",".join(str(g) for g in ngl),
        "-o",
        "json",
    ]
    try:
        res = runner(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"llama-bench a dépassé {timeout}s — abandonné") from exc
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError("llama-bench a échoué : " + " | ".join(tail))
    try:
        raw = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("sortie llama-bench illisible (pas du JSON)") from exc
    rows = []
    for r in raw:
        kind = "pp" if int(r.get("n_prompt", 0)) > 0 else "tg"
        rows.append(
            {
                "threads": int(r.get("n_threads", 0)),
                "ngl": int(r.get("n_gpu_layers", 0)),
                "kind": kind,
                "ts": float(r.get("avg_ts", 0.0)),
            }
        )
    return rows


def pick_best(rows: list[dict]) -> dict | None:
    """Meilleure combinaison (threads, ngl) : la GÉNÉRATION (tg) tranche — c'est
    la vitesse vécue ; le prefill (pp) départage. Renvoie
    {threads, ngl, tg_ts, pp_ts} ou None si aucune mesure tg."""
    combos: dict[tuple[int, int], dict] = {}
    for r in rows:
        c = combos.setdefault((r["threads"], r["ngl"]), {"tg_ts": 0.0, "pp_ts": 0.0})
        c["tg_ts" if r["kind"] == "tg" else "pp_ts"] = r["ts"]
    scored = [
        {"threads": t, "ngl": g, **v} for (t, g), v in combos.items() if v["tg_ts"] > 0
    ]
    if not scored:
        return None
    return max(scored, key=lambda c: (c["tg_ts"], c["pp_ts"]))
