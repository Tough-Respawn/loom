# loom/setup/topology.py
"""Calibration AGNOSTIQUE du contexte : découverte de topologie + pente MESURÉE.

Remplace la formule « KV théorique vs RAM » de bench.py, fausse deux fois sur le
parc réel (audit + sondes du 2026-07-18) :
- elle supposait le KV en f16 alors que l'exécutant tourne en q8_0 (×1,9) ;
- elle ignorait l'attention à fenêtre glissante (qwen35moe : ~9 Ko/token RÉELS
  contre 43,5 théoriques, ×5) ;
- elle ne modélisait qu'UNE topologie (tout-en-RAM) quand Loom en exploite trois.

Principes (chacun répond à un pattern de l'audit) :
1. LE CONSEILLEUR SIMULE L'EXÉCUTANT : la sonde lance llama-server avec la ligne
   de commande de `server_args.py` — jamais ses propres flags. (P2)
2. LA PENTE MESURÉE FAIT FOI : deux chargements à deux contextes, la différence
   de mémoire donne le coût marginal RÉEL par token — aucune formule de header
   ne survit aux architectures modernes. (P1, la leçon des sondes)
3. DÉTERMINISME : les budgets partent de la mémoire TOTALE (moins des marges
   fixes), jamais de la mémoire disponible du moment. (P3)
4. ON N'ÉCRIT QUE DU VÉRIFIÉ : la capacité extrapolée est bornée par une échelle
   de VITESSE mesurée en profondeur — on recommande le dernier barreau où le
   décode tient, pas ce que la droite promet. (P4)
5. LA DÉCISION PORTE SON MÉCANISME : la trace dit quelle contrainte a mordu
   (capacité, vitesse, budget temps, limite du modèle). (P6)

Le premier run llama-server d'une session mesure des débits ÷2 (caches froids,
allocation pinnée) : chaque sonde de vitesse est précédée d'un warmup jetable.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field

from loom.runtime.server_args import build_server_args

TOPO_MOE_HYBRIDE = "moe_hybride"  # experts en RAM, attention + KV en VRAM
TOPO_GPU_DENSE = "gpu_dense"  # tout le modèle + KV en VRAM
TOPO_RAM = "ram"  # pas de GPU exploitable : poids + KV en RAM

# Marge RAM laissée à l'OS quand le KV vit en RAM (déterministe : sur la RAM TOTALE).
_OS_RAM_BUDGET_MB = 3072
# Le décode doit garder au moins cette fraction du débit de référence — en dessous,
# on considère le barreau invalide (spill/dégradation) et on garde le précédent.
_TG_FLOOR_RATIO = 0.7
# Fraction du contexte réellement REMPLIE par la sonde de profondeur.
_DEPTH_FILL = 0.85
_FLOOR_CTX = 4096
_STEP_CTX = 2048


def discover_topology(meta: dict, gpu_backend: bool, vram_total_mb: int) -> str:
    """Choisit la topologie candidate depuis le matériel et le modèle — c'est la
    découverte qui décide, pas une hypothèse d'auteur. MoE + GPU -> hybride
    (doctrine mesurée du parc : experts en RAM, attention + KV en VRAM)."""
    if not gpu_backend or vram_total_mb <= 0:
        return TOPO_RAM
    if meta.get("expert_count"):
        return TOPO_MOE_HYBRIDE
    return TOPO_GPU_DENSE


def kv_slope(rungs: list[tuple[int, int]]) -> tuple[float, float]:
    """(pente octets/token, base Mo) depuis ≥2 barreaux (ctx, mémoire_Mo) mesurés.
    llama-server alloue le cache KV ENTIER au chargement : la mémoire à vide suffit,
    pas besoin de générer. Deux points = une droite ; plus = moindres carrés."""
    if len(rungs) < 2:
        raise ValueError("il faut au moins 2 barreaux (ctx, mem_mb)")
    n = len(rungs)
    sx = sum(c for c, _ in rungs)
    sy = sum(m for _, m in rungs)
    sxx = sum(c * c for c, _ in rungs)
    sxy = sum(c * m for c, m in rungs)
    denom = n * sxx - sx * sx
    if denom == 0:
        raise ValueError("barreaux au même contexte : pente incalculable")
    slope_mb_per_tok = (n * sxy - sx * sy) / denom
    base_mb = (sy - slope_mb_per_tok * sx) / n
    return slope_mb_per_tok * 1024 * 1024, base_mb


def capacity_ctx(
    slope_bytes: float,
    base_mb: float,
    budget_mb: int,
    model_limit: int,
    floor: int = _FLOOR_CTX,
    step: int = _STEP_CTX,
) -> int:
    """Plus grand contexte dont la mémoire PRÉDITE par la pente tient dans le budget,
    borné par la limite du modèle, arrondi au multiple de `step` inférieur."""
    if slope_bytes <= 0:
        return min(model_limit, floor)
    tokens = (budget_mb - base_mb) * 1024 * 1024 / slope_bytes
    ctx = int(min(model_limit, max(floor, (tokens // step) * step)))
    return ctx


@dataclass
class ProbeResult:
    ctx: int
    mem_mb: int  # VRAM (topologies GPU) ou RSS du process (topologie RAM)
    tg_ts: float | None = None  # décode t/s à ~85 % de profondeur (si sondé)
    pp_ts: float | None = None


@dataclass
class ServerProbe:
    """Sonde réelle : lance llama-server avec les flags EXACTS de l'exécutant, lit
    la mémoire, optionnellement mesure les débits en profondeur, tue l'arbre.
    Tout est injectable pour les tests (aucun subprocess dans le cœur pur)."""

    server_bin: str
    model_path: str
    threads: int
    ngl: int
    topology: str
    mmproj_path: str | None = None
    cpu_moe: bool = False
    n_cpu_moe: int | None = None
    port: int = 8131
    health_timeout_s: int = 600
    # Injectables (tests) — défauts = implémentations réelles.
    popen: object = subprocess.Popen
    kill: object = None
    vram_mb: object = None
    warmed_up: bool = field(default=False, init=False)

    def _measure_mem(self, proc) -> int:
        if self.topology == TOPO_RAM:
            import psutil

            return int(psutil.Process(proc.pid).memory_info().rss // (1024 * 1024))
        if self.vram_mb is not None:
            return int(self.vram_mb())
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        return int(out.splitlines()[0])

    def _kill(self, proc) -> None:
        if self.kill is not None:
            self.kill(proc)
            return
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            timeout=30,
        )

    def _tokens_of(self, text: str) -> int:
        """Compte de tokens par LE serveur qui tourne (/tokenize) — agnostique au
        modèle. Repli conservateur (1 token/2 caractères) si l'endpoint manque :
        surestimer les tokens raccourcit le prompt, jamais l'inverse."""
        try:
            body = json.dumps({"content": text}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/tokenize",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return max(1, len(json.loads(r.read()).get("tokens") or []))
        except Exception:  # noqa: BLE001 - repli prudent, jamais bloquant
            return max(1, len(text) // 2)

    def _completion(self, prompt: str, n_predict: int) -> dict:
        body = json.dumps(
            {
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.0,
                "cache_prompt": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())

    def run(self, ctx: int, depth_tokens: int | None) -> ProbeResult:
        args = build_server_args(
            server_bin=self.server_bin,
            model_path=self.model_path,
            port=self.port,
            context=ctx,
            n_gpu_layers=self.ngl,
            threads=self.threads,
            mmproj_path=self.mmproj_path,
            gpu_tuning=self.topology != TOPO_RAM,
            n_parallel=1,
            cpu_moe=self.cpu_moe,
            n_cpu_moe=self.n_cpu_moe,
        )
        proc = self.popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.health_timeout_s:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/health", timeout=3
                    ) as r:
                        if r.status == 200:
                            break
                except Exception:
                    time.sleep(3)
            else:
                raise RuntimeError(f"chargement KO à ctx={ctx} (health timeout)")
            res = ProbeResult(ctx=ctx, mem_mb=self._measure_mem(proc))
            if depth_tokens:
                phrase = "La pente mesurée vaut mieux que la formule du header. "
                if not self.warmed_up:
                    # 1er run de la session = caches froids, débits ÷2 : on purge.
                    self._completion(phrase * 40, 16)
                    self.warmed_up = True
                # Tokens/répétition MESURÉS auprès du serveur (/tokenize) — jamais
                # estimés : la densité varie du simple au triple selon le tokenizer,
                # et une estimation a réellement fait déborder le contexte (400) au
                # premier test de vérité de ce module, le 2026-07-18.
                tok_per_rep = self._tokens_of(phrase)
                # Ne jamais dépasser la fenêtre : profondeur bornée par le contexte
                # moins la génération et une marge (template/BOS).
                depth = min(depth_tokens, ctx - 96 - 256)
                reps = max(4, depth // max(1, tok_per_rep))
                resp = self._completion(phrase * reps, 96)
                t = resp.get("timings") or {}
                res.tg_ts = round(t.get("predicted_per_second") or 0.0, 1)
                res.pp_ts = round(t.get("prompt_per_second") or 0.0, 1)
            return res
        finally:
            self._kill(proc)
            time.sleep(4)  # laisser la mémoire se libérer avant le barreau suivant


def calibrate(
    probe,
    meta: dict,
    *,
    topology: str,
    budget_mb: int,
    time_budget_s: int = 900,
    progress=None,
) -> dict:
    """Orchestrateur : pente mesurée -> capacité -> échelle de VITESSE -> décision
    TRACÉE. `probe` expose run(ctx, depth_tokens|None) -> ProbeResult.

    Renvoie {context, mode, mecanisme, slope_kb_tok, capacity_ctx, rungs, vitesses,
    valide_jusqua} — `mecanisme` nomme la contrainte qui a mordu.
    """
    say = progress or (lambda _msg: None)
    t0 = time.monotonic()
    model_limit = int(meta.get("context_length") or 32768)

    # 1. Pente : deux chargements à vide, petits donc rapides.
    rungs = []
    for ctx in (8192, 16384):
        say(f"pente : chargement à ctx={ctx}…")
        r = probe.run(ctx, None)
        rungs.append((r.ctx, r.mem_mb))
    slope_bytes, base_mb = kv_slope(rungs)
    cap = capacity_ctx(slope_bytes, base_mb, budget_mb, model_limit)

    # 2. Échelle de vitesse : barreaux géométriques ≤ capacité ; on ne recommande
    # que du décode VÉRIFIÉ en profondeur — jamais l'extrapolation seule.
    vitesses: list[dict] = []
    tg_ref: float | None = None
    valide = 0
    mecanisme = "capacité (pente mesurée)"
    ladder = [c for c in (16384, 32768, 65536, 131072, 262144) if c <= cap]
    if not ladder and cap >= _FLOOR_CTX:
        ladder = [cap]
    for ctx in ladder:
        elapsed = time.monotonic() - t0
        # >= : un budget ATTEINT arrête (et budget 0 = aucun barreau, même quand
        # l'horloge n'a pas encore tické — monotonic a ~15 ms de résolution sur
        # Windows < 3.13, un `>` strict laissait passer toute l'échelle).
        if elapsed >= time_budget_s:
            mecanisme = (
                f"budget temps ({time_budget_s}s) — vitesse validée jusqu'à {valide}"
            )
            break
        depth = int(ctx * _DEPTH_FILL)
        say(f"vitesse : ctx={ctx}, profondeur ~{depth} tokens…")
        try:
            r = probe.run(ctx, depth)
        except Exception as exc:  # noqa: BLE001 - un barreau qui casse N'EST PAS fatal :
            # il devient le mécanisme d'arrêt (dernier barreau sain conservé). Leçon du
            # test de vérité du 2026-07-18 : un HTTPError tuait toute la calibration.
            mecanisme = (
                f"échec du barreau ctx={ctx} ({type(exc).__name__}: {exc}) — "
                "dernier barreau sain conservé"
            )
            break
        vitesses.append(
            {"ctx": ctx, "tg_ts": r.tg_ts, "pp_ts": r.pp_ts, "mem_mb": r.mem_mb}
        )
        if not r.tg_ts:
            mecanisme = f"débit illisible à ctx={ctx} — dernier barreau sain conservé"
            break
        if tg_ref is None:
            tg_ref = r.tg_ts
        if r.tg_ts < tg_ref * _TG_FLOOR_RATIO:
            mecanisme = (
                f"vitesse : décode {r.tg_ts} t/s < {_TG_FLOOR_RATIO:.0%} de la référence "
                f"{tg_ref} t/s à ctx={ctx} (spill/dégradation) — barreau précédent conservé"
            )
            break
        valide = ctx
    else:
        if valide == cap:
            mecanisme = "capacité (pente mesurée), vitesse validée à chaque barreau"
        elif valide:
            mecanisme = f"limite du modèle ou capacité atteinte — vitesse validée jusqu'à {valide}"

    context = max(_FLOOR_CTX, valide or min(cap, _FLOOR_CTX))
    return {
        "context": int(context),
        "mode": topology,
        "mecanisme": mecanisme,
        "slope_kb_tok": round(slope_bytes / 1024, 1),
        "base_mb": round(base_mb),
        "budget_mb": budget_mb,
        "capacity_ctx": cap,
        "rungs": rungs,
        "vitesses": vitesses,
        "valide_jusqua": valide,
        "duree_s": round(time.monotonic() - t0),
    }


def gpu_vram_total_mb() -> int:
    """VRAM totale via nvidia-smi (0 si absent) — déterministe, contrairement à
    memory.free qui dépend de ce qui tourne."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001 - pas de nvidia-smi = pas de GPU NVIDIA
        return 0


def memory_budget_mb(
    topology: str, vram_total_mb: int, ram_total_mb: int, headroom_mb: int
) -> int:
    """Budget mémoire DÉTERMINISTE pour poids-GPU + KV selon la topologie :
    totaux moins marges fixes, jamais la mémoire disponible du moment (P3)."""
    if topology == TOPO_RAM:
        return max(0, ram_total_mb - _OS_RAM_BUDGET_MB)
    return max(0, vram_total_mb - headroom_mb)
