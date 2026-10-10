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

Un llama-server qui vient de démarrer mesure des débits ÷2 (caches froids,
allocation pinnée) : chaque run relançant un serveur neuf, chaque mesure de vitesse
est précédée de son propre warmup jetable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field

from loom.runtime.effective import launch_flags
from loom.runtime.server_args import build_server_args

TOPO_MOE_HYBRIDE = "moe_hybride"  # experts en RAM, attention + KV en VRAM
TOPO_GPU_DENSE = "gpu_dense"  # tout le modèle + KV en VRAM
TOPO_RAM = "ram"  # pas de GPU exploitable : poids + KV en RAM

# Réserver une marge fixe à l'OS lorsque le KV occupe la RAM.
_OS_RAM_BUDGET_MB = 3072
# Rejeter un barreau dont le débit révèle un spill ou une dégradation excessive.
_TG_FLOOR_RATIO = 0.7
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
    mem_mb: int  # selon ServerProbe.memory_mode : device, delta de RAM (UMA) ou RSS
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
    # Simuler le nombre réel de slots pour mesurer aussi le coût de l'isolation KV.
    n_parallel: int = 1
    # Batchs de prefill sondés par probe_ubatch (None = défauts llama-server).
    ubatch: int | None = None
    batch: int | None = None
    # Checkpoints des hybrides : chacun pèse l'état récurrent complet, la mémoire
    # mesurée doit être celle de l'exécutant (model.toml / [server] défaut machine).
    checkpoint_min_step: int | None = None
    ctx_checkpoints: int | None = None
    # Profil matériel de l'exécutant (`--list-devices`) : les flags machine (profil
    # GPU, mémoire unifiée) en sont dérivés par effective.launch_flags, comme dans
    # serve.py et swap.py. Sans profil (tests, anciens appelants) la topologie décide.
    profile: object = None
    port: int = 8131
    health_timeout_s: int = 600
    popen: object = subprocess.Popen
    kill: object = None
    vram_mb: object = None  # () -> Mo utilisés sur le device (GPU discret)
    ram_avail: object = None  # () -> Mo de RAM disponible (mémoire unifiée)
    _ram_before: int = field(default=0, init=False, repr=False)

    def _flags(self) -> tuple[bool, bool]:
        """(gpu_tuning, unified_memory) de l'exécutant."""
        if self.profile is not None:
            lf = launch_flags(self.profile, None)
            return lf.gpu_tuning, lf.unified_memory
        return self.topology != TOPO_RAM, False

    @property
    def memory_mode(self) -> str:
        """Ce que `mem_mb` MESURE — une définition par type de machine :
        - "rss" (pas de GPU) : working set du process, poids mmap compris ;
        - "ram_delta" (mémoire unifiée) : RAM disponible du système AVANT le lancement
          moins APRÈS /health. La VRAM y est la RAM : ce delta compte UNE fois poids,
          KV et buffers du pilote, que ni nvidia-smi (absent sur AMD) ni le RSS
          (allocations du pilote hors process) ne voient. Suppose une machine calme ;
        - "device" (GPU discret) : mémoire utilisée du device (vram_mb() ou nvidia-smi)."""
        gpu_tuning, unified = self._flags()
        if not gpu_tuning:
            return "rss"
        return "ram_delta" if unified else "device"

    def _ram_available(self) -> int:
        if self.ram_avail is not None:
            return int(self.ram_avail())
        from loom.runtime.hardware import ram_available_mb

        return ram_available_mb()

    def _measure_mem(self, proc) -> int:
        mode = self.memory_mode
        if mode == "rss":
            import psutil

            return int(psutil.Process(proc.pid).memory_info().rss // (1024 * 1024))
        if mode == "ram_delta":
            return max(0, self._ram_before - self._ram_available())
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
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=30,
            )
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

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

    def _completion(
        self, prompt: str, n_predict: int, cache_prompt: bool = False
    ) -> dict:
        body = json.dumps(
            {
                "prompt": prompt,
                "n_predict": n_predict,
                "temperature": 0.0,
                "cache_prompt": cache_prompt,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())

    def _start(self, ctx: int):
        """Lance llama-server avec les flags EXACTS de l'exécutant et attend /health.
        Renvoie le process ; le tue et lève si le chargement échoue."""
        gpu_tuning, unified = self._flags()
        args = build_server_args(
            server_bin=self.server_bin,
            model_path=self.model_path,
            port=self.port,
            context=ctx,
            n_gpu_layers=self.ngl,
            threads=self.threads,
            mmproj_path=self.mmproj_path,
            gpu_tuning=gpu_tuning,
            unified_memory=unified,
            n_parallel=self.n_parallel,
            cpu_moe=self.cpu_moe,
            n_cpu_moe=self.n_cpu_moe,
            ubatch=self.ubatch,
            batch=self.batch,
            checkpoint_min_step=self.checkpoint_min_step,
            ctx_checkpoints=self.ctx_checkpoints,
        )
        if self.memory_mode == "ram_delta":
            # Référence prise juste avant le lancement : le delta à /health est la
            # mémoire que CE serveur a prise au système.
            self._ram_before = self._ram_available()
        proc = self.popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.health_timeout_s:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=3
                ) as r:
                    if r.status == 200:
                        return proc
            except Exception:
                time.sleep(3)
        self._kill(proc)
        raise RuntimeError(f"chargement KO à ctx={ctx} (health timeout)")

    def run(self, ctx: int, depth_tokens: int | None) -> ProbeResult:
        proc = self._start(ctx)
        try:
            res = ProbeResult(ctx=ctx, mem_mb=self._measure_mem(proc))
            if depth_tokens:
                phrase = "La pente mesurée vaut mieux que la formule du header. "
                # Serveur NEUF à chaque run (caches froids, allocation pinnée) : un
                # warmup jetable précède CHAQUE mesure, pas seulement la première.
                self._completion(phrase * 40, 16)
                # Mesurer la tokenisation réelle car sa densité varie fortement selon le modèle.
                tok_per_rep = self._tokens_of(phrase)
                # Réserver la génération et le surcoût du template dans la fenêtre.
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

    def probe_isolation(self, ctx: int = 4096) -> tuple[int, int]:
        """Sonde d'isolation du cache : (retraités au 1er passage, retraités au
        RETOUR). Séquence A -> B (pollution du slot) -> A, avec cache_prompt et
        1 slot : exactement le scénario des appels annexes de Loom (titre,
        reflect) qui écrasent le slot de la conversation.

        Un modèle couvert par le prompt-cache RAM natif recycle le préfixe de A
        au retour (retraités ~= le suffixe, quelques tokens) ; un modèle à
        mémoire hybride/SWA (exclu du cache natif) re-préfille TOUT (retraités
        ~= 1er passage). Verdict par isolation_needed() sur ces deux mesures —
        timings.prompt_n = tokens réellement RETRAITÉS (sémantique vérifiée dans
        les tests upstream de llama-server, test_slot_save.py)."""
        phrase_a = "Le cache de conversation doit survivre aux appels annexes. "
        phrase_b = "Un texte sans aucun préfixe commun vient occuper le slot. "
        prompt_a = phrase_a * 60
        proc = self._start(ctx)
        try:
            r1 = self._completion(prompt_a, 8, cache_prompt=True)
            self._completion(phrase_b * 60, 8, cache_prompt=True)
            r3 = self._completion(prompt_a + "Et maintenant ?", 8, cache_prompt=True)
            first = int((r1.get("timings") or {}).get("prompt_n") or 0)
            back = int((r3.get("timings") or {}).get("prompt_n") or 0)
            return first, back
        finally:
            self._kill(proc)
            time.sleep(4)


def isolation_needed(
    prompt_first: int, prompt_back: int, recurrent: bool | None = False
) -> bool:
    """Verdict de la sonde d'isolation : True si le cache n'a PAS survécu à la
    pollution (le retour a retraité l'essentiel du prompt -> il faut isoler les
    appels annexes dans un 2e slot). En pratique la mesure est bimodale : retour
    ~= quelques tokens (cache natif) ou ~= 100 % (hybride) — 50 % tranche net.
    Mesure illisible (1er passage vide) -> False : on n'impose pas un doublement
    de KV sans preuve.

    Mémoire récurrente -> True quoi que dise la sonde : un binaire récent y
    rattrape A -> B -> A par le cache RAM, mais ce repli casse dès que le préfixe
    bouge ou que le cache RAM est plein (Bonsai 2, 2026-09-30)."""
    if recurrent:
        return True
    if prompt_first <= 0:
        return False
    return prompt_back >= 0.5 * prompt_first


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

    rungs = []
    for ctx in (8192, 16384):
        say(f"pente : chargement à ctx={ctx}…")
        r = probe.run(ctx, None)
        rungs.append((r.ctx, r.mem_mb))
    slope_bytes, base_mb = kv_slope(rungs)
    cap = capacity_ctx(slope_bytes, base_mb, budget_mb, model_limit)

    # Ne recommander que les profondeurs dont le débit a été réellement vérifié.
    vitesses: list[dict] = []
    tg_ref: float | None = None
    valide = 0
    mecanisme = "capacité (pente mesurée)"
    ladder = [c for c in (16384, 32768, 65536, 131072, 262144) if c <= cap]
    if not ladder and cap >= _FLOOR_CTX:
        ladder = [cap]
    for ctx in ladder:
        elapsed = time.monotonic() - t0
        # `>=` garantit qu'un budget nul n'exécute aucun barreau malgré la résolution d'horloge.
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
            # Une sonde refusée termine l'échelle en conservant le dernier barreau sain.
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
    if not valide:
        # Un plancher n'est pas une mesure : le dire, pour que personne ne lise
        # « 4096 » comme un contexte validé en vitesse.
        mecanisme += (
            f" — contexte {context} = repli NON validé (aucun barreau de vitesse validé)"
        )
    return {
        "context": int(context),
        "valide": bool(valide),
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
    topology: str,
    vram_total_mb: int,
    ram_total_mb: int,
    headroom_mb: int,
    uma: bool = False,
) -> int:
    """Budget mémoire DÉTERMINISTE pour poids-GPU + KV selon la topologie :
    totaux moins marges fixes, jamais la mémoire disponible du moment (P3).
    Mémoire unifiée (`uma`) : le device EST la RAM — on la compte une fois, bornée
    par ce que le pilote annonce ET par la RAM moins la marge OS. C'est la même
    quantité que la sonde mesure alors (ServerProbe.memory_mode « ram_delta »)."""
    if topology == TOPO_RAM:
        return max(0, ram_total_mb - _OS_RAM_BUDGET_MB)
    if uma:
        device = min(vram_total_mb, ram_total_mb - _OS_RAM_BUDGET_MB)
        return max(0, device - headroom_mb)
    return max(0, vram_total_mb - headroom_mb)
