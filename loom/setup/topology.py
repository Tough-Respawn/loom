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
import os
import re
import subprocess
import sys
import tempfile
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
    # Tokens RÉELLEMENT traités par la mesure (timings du serveur) : un débit sans
    # sa quantité ne se compare pas.
    prompt_n: int | None = None
    predicted_n: int | None = None
    # Checkpoints d'état récurrent EFFECTIVEMENT créés pendant la mesure, lus dans le
    # journal du serveur (parse_checkpoints) : {effectifs, plafond, crees, taille_mb,
    # min_spacing, source} — ou {source: "non mesuré : …"} quand le journal est muet.
    checkpoints: dict | None = None


_RE_CP_CREATED = re.compile(
    r"id\s+(\d+)\s*\|.*?created context checkpoint (\d+) of (\d+) \(.*?size = "
    r"([\d.]+) MiB\)"
)
_RE_CP_ENABLED = re.compile(
    r"context checkpoints enabled, max = (\d+), min spacing = (\d+)"
)
# Événements qui changent le nombre de checkpoints d'un slot, d'après les messages de
# tools/server/server-context.cpp (llama.cpp de Loom, 2026-10) :
# - un de moins : « erased invalidated » (sans recréation), « erasing old » / « erasing
#   context checkpoint too close » / « superseding » (une création suit et refixe) ;
# - zéro : « clearing prompt with N tokens » (prompt_clear -> server_prompt::clear vide
#   la liste) et « clearing cache for lora change » ;
# - N : « restored N context checkpoint(s) from <fichier> ».
_RE_CP_SLOT = re.compile(r"id\s+(\d+)\s*\|")
_RE_CP_MOINS_UN = re.compile(
    r"erased invalidated context checkpoint|erasing (?:old )?context checkpoint"
    r"|superseding context checkpoint"
)
_RE_CP_ZERO = re.compile(r"clearing prompt with|clearing cache for lora change")
_RE_CP_RESTORED = re.compile(r"restored (\d+) context checkpoint\(s\) from")
# Rechargement d'un prompt depuis le cache RAM : le slot reprend les checkpoints de
# l'entrée, dont le journal ne donne pas le nombre à cet endroit.
_RE_CP_CACHE_LOAD = re.compile(r"found better prompt with")


def _log_tail(text: str, n: int = 4) -> str:
    """Fin du journal serveur pour un message d'erreur : les lignes d'erreur (« E »)
    d'abord, sinon les dernières lignes ; "" sans journal."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    errs = [ln for ln in lines if " E " in f" {ln} "]
    pick = (errs or lines)[-n:]
    return " — journal serveur : " + " | ".join(ln[:200] for ln in pick)


def parse_checkpoints(text: str) -> dict:
    """Checkpoints d'état récurrent pendant une mesure, d'après le journal de
    llama-server (verbosité 4), suivis ÉVÉNEMENT PAR ÉVÉNEMENT et par slot.
    « id S | … created context checkpoint N of M (…, size = X MiB) » : N est le
    nombre vivant dans le slot S après création, M le plafond PAR SLOT
    (`ctx_checkpoints`, 32 par défaut — un maximum, pas le nombre créé). Suppressions
    et remises à zéro (cf. _RE_CP_*) décrémentent ou vident le slot : deux créations
    puis « clearing prompt » laissent 0 vivant, pas 2 (revue #14).

    Renvoie `effectifs` (vivants en FIN de mesure, tous slots), `pic` (maximum
    simultané pendant la mesure — c'est lui qui a occupé la RAM), `par_slot` (fin),
    `max_par_slot`, `plafond`, `crees`, `supprimes`, `reinitialisations`, `taille_mb`,
    `source`, et `incertain` quand un rechargement depuis le cache de prompts rend un
    compte inconnu. Sans aucune ligne : non mesuré, dit tel quel. Jamais de None :
    le résultat finit dans local.toml."""
    text = text or ""
    enabled = _RE_CP_ENABLED.search(text)
    par_slot: dict[str, int] = {}
    pic = 0
    crees = supprimes = reinit = 0
    plafond = taille = None
    incertain = ""
    vu_evenement = False
    for line in text.splitlines():
        if _RE_CP_CACHE_LOAD.search(line):
            incertain = (
                "prompt rechargé depuis le cache de prompts RAM : compte du slot "
                "inconnu jusqu'à la prochaine création"
            )
            continue
        m_slot = _RE_CP_SLOT.search(line)
        if not m_slot:
            continue
        slot = str(int(m_slot.group(1)))
        m = _RE_CP_CREATED.search(line)
        if m:
            par_slot[slot] = int(m.group(2))
            plafond, taille = int(m.group(3)), round(float(m.group(4)), 1)
            crees += 1
        elif _RE_CP_MOINS_UN.search(line):
            par_slot[slot] = max(0, par_slot.get(slot, 0) - 1)
            supprimes += 1
        elif _RE_CP_ZERO.search(line):
            if slot not in par_slot and not vu_evenement:
                continue  # remise à zéro d'un slot vide avant toute création : neutre
            par_slot[slot] = 0
            reinit += 1
        else:
            m_r = _RE_CP_RESTORED.search(line)
            if not m_r:
                continue
            par_slot[slot] = int(m_r.group(1))
        vu_evenement = True
        pic = max(pic, sum(par_slot.values()))
    if vu_evenement and par_slot:
        par_slot = dict(sorted(par_slot.items(), key=lambda kv: int(kv[0])))
        if plafond is None and enabled:
            plafond = int(enabled.group(1))
        out = {
            "effectifs": sum(par_slot.values()),
            "pic": pic,
            "par_slot": par_slot,
            "max_par_slot": max(par_slot.values()),
            "crees": crees,
            "supprimes": supprimes,
            "reinitialisations": reinit,
            "source": "journal serveur (créations, suppressions et remises à zéro "
            "suivies par slot)",
        }
        if plafond is not None:
            out["plafond"] = plafond
        if taille is not None:
            out["taille_mb"] = taille
        if enabled:
            out["min_spacing"] = int(enabled.group(2))
        if incertain:
            out["incertain"] = incertain
        return out
    if enabled:
        return {
            "effectifs": 0,
            "plafond": int(enabled.group(1)),
            "crees": 0,
            "min_spacing": int(enabled.group(2)),
            "source": "journal serveur : checkpoints activés, aucun créé pendant la mesure",
        }
    if not text.strip():
        return {"source": "non mesuré : journal serveur vide"}
    return {
        "source": "non mesuré : aucune ligne de checkpoint dans le journal serveur "
        "(modèle sans état récurrent, ou journal muet)"
    }


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
    # Verbosité du journal serveur, celle de l'exécutant ([server] log_verbosity, 4 par
    # défaut) : en 3, llama-server n'écrit AUCUNE ligne de checkpoint (vérifié sur
    # Bonsai 2, 2026-10-10), et le compte effectif serait toujours « non mesuré ».
    log_verbosity: int = 4
    port: int = 8131
    health_timeout_s: int = 600
    popen: object = subprocess.Popen
    kill: object = None
    vram_mb: object = None  # () -> Mo utilisés sur le device (GPU discret)
    ram_avail: object = None  # () -> Mo de RAM disponible (mémoire unifiée)
    _ram_before: int = field(default=0, init=False, repr=False)
    # Journal du serveur sondé (stdout + stderr), fichier temporaire par lancement :
    # lu après la mesure (checkpoints effectifs), puis supprimé.
    _log_fh: object = field(default=None, init=False, repr=False)
    _log_path: str | None = field(default=None, init=False, repr=False)

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
        self,
        prompt: str,
        n_predict: int,
        cache_prompt: bool = False,
        id_slot: int | None = None,
    ) -> dict:
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "cache_prompt": cache_prompt,
        }
        if id_slot is not None:
            # Slot EXPLICITE : rejouer le routage de Loom (conversation sur 0, annexes
            # sur 1 quand il existe), pas le choix du serveur.
            payload["id_slot"] = int(id_slot)
        body = json.dumps(payload).encode()
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
        if self.log_verbosity:
            # Sans --log-file (le journal est capturé sur stdout/stderr) : -lv seul.
            args += ["-lv", str(self.log_verbosity)]
        if self.memory_mode == "ram_delta":
            # Référence prise juste avant le lancement : le delta à /health est la
            # mémoire que CE serveur a prise au système.
            self._ram_before = self._ram_available()
        # Journal du serveur dans un fichier temporaire (pas un tube : rien à vider) ;
        # stdout ET stderr, llama.cpp loggant sur l'un ou l'autre selon les builds.
        self._read_log()
        fh = tempfile.NamedTemporaryFile(
            "wb", prefix="loom-sonde-", suffix=".log", delete=False
        )
        self._log_fh, self._log_path = fh, fh.name
        try:
            proc = self.popen(args, stdout=fh, stderr=subprocess.STDOUT)
        except BaseException:
            self._read_log()  # pas de journal orphelin
            raise
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.health_timeout_s:
            code = proc.poll() if hasattr(proc, "poll") else None
            if code is not None:
                # Serveur MORT (GGUF introuvable, OOM au chargement…) : inutile
                # d'attendre le timeout de /health ; la cause est dans son journal.
                raise RuntimeError(
                    f"chargement KO à ctx={ctx} (llama-server sorti, code {code})"
                    + _log_tail(self._read_log())
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=3
                ) as r:
                    if r.status == 200:
                        return proc
            except Exception:
                time.sleep(3)
        self._kill(proc)
        raise RuntimeError(
            f"chargement KO à ctx={ctx} (health timeout)" + _log_tail(self._read_log())
        )

    def _read_log(self) -> str:
        """Ferme, lit et SUPPRIME le journal du dernier serveur lancé ; "" sans journal.
        Best-effort : un fichier illisible ou verrouillé ne fait pas échouer la mesure."""
        fh, path = self._log_fh, self._log_path
        self._log_fh, self._log_path = None, None
        if fh is None:
            return ""
        try:
            fh.close()
        except Exception:  # noqa: BLE001 - best-effort
            pass
        text = ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:  # noqa: BLE001 - best-effort
            text = ""
        # Juste après taskkill, le serveur peut tenir encore le fichier (WinError 32,
        # vu en réel) : quelques essais espacés avant d'abandonner.
        for essai in range(6):
            try:
                os.remove(path)
                break
            except FileNotFoundError:
                break
            except OSError:
                if essai < 5:
                    time.sleep(0.5)
        return text

    def run(self, ctx: int, depth_tokens: int | None) -> ProbeResult:
        proc = self._start(ctx)
        journal = ""
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
                res.prompt_n = int(t.get("prompt_n") or 0) or None
                res.predicted_n = int(t.get("predicted_n") or 0) or None
        finally:
            self._kill(proc)
            time.sleep(4)  # laisser la mémoire se libérer avant le barreau suivant
            # Après l'attente : le serveur est mort et a lâché son journal.
            journal = self._read_log()
        # Checkpoints EFFECTIVEMENT créés pendant cette mesure (le plafond n'est qu'un
        # maximum) : lus dans le journal du serveur, « non mesuré » quand il est muet.
        res.checkpoints = parse_checkpoints(journal)
        return res

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
            self._read_log()
            time.sleep(4)

    def verify_cache(self, ctx: int = 4096) -> dict:
        """Vérification du cache avec la configuration FINALE : la séquence RÉELLE de
        Loom — conversation A sur le slot 0, appel annexe B routé sur le slot final
        (1 s'il y en a deux, sinon 0), retour de A sur le slot 0 — réutilise-t-elle le
        cache ? Renvoie {first, back, annex_slot, slots, reused} ; `back` = tokens
        retraités au retour (timings.prompt_n), `reused` par cache_reused()."""
        annex_slot = 1 if self.n_parallel >= 2 else 0
        phrase_a = "La conversation garde son cache quand les annexes sont routées. "
        phrase_b = "Un titre ou une réflexion annexe vient occuper un autre slot. "
        prompt_a = phrase_a * 60
        proc = self._start(ctx)
        try:
            r1 = self._completion(prompt_a, 8, cache_prompt=True, id_slot=0)
            self._completion(phrase_b * 60, 8, cache_prompt=True, id_slot=annex_slot)
            r3 = self._completion(
                prompt_a + "Et maintenant ?", 8, cache_prompt=True, id_slot=0
            )
            first = int((r1.get("timings") or {}).get("prompt_n") or 0)
            back = int((r3.get("timings") or {}).get("prompt_n") or 0)
            return {
                "first": first,
                "back": back,
                "annex_slot": annex_slot,
                "slots": int(self.n_parallel),
                "reused": cache_reused(first, back),
            }
        finally:
            self._kill(proc)
            self._read_log()
            time.sleep(4)


def model_server_bin(mt: dict, default_bin: str) -> str:
    """Le binaire que l'EXÉCUTANT lancera pour ce modèle : `server_bin` du model.toml
    (un build qui porte une PR pas encore mergée), sinon le [server].bin global —
    même précédence que swap._model_cmd / serve.build_launch. La sonde, la détection
    matérielle et le build tracé doivent porter sur CE binaire."""
    own = str((mt or {}).get("server_bin") or "").strip()
    return own or str(default_bin)


def probe_slots(server_cfg: dict, isolation: bool | None) -> int:
    """Slots de la sonde = ceux de l'exécutant : [server] n_parallel global, monté à 2
    au minimum quand l'isolation est nécessaire (server_args.resolve_parallel)."""
    from loom.runtime.server_args import resolve_parallel

    base = int((server_cfg or {}).get("n_parallel") or 1)
    return resolve_parallel(base, bool(isolation))


def probe_batches(mt: dict, server_cfg: dict) -> tuple[int | None, int | None]:
    """(ubatch, batch) que l'EXÉCUTANT appliquera à ce modèle : model.toml, sinon le
    repli machine [server] ubatch/batch, sinon rien (défauts llama-server). La sonde
    doit démarrer avec les mêmes : vu le 2026-10-10, elle tournait en -ub 512 quand
    l'exécutant servait en 2048."""
    mt = mt or {}
    server_cfg = server_cfg or {}
    ub = mt.get("ubatch") or server_cfg.get("ubatch")
    b = mt.get("batch") or server_cfg.get("batch")
    return (int(ub) if ub else None, int(b) if b else None)


def cache_reused(prompt_first: int, prompt_back: int) -> bool | None:
    """Verdict de la vérification finale : True si le retour a retraité moins de la
    moitié du prompt (cache réutilisé), False sinon, None si la mesure est illisible.
    Même seuil bimodal que la sonde d'isolation."""
    if prompt_first <= 0:
        return None
    return prompt_back < 0.5 * prompt_first


def cache_check_text(v: dict) -> str:
    """Résumé lisible d'un résultat de verify_cache()."""
    return (
        f"retour {v.get('back')}/{v.get('first')} tokens retraités, annexe sur le slot "
        f"{v.get('annex_slot')}, {v.get('slots')} slot(s)"
    )


def isolation_text(needed: bool | None, first: int, back: int) -> str:
    """Libellé HONNÊTE du verdict d'isolation : ce que la mesure a montré, et pourquoi on
    isole quand même. Vu sur Ornith (2026-10-10) : « cache PERDU (retour 6/721) » alors
    que 6 tokens retraités = cache survécu ; les 2 slots étaient IMPOSÉS par la mémoire
    récurrente (isolation_needed, recurrent=True)."""
    if needed is None:
        return "illisible (réglage inchangé)."
    retour = f"retour {back}/{first} tokens retraités"
    if not needed:
        return f"cache survit à la pollution ({retour}) -> 1 slot suffit."
    if first > 0 and back < 0.5 * first:
        return (
            f"cache survit ici à la pollution ({retour}) mais la mémoire est "
            "récurrente : 2 slots imposés (le repli du cache RAM casse dès que le "
            "préfixe bouge)."
        )
    return f"cache PERDU après pollution du slot ({retour}) -> 2 slots pour ce modèle."


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


def _point(r, base: dict) -> dict:
    """Un point de calibration avec ce que la sonde a rapporté en plus des chiffres
    (checkpoints effectifs) — rien d'inventé quand elle ne rapporte rien."""
    cp = getattr(r, "checkpoints", None)
    if isinstance(cp, dict):
        base["checkpoints"] = cp
    return base


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
    # Chaque point garde ce que la sonde a rapporté (checkpoints effectifs du journal
    # serveur) : la pente se lit avec, l'archive le conserve (revue #14).
    rungs_detail: list[dict] = []
    for ctx in (8192, 16384):
        say(f"pente : chargement à ctx={ctx}…")
        r = probe.run(ctx, None)
        rungs.append((r.ctx, r.mem_mb))
        rungs_detail.append(_point(r, {"ctx": r.ctx, "mem_mb": r.mem_mb}))
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
            _point(
                r, {"ctx": ctx, "tg_ts": r.tg_ts, "pp_ts": r.pp_ts, "mem_mb": r.mem_mb}
            )
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
        mecanisme += f" — contexte {context} = repli NON validé (aucun barreau de vitesse validé)"
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
        "rungs_detail": rungs_detail,
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
