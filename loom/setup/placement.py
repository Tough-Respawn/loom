# loom/setup/placement.py
"""Placement MESURÉ des poids d'un modèle : où vivent les couches denses et les experts.

Jusqu'au 2026-10-09 une règle décidait (MoE + GPU -> experts en RAM, `cpu_moe = true`
posé par /add-model, `discover_topology` imposant la topologie). Mesuré ce jour-là sur
Ornith 35B-A3B Q8_0 (Radeon 860M, mémoire unifiée) : tout sur GPU = +21 % de prefill et
+17-19 % de génération par rapport aux experts sur CPU. La règle venait d'une sonde du
21/07 qui n'avait mesuré que le prefill, à une répétition, avec ±12 % de bruit.

Objectif (revue du 2026-10-10) : MAXIMISER LA GÉNÉRATION AU CONTEXTE UTILE, sous
contraintes de mémoire, de conservation du cache et, si souhaité, d'un délai maximal de
prefill. Le prefill départage les ex æquo.

Principes :
- les CANDIDATS viennent de la faisabilité mémoire (profil GGUF quand il existe),
  jamais d'une doctrine ; la configuration ACTUELLE est la ligne de base si elle tient
  d'après la même estimation ; ce qu'on écarte est tracé dans les non-explorés avec sa
  raison, jamais « moins performant » ;
- une sonde PAR CANDIDAT, avec les flags exacts de l'exécutant (ServerProbe) ;
- PRÉSÉLECTION rapide à profondeur fixe, puis les FINALISTES comparés au même contexte
  utile et à la même profondeur ; un candidat unique est quand même VALIDÉ ;
- une MARGE de changement : on ne quitte la base que si le gain la dépasse (politique,
  pas une mesure du bruit), et la décision porte son mécanisme.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field, replace

from loom.runtime.hardware import recommend_gpu_layers

#: Gain minimal de génération (en %) pour quitter le placement de base.
PLACEMENT_MARGIN_PCT = 5.0
#: À génération ÉQUIVALENTE (écart dans la marge ou la dispersion), gain de prefill
#: minimal (en %) pour que le prefill départage, même contre la base. Garder l'actuel
#: vaut pour l'incertitude, pas quand un axe est net sans perte sur l'autre (Ornith,
#: 3e /rebench du 2026-10-10 : même génération, prefill 212 contre 124 t/s).
PLACEMENT_PP_TIEBREAK_PCT = 10.0
#: Contexte et prompt de la PRÉSÉLECTION : prefill long (le levier n'existe pas à 128
#: tokens) et génération mesurée à la profondeur de ce prompt.
PLACEMENT_PROBE_CTX = 8192
PLACEMENT_PROBE_PROMPT = 4096
#: Répétitions par candidat (moyenne) : une seule mesure n'a pas d'écart-type.
PLACEMENT_REPS = 2
#: Plafond de répétitions quand le classement reste incertain (écart entre deux
#: candidats dans leur dispersion mesurée) : on affine le tandem, pas tout le monde.
PLACEMENT_MAX_REPS = 4
#: Finalistes : à moins de ce % du meilleur tg en présélection (la base y est toujours).
PLACEMENT_FINALIST_PCT = 15.0
PLACEMENT_MAX_FINALISTS = 3
#: Profondeur de la comparaison finale : moitié du contexte utile, bornée (coût : un
#: prefill de cette taille par candidat et par répétition). Choix provisoire.
PLACEMENT_FINAL_DEPTH_RATIO = 0.5
PLACEMENT_FINAL_DEPTH_MAX = 16384
#: Budget temps de la sonde de placement (s) : au-delà, on décide sur l'acquis.
PLACEMENT_TIME_BUDGET_S = 900
#: Répétitions de la validation du réglage FINAL complet au contexte calibré.
PLACEMENT_FINAL_REPS = 2
#: Marge RAM laissée à l'OS quand le « device » est la RAM (mémoire unifiée).
_OS_RAM_BUDGET_MB = 3072
_MIB = 1024 * 1024


@dataclass(frozen=True)
class Placement:
    """Un emplacement candidat des poids, traduit en flags llama-server."""

    label: str  # cpu | gpu_total | gpu_partiel | experts_cpu | experts_partiel
    ngl: int  # -ngl (999 = tout ce qui n'est pas offloadé ailleurs)
    cpu_moe: bool = False  # --cpu-moe : tous les experts en RAM
    n_cpu_moe: int | None = (
        None  # --n-cpu-moe N : experts des N premières couches en RAM
    )
    estime: bool = False  # faisabilité seulement estimée (partiel) : peut échouer
    actuel: bool = False  # configuration actuelle du model.toml (ligne de base)
    faisabilite: str = ""  # comment la faisabilité a été établie (trace)
    # Couple (ubatch, batch) de la configuration COMPLÈTE comparée (finalistes x
    # batchs) ; None = les batchs de la sonde telle que fabriquée.
    ubatch: int | None = None
    batch: int | None = None
    # Threads de la configuration COMPLÈTE : réglés sur ce placement AVANT la finale
    # quand il a du calcul CPU (remarque de méthode, 2026-10-10) ; None = ceux de la
    # sonde telle que fabriquée (machine). Pas dans la clé : un finaliste n'a qu'une
    # valeur en finale, et la clé reste celle que les écrivains du model.toml lisent.
    threads: int | None = None

    @property
    def key(self) -> str:
        """Identifiant qui porte les PARAMÈTRES : deux partiels ne se confondent pas,
        ni deux couples de batchs du même placement."""
        if self.label == "experts_partiel":
            base = f"experts_partiel_n{self.n_cpu_moe}"
        elif self.label == "gpu_partiel":
            base = f"gpu_partiel_ngl{self.ngl}"
        else:
            base = self.label
        # Les DEUX paramètres : (512, 4096) et (512, 2048) ne se confondent pas.
        return f"{base}@ub{self.ubatch}@b{self.batch}" if self.ubatch else base

    @classmethod
    def from_flags(
        cls,
        ngl: int,
        cpu_moe: bool,
        n_cpu_moe: int | None,
        n_layers: int | None = None,
        **kw,
    ) -> Placement:
        """Le placement que décrivent des flags llama-server (ceux d'une sonde, d'un
        model.toml résolu…)."""
        if n_cpu_moe is not None:
            return cls("experts_partiel", 999, n_cpu_moe=int(n_cpu_moe), **kw)
        if cpu_moe:
            return cls("experts_cpu", 999, cpu_moe=True, **kw)
        ngl = int(ngl)
        if ngl <= 0:
            return cls("cpu", 0, **kw)
        if ngl >= 999 or (n_layers and ngl > int(n_layers)):
            # Au-delà du nombre de couches (99 sur 41, 999…), llama.cpp offloade toutes
            # les couches et la sortie : c'est le total.
            return cls("gpu_total", 999, **kw)
        # Réglage EXACT conservé jusqu'à la frontière : -ngl n_layers n'est pas le total
        # pour llama.cpp — la sortie passe EN PREMIER sur le device et c'est la couche 0
        # qui reste sur CPU (« 42/43 », src/llama-model.cpp:1619-1644).
        return cls("gpu_partiel", ngl, **kw)

    def describe(self) -> str:
        if self.label == "cpu":
            txt = "CPU seul (ngl 0)"
        elif self.label == "gpu_total":
            txt = "tout sur GPU (ngl 999)"
        elif self.label == "gpu_partiel":
            txt = f"offload partiel (ngl {self.ngl}" + (
                ", estimé)" if self.estime else ", réglage exact)"
            )
        elif self.label == "experts_cpu":
            txt = "denses sur GPU, experts sur CPU (--cpu-moe)"
        else:
            txt = (
                f"denses sur GPU, experts de {self.n_cpu_moe} couches sur CPU (estimé)"
            )
        if self.ubatch:
            txt += f" — ub {self.ubatch}/b {self.batch}"
        if self.threads:
            txt += f" — {self.threads} threads"
        return txt + (" — configuration actuelle" if self.actuel else "")


@dataclass(frozen=True)
class PrefillConstraint:
    """Garde-fou EXPLICITE : « traiter `new_tokens` nouveaux tokens en moins de
    `max_seconds` ». Un candidat qui ne le tient pas est écarté — sauf si aucun ne le
    tient : la contrainte est alors insatisfiable et la génération décide seule."""

    new_tokens: int
    max_seconds: float

    def seconds(self, pp_ts: float) -> float:
        return self.new_tokens / pp_ts if pp_ts and pp_ts > 0 else math.inf


def constraints_from_config(raw: dict) -> tuple[PrefillConstraint | None, float | None]:
    """(contrainte prefill explicite, plancher relatif) depuis la table [placement] de
    config/local.toml : `prefill_new_tokens` + `prefill_max_s` (N tokens en T s) et
    `prefill_floor_ratio` (0-1, choix de confort). Absents = aucune contrainte : la
    génération décide, le prefill départage."""
    tbl = (raw or {}).get("placement") or {}
    prefill = None
    if tbl.get("prefill_new_tokens") and tbl.get("prefill_max_s"):
        prefill = PrefillConstraint(
            int(tbl["prefill_new_tokens"]), float(tbl["prefill_max_s"])
        )
    floor = tbl.get("prefill_floor_ratio")
    return prefill, (float(floor) if floor else None)


#: Architectures dont la formule de l'état récurrent est vérifiée contre un journal
#: llama.cpp (Bonsai 2, qwen35 : 149,625 Mio calculés pour 149,626 observés). Hors de
#: cette liste, la formule peut ne pas s'appliquer (KDA, attention linéaire MiniMax…).
ETAT_RECURRENT_VALIDE = frozenset({"qwen35", "qwen35moe", "qwen3next"})


def inconnues_decisives(profile, meta: dict | None) -> list[str]:
    """Ce qui rend l'estimation mémoire INCERTAINE pour le précontrôle (revue n°16 :
    « des métadonnées incomplètes doivent rester signalées comme incertaines ») — vide
    = métadonnées complètes. Seules les données DÉCISIVES comptent : une clé
    optionnelle absente (sliding_window, expert_count d'un dense) n'est pas un manque."""
    meta = meta or {}
    out: list[str] = []
    w = getattr(profile, "weights", None)
    if not w:
        out.append("catalogue des tenseurs absent (poids inconnus)")
    else:
        parties = meta.get("split_count")
        if parties and int(parties) > 1:
            out.append(
                f"GGUF en {parties} parties : catalogue partiel (première partie seule)"
            )
        n = meta.get("n_layers")
        par = w.get("par_couche") or []
        if n and len(par) != int(n):
            out.append(f"catalogue incomplet ({len(par)} couches listées sur {n})")
    prov = getattr(profile, "provenance", None) or {}
    if str(prov.get("couches_attention", "inconnu")).startswith("inconnu"):
        out.append("couches d'attention inconnues")
    if str(prov.get("kv", "inconnu")).startswith("inconnu"):
        out.append("dimensions du KV inconnues (forfait de secours)")
    if str(prov.get("couches_swa", "")).startswith("inconnu"):
        out.append("fenêtre glissante sans motif (couches SWA inconnues)")
    if meta.get("key_length_mla") or meta.get("kv_lora_rank"):
        out.append("attention MLA (cache K seul) : formule du KV inadaptée")
    if meta.get("shared_kv_layers"):
        out.append("KV partagé entre couches : formule du KV inadaptée")
    if meta.get("key_length_swa") or meta.get("value_length_swa"):
        out.append("têtes SWA dédiées : formule du KV inadaptée")
    if meta.get("head_count_kv_array"):
        out.append("head_count_kv par couche (tableau) : formule du KV inadaptée")
    if getattr(profile, "recurrent_layers", None):
        arch = getattr(profile, "architecture", None)
        if arch not in ETAT_RECURRENT_VALIDE:
            out.append(
                f"état récurrent : architecture {arch or '?'} hors liste validée"
            )
        elif not profile.recurrent_state_bytes:
            out.append("état récurrent inconnu (dimensions absentes)")
    return out


class AucunPlacementFaisable(RuntimeError):
    """ÉTAPE 2 : aucun placement (configuration actuelle et CPU seul compris) ne tient au
    contexte UTILE d'après l'estimation : aucun placement comparé, calibration non
    lancée, rien d'appliqué — l'appelant le dit. Le précontrôle (DemarrageImpossible)
    a laissé passer le démarrage au plancher : c'est le contexte DEMANDÉ qui ne tient
    pas ; llama-bench (loom-setup) et la sonde d'isolation ont pu charger le modèle."""


@dataclass
class PlacementPlan:
    candidates: list[Placement]
    non_explores: list[dict] = field(default_factory=list)  # [{key, raison}]

    @property
    def aucun_faisable(self) -> bool:
        """Résultat EXPLICITE : aucun candidat ne passe l'estimation (device et hôte)."""
        return not self.candidates

    @property
    def raison(self) -> str:
        """Pourquoi rien n'est faisable (chaque refus chiffré) ; "" sinon."""
        if self.candidates:
            return ""
        refus = [f"{n['key']} : {n['raison']}" for n in self.non_explores]
        return "aucun placement faisable d'après l'estimation mémoire" + (
            " — " + " ; ".join(refus) if refus else ""
        )


def useful_context(
    model_ctx: int | None, global_ctx: int | None, model_limit: int | None
) -> int:
    """Contexte UTILE : celui que l'exécutant servira — le `context` du modèle, sinon
    le [server] context de la machine, sinon le contexte de sonde ; borné par la
    limite déclarée du modèle, jamais sous 4 096. Les estimations de faisabilité et
    les comparaisons se font à ce contexte, pas à une constante."""
    ctx = int(model_ctx or global_ctx or PLACEMENT_PROBE_CTX)
    if model_limit:
        ctx = min(ctx, int(model_limit))
    return max(4096, ctx)


def batch_couples(current: tuple | None) -> list[tuple[int, int]]:
    """Les couples (ubatch, batch) comparés sur les finalistes : celui de l'EXÉCUTANT
    d'abord (model.toml, sinon machine, sinon défauts llama-server 512/2048 — c'est la
    base), puis l'alternative du parc (bench.UBATCH_CANDIDATES). Deux au plus : le
    2x2 est le point de départ économe validé par la revue du 2026-10-10."""
    from loom.setup.bench import UBATCH_CANDIDATES

    cur = (
        (int(current[0]), int(current[1] or 0) or None)
        if current and current[0]
        else UBATCH_CANDIDATES[0]
    )
    if cur[1] is None:
        cur = (cur[0], max(cur[0], UBATCH_CANDIDATES[0][1]))
    out = [cur] + [tuple(c) for c in UBATCH_CANDIDATES if tuple(c) != cur]
    return out[:2]


def final_depth(ctx: int) -> int:
    """Profondeur (tokens déjà en contexte) de la comparaison finale à `ctx`."""
    return max(
        256, min(PLACEMENT_FINAL_DEPTH_MAX, int(ctx * PLACEMENT_FINAL_DEPTH_RATIO))
    )


def kv_estimate_mb(profile, ctx: int, *, gpu_tuning: bool, slots: int = 1) -> int:
    """Mio de cache KV au contexte `ctx` avec le type de cache de l'EXÉCUTANT (q8_0
    sous profil GPU, f16 sinon) et `slots` slots — cf. ModelProfile.kv_bytes."""
    kv_type = "q8_0" if gpu_tuning else "f16"
    return int(profile.kv_bytes(ctx, kv_type, slots) // _MIB)


#: Checkpoints d'état récurrent par slot quand model.toml ne fixe pas `ctx_checkpoints`
#: (valeur observée du serveur, cf. config.ModelConfig.ctx_checkpoints). Provisoire :
#: documentée, remplacée par la valeur du model.toml quand elle existe.
DEFAULT_CTX_CHECKPOINTS = 32


def memory_estimate_mb(
    profile,
    ctx: int,
    *,
    gpu_tuning: bool,
    slots: int = 1,
    checkpoints: int | None = None,
) -> dict:
    """Mémoire PAR CONTEXTE que l'exécutant allouera au-delà des poids, VENTILÉE par
    emplacement : côté device, le cache KV au contexte utile (type de cache de
    l'exécutant, `slots` slots) et l'état récurrent VIVANT des hybrides (il suit le KV) ;
    côté hôte, les `checkpoints` instantanés par slot — le serveur les garde dans des
    tableaux RAM, jamais sur le device. Les imputer à la VRAM faisait perdre à un modèle
    de 16 Gio son candidat tout-GPU sur 24 Gio (revue 2026-10-10). Avant, seule la
    pente mesurée voyait les checkpoints (Bonsai 2 : 32 x 150 Mio par slot)."""
    cp = DEFAULT_CTX_CHECKPOINTS if checkpoints is None else int(checkpoints)
    kv_mb = kv_estimate_mb(profile, ctx, gpu_tuning=gpu_tuning, slots=slots)
    rec_mb = int(profile.recurrent_bytes(slots=slots, checkpoints=cp) // _MIB)
    live_mb = int(profile.recurrent_bytes(slots=slots, checkpoints=0) // _MIB)
    cp_mb = rec_mb - live_mb
    return {
        "kv_mb": kv_mb,
        "recurrent_mb": rec_mb,
        "recurrent_live_mb": live_mb,
        "checkpoints_mb": cp_mb,
        "device_mb": kv_mb + live_mb,
        "host_mb": cp_mb,
        "total_mb": kv_mb + rec_mb,
        "checkpoints": cp,
        "slots": max(1, int(slots)),
    }


def host_budget_mb(ram_total_mb: int) -> int:
    """RAM disponible pour ce que le serveur garde côté hôte (poids CPU, KV des couches
    CPU, checkpoints) : la RAM moins la marge OS. En mémoire unifiée c'est aussi le
    plafond de la SOMME device + hôte, comptée une fois."""
    return max(0, int(ram_total_mb) - _OS_RAM_BUDGET_MB)


def device_budget_mb(
    vram_total_mb: int, ram_total_mb: int, uma: bool, headroom_mb: int
) -> int:
    """Mémoire device disponible pour poids + KV. Mémoire unifiée : le device EST la
    RAM, on la compte une fois et on garde la marge OS ; discret : la VRAM seule."""
    if uma:
        return max(
            0, min(vram_total_mb, ram_total_mb - _OS_RAM_BUDGET_MB) - headroom_mb
        )
    return max(0, vram_total_mb - headroom_mb)


def placement_from_config(mt: dict, *, n_layers: int | None) -> Placement | None:
    """La configuration ACTUELLE du model.toml traduite en Placement (actuel=True),
    None si le fichier ne fixe rien d'explicite."""
    mt = mt or {}
    if mt.get("n_cpu_moe") is not None:
        return Placement(
            "experts_partiel", 999, n_cpu_moe=int(mt["n_cpu_moe"]), actuel=True
        )
    if mt.get("cpu_moe"):
        return Placement("experts_cpu", 999, cpu_moe=True, actuel=True)
    ngl = mt.get("n_gpu_layers")
    if ngl is None:
        return None
    return Placement.from_flags(int(ngl), False, None, n_layers, actuel=True)


def current_placement(
    mt: dict,
    *,
    n_layers: int | None,
    size_mb: int,
    profile,
    override_ngl: int | None = None,
    headroom: int = 1024,
) -> Placement | None:
    """La configuration ACTUELLE telle que l'EXÉCUTANT la résout (actuel=True) : réglages
    explicites du model.toml d'abord, sinon le MÊME résolveur que serve.py / swap.py
    (runtime.ngl.resolve_ngl : override machine, puis VRAM libre). Un `cpu_moe = false`
    écrit ne vaut pas 999 : sur un GPU de 8 Go le runtime tourne en 8 couches, c'est 8
    qui doit être la référence. Sans profil matériel : seuls les explicites comptent."""
    explicit = placement_from_config(mt, n_layers=n_layers)
    if explicit is not None or profile is None:
        return explicit
    from loom.config import ModelConfig
    from loom.runtime.ngl import resolve_ngl

    mt = mt or {}
    layers = int(n_layers or mt.get("n_layers") or 0)
    model = ModelConfig(
        repo=str(mt.get("repo") or ""),
        filename=str(mt.get("filename") or ""),
        n_layers=layers,
        size_mb=int(size_mb or mt.get("size_mb") or 0),
        n_gpu_layers=mt.get("n_gpu_layers"),
        cpu_moe=bool(mt.get("cpu_moe")),
        n_cpu_moe=mt.get("n_cpu_moe"),
    )
    ngl = int(resolve_ngl(model, profile, override_ngl, headroom))
    why = "configuration actuelle (résolue comme l'exécutant)"
    return Placement.from_flags(ngl, False, None, layers, actuel=True, faisabilite=why)


def _split_mb(
    pl: Placement,
    *,
    profile,
    model_size_mb: int,
    kv_mb: int,
    host_extra_mb: int,
    layers,
    mmproj_mb: int = 0,
) -> tuple[int, int, str]:
    """(device Mo, hôte Mo, source) de ce que le placement alloue de chaque côté : poids
    offloadés + KV et état vivant des couches offloadées sur le device ; poids restés
    sur CPU + KV de ces couches + `host_extra_mb` (checkpoints) + `mmproj_mb` (chargé
    en RAM : --no-mmproj-offload) sur l'hôte. Catalogue des tenseurs quand il existe
    (règle des couches de llama.cpp, cf. ModelProfile.device_layers), sinon proportion
    de la taille du fichier."""
    host_extra_mb = int(host_extra_mb or 0) + int(mmproj_mb or 0)
    frac = 1.0  # part des couches (donc du KV / état vivant) sur le device
    if pl.label == "cpu":
        frac = 0.0
    elif pl.label == "gpu_partiel" and layers:
        frac = min(1.0, pl.ngl / layers)
    flags = dict(ngl=pl.ngl, cpu_moe=pl.cpu_moe, n_cpu_moe=pl.n_cpu_moe)
    b = profile.gpu_bytes(**flags) if profile is not None else None
    if b is not None:
        # KV et état vivant vivent avec LEUR couche (src/llama-kv-cache.cpp:212-221,
        # src/llama-memory-recurrent.cpp:83-92) : prorata des couches à mémoire de
        # contexte réellement offloadées, pas ngl/n.
        porteuses = set(profile.attention_layers) | set(profile.recurrent_layers)
        dev_layers = profile.device_layers(ngl=pl.ngl)
        if porteuses:
            frac = len(dev_layers & porteuses) / len(porteuses)
        elif layers:
            frac = min(1.0, len(dev_layers) / layers)
        kv_dev = int(kv_mb * frac)
        dev_w = b // _MIB
        host_w = int(profile.host_bytes(**flags) or 0) // _MIB
        return dev_w + kv_dev, host_w + (kv_mb - kv_dev) + host_extra_mb, "catalogue"
    kv_dev = int(kv_mb * frac)
    kv_host = kv_mb - kv_dev
    if pl.cpu_moe:
        # Experts en RAM : les denses seuls sur le device, poids inconnus sans catalogue
        # (supposés tenir) ; côté hôte on majore par le fichier entier.
        return kv_dev, model_size_mb + kv_host + host_extra_mb, "proportion (majorant)"
    part = frac
    if pl.n_cpu_moe is not None and layers:
        part = 1.0 - pl.n_cpu_moe / layers  # part d'experts gardée (approximation)
    dev_w = int(model_size_mb * part)
    return dev_w + kv_dev, model_size_mb - dev_w + kv_host + host_extra_mb, "proportion"


def _fits(
    pl: Placement,
    *,
    profile,
    model_size_mb: int,
    kv_mb: int,
    budget: int,
    layers: int,
    host_extra_mb: int = 0,
    host_budget: int | None = None,
    uma: bool = False,
    mmproj_mb: int = 0,
):
    """(tient ?, trace) : le placement contre DEUX plafonds. Device (`budget`) : poids
    offloadés + KV/état vivant. Hôte (`host_budget`, RAM moins la marge OS) : poids CPU
    + KV des couches CPU + checkpoints + mmproj. Mémoire unifiée : le device reste borné
    par son plafond (heap), et la SOMME device + hôte par la RAM, comptée une seule
    fois."""
    dev, host, src = _split_mb(
        pl,
        profile=profile,
        model_size_mb=model_size_mb,
        kv_mb=kv_mb,
        host_extra_mb=host_extra_mb,
        layers=layers,
        mmproj_mb=mmproj_mb,
    )
    approx = "~" if src.startswith("proportion") else ""
    extra = f"checkpoints {host_extra_mb} Mo" + (
        f" + mmproj {int(mmproj_mb)} Mo" if mmproj_mb else ""
    )
    if pl.label == "cpu" and host_budget is not None:
        # CPU seul : tout est côté hôte (poids, KV, checkpoints), rien sur le device.
        ok = host <= host_budget
        return ok, (
            f"{src} : {approx}{host} Mo hôte (poids + KV + {extra}) pour "
            f"{host_budget} Mo de RAM"
        )
    if host_budget is None:
        return (
            dev <= budget,
            f"{src} : {approx}{dev} Mo (poids device + KV) pour {budget} Mo",
        )
    if uma:
        ok = dev <= budget and dev + host <= host_budget
        return ok, (
            f"{src} : {approx}{dev} Mo device (poids + KV) pour {budget} Mo ; "
            f"{approx}{dev + host} Mo au total (mémoire unifiée, hôte {host} Mo dont "
            f"{extra}, comptée une fois) pour {host_budget} Mo de RAM"
        )
    ok = dev <= budget and host <= host_budget
    return ok, (
        f"{src} : {approx}{dev} Mo device (poids + KV) pour {budget} Mo ; "
        f"{approx}{host} Mo hôte (poids CPU + {extra}) pour {host_budget} Mo de RAM"
    )


def _partial_experts(
    *, profile, layers, model_size_mb, kv_mb, budget, **fit_kw
) -> int | None:
    """Plus petit N tel que « experts des N premières couches sur CPU » tient (des deux
    côtés : device ET hôte)."""
    if layers <= 1:
        return None
    if profile is not None and profile.weights:
        for n in range(1, layers):
            ok, _ = _fits(
                Placement("experts_partiel", 999, n_cpu_moe=n),
                profile=profile,
                model_size_mb=model_size_mb,
                kv_mb=kv_mb,
                budget=budget,
                layers=layers,
                **fit_kw,
            )
            if ok:
                return n
        return None
    if model_size_mb <= 0:
        return None
    deficit = model_size_mb + kv_mb - budget
    n = -(-deficit * layers // model_size_mb)  # arrondi supérieur
    return int(min(layers - 1, max(1, n)))


def _partial_dense(*, profile, layers, model_size_mb, kv_mb, budget, **fit_kw) -> int:
    """Plus grand -ngl qui tient (0 = rien), des deux côtés : device ET hôte."""
    if not layers:
        return 0
    if profile is not None and profile.weights:
        # Dès -ngl n_layers : toutes les couches sauf la 0, sortie comprise
        # (src/llama-model.cpp:1619-1644) — le plus grand partiel, pas le total.
        for k in range(layers, 0, -1):
            ok, _ = _fits(
                Placement("gpu_partiel", k),
                profile=profile,
                model_size_mb=model_size_mb,
                kv_mb=kv_mb,
                budget=budget,
                layers=layers,
                **fit_kw,
            )
            if ok:
                return k
        return 0
    reco = recommend_gpu_layers(budget, model_size_mb + kv_mb, layers, 0)
    return int(reco) if 0 < reco < 999 else 0


def plan_placements(
    *,
    moe: bool,
    n_layers: int | None,
    model_size_mb: int,
    kv_mb: int,
    gpu_backend: bool,
    vram_total_mb: int,
    ram_total_mb: int,
    uma: bool,
    headroom_mb: int,
    current: Placement | None = None,
    profile=None,
    host_extra_mb: int = 0,
    mmproj_mb: int = 0,
) -> PlacementPlan:
    """Candidats faisables (le plus sûr en premier, ou la configuration ACTUELLE) et
    liste de ce qu'on choisit de NE PAS mesurer, avec sa raison.

    - sans GPU : CPU seul ;
    - MoE : experts sur CPU (s'ils tiennent en RAM), puis tout GPU si poids + KV
      tiennent, sinon DEUX partiels estimés (serré, prudent) ; CPU seul non exploré ;
    - dense : tout GPU si ça tient, sinon deux offloads partiels (serré, prudent), sinon
      CPU ; CPU seul non exploré tant qu'un candidat GPU existe.
    `kv_mb` est la mémoire par contexte côté DEVICE au contexte utile (KV + état
    récurrent vivant), `host_extra_mb` celle côté HÔTE (checkpoints), `mmproj_mb` le
    projecteur multimodal chargé en RAM par chaque démarrage, `profile` le profil
    GGUF. Chaque candidat est vérifié des deux côtés (cf. _fits) — CPU seul et
    la configuration actuelle compris. Rien ne tient : plan VIDE (`aucun_faisable`,
    `raison`), à traiter explicitement par l'appelant."""
    non: list[dict] = []
    sans_gpu = not gpu_backend or vram_total_mb <= 0
    budget = (
        0
        if sans_gpu
        else device_budget_mb(vram_total_mb, ram_total_mb, uma, headroom_mb)
    )
    layers = int(n_layers or 0)
    kw = dict(
        profile=profile,
        model_size_mb=model_size_mb,
        kv_mb=kv_mb,
        budget=budget,
        layers=layers,
        host_extra_mb=int(host_extra_mb or 0),
        host_budget=host_budget_mb(ram_total_mb),
        uma=uma and not sans_gpu,
        mmproj_mb=int(mmproj_mb or 0),
    )
    cands: list[Placement] = []

    def _check(pl: Placement):
        return _fits(pl, **kw)

    def _add_if_fits(pl: Placement, prefixe: str = "") -> bool:
        ok, why = _check(pl)
        if ok:
            cands.append(replace(pl, faisabilite=prefixe + why))
        else:
            non.append({"key": pl.key, "raison": f"non exploré : ne tient pas — {why}"})
        return ok

    if sans_gpu:
        # Sans GPU, CPU seul est le SEUL candidat — vérifié en RAM comme les autres.
        _add_if_fits(Placement("cpu", 0), prefixe="sans GPU exploitable — ")
        return _with_current(cands, current, non, _check)

    if moe:
        _add_if_fits(Placement("experts_cpu", 999, cpu_moe=True))
        if _add_if_fits(Placement("gpu_total", 999)):
            non.append(
                {
                    "key": "experts_partiel",
                    "raison": "non exploré : tout-GPU tient, un partiel ne rendrait que "
                    "de la mémoire (économie de mesure)",
                }
            )
        else:
            n = _partial_experts(**kw)
            if n is not None:
                _add_if_fits(
                    Placement("experts_partiel", 999, n_cpu_moe=n, estime=True)
                )
                n_prudent = min(layers - 1, n + max(1, math.ceil(layers * 0.15)))
                if n_prudent > n:
                    _add_if_fits(
                        Placement(
                            "experts_partiel", 999, n_cpu_moe=n_prudent, estime=True
                        )
                    )
    else:
        if not _add_if_fits(Placement("gpu_total", 999)):
            k = _partial_dense(**kw)
            if k > 0:
                _add_if_fits(Placement("gpu_partiel", k, estime=True))
                k_prudent = k - max(1, math.ceil(layers * 0.1))
                if k_prudent > 0:
                    _add_if_fits(Placement("gpu_partiel", k_prudent, estime=True))
    if not cands:
        # Aucun candidat GPU retenu (refus device ou hôte) : CPU seul, VÉRIFIÉ en RAM
        # comme les autres (il était ajouté sans contrôle — revue #14). Refusé lui
        # aussi, le plan n'est vide que si _with_current refuse aussi la configuration
        # actuelle.
        _add_if_fits(
            Placement("cpu", 0),
            prefixe="rien ne tient sur le device, CPU seul — ",
        )
    if any(c.label != "cpu" for c in cands):
        non.append(
            {
                "key": "cpu",
                "raison": "non exploré : déprioritisé, GPU disponible — CPU seul n'est pas "
                "démontré dominé, hors budget de mesure",
            }
        )
    return _with_current(cands, current, non, _check)


def _with_current(
    cands: list[Placement], current: Placement | None, non: list[dict], check=None
):
    """La configuration actuelle devient la ligne de base (candidat 0), ajoutée si elle
    n'est pas parmi les candidats générés ; elle sort des non-explorés. Elle passe la
    MÊME vérification (`check(pl) -> (tient, trace)`) : infaisable d'après
    l'estimation, elle n'est ni comparée ni base, et la trace le dit."""
    if current is None:
        return PlacementPlan(cands, non)
    idx = next((i for i, c in enumerate(cands) if c.key == current.key), None)
    if idx is None:
        ok, why = check(current) if check is not None else (True, "")
        if not ok:
            non = [n for n in non if n["key"] != current.key]
            non.append(
                {
                    "key": current.key,
                    # « Non comparée », pas « non mesurée » : la sonde d'isolation l'a
                    # déjà chargée avant ce contrôle (revue #15).
                    "raison": "configuration actuelle non comparée : ne tient pas "
                    f"d'après l'estimation — {why}",
                }
            )
            return PlacementPlan(cands, non)
        base = replace(
            current,
            actuel=True,
            faisabilite="configuration actuelle" + (f" — {why}" if why else ""),
        )
        cands = [base] + cands
    else:
        base = replace(cands[idx], actuel=True)
        cands = [base] + cands[:idx] + cands[idx + 1 :]
    non = [n for n in non if n["key"] != base.key and not base.key.startswith(n["key"])]
    return PlacementPlan(cands, non)


def placement_candidates(**kw) -> list[Placement]:
    """Compatibilité : les candidats seuls (cf. plan_placements)."""
    return plan_placements(**kw).candidates


# ── précontrôle mémoire AVANT tout chargement (revue n°16, 2026-10-10) ───────────
#
# Deux étapes : (1) avant tout chargement, la faisabilité des démarrages des sondes
# (llama-bench compris) ; (2) après l'isolation, le contrôle au contexte utile (plan
# de l'étape « placement », inchangé). « Le contexte demandé ne tient pas » n'est pas
# « impossible de démarrer » : un démarrage plus modeste est pris quand il tient ;
# des métadonnées incomplètes restent incertaines (flux inchangé).

#: Contexte plancher du précontrôle : celui de useful_context (jamais moins).
PRECONTROLE_CTX = 4096
#: Checkpoints qu'un prompt brut crée au plus dans son slot
#: (tools/server/server-context.cpp:3986-4061).
CHECKPOINTS_PAR_PROMPT = 2
#: Listes de checkpoints vivantes au pic de la sonde d'isolation (A -> B -> A', 1 slot,
#: ServerProbe.probe_isolation) : celle du slot et les copies des entrées A et B du
#: cache de prompts RAM — prompt_save COPIE les checkpoints (server-task.cpp:1782-1791),
#: l'éviction ne borne que la liste du slot (server-context.cpp:2543-2551).
ISOLATION_LISTES = 3
#: États de séquence complets (KV + état récurrent) copiés dans ce cache au pic : A et
#: B sauvegardés (server-context.cpp:308-332 et 1775-1786).
ISOLATION_ETATS = 2
#: Tokens d'un prompt de la sonde d'isolation, borne haute (721 et 781 mesurés).
ISOLATION_TOKENS = 1024
#: Cellules de KV d'un test llama-bench : -p 128 / -n 16 arrondis à 256
#: (src/llama-context.cpp:369), une séquence, cache f16 (défaut de l'outil).
LLAMA_BENCH_CELLS = 256


class DemarrageImpossible(RuntimeError):
    """Le précontrôle conclut AVANT tout chargement qu'aucun démarrage ne tient.
    `etabli` : impossibilité physique établie (allocations certaines > mémoire
    physique) ; sinon hors budget même au contexte plancher. Distincte
    d'AucunPlacementFaisable (contrôle au contexte utile, après l'isolation)."""

    def __init__(self, raison: str, *, etabli: bool, details: dict | None = None):
        super().__init__(raison)
        self.etabli = bool(etabli)
        self.details = details or {}


class PlacementNonValide(RuntimeError):
    """Aucun placement validé par la mesure, et le repli — les flags actuels — ne tient
    pas aux chargements de pente de la calibration d'après l'estimation
    (repli_calibration) : y revenir lancerait un chargement condamné. Calibration non
    lancée, rien d'appliqué."""


def raison_repli_condamne(mecanisme: str | None, repli: dict) -> str:
    """Raison de PlacementNonValide (loom-setup et /rebench disent la même chose) : les
    échecs des candidats (mécanisme de probe_placement) et pourquoi le repli ne tient
    pas."""
    meca = mecanisme or "sonde de placement illisible"
    slots = int(repli.get("slots") or 1)
    s = "s" if slots > 1 else ""
    return (
        f"aucun placement validé ({meca}) et le repli sur la configuration actuelle ne "
        f"tient pas aux chargements de pente de la calibration ({repli.get('ctx')} x "
        f"{slots} slot{s}) d'après l'estimation — {repli.get('raison')}"
    )


def lire_mmproj(path) -> dict:
    """Le projecteur multimodal que la sonde passera (--mmproj) : {mb, bloquant}.
    Absent ou en-tête rejeté : llama-server échoue au chargement, après le modèle
    principal (load_model renvoie false quand mtmd_init_from_file échoue,
    tools/server/server-context.cpp:1295-1299) — bloquant, impossibilité établie. Type
    de valeur inconnu du lecteur : taille inconnue (mb None), le précontrôle sera
    incertain. Sinon les Mo de son catalogue : une allocation hôte certaine
    (--no-mmproj-offload)."""
    from pathlib import Path

    from loom.runtime.gguf_meta import TypeGGUFInconnu, read_gguf_meta

    p = Path(path)
    echec = "la sonde passerait --mmproj et llama-server échouerait au chargement"
    if not p.is_file():
        # Portée dite : c'est la SONDE qui échouerait — loom serve, lui, télécharge le
        # mmproj manquant avant de lancer llama-server (serve.ensure_all_models).
        return {
            "mb": 0,
            "bloquant": f"mmproj absent ({p.name}) : {echec} — à télécharger d'abord "
            "(loom serve le récupère au démarrage)",
        }
    try:
        w = read_gguf_meta(p).get("weights") or {}
    except TypeGGUFInconnu:
        return {"mb": None, "bloquant": None}
    except (ValueError, OSError) as exc:
        return {"mb": 0, "bloquant": f"mmproj illisible ({p.name} : {exc}) : {echec}"}
    return {"mb": int(w.get("total", 0) or 0) // _MIB, "bloquant": None}


def placement_brut(ngl, cpu_moe, n_cpu_moe, n_layers) -> Placement:
    """Placement aux flags EXACTS d'un démarrage (-ngl, --cpu-moe, --n-cpu-moe).
    Placement.from_flags fait passer --cpu-moe avant -ngl (experts_cpu en ngl 999) ;
    ici -ngl reste celui qui sera lancé : « -ngl 0 --cpu-moe » ne met rien sur le
    device."""
    ngl = int(ngl or 0)
    if ngl <= 0:
        return Placement("cpu", 0)
    if n_cpu_moe is not None:
        return Placement("experts_partiel", ngl, n_cpu_moe=int(n_cpu_moe))
    if cpu_moe:
        return Placement("experts_cpu", ngl, cpu_moe=True)
    return Placement.from_flags(ngl, False, None, n_layers)


def _capacite(hw, ram_total_mb: int) -> tuple[dict, list[str]]:
    """Capacité PHYSIQUE connue (VRAM lue par --list-devices, un seul GPU, RAM) et ce
    qui la rend inconnue. 0 veut dire « inconnue », jamais « capacité nulle »."""
    inconnues: list[str] = []
    has_gpu = bool(getattr(hw, "has_gpu", False))
    vram = int(getattr(hw, "vram_total_mb", 0) or 0) if has_gpu else 0
    if has_gpu and vram <= 0:
        inconnues.append("VRAM totale inconnue (aucun total lu par --list-devices)")
    if has_gpu and int(getattr(hw, "gpu_count", 0) or 0) > 1:
        inconnues.append(
            f"plusieurs GPU listés ({hw.gpu_count}) : capacité non modélisée"
        )
    if int(ram_total_mb or 0) <= 0:
        inconnues.append("RAM totale inconnue")
    return {"vram_mb": vram, "ram_mb": int(ram_total_mb or 0)}, inconnues


def _residence(hw) -> dict:
    """Poids côté CPU RÉSIDENTS certains seulement quand Loom impose --load-mode none
    au serveur : profil GPU hors mémoire unifiée, c.-à-d. GPU discret détecté
    (server_args.no_mmap_args). Ailleurs, le mmap peut rester actif (src/llama-model
    .cpp:1542-1551) : des pages de fichier, pas une allocation certaine."""
    if getattr(hw, "has_gpu", False) and getattr(hw, "vram_is_discrete", False):
        return {
            "certaine": True,
            "raison": "GPU discret : --load-mode none, poids côté CPU résidents",
        }
    if getattr(hw, "has_gpu", False):
        return {
            "certaine": False,
            "raison": "mémoire unifiée présumée (classification heuristique) : mmap "
            "possible, poids côté CPU non certains",
        }
    return {
        "certaine": False,
        "raison": "CPU seul : poids mappés depuis le fichier (pagination possible, pas "
        "d'échec certain)",
    }


def precontrole(
    profile,
    meta: dict | None,
    *,
    model_size_mb: int,
    hw,
    gpu_backend: bool,
    vram_total_mb: int,
    ram_total_mb: int,
    uma: bool,
    headroom_mb: int,
    base_slots: int,
    ctx_checkpoints: int | None,
    current: Placement | None,
    mmproj_mb: int | None = 0,
) -> dict:
    """Verdict AVANT tout chargement sur les démarrages SERVEUR :

    - `impossible` (établi) : métadonnées et capacité physique connues, poids côté CPU
      résidents (GPU discret), et la somme des allocations CERTAINES du démarrage
      minimal — poids chargés, état récurrent x 1, mmproj ; KV compté 0 — dépasse
      VRAM + RAM. Chaque octet résident est d'un seul côté : vrai pour TOUT placement,
      sans pagination (fichier d'échange, repli de la mémoire épinglée) ;
    - `hors_budget` : métadonnées complètes et plan VIDE au contexte plancher (4096 par
      slot, slots de base) avec la MÊME comptabilité que l'étape 2 (`vram_total_mb`,
      `uma`, checkpoints, configuration actuelle de ce parcours) : par monotonie
      (contexte >= 4096, slots >= base), l'étape 2 refuserait de toute façon. Une
      capacité physique inconnue n'y change rien (même VRAM que l'étape 2) : elle
      interdit seulement l'« établi » ;
    - `incertain` : métadonnées incomplètes — flux inchangé ;
    - `faisable` sinon.
    """
    meta = meta or {}
    inconnues = inconnues_decisives(profile, meta)
    if mmproj_mb is None:
        inconnues.append("mmproj : taille inconnue (type GGUF inconnu du lecteur)")
    capacite, inc_cap = _capacite(hw, ram_total_mb)
    toutes = inconnues + inc_cap
    complet = not inconnues  # MÉTADONNÉES complètes
    capacite_connue = not inc_cap
    residence = _residence(hw)
    loaded = profile.loaded_bytes() if profile is not None else None
    poids = int(loaded // _MIB) if loaded else 0
    etat = int((profile.recurrent_live_bytes() if profile is not None else 0) // _MIB)
    mmproj = int(mmproj_mb or 0)
    borne = {
        "poids_mb": poids,
        "etat_mb": etat,
        "mmproj_mb": mmproj,
        "kv_mb": 0,
        "total_mb": poids + etat + mmproj,
    }
    slots = max(1, int(base_slots or 1))
    est = memory_estimate_mb(
        profile,
        PRECONTROLE_CTX,
        gpu_tuning=bool(getattr(hw, "has_gpu", False)),
        slots=slots,
        checkpoints=ctx_checkpoints,
    )
    plan0 = plan_placements(
        moe=bool(meta.get("expert_count")),
        n_layers=meta.get("n_layers"),
        model_size_mb=int(model_size_mb or 0),
        kv_mb=est["device_mb"],
        host_extra_mb=est["host_mb"],
        gpu_backend=bool(gpu_backend),
        vram_total_mb=int(vram_total_mb or 0),
        ram_total_mb=int(ram_total_mb or 0),
        uma=bool(uma),
        headroom_mb=int(headroom_mb),
        current=current,
        profile=profile,
        mmproj_mb=mmproj,  # chaque démarrage le charge en RAM (--no-mmproj-offload)
    )
    postes = {
        "poids_mb": poids or int(model_size_mb or 0),
        "kv_mb": est["kv_mb"],
        "etat_vivant_mb": est["recurrent_live_mb"],
        "checkpoints_mb": est["checkpoints_mb"],
        "checkpoints_par_slot": est["checkpoints"],
        "mmproj_mb": mmproj,
    }
    res = {
        "verdict": "faisable",
        "etabli": False,
        "complet": complet,
        "capacite_connue": capacite_connue,
        "inconnues": toutes,
        "capacite": capacite,
        "residence": residence,
        "borne": borne,
        "plan_plancher": {
            "ctx": PRECONTROLE_CTX,
            "slots": slots,
            "candidats": [c.key for c in plan0.candidates],
            "non_explores": plan0.non_explores,
            "postes": postes,
        },
        "raison": "",
    }
    physique = capacite["vram_mb"] + capacite["ram_mb"]
    s = "s" if slots > 1 else ""
    if not capacite_connue:
        non_etabli = f"capacité physique non établie : {', '.join(inc_cap)}"
    elif not residence["certaine"]:
        non_etabli = residence["raison"]
    else:
        non_etabli = (
            f"allocations certaines {borne['total_mb']} Mo ≤ mémoire physique "
            f"{physique} Mo"
        )
    if (
        complet
        and capacite_connue
        and residence["certaine"]
        and borne["total_mb"] > physique
    ):
        res.update(
            verdict="impossible",
            etabli=True,
            raison=(
                "démarrage impossible d'après l'estimation : allocations certaines au "
                f"démarrage minimal = poids chargés {poids} Mo + état récurrent {etat} Mo"
                f" + mmproj {mmproj} Mo = {borne['total_mb']} Mo > mémoire physique "
                f"(VRAM {capacite['vram_mb']} + RAM {capacite['ram_mb']} = {physique} "
                "Mo), pour tout placement, sans pagination (GPU discret : poids côté "
                "CPU résidents)"
            ),
        )
    elif complet and not plan0.candidates:
        refus = " ; ".join(f"{n['key']} : {n['raison']}" for n in plan0.non_explores)
        conseil = ""
        if postes["checkpoints_mb"] > postes["kv_mb"] + postes["etat_vivant_mb"]:
            conseil = (
                " — les checkpoints dominent : un ctx_checkpoints plus bas réduirait "
                "ce poste"
            )
        res.update(
            verdict="hors_budget",
            raison=(
                f"hors budget même au contexte plancher ({PRECONTROLE_CTX} par slot, "
                f"{slots} slot{s}) : aucun placement ne tient d'après l'estimation — "
                f"{refus} ; postes au plancher : poids {postes['poids_mb']} Mo, KV "
                f"{postes['kv_mb']} Mo, état vivant {postes['etat_vivant_mb']} Mo, "
                f"checkpoints {postes['checkpoints_mb']} Mo "
                f"({postes['checkpoints_par_slot']} par slot)"
                + (f", mmproj {mmproj} Mo" if mmproj else "")
                + f"{conseil}. Non établi physiquement ({non_etabli})"
            ),
        )
    elif not complet:
        res.update(
            verdict="incertain",
            raison=(
                f"précontrôle incertain : {', '.join(toutes)} — flux inchangé, le serveur "
                "tranchera"
            ),
        )
    return res


def texte_etape2(
    plan: PlacementPlan,
    *,
    ctx: int,
    slots: int,
    estimation: dict,
    precontrole: dict | None = None,
) -> str:
    """Raison du refus de l'ÉTAPE 2 (contrôle au contexte utile, après l'isolation) :
    c'est le contexte DEMANDÉ qui ne tient pas, pas le démarrage (revue n°16) — le
    contexte, les slots retenus, les postes chiffrés, ce que le précontrôle a établi au
    plancher (« passait » seulement s'il l'a jugé faisable) et les leviers RÉELS :
    context au-dessus du plancher, checkpoints s'il y en a, slots au-delà du plancher."""
    s = "s" if int(slots) > 1 else ""
    est = estimation or {}
    txt = (
        f"le contexte utile {int(ctx)} ({int(slots)} slot{s}) ne tient avec aucun "
        f"placement — {plan.raison} ; postes à ce contexte : KV {est.get('kv_mb', 0)} "
        f"Mo, état vivant {est.get('recurrent_live_mb', 0)} Mo, checkpoints "
        f"{est.get('checkpoints_mb', 0)} Mo"
    )
    pc = precontrole or {}
    if not pc.get("verdict"):
        return txt
    plancher = int((pc.get("plan_plancher") or {}).get("slots") or 1)
    if pc["verdict"] == "faisable":
        sp = "s" if plancher > 1 else ""
        txt += (
            f" ; le démarrage au plancher ({PRECONTROLE_CTX} par slot, {plancher} "
            f"slot{sp}) passait (précontrôle)"
        )
    elif pc["verdict"] == "incertain":
        txt += " ; précontrôle incertain : faisabilité du démarrage au plancher non établie"
    leviers = []
    if int(ctx) > PRECONTROLE_CTX:
        leviers.append("un context plus bas")
    if int(est.get("checkpoints_mb", 0) or 0) > 0:
        leviers.append("un ctx_checkpoints plus bas")
    if int(slots) > plancher:
        leviers.append("moins de slots")
    if leviers:
        txt += f" — leviers : {', '.join(leviers)}"
    return txt


def precontrole_texte(res: dict) -> str:
    """Une ligne lisible du verdict du précontrôle (console, verdict /rebench)."""
    if not res:
        return "précontrôle : non fait"
    if res.get("verdict") in ("impossible", "hors_budget", "incertain"):
        raison = str(res.get("raison") or "")
        return raison if raison.startswith("précontrôle") else f"précontrôle : {raison}"
    plan = res.get("plan_plancher") or {}
    slots = int(plan.get("slots") or 1)
    s = "s" if slots > 1 else ""
    txt = (
        f"précontrôle : faisable — au plancher ({plan.get('ctx')} par slot, {slots} "
        f"slot{s}) : {', '.join(plan.get('candidats') or [])}"
    )
    if res.get("capacite_connue") is False:
        # Le plan repose sur une capacité que Loom ne connaît pas : dit, jamais tu.
        txt += f" — capacité physique non établie ({', '.join(res.get('inconnues') or [])})"
    return txt


def _flags_bruts(flags: dict) -> dict:
    return {
        "ngl": int(flags.get("ngl") or 0),
        "cpu_moe": bool(flags.get("cpu_moe")),
        "n_cpu_moe": flags.get("n_cpu_moe"),
    }


def _flags_tiennent(
    profile,
    meta: dict,
    *,
    flags: dict,
    kv_mb: int,
    host_extra_mb: int,
    model_size_mb: int,
    gpu_backend: bool,
    vram_total_mb: int,
    ram_total_mb: int,
    uma: bool,
    headroom_mb: int,
    mmproj_mb: int = 0,
) -> tuple[bool, str]:
    """(tient ?, trace) : un démarrage aux flags BRUTS (placement_brut), contre les deux
    plafonds de _fits — `kv_mb` côté device, `host_extra_mb` et `mmproj_mb` côté hôte."""
    n = meta.get("n_layers")
    sans_gpu = not gpu_backend or int(vram_total_mb or 0) <= 0
    budget = (
        0
        if sans_gpu
        else device_budget_mb(int(vram_total_mb), int(ram_total_mb), uma, headroom_mb)
    )
    # Sans device dans le build, llama.cpp n'offloade rien, quel que soit -ngl.
    brut = placement_brut(
        0 if sans_gpu else flags["ngl"], flags["cpu_moe"], flags["n_cpu_moe"], n
    )
    return _fits(
        brut,
        profile=profile,
        model_size_mb=int(model_size_mb or 0),
        kv_mb=int(kv_mb),
        budget=budget,
        layers=int(n or 0),
        host_extra_mb=int(host_extra_mb or 0),
        host_budget=host_budget_mb(ram_total_mb),
        uma=bool(uma) and not sans_gpu,
        mmproj_mb=int(mmproj_mb or 0),
    )


def repli_calibration(
    profile,
    meta: dict | None,
    *,
    flags: dict,
    complet: bool,
    slots: int,
    ctx: int,
    model_size_mb: int,
    gpu_backend: bool,
    vram_total_mb: int,
    ram_total_mb: int,
    uma: bool,
    headroom_mb: int,
    gpu_tuning: bool,
    mmproj_mb: int = 0,
) -> dict:
    """Le REPLI de la calibration — les flags actuels, quand aucun placement n'est
    validé — tient-il à ses barreaux de PENTE (topology.calibrate les charge tous, sans
    protection) ? Jugé au plus grand, `ctx` x `slots` retenus — l'estimation croît avec
    le contexte, il couvre les autres —, sans checkpoint : ces chargements sont NUS
    (aucun prompt, donc ni checkpoint ni copie du cache de prompts) ; mmproj en RAM.
    {tient: bool | None, ctx, slots, raison} ; None : métadonnées incomplètes, rien
    n'est conclu (le serveur tranchera). Ni le plancher de la sonde d'isolation (4096 x
    1), ni le contexte utile : la calibration peut trouver plus petit que lui, et ses
    barreaux de vitesse tolèrent un échec."""
    meta = meta or {}
    slots = max(1, int(slots or 1))
    base = {"tient": None, "ctx": int(ctx), "slots": slots}
    if not complet:
        return {**base, "raison": "métadonnées incomplètes : rien n'est conclu"}
    est = memory_estimate_mb(
        profile,
        int(ctx),
        gpu_tuning=gpu_tuning,
        slots=slots,
        checkpoints=0,
    )
    ok, why = _flags_tiennent(
        profile,
        meta,
        flags=_flags_bruts(flags),
        kv_mb=est["device_mb"],
        host_extra_mb=est["host_mb"],
        model_size_mb=model_size_mb,
        gpu_backend=gpu_backend,
        vram_total_mb=vram_total_mb,
        ram_total_mb=ram_total_mb,
        uma=uma,
        headroom_mb=headroom_mb,
        mmproj_mb=mmproj_mb,
    )
    return {**base, "tient": bool(ok), "raison": why}


def demarrage_isolation(
    profile,
    meta: dict | None,
    *,
    flags: dict,
    complet: bool,
    model_size_mb: int,
    gpu_backend: bool,
    vram_total_mb: int,
    ram_total_mb: int,
    uma: bool,
    headroom_mb: int,
    gpu_tuning: bool,
    ctx_checkpoints: int | None = None,
    mmproj_mb: int = 0,
) -> dict:
    """Démarrage de la sonde d'isolation : TOUJOURS 1 slot, 4096 tokens (à 2 slots,
    l'appel B part sur le slot libre et la pollution n'a jamais lieu). Les flags
    prévus (bruts) s'ils tiennent — comptabilité de la sonde au pic, côté hôte :
    ISOLATION_LISTES listes de min(CHECKPOINTS_PAR_PROMPT, `ctx_checkpoints`)
    checkpoints, ISOLATION_ETATS états de séquence copiés dans le cache de prompts RAM
    et le mmproj —, sinon, données complètes, le premier candidat du plan à 4096 x 1 ; mémoire
    récurrente : verdict imposé, sonde non lancée. Données incomplètes : le démarrage
    prévu, inchangé (le serveur tranchera). `prevu_tient` : le démarrage prévu tient-il
    d'après l'estimation (None : inconnu) — information de trace ; la garde du repli
    de la calibration, elle, juge son premier chargement (repli_calibration)."""
    meta = meta or {}
    flags = _flags_bruts(flags)
    base = {
        "flags": flags,
        "slots": 1,
        "ctx": PRECONTROLE_CTX,
        "lancer": True,
        "modeste": False,
        "prevu_tient": None,
    }
    if not complet:
        return {
            **base,
            "raison": "métadonnées incomplètes : démarrage prévu, 1 slot (le serveur "
            "tranchera)",
        }
    base["prevu_tient"] = False
    n = meta.get("n_layers")
    cp_slot = CHECKPOINTS_PAR_PROMPT
    if ctx_checkpoints is not None:
        cp_slot = min(cp_slot, max(0, int(ctx_checkpoints)))
    est = memory_estimate_mb(
        profile,
        PRECONTROLE_CTX,
        gpu_tuning=gpu_tuning,
        slots=1,
        checkpoints=ISOLATION_LISTES * cp_slot,
    )
    # Le cache de prompts RAM (actif par défaut, --cache-ram 8192) garde A et B en
    # états COMPLETS au pic : KV de leurs tokens + état récurrent, côté hôte.
    kv_type = "q8_0" if gpu_tuning else "f16"
    etats = profile.recurrent_live_bytes() + profile.kv_bytes(
        ISOLATION_TOKENS, kv_type, 1
    )
    hote_mb = est["host_mb"] + int(ISOLATION_ETATS * etats // _MIB)
    ok, why = _flags_tiennent(
        profile,
        meta,
        flags=flags,
        kv_mb=est["device_mb"],
        host_extra_mb=hote_mb,
        model_size_mb=model_size_mb,
        gpu_backend=gpu_backend,
        vram_total_mb=vram_total_mb,
        ram_total_mb=ram_total_mb,
        uma=uma,
        headroom_mb=headroom_mb,
        mmproj_mb=mmproj_mb,
    )
    if ok:
        return {
            **base,
            "prevu_tient": True,
            "raison": f"démarrage prévu à {PRECONTROLE_CTX} x 1 slot : {why}",
        }
    if meta.get("recurrent"):
        return {
            **base,
            "lancer": False,
            "raison": "mémoire récurrente : verdict d'isolation imposé — le démarrage "
            f"prévu ne tient pas à {PRECONTROLE_CTX} x 1 ({why}), sonde non lancée",
        }
    plan1 = plan_placements(
        moe=bool(meta.get("expert_count")),
        n_layers=n,
        model_size_mb=int(model_size_mb or 0),
        kv_mb=est["device_mb"],
        host_extra_mb=hote_mb,
        gpu_backend=bool(gpu_backend),
        vram_total_mb=int(vram_total_mb or 0),
        ram_total_mb=int(ram_total_mb or 0),
        uma=bool(uma),
        headroom_mb=int(headroom_mb),
        current=None,
        profile=profile,
        mmproj_mb=mmproj_mb,
    )
    if not plan1.candidates:
        return {
            **base,
            "lancer": False,
            "raison": f"le démarrage prévu ne tient pas ({why}) et aucun démarrage plus "
            f"modeste ne tient à {PRECONTROLE_CTX} x 1 : sonde non lancée",
        }
    pl = plan1.candidates[0]
    return {
        **base,
        "flags": {
            "ngl": int(pl.ngl),
            "cpu_moe": bool(pl.cpu_moe),
            "n_cpu_moe": pl.n_cpu_moe,
        },
        "modeste": True,
        "placement": pl.key,
        "raison": f"le démarrage prévu ne tient pas à {PRECONTROLE_CTX} x 1 ({why}) : "
        f"sonde sur {pl.describe()}",
    }


def filtre_llama_bench(
    profile,
    *,
    ngl: list[int],
    ncmoe: int,
    complet: bool,
    hw,
    vram_total_mb: int,
    ram_total_mb: int,
    uma: bool,
    headroom_mb: int,
    meme_binaire: bool = True,
) -> dict:
    """-ngl de llama-bench dont la borne DEVICE (poids offloadés selon la règle de
    llama.cpp, KV f16 de LLAMA_BENCH_CELLS cellules et état récurrent d'une séquence
    sur les couches offloadées) dépasse le budget device — le MÊME que le précontrôle
    et l'étape 2 (device_budget_mb sur la VRAM de l'appelant, repli nvidia-smi compris,
    marge comprise) : retirés (un seul -ngl qui échoue fait échouer toute
    l'invocation). Liste vide : repli sur -ngl 0. Données incomplètes, aucune VRAM
    connue, plusieurs GPU ou llama-bench d'un autre binaire : inchangée."""
    liste = [int(g) for g in ngl]
    res = {"ngl": liste, "ncmoe": int(ncmoe or 0), "retires": [], "note": ""}
    vram = int(vram_total_mb or 0) or int(getattr(hw, "vram_total_mb", 0) or 0)
    if not complet:
        res["note"] = "métadonnées incomplètes : liste inchangée"
        return res
    if not meme_binaire:
        res["note"] = (
            "llama-bench vient d'un autre binaire que la sonde : liste inchangée"
        )
        return res
    if (
        not getattr(hw, "has_gpu", False)
        or vram <= 0
        or int(getattr(hw, "gpu_count", 0) or 0) > 1
    ):
        res["note"] = "capacité VRAM inconnue ou non modélisée : liste inchangée"
        return res
    budget = device_budget_mb(vram, int(ram_total_mb or 0), uma, int(headroom_mb))
    garde = []
    for g in liste:
        couches = profile.device_layers(ngl=g)
        poids = profile.gpu_bytes(ngl=g, n_cpu_moe=int(ncmoe) if ncmoe else None) or 0
        kv = profile.kv_bytes(LLAMA_BENCH_CELLS, "f16", 1, layers=couches)
        etat = profile.recurrent_live_bytes(layers=couches)
        dev_mb = int((poids + kv + etat) // _MIB)
        if dev_mb > budget:
            res["retires"].append(
                {
                    "ngl": g,
                    "raison": f"dépasse le budget device : {dev_mb} Mo sur le device > "
                    f"{budget} Mo (VRAM {vram} Mo, marge {int(headroom_mb)} Mo)",
                }
            )
        else:
            garde.append(g)
    if garde:
        res["ngl"] = garde
        if garde == [0] and any(g != 0 for g in liste):
            res["note"] = (
                "tous les -ngl > 0 dépassent le budget device : -ngl 0 seul (CPU)"
            )
    else:
        res.update(
            ngl=[0],
            ncmoe=0,
            note="tous les -ngl candidats dépassent le budget device : repli sur -ngl "
            "0 (CPU)",
        )
    return res


def _agreger(echantillons: list[dict]) -> dict:
    """Mesure d'un candidat depuis ses échantillons : moyennes, mémoire max, nombre,
    dispersion de la génération (étendue relative à la moyenne, en %) et les
    échantillons eux-mêmes — un débit sans son bruit ni sa quantité ne se compare pas."""
    tgs = [e["tg_ts"] for e in echantillons if e.get("tg_ts")]
    pps = [e["pp_ts"] for e in echantillons if e.get("pp_ts")]
    if not tgs:
        return {"echec": "débit illisible"}
    mean = sum(tgs) / len(tgs)
    out = {
        "tg_ts": round(mean, 2),
        "pp_ts": round(sum(pps) / len(pps), 2) if pps else 0.0,
        "mem_mb": max(int(e.get("mem_mb") or 0) for e in echantillons),
        "n": len(tgs),
        "tg_disp_pct": round((max(tgs) - min(tgs)) / mean * 100, 1)
        if mean > 0
        else 0.0,
        "echantillons": list(echantillons),
    }
    # Checkpoints EFFECTIFS (journal serveur, cf. topology.parse_checkpoints) : le
    # maximum des échantillons, ou la raison pour laquelle rien n'a été mesuré. Rien
    # quand la sonde ne rapporte pas de checkpoints du tout.
    cps = [
        e["checkpoints"] for e in echantillons if isinstance(e.get("checkpoints"), dict)
    ]
    if cps:
        vus = [c for c in cps if c.get("effectifs") is not None]
        if vus:
            top = max(vus, key=lambda c: int(c["effectifs"]))
            out["checkpoints_effectifs"] = int(top["effectifs"])  # vivants en fin
            # Pic simultané (la RAM réellement occupée), le plus haut des échantillons.
            out["checkpoints_pic"] = max(int(c.get("pic", c["effectifs"])) for c in vus)
            if top.get("max_par_slot") is not None:
                out["checkpoints_max_par_slot"] = int(top["max_par_slot"])
            if top.get("plafond") is not None:
                out["checkpoints_plafond"] = int(top["plafond"])  # par slot
            taille = next(
                (c["taille_mb"] for c in vus if c.get("taille_mb") is not None), None
            )
            if taille is not None:
                out["checkpoint_mb"] = float(taille)
        dernier = vus[-1] if vus else cps[-1]
        detail = str(dernier.get("source", ""))
        incertains = [c["incertain"] for c in cps if c.get("incertain")]
        if incertains:
            detail += f" ; incertain : {incertains[-1]}"
        out["checkpoints_detail"] = detail
    return out


def checkpoints_text(mesure: dict) -> str:
    """Suffixe lisible d'une mesure agrégée : « ; checkpoints effectifs 4 (2 par slot au
    plus, plafond 32 par slot, 149.6 Mio chacun, ~598 Mio) », avec « pic N » quand des
    checkpoints ont été supprimés en cours de mesure (la mémoire se chiffre au pic),
    « ; checkpoints : non mesuré : … », ou "" quand rien n'a été rapporté."""
    n = (mesure or {}).get("checkpoints_effectifs")
    if n is not None:
        pic = int(mesure.get("checkpoints_pic") or n)
        parts = []
        if pic > int(n):
            parts.append(f"pic {pic}")
        if mesure.get("checkpoints_max_par_slot") is not None:
            parts.append(f"{mesure['checkpoints_max_par_slot']} par slot au plus")
        if mesure.get("checkpoints_plafond"):
            parts.append(f"plafond {mesure['checkpoints_plafond']} par slot")
        txt = f" ; checkpoints effectifs {n}"
        if mesure.get("checkpoint_mb"):
            taille = float(mesure["checkpoint_mb"])
            au_pic = " au pic" if pic > int(n) else ""
            parts.append(f"{taille} Mio chacun, ~{round(taille * pic)} Mio{au_pic}")
        detail = str(mesure.get("checkpoints_detail") or "")
        if "incertain" in detail:
            parts.append(detail[detail.index("incertain") :])
        return txt + (f" ({', '.join(parts)})" if parts else "")
    detail = (mesure or {}).get("checkpoints_detail")
    return f" ; checkpoints : {detail}" if detail else ""


def _incertains(mesures: dict[str, dict]) -> list[str]:
    """Clés dont le classement contre le meilleur reste INCERTAIN : écart de moyennes
    inférieur ou égal à la plus grande des deux dispersions. Vide = classement net."""
    valid = {k: v for k, v in mesures.items() if "tg_ts" in v}
    if len(valid) < 2:
        return []
    ordre = sorted(valid, key=lambda k: valid[k]["tg_ts"], reverse=True)
    best = ordre[0]
    tg_b = valid[best]["tg_ts"]
    d_b = float(valid[best].get("tg_disp_pct") or 0)
    flous = []
    for k in ordre[1:]:
        tg_k = valid[k]["tg_ts"]
        ecart = (tg_b / tg_k - 1) * 100 if tg_k else math.inf
        if ecart <= max(d_b, float(valid[k].get("tg_disp_pct") or 0)):
            flous.append(k)
    return [best] + flous if flous else []


def _configure(sonde, c, *, threads: int | None = None):
    """Configuration COMPLÈTE : le couple (ubatch, batch) et les threads du candidat
    remplacent ceux de la sonde fabriquée — sur une COPIE, une sonde par configuration.
    Rien de posé quand le candidat ne fixe rien (la sonde reste celle de la machine)."""
    th = threads if threads is not None else getattr(c, "threads", None)
    ub = getattr(c, "ubatch", None)
    if ub is None and th is None:
        return sonde
    sonde = copy.copy(sonde)
    if ub is not None:
        sonde.ubatch = ub
        sonde.batch = getattr(c, "batch", None)
    if th is not None:
        sonde.threads = int(th)
    return sonde


def _mesurer(
    make_probe,
    cands,
    ctx,
    depth,
    reps,
    say,
    *,
    deadline: float | None = None,
    max_reps: int | None = None,
) -> dict[str, dict]:
    """Une sonde par candidat, `reps` TOURS à (ctx, depth) : les candidats ALTERNENT
    à chaque tour (A B A B, pas A A B B) pour que dérive thermique et caches ne
    favorisent pas le dernier mesuré. Puis, tant que le classement reste incertain
    (`_incertains`) et que le plafond `max_reps` et le `deadline` le permettent, un
    tour de plus pour les seuls candidats indécis. Un candidat qui casse est écarté
    et nommé (jamais fatal)."""
    plafond = max(reps, int(max_reps or PLACEMENT_MAX_REPS))
    sondes: dict[str, object] = {}
    samples: dict[str, list[dict]] = {}
    out: dict[str, dict] = {}
    for c in cands:
        try:
            sondes[c.key] = _configure(make_probe(c), c)
            samples[c.key] = []
        except Exception as exc:  # noqa: BLE001 - un candidat qui casse n'est PAS fatal
            out[c.key] = {"echec": f"{type(exc).__name__}: {exc}"}
    by_key = {c.key: c for c in cands}

    def _tour(keys, numero):
        for k in keys:
            c = by_key[k]
            say(f"placement {k} ({c.describe()}) : mesure {numero} à ctx {ctx}…")
            try:
                r = sondes[k].run(ctx, depth)
            except Exception as exc:  # noqa: BLE001 - écarté et nommé, jamais fatal
                out[k] = {"echec": f"{type(exc).__name__}: {exc}"}
                sondes.pop(k, None)
                continue
            # Pas de None dans un échantillon : il finit dans local.toml (sans null).
            ech = {
                "tg_ts": float(r.tg_ts or 0) or None,
                "pp_ts": float(r.pp_ts or 0) or None,
                "mem_mb": int(r.mem_mb or 0),
                "prompt_n": getattr(r, "prompt_n", None),
                "predicted_n": getattr(r, "predicted_n", None),
                "checkpoints": getattr(r, "checkpoints", None),
            }
            samples[k].append({kk: vv for kk, vv in ech.items() if vv is not None})

    tours = 0
    for _ in range(reps):
        tours += 1
        _tour([k for k in sondes], tours)
    for k in list(sondes):
        out[k] = _agreger(samples[k])
    # Affinage : un tour de plus pour le tandem incertain, dans le plafond et le budget.
    while tours < plafond:
        if deadline is not None and time.monotonic() >= deadline:
            break
        flous = [k for k in _incertains(out) if k in sondes]
        if len(flous) < 2:
            break
        tours += 1
        _tour(flous, tours)
        for k in flous:
            if k in sondes:
                out[k] = _agreger(samples[k])
    return out


def _non_explores_txt(non: list[dict]) -> str:
    return "".join(f" ; {n['key']} : {n['raison']}" for n in non)


def probe_placement(
    make_probe,
    candidates: list[Placement],
    *,
    ctx: int = PLACEMENT_PROBE_CTX,
    depth: int = PLACEMENT_PROBE_PROMPT,
    reps: int = PLACEMENT_REPS,
    margin_pct: float = PLACEMENT_MARGIN_PCT,
    progress=None,
    useful_ctx: int | None = None,
    non_explores: list[dict] | None = None,
    prefill: PrefillConstraint | None = None,
    pp_floor_ratio: float | None = None,
    time_budget_s: float = PLACEMENT_TIME_BUDGET_S,
    batch_couples: list[tuple[int, int]] | None = None,
    thread_options: list[ThreadsOption] | None = None,
) -> dict | None:
    """Sonde les candidats avec le VRAI serveur (`make_probe(placement)` renvoie une
    sonde exposant `.run(ctx, depth) -> ProbeResult`, cf. topology.ServerProbe).

    - un seul candidat : VALIDÉ une fois au contexte utile (charge, génère), non comparé
      — avec `batch_couples`, ses couples de batchs, eux, sont comparés ;
    - plusieurs : PRÉSÉLECTION à (ctx, depth) `reps` fois (couple de batchs actuel),
      puis les FINALISTES (la base + les meilleurs à PLACEMENT_FINALIST_PCT du meilleur,
      PLACEMENT_MAX_FINALISTS au plus) remesurés au contexte utile et à
      final_depth(utile) — x chaque couple de `batch_couples` : des CONFIGURATIONS
      COMPLÈTES au même contexte, à la même profondeur, aux mêmes slots, en tours
      alternés ; la décision (`pick_placement`) porte sur cette dernière mesure, la
      base étant la configuration actuelle EXACTE (placement + couple de l'exécutant).
      Avec `thread_options`, chaque finaliste à calcul CPU voit d'abord SES threads
      réglés (probe_threads, au contexte et à la profondeur de la finale, couple
      actuel) : la finale compare des configurations chacune à son réglage, et non un
      experts-CPU aux threads de la machine contre un tout-GPU qui s'en moque.
      Rien de mesurable -> placement None, l'échec de chaque candidat nommé dans le
      mécanisme (on n'écrit jamais une valeur inventée) ; aucun candidat -> None.

    Renvoie {placement (None si validation en échec ; porte ubatch/batch quand des
    couples ont été comparés, threads quand ils ont été réglés), baseline, tg_ts,
    pp_ts, gain_pct, mesures (phase décisive, par clé), preselection, finalistes,
    threads_finalistes (par finaliste : verdict de probe_threads, ou non_explore),
    ctx_final, depth_final, couples, non_explores, compare, mecanisme}."""
    say = progress or (lambda _m: None)
    non = list(non_explores or [])
    couples = [(int(c[0]), int(c[1])) for c in (batch_couples or [])]
    if not candidates:
        return None
    t0 = time.monotonic()
    deadline = t0 + float(time_budget_s)
    deeper = bool(useful_ctx and int(useful_ctx) > ctx)
    ctx_final = int(useful_ctx) if deeper else ctx
    depth_final = final_depth(ctx_final) if deeper else depth
    # Avec des couples, les finalistes sont TOUJOURS remesurés x couples, même à contexte
    # utile court : quatre configurations complètes dans les mêmes conditions.
    two_phase = deeper or bool(couples)
    th_fin: dict[str, dict] = {}

    def _configs(placements):
        """Configurations complètes, couple-major (A@c1, B@c1, A@c2, B@c2) : la base
        (candidat 0 avec le couple ACTUEL) reste la première."""
        if not couples:
            return list(placements)
        # « configuration actuelle » ne vaut que pour le couple ACTUEL (couples[0]).
        return [
            replace(c, ubatch=ub, batch=b, actuel=c.actuel and (ub, b) == couples[0])
            for ub, b in couples
            for c in placements
        ]

    def _tune_threads(pl: Placement) -> Placement:
        """Threads de CE finaliste, réglés avant la finale (couple actuel, contexte et
        profondeur de la finale). Tout GPU : non exploré, dit. Illisible : conservés."""
        if not thread_options:
            return pl
        if not needs_cpu_compute(pl):
            th_fin[pl.key] = {
                "non_explore": f"non exploré : {pl.key} sans calcul CPU attendu",
                "placement": pl.key,
            }
            return pl
        if time.monotonic() >= deadline:
            th_fin[pl.key] = {
                "non_explore": f"non exploré : budget temps épuisé avant {pl.key}",
                "placement": pl.key,
            }
            return pl
        base_cfg = _configs([pl])[0]  # avec le couple ACTUEL quand il y en a
        say(f"threads de {pl.key} avant la finale (à ctx {ctx_final})…")
        th = probe_threads(
            lambda o: _configure(make_probe(base_cfg), base_cfg, threads=o.threads),
            thread_options,
            ctx=ctx_final,
            depth=depth_final,
            reps=reps,
            margin_pct=margin_pct,
            progress=say,
            prefill=prefill,
            pp_floor_ratio=pp_floor_ratio,
        )
        if not th:
            th_fin[pl.key] = {
                "non_explore": f"sonde de threads illisible sur {pl.key} : threads "
                "actuels conservés",
                "placement": pl.key,
            }
            return pl
        th["placement"] = pl.key
        th_fin[pl.key] = th
        return replace(pl, threads=int(th["threads"]))

    def _res(**kw):
        base = {
            "baseline": _configs([candidates[0]])[0].key,
            "non_explores": non,
            "ctx_final": ctx_final,
            "depth_final": depth_final,
            "couples": couples,
            "threads_finalistes": th_fin,
        }
        base.update(kw)
        return base

    if len(candidates) == 1:
        seul = candidates[0]
        configs = _configs([seul])
        if len(configs) == 1:
            mes = _mesurer(
                make_probe, configs, ctx_final, depth_final, 1, say, max_reps=1
            )
        else:
            # Rien à comparer entre placements, mais le couple de batchs, lui, se mesure.
            mes = _mesurer(
                make_probe,
                configs,
                ctx_final,
                depth_final,
                reps,
                say,
                deadline=deadline,
            )
        if not any("tg_ts" in v for v in mes.values()):
            m = next(iter(mes.values()))
            return _res(
                placement=None,
                tg_ts=None,
                pp_ts=None,
                gain_pct=None,
                mesures=mes,
                preselection=mes,
                finalistes=[],
                compare=False,
                mecanisme=(
                    f"{seul.key} : seul candidat faisable, validation en ÉCHEC "
                    f"({m.get('echec', 'débit illisible')})" + _non_explores_txt(non)
                ),
            )
        if len(configs) == 1:
            m = mes[seul.key]
            return _res(
                placement=seul,
                tg_ts=m["tg_ts"],
                pp_ts=m["pp_ts"],
                gain_pct=None,
                mesures=mes,
                preselection=mes,
                finalistes=[seul.key],
                compare=False,
                mecanisme=(
                    f"{seul.key} : seul candidat faisable — validé à ctx {ctx_final} "
                    f"(génération {m['tg_ts']} t/s, prefill {m['pp_ts']} t/s), non comparé"
                    + _non_explores_txt(non)
                ),
            )
        best, mecanisme = pick_placement(
            mes, configs, margin_pct, prefill=prefill, pp_floor_ratio=pp_floor_ratio
        )
        m = mes.get(best.key) or {}
        return _res(
            placement=best,
            tg_ts=m.get("tg_ts"),
            pp_ts=m.get("pp_ts"),
            gain_pct=None,
            mesures=mes,
            preselection=mes,
            finalistes=[c.key for c in configs],
            compare=sum(1 for v in mes.values() if "tg_ts" in v) >= 2,
            mecanisme=(
                f"{seul.label} : seul placement faisable, seuls les batchs comparés — "
                + mecanisme
                + _non_explores_txt(non)
            ),
        )

    # Présélection avec le couple ACTUEL (couples[0]) : un run par placement et par tour.
    presel = (
        [replace(c, ubatch=couples[0][0], batch=couples[0][1]) for c in candidates]
        if couples
        else list(candidates)
    )
    orig = {p.key: c for p, c in zip(presel, candidates)}
    phase1 = _mesurer(make_probe, presel, ctx, depth, reps, say, deadline=deadline)
    if not any("tg_ts" in v for v in phase1.values()):
        # Rien de mesurable : aucun placement élu (jamais de valeur inventée), mais
        # l'échec de CHAQUE candidat est nommé — la sortie « repli condamné » le cite.
        echecs = " ; ".join(
            f"{k} : {v.get('echec', 'débit illisible')}" for k, v in phase1.items()
        )
        return _res(
            placement=None,
            tg_ts=None,
            pp_ts=None,
            gain_pct=None,
            mesures=phase1,
            preselection=phase1,
            finalistes=[],
            compare=False,
            mecanisme=f"toutes les mesures en échec — {echecs}"
            + _non_explores_txt(non),
        )
    notes = ""
    configs = presel
    if two_phase:
        valid = {k: v for k, v in phase1.items() if "tg_ts" in v}
        best_tg = max(v["tg_ts"] for v in valid.values())
        base_key = presel[0].key
        seuil = best_tg * (1 - PLACEMENT_FINALIST_PCT / 100)
        autres = sorted(
            (c for c in presel if c.key in valid and c.key != base_key),
            key=lambda c: valid[c.key]["tg_ts"],
            reverse=True,
        )
        finalists = [c for c in presel if c.key == base_key and c.key in valid]
        finalists += [c for c in autres if valid[c.key]["tg_ts"] >= seuil][
            : PLACEMENT_MAX_FINALISTS - len(finalists)
        ]
        if len(finalists) < 2 and autres:
            # Au moins DEUX placements en finale : le meilleur alternatif y va même s'il
            # est loin derrière en présélection — avec des couples de batchs, le
            # classement peut bouger (Ornith : experts-CPU n'avait jamais été mesuré
            # en ub 512), et la trace doit le dire par une mesure, pas par une coupe.
            finalists.append(autres[0])
        # Threads de chaque finaliste à calcul CPU, réglés AVANT la finale.
        regles = [_tune_threads(orig[c.key]) for c in finalists]
        if th_fin:
            notes += " ; threads réglés avant la finale : " + ", ".join(
                f"{k} -> t{v['threads']}"
                if "threads" in v
                else f"{k} : {v['non_explore']}"
                for k, v in th_fin.items()
            )
        # Configurations complètes : finalistes x couples (couple actuel d'abord).
        configs = _configs(regles)
        if time.monotonic() >= deadline:
            notes += (
                f" ; budget temps ({time_budget_s:g} s) épuisé : finalistes non remesurés "
                "au contexte utile"
            )
            mesures: dict[str, dict] = {}
        else:
            # Finalistes mesurés ENSEMBLE (tours alternés, affinage du tandem incertain).
            mesures = _mesurer(
                make_probe,
                configs,
                ctx_final,
                depth_final,
                reps,
                say,
                deadline=deadline,
            )
        finalistes = [c.key for c in finalists]
        if not any("tg_ts" in v for v in mesures.values()):
            # Rien de remesuré : décider sur la présélection, en le disant.
            mesures = phase1
            configs = presel
            notes += " ; décision sur la présélection (aucun finaliste remesuré)"
            ctx_final, depth_final = ctx, depth
    else:
        mesures = phase1
        finalistes = [k for k, v in phase1.items() if "tg_ts" in v]
    best, mecanisme = pick_placement(
        mesures, configs, margin_pct, prefill=prefill, pp_floor_ratio=pp_floor_ratio
    )
    base = configs[0]
    gain = None
    if best.key != base.key and "tg_ts" in mesures.get(base.key, {}):
        gain = round(
            (mesures[best.key]["tg_ts"] / mesures[base.key]["tg_ts"] - 1) * 100, 1
        )
    m_best = mesures.get(best.key) or {}
    return _res(
        placement=best,
        tg_ts=m_best.get("tg_ts"),
        pp_ts=m_best.get("pp_ts"),
        gain_pct=gain,
        mesures=mesures,
        preselection=phase1,
        finalistes=finalistes,
        ctx_final=ctx_final,
        depth_final=depth_final,
        compare=sum(1 for v in mesures.values() if "tg_ts" in v) >= 2,
        mecanisme=mecanisme + notes + _non_explores_txt(non),
    )


@dataclass(frozen=True)
class ThreadsOption:
    """Un nombre de threads candidat, sondé sur le placement ÉLU (clé `t<n>`)."""

    threads: int
    actuel: bool = False
    # Jamais posés : la sonde de threads ne touche pas aux batchs de la configuration.
    ubatch: int | None = None
    batch: int | None = None

    @property
    def key(self) -> str:
        return f"t{self.threads}"

    def describe(self) -> str:
        return f"{self.threads} threads" + (" — actuel" if self.actuel else "")


def thread_options(
    current: int, logical: int, physical: int | None
) -> list[ThreadsOption]:
    """Candidats de threads : l'ACTUEL d'abord (la base), puis les candidats du parc
    (bench.thread_candidates : physiques/2, physiques, logiques) hors doublon, deux
    au plus — un balayage économe sur la configuration élue."""
    from loom.setup.bench import thread_candidates

    cur = int(current)
    autres = [c for c in thread_candidates(int(logical), physical) if c != cur]
    return [ThreadsOption(cur, actuel=True)] + [ThreadsOption(c) for c in autres[:2]]


def needs_cpu_compute(placement: Placement) -> bool:
    """Du calcul CPU à régler ? Tout GPU : non (threads sans effet attendu — non
    exploré, et la trace le dit). Experts ou couches sur CPU, CPU seul : oui."""
    return placement.label != "gpu_total"


def probe_threads(
    make_probe,
    options: list[ThreadsOption],
    *,
    ctx: int,
    depth: int,
    reps: int = PLACEMENT_REPS,
    margin_pct: float = PLACEMENT_MARGIN_PCT,
    progress=None,
    time_budget_s: float = PLACEMENT_TIME_BUDGET_S,
    prefill: PrefillConstraint | None = None,
    pp_floor_ratio: float | None = None,
) -> dict | None:
    """Sonde les threads sur la configuration ÉLUE (`make_probe(option)` renvoie une
    sonde de cette configuration avec `option.threads`), au même contexte et à la
    même profondeur que la finale, tours alternés, échantillons conservés ; la
    génération décide, le prefill départage, l'actuel reste en cas d'indécision, et
    les CONTRAINTES de prefill s'appliquent comme au placement (mêmes règles que
    pick_placement). Rien de mesurable -> None."""
    say = progress or (lambda _m: None)
    if not options:
        return None
    deadline = time.monotonic() + float(time_budget_s)
    mesures = _mesurer(make_probe, options, ctx, depth, reps, say, deadline=deadline)
    if not any("tg_ts" in v for v in mesures.values()):
        return None
    best, mecanisme = pick_placement(
        mesures, options, margin_pct, prefill=prefill, pp_floor_ratio=pp_floor_ratio
    )
    base = options[0]
    gain = None
    if best.key != base.key and "tg_ts" in mesures.get(base.key, {}):
        gain = round(
            (mesures[best.key]["tg_ts"] / mesures[base.key]["tg_ts"] - 1) * 100, 1
        )
    m = mesures.get(best.key) or {}
    return {
        "threads": int(best.threads),
        "baseline": int(base.threads),
        "tg_ts": m.get("tg_ts"),
        "pp_ts": m.get("pp_ts"),
        "gain_pct": gain,
        "mesures": mesures,
        "ctx": int(ctx),
        "depth": int(depth),
        "compare": sum(1 for v in mesures.values() if "tg_ts" in v) >= 2,
        "mecanisme": mecanisme,
    }


def validate_final(
    probe,
    *,
    ctx: int,
    depth: int,
    n_layers: int | None = None,
    reps: int = PLACEMENT_FINAL_REPS,
    reference_tg: float | None = None,
    margin_pct: float = PLACEMENT_MARGIN_PCT,
    progress=None,
    prefill: PrefillConstraint | None = None,
    prefill_insatisfiable: bool = False,
) -> dict:
    """Valide le RÉGLAGE FINAL complet — la sonde `probe` porte le placement élu, les
    slots décidés et les batchs mesurés — au contexte CALIBRÉ `ctx` et à la profondeur
    de la comparaison, `reps` fois (warmup à chaque démarrage). Le verdict n'assemble
    ainsi plus des mesures prises avec des paramètres différents.

    `reference_tg` = la génération qui a fait élire le placement : `coherent` dit si la
    configuration complète la reproduit (pas plus bas que la marge ou la dispersion) ;
    None sans référence. Un échec est nommé, jamais fatal."""
    say = progress or (lambda _m: None)
    pl = Placement.from_flags(
        int(getattr(probe, "ngl", 999) or 0),
        bool(getattr(probe, "cpu_moe", False)),
        getattr(probe, "n_cpu_moe", None),
        n_layers,
    )
    base = {
        "ctx": int(ctx),
        "depth": int(depth),
        "slots": int(getattr(probe, "n_parallel", 1) or 1),
        "ubatch": getattr(probe, "ubatch", None),
        "batch": getattr(probe, "batch", None),
        "placement": pl.key,
    }
    echantillons: list[dict] = []
    for i in range(max(1, int(reps))):
        say(f"réglage final {pl.key} : mesure {i + 1} à ctx {ctx}, profondeur {depth}…")
        try:
            r = probe.run(ctx, depth)
        except Exception as exc:  # noqa: BLE001 - nommé, jamais fatal
            return {**base, "echec": f"{type(exc).__name__}: {exc}"}
        ech = {
            "tg_ts": float(r.tg_ts or 0) or None,
            "pp_ts": float(r.pp_ts or 0) or None,
            "mem_mb": int(r.mem_mb or 0),
            "prompt_n": getattr(r, "prompt_n", None),
            "predicted_n": getattr(r, "predicted_n", None),
            "checkpoints": getattr(r, "checkpoints", None),
        }
        echantillons.append({k: v for k, v in ech.items() if v is not None})
    m = _agreger(echantillons)
    if "echec" in m:
        return {**base, **m}
    out = {**base, **m, "coherent": None}
    if reference_tg:
        ecart = (m["tg_ts"] / float(reference_tg) - 1) * 100
        out["reference_tg"] = float(reference_tg)
        out["ecart_pct"] = round(ecart, 1)
        out["coherent"] = bool(
            ecart >= -max(margin_pct, float(m.get("tg_disp_pct") or 0))
        )
    if prefill is not None:
        # La contrainte explicite se VÉRIFIE sur le réglage final, pas seulement sur les
        # candidats. Résultat DISTINCT de la cohérence de génération (revue #14 : un
        # `coherent=False` sans `ecart_pct` cassait l'affichage, et une génération
        # identique passait pour « ne reproduit pas (+0 %) »). Violée alors qu'un
        # candidat la tenait : BLOQUANT (rien n'est appliqué). Insatisfiable sur cette
        # machine (aucun candidat ne la tenait) : avertissement, comme au placement.
        secondes = prefill.seconds(float(m.get("pp_ts") or 0))
        respectee = secondes <= prefill.max_seconds
        pc = {
            "new_tokens": prefill.new_tokens,
            "max_seconds": prefill.max_seconds,
            "secondes": round(secondes, 1) if secondes != math.inf else None,
            "respectee": bool(respectee),
        }
        if not respectee and prefill_insatisfiable:
            pc["insatisfiable"] = True
        out["prefill_contrainte"] = pc
        if not respectee and not prefill_insatisfiable:
            out["bloquant"] = "contrainte prefill non respectée : " + _prefill_txt(pc)
    return out


def _prefill_txt(pc: dict) -> str:
    """« 2000 tokens en 13.3 s > 10 s » (ou « prefill illisible »)."""
    lim = f"{float(pc['max_seconds']):g} s"
    if pc.get("secondes") is None:
        return f"{pc['new_tokens']} tokens, prefill illisible (limite {lim})"
    signe = "≤" if pc.get("respectee") else ">"
    return f"{pc['new_tokens']} tokens en {pc['secondes']} s {signe} {lim}"


def prefill_satisfiable(
    mesures: dict, prefill: PrefillConstraint | None
) -> bool | None:
    """La contrainte explicite était-elle TENUE par au moins un candidat mesuré ?
    None sans contrainte ou sans mesure exploitable."""
    if prefill is None:
        return None
    pps = [
        float(v.get("pp_ts") or 0)
        for v in (mesures or {}).values()
        if isinstance(v, dict) and v.get("pp_ts")
    ]
    if not pps:
        return None
    return any(prefill.seconds(pp) <= prefill.max_seconds for pp in pps)


def final_checks_text(fin: dict) -> str:
    """Suffixe lisible des contrôles du réglage final, chacun pour ce qu'il est : la
    cohérence de GÉNÉRATION (seulement si une référence existe) et la contrainte de
    PREFILL (durée mesurée face à la limite)."""
    fin = fin or {}
    out = ""
    if fin.get("coherent") is True and fin.get("ecart_pct") is not None:
        out += (
            " ; génération cohérente avec la mesure de placement "
            f"({fin['ecart_pct']:+.1f} %)"
        )
    elif fin.get("coherent") is False and fin.get("ecart_pct") is not None:
        out += (
            " ; génération : ne reproduit pas la mesure de placement "
            f"({fin['ecart_pct']:+.1f} %) — à appliquer avec prudence"
        )
    pc = fin.get("prefill_contrainte")
    if pc:
        if pc.get("respectee"):
            out += f" ; contrainte prefill respectée ({_prefill_txt(pc)})"
        elif pc.get("insatisfiable"):
            out += (
                f" ; contrainte prefill NON respectée ({_prefill_txt(pc)}), "
                "insatisfiable sur cette machine : aucun candidat ne la tenait"
            )
        else:
            out += f" ; contrainte prefill NON respectée ({_prefill_txt(pc)})"
    return out


def pick_placement(
    mesures: dict[str, dict],
    candidates: list[Placement],
    margin_pct: float = PLACEMENT_MARGIN_PCT,
    *,
    prefill: PrefillConstraint | None = None,
    pp_floor_ratio: float | None = None,
) -> tuple[Placement, str]:
    """(placement retenu, mécanisme). La génération tranche ; la ligne de base
    (candidat 0 = configuration actuelle quand elle est connue) n'est quittée que si
    une alternative la bat de plus de `margin_pct` ; entre alternatives équivalentes au
    tg, le prefill départage. Un candidat sans mesure (échec) est écarté et nommé.

    Contraintes optionnelles : `prefill` (N tokens en T s, explicite) écarte ceux qui ne
    la tiennent pas, sauf si aucun ne la tient ; `pp_floor_ratio` (plancher relatif au
    meilleur prefill) est un CHOIX DE CONFORT qui peut sacrifier de la génération."""
    by_key = {c.key: c for c in candidates}
    valid = {
        k: v
        for k, v in mesures.items()
        if k in by_key and float(v.get("tg_ts") or 0) > 0
    }
    notes = [
        f"{k} : {v.get('echec', 'sans mesure')}"
        for k, v in mesures.items()
        if k not in valid
    ]
    if prefill is not None and valid:
        viol = {
            k: prefill.seconds(float(v.get("pp_ts") or 0))
            for k, v in valid.items()
            if prefill.seconds(float(v.get("pp_ts") or 0)) > prefill.max_seconds
        }
        if viol and len(viol) < len(valid):
            for k, s in viol.items():
                notes.append(
                    f"{k} écarté : contrainte prefill ({prefill.new_tokens} tokens en "
                    f"{s:.1f} s > {prefill.max_seconds:g} s)"
                )
            valid = {k: v for k, v in valid.items() if k not in viol}
        elif viol:
            notes.append(
                f"aucun candidat ne satisfait la contrainte prefill "
                f"({prefill.new_tokens} tokens en {prefill.max_seconds:g} s) : décision "
                "à la génération seule"
            )
    if pp_floor_ratio and valid:
        best_pp = max(float(v.get("pp_ts") or 0) for v in valid.values())
        low = {
            k
            for k, v in valid.items()
            if float(v.get("pp_ts") or 0) < best_pp * pp_floor_ratio
        }
        if low and len(low) < len(valid):
            for k in sorted(low):
                notes.append(
                    f"{k} écarté : prefill sous {pp_floor_ratio:.0%} du meilleur "
                    "(plancher relatif = choix de confort, peut sacrifier de la génération)"
                )
            valid = {k: v for k, v in valid.items() if k not in low}
    suffixe = (" ; " + " ; ".join(notes)) if notes else ""
    base = candidates[0]
    if not valid:
        return base, f"aucune mesure exploitable, {base.key} conservé{suffixe}"
    if base.key not in valid:
        # La base n'a pas pu être mesurée (ou a été écartée) : la meilleure alternative.
        key = max(valid, key=lambda k: (valid[k]["tg_ts"], valid[k].get("pp_ts") or 0))
        return by_key[
            key
        ], f"{key} retenu (base {base.key} non mesurée ou écartée){suffixe}"
    base_tg = valid[base.key]["tg_ts"]
    disp_base = float(valid[base.key].get("tg_disp_pct") or 0)
    seuil = base_tg * (1 + margin_pct / 100)
    gagnants = {k: v for k, v in valid.items() if k != base.key and v["tg_ts"] > seuil}
    # La marge est une POLITIQUE de changement, pas une mesure du bruit : un gain qui
    # tient dans la dispersion mesurée des deux candidats est indécis, pas un gain.
    indecis: dict[str, tuple[float, float]] = {}
    for k, v in list(gagnants.items()):
        gain_k = (v["tg_ts"] / base_tg - 1) * 100
        disp = max(disp_base, float(v.get("tg_disp_pct") or 0))
        if gain_k <= disp:
            indecis[k] = (gain_k, disp)
            del gagnants[k]
    if not gagnants:
        if len(valid) == 1:
            return base, f"{base.key} : seul candidat mesuré{suffixe}"
        # Génération ÉQUIVALENTE à la base — écart dans la MARGE explicite, et elle
        # seule : une forte dispersion déclenche un affinage ou une indécision, elle
        # n'élargit pas la perte de génération acceptable (revue P1 : experts-CPU à
        # -6,7 % passait pour équivalent grâce à sa dispersion de 6,7 %). Le prefill
        # départage alors, même contre la base, s'il est NETTEMENT meilleur.
        pp_base = float(valid[base.key].get("pp_ts") or 0)
        equivalents_base = {
            k: v
            for k, v in valid.items()
            if k != base.key and abs((v["tg_ts"] / base_tg - 1) * 100) <= margin_pct
        }
        if equivalents_base and pp_base > 0:
            k_pp = max(
                equivalents_base, key=lambda k: float(valid[k].get("pp_ts") or 0)
            )
            pp_k = float(valid[k_pp].get("pp_ts") or 0)
            if pp_k > pp_base * (1 + PLACEMENT_PP_TIEBREAK_PCT / 100):
                ecart_tg = (valid[k_pp]["tg_ts"] / base_tg - 1) * 100
                gain_pp = (pp_k / pp_base - 1) * 100
                return by_key[k_pp], (
                    f"{k_pp} adopté : génération équivalente à {base.key} "
                    f"({valid[k_pp]['tg_ts']} contre {base_tg}, {ecart_tg:+.0f} %), "
                    f"départagé au prefill : {pp_k} t/s contre {pp_base} ({gain_pp:+.0f} %, "
                    f"au-dessus de {PLACEMENT_PP_TIEBREAK_PCT:g} %){suffixe}"
                )
        meilleur = max(
            (k for k in valid if k != base.key), key=lambda k: valid[k]["tg_ts"]
        )
        ecart = (valid[meilleur]["tg_ts"] / base_tg - 1) * 100
        disp = max(disp_base, float(valid[meilleur].get("tg_disp_pct") or 0))
        if meilleur in indecis or abs(ecart) <= disp:
            return base, (
                f"{base.key} conservé — indécis : {meilleur} à {ecart:+.0f} % de tg "
                f"({valid[meilleur]['tg_ts']} contre {base_tg}), écart dans la dispersion "
                f"mesurée ({disp:.0f} %){suffixe}"
            )
        return base, (
            f"{base.key} conservé : {meilleur} mesuré à {ecart:+.0f} % de tg "
            f"({valid[meilleur]['tg_ts']} contre {base_tg}), sous la marge de "
            f"{margin_pct:g} %{suffixe}"
        )
    if indecis:
        suffixe += " ; " + " ; ".join(
            f"{k} : {g:+.0f} % mais dispersion mesurée {d:.0f} % — indécis"
            for k, (g, d) in indecis.items()
        )
    best_tg = max(v["tg_ts"] for v in gagnants.values())
    # Alternatives équivalentes entre elles (sous la marge) : le prefill tranche.
    equivalents = {
        k: v
        for k, v in gagnants.items()
        if v["tg_ts"] >= best_tg * (1 - margin_pct / 100)
    }
    key = max(
        equivalents,
        key=lambda k: (equivalents[k].get("pp_ts") or 0, equivalents[k]["tg_ts"]),
    )
    gain = (valid[key]["tg_ts"] / base_tg - 1) * 100
    perdants = [
        f"{k} ({valid[k]['tg_ts']} t/s)" for k in valid if k not in (key, base.key)
    ]
    perdants_txt = (
        " ; mesurés moins performants : " + ", ".join(perdants) if perdants else ""
    )
    return by_key[key], (
        f"{key} adopté : génération {valid[key]['tg_ts']} t/s contre {base_tg} "
        f"({base.key}), {gain:+.0f} %, au-dessus de la marge de {margin_pct:g} %"
        f"{perdants_txt}{suffixe}"
    )
