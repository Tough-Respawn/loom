# loom/setup/placement.py
"""Placement MESURÉ des poids d'un modèle : où vivent les couches denses et les experts.

Jusqu'au 2026-10-09 une règle décidait (MoE + GPU -> experts en RAM, `cpu_moe = true`
posé par /add-model, `discover_topology` imposant la topologie). Mesuré ce jour-là sur
Ornith 35B-A3B Q8_0 (Radeon 860M, mémoire unifiée) : tout sur GPU = +21 % de prefill et
+17-19 % de génération par rapport aux experts sur CPU. La règle venait d'une sonde du
21/07 qui n'avait mesuré que le prefill, à une répétition, avec ±12 % de bruit.

Principes (ceux de topology.py, appliqués au placement) :
- les CANDIDATS viennent de la faisabilité mémoire, jamais d'une doctrine ;
- une sonde PAR CANDIDAT, avec les flags exacts de l'exécutant (ServerProbe), pour
  qu'un échec mémoire n'emporte pas les autres mesures ;
- les DEUX axes au point de fonctionnement : prefill sur un prompt long et génération
  en profondeur, plusieurs répétitions ;
- une MARGE de bruit : on ne quitte le placement le plus simple (candidat 0) que si le
  gain la dépasse, et la décision porte son mécanisme.
"""

from __future__ import annotations

from dataclasses import dataclass

from loom.runtime.hardware import recommend_gpu_layers

#: Gain minimal de génération (en %) pour quitter le placement de base.
PLACEMENT_MARGIN_PCT = 5.0
#: Contexte et prompt de la sonde : prefill long (le levier n'existe pas à 128 tokens)
#: et génération mesurée à la profondeur de ce prompt.
PLACEMENT_PROBE_CTX = 8192
PLACEMENT_PROBE_PROMPT = 4096
#: Répétitions par candidat (moyenne) : une seule mesure n'a pas d'écart-type.
PLACEMENT_REPS = 2
#: Marge RAM laissée à l'OS quand le « device » est la RAM (mémoire unifiée).
_OS_RAM_BUDGET_MB = 3072


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

    def describe(self) -> str:
        if self.label == "cpu":
            return "CPU seul (ngl 0)"
        if self.label == "gpu_total":
            return "tout sur GPU (ngl 999)"
        if self.label == "gpu_partiel":
            return f"offload partiel (ngl {self.ngl}, estimé)"
        if self.label == "experts_cpu":
            return "denses sur GPU, experts sur CPU (--cpu-moe)"
        return f"denses sur GPU, experts de {self.n_cpu_moe} couches sur CPU (estimé)"


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


def placement_candidates(
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
) -> list[Placement]:
    """Candidats faisables, le plus simple (ou le plus sûr) EN PREMIER = ligne de base.

    - sans GPU : CPU seul ;
    - MoE : experts sur CPU (tient toujours si les denses tiennent), puis tout GPU si
      poids + KV tiennent, sinon un partiel ESTIMÉ (part d'experts à laisser sur CPU
      proportionnelle au déficit) ;
    - dense : tout GPU si ça tient, sinon l'offload proportionnel (estimé), sinon CPU."""
    if not gpu_backend or vram_total_mb <= 0:
        return [Placement("cpu", 0)]
    budget = device_budget_mb(vram_total_mb, ram_total_mb, uma, headroom_mb)
    need = model_size_mb + kv_mb
    layers = int(n_layers or 0)
    if moe:
        out = [Placement("experts_cpu", 999, cpu_moe=True)]
        if need <= budget:
            out.append(Placement("gpu_total", 999))
        elif layers > 1 and model_size_mb > 0:
            deficit = need - budget
            n = -(-deficit * layers // model_size_mb)  # arrondi supérieur
            n = int(min(layers - 1, max(1, n)))
            out.append(Placement("experts_partiel", 999, n_cpu_moe=n, estime=True))
        return out
    if need <= budget:
        return [Placement("gpu_total", 999)]
    reco = recommend_gpu_layers(budget, need, layers, 0) if layers else 0
    if 0 < reco < 999:
        return [Placement("gpu_partiel", reco, estime=True)]
    return [Placement("cpu", 0)]


def probe_placement(
    make_probe,
    candidates: list[Placement],
    *,
    ctx: int = PLACEMENT_PROBE_CTX,
    depth: int = PLACEMENT_PROBE_PROMPT,
    reps: int = PLACEMENT_REPS,
    margin_pct: float = PLACEMENT_MARGIN_PCT,
    progress=None,
) -> dict | None:
    """Sonde chaque candidat avec le VRAI serveur (`make_probe(placement)` renvoie une
    sonde exposant `.run(ctx, depth) -> ProbeResult`, cf. topology.ServerProbe), `reps`
    fois, et élit par `pick_placement`. Un candidat qui échoue (mémoire, chargement)
    est écarté et nommé ; rien de mesurable -> None (on n'écrit jamais une valeur
    inventée). Un seul candidat faisable n'est pas sondé : rien à comparer."""
    say = progress or (lambda _m: None)
    if not candidates:
        return None
    if len(candidates) == 1:
        seul = candidates[0]
        return {
            "placement": seul,
            "baseline": seul.label,
            "tg_ts": None,
            "pp_ts": None,
            "gain_pct": None,
            "mesures": {},
            "mecanisme": f"{seul.label} : seul candidat faisable (non sondé)",
        }
    mesures: dict[str, dict] = {}
    for c in candidates:
        say(f"placement {c.label} ({c.describe()}) : {reps} mesure(s)…")
        tgs: list[float] = []
        pps: list[float] = []
        mems: list[int] = []
        try:
            sonde = make_probe(c)
            for _ in range(reps):
                r = sonde.run(ctx, depth)
                if r.tg_ts:
                    tgs.append(float(r.tg_ts))
                if r.pp_ts:
                    pps.append(float(r.pp_ts))
                mems.append(int(r.mem_mb or 0))
        except Exception as exc:  # noqa: BLE001 - un candidat qui casse n'est PAS fatal
            mesures[c.label] = {"echec": f"{type(exc).__name__}: {exc}"}
            continue
        if not tgs:
            mesures[c.label] = {"echec": "débit illisible"}
            continue
        mesures[c.label] = {
            "tg_ts": round(sum(tgs) / len(tgs), 2),
            "pp_ts": round(sum(pps) / len(pps), 2) if pps else 0.0,
            "mem_mb": max(mems) if mems else 0,
        }
    if not any("tg_ts" in v for v in mesures.values()):
        return None
    best, mecanisme = pick_placement(mesures, candidates, margin_pct)
    base = candidates[0]
    gain = None
    if best.label != base.label and "tg_ts" in mesures.get(base.label, {}):
        gain = round(
            (mesures[best.label]["tg_ts"] / mesures[base.label]["tg_ts"] - 1) * 100, 1
        )
    return {
        "placement": best,
        "baseline": base.label,
        "tg_ts": mesures[best.label]["tg_ts"],
        "pp_ts": mesures[best.label]["pp_ts"],
        "gain_pct": gain,
        "mesures": mesures,
        "mecanisme": mecanisme,
    }


def pick_placement(
    mesures: dict[str, dict],
    candidates: list[Placement],
    margin_pct: float = PLACEMENT_MARGIN_PCT,
) -> tuple[Placement, str]:
    """(placement retenu, mécanisme). La génération tranche ; la ligne de base
    (candidat 0) n'est quittée que si une alternative la bat de plus de `margin_pct` ;
    entre alternatives équivalentes au tg, le prefill départage. Un candidat sans
    mesure (échec) est écarté et nommé dans le mécanisme."""
    by_label = {c.label: c for c in candidates}
    valid = {
        k: v
        for k, v in mesures.items()
        if k in by_label and float(v.get("tg_ts") or 0) > 0
    }
    echecs = [
        f"{k} : {v.get('echec', 'sans mesure')}"
        for k, v in mesures.items()
        if k not in valid
    ]
    suffixe = (" ; " + " ; ".join(echecs)) if echecs else ""
    base = candidates[0]
    if not valid:
        return base, f"aucune mesure exploitable, {base.label} conservé{suffixe}"
    if base.label not in valid:
        # La base n'a pas pu être mesurée : la meilleure alternative mesurée.
        label = max(
            valid, key=lambda k: (valid[k]["tg_ts"], valid[k].get("pp_ts") or 0)
        )
        return by_label[label], f"{label} retenu (base non mesurée){suffixe}"
    base_tg = valid[base.label]["tg_ts"]
    seuil = base_tg * (1 + margin_pct / 100)
    gagnants = {
        k: v for k, v in valid.items() if k != base.label and v["tg_ts"] > seuil
    }
    if not gagnants:
        if len(valid) == 1:
            return base, f"{base.label} : seul candidat mesuré{suffixe}"
        meilleur = max(
            (k for k in valid if k != base.label), key=lambda k: valid[k]["tg_ts"]
        )
        ecart = (valid[meilleur]["tg_ts"] / base_tg - 1) * 100
        return base, (
            f"{base.label} conservé : {meilleur} à {ecart:+.0f} % de tg, sous la marge "
            f"de {margin_pct:g} %{suffixe}"
        )
    best_tg = max(v["tg_ts"] for v in gagnants.values())
    # Alternatives équivalentes entre elles (sous la marge) : le prefill tranche.
    equivalents = {
        k: v
        for k, v in gagnants.items()
        if v["tg_ts"] >= best_tg * (1 - margin_pct / 100)
    }
    label = max(
        equivalents,
        key=lambda k: (equivalents[k].get("pp_ts") or 0, equivalents[k]["tg_ts"]),
    )
    gain = (valid[label]["tg_ts"] / base_tg - 1) * 100
    return by_label[label], (
        f"{label} adopté : génération {valid[label]['tg_ts']} t/s contre {base_tg} "
        f"({base.label}), {gain:+.0f} %, au-dessus de la marge de {margin_pct:g} %{suffixe}"
    )
