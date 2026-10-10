"""Profil ARCHITECTURAL d'un modèle, bâti sur le header GGUF, pour piloter le bench.

Il répond aux questions dont dépendent les candidats de placement et le budget :
- quelles couches portent un cache KV (attention pleine, fenêtre glissante) et
  lesquelles n'en ont pas (récurrentes : GDN, Mamba, RWKV) ;
- combien pèse le KV au contexte VISÉ, avec le type de cache de l'EXÉCUTANT (q8_0
  sous profil GPU, f16 sinon) et le nombre de slots ;
- où pèsent les poids (experts, attention, FFN, récurrence, embeddings, sortie), par
  couche, donc combien va sur le device pour un -ngl / --cpu-moe / --n-cpu-moe.

Chaque donnée porte sa PROVENANCE : « déclaré » (clé du header), « déduit » (calcul
ou catalogue des tenseurs), « inconnu » (repli, borne haute). Le profil choisit les
essais pertinents ; il n'impose jamais une doctrine (« MoE donc experts CPU »).
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Octets par élément du cache KV selon le type (bloc q8_0 = 32 éléments + échelle).
_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}
#: KV/token de repli (f16) si le header ne porte pas les dimensions d'attention
#: (~valeur d'un 8B dense GQA : 36 couches x 1024 dims KV x 2 (K+V) x 2 octets).
KV_FALLBACK_BYTES_PER_TOKEN = 150_000
_MIB = 1024 * 1024


def _recurrent_state_bytes(meta: dict, n_recurrent: int) -> int | None:
    """Octets f32 de l'état récurrent d'une séquence, formule de llama.cpp
    (llama-hparams n_embd_r / n_embd_s) : RWKV `wkv.head_size` (2 décalages x n_embd +
    n_embd x head_size), LFM2 `shortconv.l_cache` (n_embd x (l_cache - 1)), Mamba/GDN
    `ssm.*` ((d_conv - 1) x (d_inner + 2 n_group d_state) + d_state x d_inner). None si
    les dimensions manquent."""
    if n_recurrent <= 0:
        return 0
    n_embd = int(meta.get("embedding_length") or 0)
    wkv = int(meta.get("wkv_head_size") or 0)
    shortconv = int(meta.get("shortconv_l_cache") or 0)
    conv = int(meta.get("ssm_conv_kernel") or 0)
    inner = int(meta.get("ssm_inner_size") or 0)
    state = int(meta.get("ssm_state_size") or 0)
    group = int(meta.get("ssm_group_count") or 1)
    if wkv and n_embd:
        par_couche = 2 * n_embd + n_embd * wkv
    elif shortconv and n_embd:
        par_couche = n_embd * (shortconv - 1)
    elif conv and inner and state:
        par_couche = (conv - 1) * (inner + 2 * group * state) + state * inner
    else:
        return None
    return int(par_couche * 4 * n_recurrent)


@dataclass
class ModelProfile:
    architecture: str | None = None
    n_layers: int | None = None
    context_limit: int | None = None
    moe: bool = False
    expert_count: int | None = None
    experts_per_token: int | None = None
    kv_heads: int | None = None
    key_length: int | None = None
    value_length: int | None = None
    swa_window: int | None = None
    swa_pattern: int | None = None
    attention_layers: list[int] = field(default_factory=list)  # couches à cache KV
    recurrent_layers: list[int] = field(default_factory=list)
    swa_layers: list[int] = field(default_factory=list)  # KV borné à la fenêtre
    recurrent: bool = False
    weights: dict | None = None  # résumé du catalogue (gguf_meta.read_gguf_meta)
    model_size_mb: int = 0
    # Octets de l'ÉTAT RÉCURRENT complet d'UNE séquence (toutes les couches récurrentes,
    # f32) : 0 sans récurrence, None si les dimensions manquent. Chaque checkpoint du
    # serveur en pèse autant (Bonsai 2 : ~150 Mio observés le 2026-09-30).
    recurrent_state_bytes: int | None = 0
    provenance: dict[str, str] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────────────

    @classmethod
    def from_meta(cls, meta: dict, model_size_mb: int = 0) -> ModelProfile:
        meta = meta or {}
        prov: dict[str, str] = {}

        def declared(key: str):
            v = meta.get(key)
            prov[key] = "déclaré" if v is not None else "inconnu"
            return v

        n_layers = declared("n_layers")
        expert_count = declared("expert_count")
        kv_heads = declared("head_count_kv")
        key_length = declared("key_length")
        if (
            key_length is None
            and meta.get("embedding_length")
            and meta.get("head_count")
        ):
            key_length = int(meta["embedding_length"]) // int(meta["head_count"])
            prov["key_length"] = "déduit (embedding_length / head_count)"
        value_length = meta.get("value_length")
        if value_length is not None:
            prov["value_length"] = "déclaré"
        elif key_length is not None:
            value_length = key_length
            prov["value_length"] = "déduit (= key_length)"
        else:
            prov["value_length"] = "inconnu"
        swa_window = declared("sliding_window")
        pattern = meta.get("sliding_window_pattern")
        swa_pattern = pattern if isinstance(pattern, int) and pattern > 0 else None
        interval = meta.get("full_attention_interval")
        recurrent = bool(meta.get("recurrent"))
        weights = meta.get("weights") or None

        n = int(n_layers or 0)
        if weights and weights.get("couches_attention") is not None:
            attention = [int(i) for i in weights["couches_attention"]]
            recurrent_layers = [int(i) for i in weights.get("couches_recurrentes", [])]
            prov["couches_attention"] = "déduit (catalogue des tenseurs)"
        elif isinstance(interval, int) and interval > 0 and n:
            attention = [i for i in range(n) if (i + 1) % interval == 0]
            recurrent_layers = [i for i in range(n) if (i + 1) % interval != 0]
            prov["couches_attention"] = f"déduit (full_attention_interval = {interval})"
        elif recurrent:
            attention = list(range(n))
            recurrent_layers = []
            prov["couches_attention"] = (
                "inconnu (mémoire récurrente sans motif déclaré : toutes les couches "
                "supposées à cache KV, borne haute)"
            )
        else:
            attention = list(range(n))
            recurrent_layers = []
            prov["couches_attention"] = "déduit (architecture sans récurrence)"

        if swa_window and swa_pattern:
            swa_layers = [i for i in attention if (i + 1) % swa_pattern != 0]
            prov["couches_swa"] = f"déduit (sliding_window_pattern = {swa_pattern})"
        elif swa_window:
            swa_layers = []
            prov["couches_swa"] = (
                "inconnu (fenêtre déclarée sans motif : toutes pleines, borne haute)"
            )
        else:
            swa_layers = []
            prov["couches_swa"] = "déclaré (pas de fenêtre glissante)"

        if not attention and not prov["couches_attention"].startswith("inconnu"):
            # Couches connues et aucune d'attention (Mamba pur…) : AUCUN cache KV —
            # ce n'est pas « KV inconnu », le forfait de secours ne s'applique pas.
            prov["kv"] = "déduit (aucune couche d'attention : pas de cache KV)"
        elif kv_heads and key_length:
            prov["kv"] = (
                "déclaré/déduit (têtes KV x (K + V) x type de cache, couches d'attention)"
            )
        else:
            prov["kv"] = (
                f"inconnu (repli {KV_FALLBACK_BYTES_PER_TOKEN // 1000} Ko/token)"
            )
        prov["poids"] = (
            weights["provenance"]
            if weights
            else "inconnu (pas de catalogue de tenseurs)"
        )
        rec_bytes = _recurrent_state_bytes(meta, len(recurrent_layers))
        if not recurrent_layers:
            rec_bytes = 0
            prov["etat_recurrent"] = "déclaré (pas de couche récurrente)"
        elif rec_bytes is None:
            prov["etat_recurrent"] = (
                "inconnu (dimensions ssm/wkv/shortconv absentes du header)"
            )
        else:
            prov["etat_recurrent"] = (
                "déduit (formule llama.cpp : conv (d_conv-1)x(d_inner + 2 n_group d_state)"
                " + état d_state x d_inner, f32, par couche récurrente)"
            )
        return cls(
            architecture=meta.get("architecture"),
            n_layers=n_layers,
            context_limit=meta.get("context_length"),
            moe=bool(expert_count),
            expert_count=expert_count,
            experts_per_token=meta.get("expert_used_count"),
            kv_heads=kv_heads,
            key_length=key_length,
            value_length=value_length,
            swa_window=swa_window,
            swa_pattern=swa_pattern,
            attention_layers=attention,
            recurrent_layers=recurrent_layers,
            swa_layers=swa_layers,
            recurrent=recurrent,
            weights=weights,
            model_size_mb=int(model_size_mb or 0),
            recurrent_state_bytes=rec_bytes,
            provenance=prov,
        )

    def recurrent_bytes(self, *, slots: int = 1, checkpoints: int = 0) -> int:
        """Octets de mémoire récurrente que le serveur allouera : par slot, l'état
        vivant plus `checkpoints` instantanés complets ; x slots. 0 si inconnu (la pente
        mesurée reste alors la seule source)."""
        if not self.recurrent_state_bytes:
            return 0
        return int(
            self.recurrent_state_bytes
            * (1 + max(0, int(checkpoints)))
            * max(1, int(slots))
        )

    # ── estimations ─────────────────────────────────────────────────────────────

    def kv_bytes(self, ctx: int, kv_type: str = "f16", slots: int = 1) -> int:
        """Octets de cache KV pour `ctx` tokens PAR SLOT, `slots` slots, cache de
        type `kv_type`. Couches récurrentes : aucun KV ; couches à fenêtre glissante :
        bornées à la fenêtre. Sans dimensions d'attention : repli conservateur."""
        slots = max(1, int(slots))
        if not self.attention_layers and not str(
            self.provenance.get("couches_attention", "inconnu")
        ).startswith("inconnu"):
            return 0  # aucune couche d'attention connue : pas de KV, pas de forfait
        if not (self.kv_heads and self.key_length):
            return int(KV_FALLBACK_BYTES_PER_TOKEN * ctx * slots)
        bpe = _BYTES_PER_ELEM.get(kv_type, 2.0)
        per_tok = self.kv_heads * (self.key_length + (self.value_length or 0)) * bpe
        swa = set(self.swa_layers)
        total = 0.0
        for i in self.attention_layers:
            toks = min(ctx, self.swa_window) if (i in swa and self.swa_window) else ctx
            total += toks * per_tok
        return int(total * slots)

    def gpu_bytes(
        self, *, ngl: int = 999, cpu_moe: bool = False, n_cpu_moe: int | None = None
    ) -> int | None:
        """Octets de poids que ce placement met sur le device, d'après le catalogue
        (None sans catalogue). llama.cpp : les `ngl` DERNIÈRES couches vont sur GPU
        (i_gpu_start = n_layer - ngl), la sortie seulement si ngl > n_layer, les
        embeddings d'entrée restent sur CPU. --cpu-moe retire tous les experts ;
        --n-cpu-moe N retire ceux des N premières couches."""
        if not self.weights:
            return None
        layers = list(self.weights.get("par_couche") or [])
        experts = list(self.weights.get("experts_par_couche") or [])
        experts += [0] * (len(layers) - len(experts))
        n = len(layers)
        k = max(0, min(int(ngl), n))
        start = n - k
        total = sum(layers[start:])
        if cpu_moe:
            total -= sum(experts[start:])
        elif n_cpu_moe is not None:
            total -= sum(experts[i] for i in range(start, n) if i < int(n_cpu_moe))
        if ngl > n:
            total += int((self.weights.get("familles") or {}).get("output", 0))
        return int(total)

    def describe(self) -> list[str]:
        """Lignes lisibles, chacune avec sa provenance."""
        p = self.provenance
        out = [
            f"architecture {self.architecture or '?'}, {self.n_layers or '?'} couches "
            f"({p.get('n_layers', 'inconnu')})",
            f"couches à cache KV : {len(self.attention_layers)}, récurrentes : "
            f"{len(self.recurrent_layers)} ({p.get('couches_attention', 'inconnu')})",
        ]
        if self.moe:
            out.append(
                f"MoE : {self.expert_count} experts ({p.get('expert_count')}), "
                f"{self.experts_per_token or '?'} actifs par token"
            )
        out.append(
            f"KV : {self.kv_heads or '?'} têtes x (K {self.key_length or '?'} + V "
            f"{self.value_length or '?'}) — value_length {p.get('value_length')} ; "
            f"fenêtre {self.swa_window or 'aucune'} ({p.get('couches_swa')})"
        )
        if self.recurrent_layers:
            if self.recurrent_state_bytes:
                out.append(
                    f"état récurrent : {self.recurrent_state_bytes // _MIB} Mio par séquence "
                    f"({len(self.recurrent_layers)} couches), autant par checkpoint "
                    f"({p.get('etat_recurrent')})"
                )
            else:
                out.append(f"état récurrent : {p.get('etat_recurrent')}")
        if self.weights:
            fam = self.weights.get("familles") or {}
            out.append(
                f"poids {self.weights.get('total', 0) // _MIB} Mio : experts "
                f"{fam.get('experts', 0) // _MIB}, attention {fam.get('attention', 0) // _MIB}, "
                f"FFN {fam.get('ffn', 0) // _MIB}, récurrence {fam.get('recurrent', 0) // _MIB}, "
                f"embeddings {fam.get('embeddings', 0) // _MIB}, sortie "
                f"{fam.get('output', 0) // _MIB} ({p.get('poids')})"
            )
        else:
            out.append(f"poids : {p.get('poids')}")
        return out
