"""Lecture du header GGUF, sans dépendance externe : paires clé/valeur (tableaux
courts compris) ET catalogue des tenseurs (noms, offsets) — jamais les données.

Sert à compléter model.toml après téléchargement (n_layers, contexte max, MoE) et à
bâtir le PROFIL du modèle (loom.runtime.model_profile) qui pilote le bench de
placement : quelles couches portent un cache KV, où pèsent les poids (attention,
experts, FFN, récurrence, embeddings, sortie), par couche.

Les tailles de tenseurs viennent des OFFSETS successifs dans la section de données,
pas d'une table des types ggml : un quant inconnu du lecteur (Bonsai PQ2_0 = type 142
d'un fork) se mesure quand même. Spécification du format :
https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

# Les chaînes et tableaux GGUF nécessitent un décodage distinct des scalaires.
_SCALAR = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "B",
    10: "Q",
    11: "q",
    12: "d",
}
# Au-delà, un tableau est un vocabulaire ou une table de signes : traversé, pas gardé.
_ARRAY_KEEP_MAX = 4096
# Bornes de gguf.cpp (GGUF_MAX_STRING_LENGTH, GGUF_MAX_ARRAY_ELEMENTS, GGML_MAX_DIMS) :
# au-delà, l'en-tête est corrompu — llama-server le rejette aussi. Sans elles, une
# longueur de 2^62 faisait lever MemoryError/OverflowError, que personne n'attrape.
_MAX_CHAINE = 1024 * 1024 * 1024
_MAX_ELEMENTS = 1024 * 1024 * 1024
_MAX_DIMS = 4
_DEFAULT_ALIGNMENT = 32
# Tenseurs d'une couche à mémoire récurrente (Mamba/GDN `ssm_*`, RWKV `time_mix*`,
# LFM2 `shortconv*`).
_RECURRENT_PREFIXES = ("ssm_", "time_mix", "shortconv")


class TypeGGUFInconnu(ValueError):
    """Type de valeur que ce lecteur ne connaît pas : le fichier peut être plus récent
    que lui, pas invalide — à distinguer d'un en-tête rejeté (pas un GGUF, version < 2,
    tronqué) que llama-server refuserait aussi."""


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    if n > _MAX_CHAINE:
        raise ValueError(f"chaîne GGUF de {n} octets : en-tête corrompu")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int, keep_arrays: bool = False):
    if vtype == 8:
        return _read_string(f)
    if vtype == 9:
        (itype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if count > _MAX_ELEMENTS:
            raise ValueError(f"tableau GGUF de {count} éléments : en-tête corrompu")
        keep = keep_arrays and count <= _ARRAY_KEEP_MAX
        vals = []
        # Traverser les tableaux garde le curseur aligné sur les clés suivantes.
        for _ in range(count):
            v = _read_value(f, itype, keep)
            if keep:
                vals.append(v)
        return vals if keep else None
    fmt = _SCALAR.get(vtype)
    if fmt is None:
        raise TypeGGUFInconnu(f"type GGUF inconnu : {vtype}")
    (v,) = struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))
    return v


def _read_tensor_infos(f, count: int) -> list[tuple[str, int, int]]:
    """[(nom, type ggml, offset dans la section de données)]. Les dimensions sont
    lues pour avancer, pas conservées : la taille se déduit des offsets."""
    out = []
    for _ in range(count):
        name = _read_string(f)
        (n_dims,) = struct.unpack("<I", f.read(4))
        if n_dims > _MAX_DIMS:
            raise ValueError(f"tenseur GGUF à {n_dims} dimensions : en-tête corrompu")
        f.read(8 * n_dims)
        (ty,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        out.append((name, ty, off))
    return out


def _layer_of(name: str) -> tuple[int | None, str]:
    """(indice de couche, nom du tenseur dans la couche) pour `blk.N.xxx`, sinon
    (None, nom complet)."""
    if name.startswith("blk."):
        parts = name.split(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            return int(parts[1]), parts[2]
    return None, name


def _weights_summary(
    infos: list[tuple[str, int, int]], data_start: int, file_size: int
) -> dict | None:
    """Poids par FAMILLE et par COUCHE depuis le catalogue (tailles par offsets)."""
    if not infos:
        return None
    infos = sorted(infos, key=lambda t: t[2])
    data_size = max(0, file_size - data_start)
    sizes: dict[str, int] = {}
    for i, (name, _ty, off) in enumerate(infos):
        end = infos[i + 1][2] if i + 1 < len(infos) else data_size
        sizes[name] = max(0, end - off)

    familles = {
        k: 0
        for k in (
            "attention",
            "experts",
            "ffn",
            "recurrent",
            "embeddings",
            "output",
            "autres",
        )
    }
    par_couche: dict[int, int] = {}
    experts_par_couche: dict[int, int] = {}
    couches_recurrentes: set[int] = set()
    couches_nextn: set[int] = set()
    couches_kv: set[int] = set()
    # 1er passage : quelles couches sont récurrentes (leurs projections `attn_qkv` /
    # `attn_gate` appartiennent au mécanisme récurrent, pas à un cache KV) et quels
    # blocs sont une tête de prédiction `nextn` (MTP, inutilisée à l'inférence : pas
    # de cache KV bien qu'elle porte une projection K — Ornith 1.5, bloc 40/41).
    for name, _ty, _off in infos:
        layer, part = _layer_of(name)
        if layer is None:
            continue
        if part.startswith(_RECURRENT_PREFIXES):
            couches_recurrentes.add(layer)
        if part.startswith("nextn."):
            couches_nextn.add(layer)
    for name, _ty, _off in infos:
        size = sizes[name]
        layer, part = _layer_of(name)
        if layer is None:
            if name.startswith("token_embd"):
                familles["embeddings"] += size
            elif name.startswith("output"):
                familles["output"] += size
            else:
                familles["autres"] += size
            continue
        par_couche[layer] = par_couche.get(layer, 0) + size
        experts_par_couche.setdefault(layer, 0)
        if "_exps" in part:
            familles["experts"] += size
            experts_par_couche[layer] += size
        elif part.startswith("ffn"):
            familles["ffn"] += size
        elif part.startswith(_RECURRENT_PREFIXES) or (
            layer in couches_recurrentes and part.startswith("attn")
        ):
            familles["recurrent"] += size
        elif part.startswith("attn"):
            familles["attention"] += size
            # Une projection K séparée (ou fusionnée hors récurrence) = un cache KV.
            if part.startswith(("attn_k.", "attn_kv", "attn_qkv")):
                couches_kv.add(layer)
        else:
            familles["autres"] += size
    n = (max(par_couche) + 1) if par_couche else 0
    return {
        "total": sum(sizes.values()),
        "familles": familles,
        "par_couche": [par_couche.get(i, 0) for i in range(n)],
        "experts_par_couche": [experts_par_couche.get(i, 0) for i in range(n)],
        "couches_attention": sorted(couches_kv - couches_recurrentes - couches_nextn),
        "couches_recurrentes": sorted(couches_recurrentes),
        "couches_nextn": sorted(couches_nextn),
        # Sortie liée : sans output.weight, llama.cpp duplique token_embd comme sortie
        # (TENSOR_DUPLICATED). La famille « output » ne le dit pas : output_norm y entre.
        "sortie_liee": "output.weight" not in sizes,
        "provenance": "déduit (catalogue des tenseurs, tailles par offsets)",
    }


def read_gguf_meta(path: str | Path) -> dict:
    """{'architecture','n_layers','context_length','expert_count','expert_used_count',
    champs d'attention ('head_count','head_count_kv','embedding_length','key_length',
    'value_length'), fenêtre glissante ('sliding_window','sliding_window_pattern'),
    'full_attention_interval' (hybrides qwen35*), 'recurrent' (bool), 'arrays'
    (tableaux courts hors tokenizer, par clé complète) et 'weights' (catalogue des
    tenseurs résumé par famille et par couche, None si le fichier n'en liste aucun).
    Les champs absents valent None.

    Lève ValueError si le fichier n'est pas un GGUF lisible — l'appelant traite ça
    en best-effort (un GGUF exotique n'empêche pas l'installation)."""
    kv: dict = {}
    infos: list[tuple[str, int, int]] = []
    data_start = 0
    file_size = 0
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("pas un fichier GGUF")
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                raise ValueError(f"GGUF v{version} non géré")
            tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
            for _ in range(kv_count):
                key = _read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                kv[key] = _read_value(f, vtype, not key.startswith("tokenizer."))
            infos = _read_tensor_infos(f, tensor_count)
            align = int(kv.get("general.alignment") or _DEFAULT_ALIGNMENT)
            data_start = -(-f.tell() // align) * align
            file_size = os.fstat(f.fileno()).st_size
    except (struct.error, MemoryError, OverflowError) as exc:
        # Header tronqué/corrompu = pas un GGUF valide (une longueur sous les bornes mais
        # au-delà du fichier peut encore tenter une allocation démesurée).
        raise ValueError(
            f"header GGUF tronqué ou corrompu ({type(exc).__name__}: {exc})"
        ) from exc

    arch = kv.get("general.architecture")

    def _int(suffix: str) -> int | None:
        v = kv.get(f"{arch}.{suffix}") if arch else None
        return int(v) if isinstance(v, int) else None

    pattern = kv.get(f"{arch}.attention.sliding_window_pattern") if arch else None
    if not isinstance(pattern, (int, list)):
        pattern = None
    split_count = kv.get("split.count")

    return {
        "architecture": arch,
        "n_layers": _int("block_count"),
        "context_length": _int("context_length"),
        "expert_count": _int("expert_count"),
        "expert_used_count": _int("expert_used_count"),
        # `key_length` donne une estimation plus juste du cache KV quand il existe.
        "head_count": _int("attention.head_count"),
        "head_count_kv": _int("attention.head_count_kv"),
        "embedding_length": _int("embedding_length"),
        "key_length": _int("attention.key_length"),
        "value_length": _int("attention.value_length"),
        "sliding_window": _int("attention.sliding_window"),
        "sliding_window_pattern": pattern,
        "full_attention_interval": _int("full_attention_interval"),
        # Dimensions de l'ÉTAT RÉCURRENT (Mamba/GDN `ssm.*`, RWKV `wkv.head_size`, LFM2
        # `shortconv.l_cache`) : chaque checkpoint du serveur pèse cet état complet.
        "ssm_conv_kernel": _int("ssm.conv_kernel"),
        "ssm_inner_size": _int("ssm.inner_size"),
        "ssm_state_size": _int("ssm.state_size"),
        "ssm_group_count": _int("ssm.group_count"),
        "wkv_head_size": _int("wkv.head_size"),
        "shortconv_l_cache": _int("shortconv.l_cache"),
        # Mémoire récurrente (Mamba/GDN `ssm.*`, RWKV `wkv.*`, LFM2 `shortconv.*`) :
        # les checkpoints n'existent qu'aux débuts de messages, un appel annexe sur
        # le slot de la conversation la fait recalculer.
        "recurrent": any(
            k.startswith(tuple(f"{arch}.{p}." for p in ("ssm", "wkv", "shortconv")))
            for k in kv
        )
        if arch
        else False,
        # Clés qui rendent l'estimation INCERTAINE (précontrôle, revue n°16) : un GGUF en
        # plusieurs parties ne liste ici que les tenseurs de la première ; MLA (cache K
        # seul), KV partagé entre couches, têtes SWA dédiées et head_count_kv par
        # couche ne suivent pas la formule uniforme du KV.
        "split_count": int(split_count) if isinstance(split_count, int) else None,
        "key_length_mla": _int("attention.key_length_mla"),
        "kv_lora_rank": _int("attention.kv_lora_rank"),
        "shared_kv_layers": _int("attention.shared_kv_layers"),
        "key_length_swa": _int("attention.key_length_swa"),
        "value_length_swa": _int("attention.value_length_swa"),
        "head_count_kv_array": bool(
            arch and isinstance(kv.get(f"{arch}.attention.head_count_kv"), list)
        ),
        "arrays": {k: v for k, v in kv.items() if isinstance(v, list)},
        "weights": _weights_summary(infos, data_start, file_size),
    }
