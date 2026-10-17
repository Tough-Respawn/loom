"""Lecture à distance de l'en-tête d'un GGUF Hugging Face (requêtes HTTP Range), pour
recommander un quant SANS le télécharger : dense ou MoE, part des experts et de
l'embedding dans les poids, nombre de paramètres, types de tenseurs.

Coût mesuré : ~11 Mio et ~3,5 s par en-tête (le vocabulaire du tokenizer y est
stocké). D'où UN SEUL en-tête par dépôt : ses proportions s'appliquent aux autres
quants, dont le format se lit dans le nom de fichier (`quant_label`) et se compare aux
formats déclarés par le llama-quantize installé (`supported_formats`)."""

from __future__ import annotations

import functools
import re
import struct
import subprocess
import urllib.request
from pathlib import Path

# Énumération ggml_type de llama.cpp (ggml.h, b11669). Les ids retirés de ggml ne sont
# plus chargeables ; un id inconnu (ex. 142, PQ2_0 du fork Prism) non plus.
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}  # fmt: skip
FULL_PRECISION = {"F32", "F16", "BF16", "F64"}

_HF_URL = "https://huggingface.co/{repo}/resolve/main/{filename}"
_CHUNK = 4 << 20
_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _http_fetch(url: str):
    """fetch(start, end) -> (octets, taille totale du fichier) par requête Range."""

    def fetch(start: int, end: int) -> tuple[bytes, int | None]:
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "loom"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            total = (r.headers.get("Content-Range") or "").rpartition("/")[2]
            return r.read(), int(total) if total.isdigit() else None

    return fetch


class _Reader:
    def __init__(self, fetch):
        self.fetch, self.buf, self.pos, self.total = fetch, b"", 0, None

    def read(self, n: int) -> bytes:
        while len(self.buf) < self.pos + n:
            start = len(self.buf)
            data, total = self.fetch(start, start + max(_CHUNK, n) - 1)
            if not data:
                raise EOFError("en-tête GGUF tronqué")
            self.buf += data
            self.total = self.total or total
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", "replace")

    def value(self, t: int):
        if t == 8:
            return self.string()
        if t == 9:
            it, n = self.u32(), self.u64()
            if it in (8, 9):  # chaînes ou tableaux imbriqués : longueur variable
                for _ in range(n):
                    self.value(it)
            else:
                self.read(_SCALAR[it] * n)
            return None  # tableaux (tokenizer) : sautés, inutiles ici
        raw = self.read(_SCALAR[t])
        if t in (4, 5):
            return struct.unpack("<I" if t == 4 else "<i", raw)[0]
        if t in (10, 11):
            return struct.unpack("<Q" if t == 10 else "<q", raw)[0]
        return raw


def read_header(fetch) -> dict:
    """Métadonnées utiles + tenseurs (nom, id de type, nb d'éléments, octets) d'un GGUF.
    Les octets d'un tenseur = écart entre offsets consécutifs (le dernier : fin du
    fichier), donc exacts quel que soit le type, même inconnu."""
    r = _Reader(fetch)
    if r.read(4) != b"GGUF":
        raise ValueError("pas un fichier GGUF")
    r.u32()  # version
    n_tensors, n_kv = r.u64(), r.u64()
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        meta[key] = r.value(r.u32())
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        dims = [r.u64() for _ in range(r.u32())]
        ttype, offset = r.u32(), r.u64()
        n_elem = 1
        for d in dims:
            n_elem *= d
        infos.append([name, ttype, n_elem, offset])
    align = meta.get("general.alignment") or 32
    data_start = -(-r.pos // align) * align
    infos.sort(key=lambda t: t[3])
    tensors = []
    for i, (name, ttype, n_elem, offset) in enumerate(infos):
        end = infos[i + 1][3] if i + 1 < len(infos) else (r.total or 0) - data_start
        tensors.append((name, ttype, n_elem, max(0, end - offset)))
    arch = meta.get("general.architecture") or ""
    return {
        "arch": arch,
        "expert_count": meta.get(f"{arch}.expert_count") or 0,
        "expert_used": meta.get(f"{arch}.expert_used_count") or 0,
        "tensors": tensors,
    }


def summarize(header: dict) -> dict:
    """Proportions qui gouvernent la vitesse : part des experts (`f_exp`, MoE) et de
    l'embedding d'entrée (`f_emb`, lu une ligne par token, sauf s'il sert aussi de
    sortie quand `output.weight` est absent)."""
    tensors = header["tensors"]
    total = sum(t[3] for t in tensors) or 1
    exp = sum(t[3] for t in tensors if "_exps" in t[0])
    has_output = any(t[0] == "output.weight" for t in tensors)
    emb = sum(t[3] for t in tensors if t[0] == "token_embd.weight") if has_output else 0
    unknown = sorted({t[1] for t in tensors if t[1] not in GGML_TYPES})
    return {
        "moe": header["expert_count"] > 1 and exp > 0,
        "expert_count": header["expert_count"],
        "expert_used": header["expert_used"],
        "f_exp": exp / total,
        "f_emb": emb / total,
        "n_params": sum(t[2] for t in tensors),
        "unknown_types": unknown,
    }


_LABEL_RE = re.compile(
    r"(?<![A-Z0-9])((?:UD-)?(?:I?Q\d+(?:_[A-Z0-9]+)*|P?T?Q\d+_\d+|MXFP4(?:_MOE)?|"
    r"NVFP4|BF16|F16|F32))(?![A-Z0-9])",
    re.IGNORECASE,
)


def quant_label(filename: str) -> str | None:
    """Format annoncé par le nom de fichier (ex. « Q4_K_M », « UD-Q4_K_XL », « PQ2_0 »),
    None si absent. Le dernier motif gagne (le nom du modèle peut contenir des chiffres)."""
    name = Path(filename).name.rsplit(".gguf", 1)[0]
    name = re.sub(r"-\d{5}-of-\d{5}$", "", name)
    found = _LABEL_RE.findall(name)
    return found[-1].upper().removeprefix("UD-") if found else None


def label_supported(label: str, supported: set[str]) -> bool:
    """Un libellé composé (Q4_K_XL, IQ4_NL_XL, Q4_K_L…) est supporté si sa base l'est."""
    if label in supported:
        return True
    base = re.sub(r"_(XXL|XL|L|M|S)$", "", label)  # Q6_K_M d'unsloth = types Q6_K
    return base in supported


@functools.lru_cache(maxsize=4)
def supported_formats(server_bin: str | None) -> frozenset[str] | None:
    """Formats que le llama.cpp INSTALLÉ connaît : ceux déclarés par son llama-quantize
    (même dossier que llama-server), plus les noms de types ggml. None si introuvable :
    on ne filtre alors pas (le test de chargement de fin d'installation reste)."""
    if not server_bin:
        return None
    folder = Path(server_bin).parent
    exe = next(
        (p for p in folder.glob("llama-quantize*") if p.suffix in ("", ".exe")), None
    )
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [str(exe), "--help"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or "") + (out.stderr or "")
    names = set(re.findall(r"^\s*\d+\s+or\s+([A-Z0-9_]+)\s*:", text, re.MULTILINE))
    return frozenset(names | set(GGML_TYPES.values())) if names else None


def analyze_repo(
    repo: str, files: list[dict], supported=None, fetch_for=None
) -> dict | None:
    """Lit l'en-tête d'UN quant du dépôt (non découpé de préférence, au format supporté)
    et rend ses proportions + le fichier lu. None si la lecture échoue (hors ligne) :
    la recommandation retombe alors sur la taille seule."""
    fetch_for = fetch_for or (
        lambda fn: _http_fetch(_HF_URL.format(repo=repo, filename=fn))
    )

    def score(f):
        label = quant_label(f["filename"]) or ""
        ok = not supported or not label or label_supported(label, supported)
        return (len(f.get("part_files") or [f["filename"]]) > 1, not ok, f["size_mb"])

    if not files:
        return None
    pick = min(files, key=score)
    try:
        info = summarize(read_header(fetch_for(pick["filename"])))
    except Exception:  # noqa: BLE001 - réseau, format : recommandation dégradée, jamais d'échec
        return None
    if (
        info["n_params"] < 100_000_000
    ):  # pas un modèle (fichier auxiliaire passé au filtre)
        return None
    info["file"] = pick["filename"]
    info["split"] = len(pick.get("part_files") or [pick["filename"]]) > 1
    return info
