"""Banc RÉEL du mode décision (loom/agent/decide.py) contre le JSON généré.

Lance un llama-server ÉPHÉMÈRE avec les flags exacts de Loom (build_server_args, ngl
résolu comme serve.py sur la VRAM libre du moment), puis, sur des tickets support :
  - décision par logits : 12 champs, une lecture par champ, préfixe en cache ;
  - même schéma en /v1/chat/completions avec response_format json_schema (grammaire).
Mesure : latence (froid = 1er ticket, chaud = suivants), accord champ à champ, justesse
sur les champs annotés. Le serveur est tué à la fin.

  uv run python evals/bench_decision.py E:/loom-models/local/text/gemma4-e4b-heretic
  uv run python evals/bench_decision.py C:/loom-models/local/text/ornith-1.5-35b-a3b --no-op-offload
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loom.agent.decide import Decider, Field, render_prompt
from loom.runtime.hardware import detect_hardware, recommend_gpu_layers
from loom.runtime.server_args import build_server_args


def _server_bin() -> str:
    """Binaire llama-server de Loom : [server] bin de config/local.toml, sinon defaults."""
    root = Path(__file__).resolve().parents[1]
    for name in ("local.toml", "defaults.toml"):
        p = root / "config" / name
        if p.exists():
            b = (
                tomllib.loads(p.read_text(encoding="utf-8"))
                .get("server", {})
                .get("bin")
            )
            if b:
                return b
    return "llama-server"


SERVER_BIN = _server_bin()

INSTRUCTIONS = "You triage customer support tickets for an online electronics store."

SCHEMA = [
    Field.enum(
        "category",
        ["billing", "technical", "shipping", "cancellation", "other"],
        "What type of support request is this?",
    ),
    Field.boolean("urgent", "Does this need handling within 24 hours?"),
    Field.enum("priority", ["low", "medium", "high", "critical"], "Support priority."),
    Field.enum("sentiment", ["angry", "neutral", "happy"], "Customer's tone."),
    Field.boolean("needs_human", "Must a human agent see this (not a bot reply)?"),
    Field.enum("language", ["en", "fr", "de", "es"], "Language of the message."),
    Field.boolean("refund_requested", "Does the customer ask for money back?"),
    Field.enum(
        "product_area",
        ["laptop", "phone", "audio", "accessory", "unknown"],
        "Product concerned.",
    ),
    Field.boolean("spam", "Is this spam or an unsolicited sales pitch?"),
    Field.boolean("contains_order_number", "Does the message contain an order number?"),
    Field.enum("reply_channel", ["email", "phone", "none"], "Best channel to reply."),
    Field.integer("complexity", 1, 5, "Effort to resolve, 1 (trivial) to 5 (hard)."),
]

# (message, vérité terrain sur les champs NON ambigus)
TICKETS = [
    (
        "I was charged twice for order #A1289 and I need this fixed today, this is "
        "unacceptable.",
        {
            "category": "billing",
            "urgent": True,
            "language": "en",
            "spam": False,
            "contains_order_number": True,
            "refund_requested": True,
        },
    ),
    (
        "Bonjour, mon casque bluetooth ne s'allume plus après la mise à jour. Que faire ?",
        {
            "category": "technical",
            "language": "fr",
            "spam": False,
            "contains_order_number": False,
            "product_area": "audio",
        },
    ),
    (
        "Hi! Boost your store's SEO with our premium backlink package, 50% off this week.",
        {"spam": True, "language": "en", "needs_human": False},
    ),
    (
        "Wo ist mein Paket? Bestellung 77-4410, seit 3 Wochen keine Lieferung.",
        {
            "category": "shipping",
            "language": "de",
            "contains_order_number": True,
            "spam": False,
        },
    ),
    (
        "Please cancel my subscription to the phone protection plan, I no longer own "
        "the phone. Thanks a lot for the great service so far!",
        {
            "category": "cancellation",
            "language": "en",
            "sentiment": "happy",
            "product_area": "phone",
            "spam": False,
        },
    ),
    (
        "Quiero devolver el portátil y que me devuelvan el dinero, la pantalla llegó rota.",
        {
            "language": "es",
            "refund_requested": True,
            "product_area": "laptop",
            "spam": False,
        },
    ),
]


def json_schema() -> dict:
    props = {}
    for f in SCHEMA:
        if f.type == "boolean":
            props[f.name] = {"type": "boolean"}
        elif f.type == "integer":
            props[f.name] = {"type": "integer", "minimum": 1, "maximum": 5}
        else:
            props[f.name] = {"type": "string", "enum": f.choices}
    return {
        "type": "object",
        "properties": props,
        "required": [f.name for f in SCHEMA],
        "additionalProperties": False,
    }


def start_server(
    model_dir: Path, port: int, extra: list[str], ngl_override: int | None
):
    cfg = tomllib.loads((model_dir / "model.toml").read_text(encoding="utf-8"))
    hw = detect_hardware(SERVER_BIN)
    if cfg.get("cpu_moe") or cfg.get("n_cpu_moe") is not None:
        ngl = 999
    elif cfg.get("n_gpu_layers") is not None:
        ngl = int(cfg["n_gpu_layers"])
    else:
        ngl = recommend_gpu_layers(hw.vram_free_mb, cfg["size_mb"], cfg["n_layers"])
    if ngl_override is not None:
        ngl = ngl_override
    args = (
        build_server_args(
            server_bin=SERVER_BIN,
            model_path=str(model_dir / cfg["filename"]),
            port=port,
            context=int(cfg.get("context") or 8192),
            n_gpu_layers=ngl,
            threads=max(1, hw.cpu_threads // 2) if hw.has_gpu else hw.cpu_threads,
            gpu_tuning=hw.has_gpu,
            n_parallel=1,
            cpu_moe=bool(cfg.get("cpu_moe")),
            n_cpu_moe=cfg.get("n_cpu_moe"),
        )
        + extra
    )
    print(f"[bench] VRAM libre {hw.vram_free_mb} MiB, ngl={ngl}")
    print("[bench]", " ".join(args))
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 900:
        try:
            if (
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code
                == 200
            ):
                print(f"[bench] serveur prêt en {time.monotonic() - t0:.0f} s")
                return proc
        except Exception:
            time.sleep(3)
        if proc.poll() is not None:
            raise SystemExit("[bench] llama-server est mort au chargement")
    proc.kill()
    raise SystemExit("[bench] health timeout")


def json_baseline(
    http: httpx.Client, base: str, context: str
) -> tuple[dict, float, dict]:
    body = {
        "messages": [
            {"role": "user", "content": render_prompt(INSTRUCTIONS, SCHEMA, context)}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "triage", "schema": json_schema()},
        },
        "temperature": 0.0,
        "max_tokens": 400,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    r = http.post(f"{base}/v1/chat/completions", json=body)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    d = r.json()
    txt = d["choices"][0]["message"]["content"] or ""
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        obj = {"_invalid": txt[:200]}
    return obj, ms, d.get("timings") or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--ngl", type=int, default=None)
    ap.add_argument("--no-op-offload", action="store_true")
    ap.add_argument(
        "--min-batch", type=int, default=None, help="GGML_OP_OFFLOAD_MIN_BATCH"
    )
    ap.add_argument(
        "--rounds", type=int, default=2, help="passes sur les tickets (la 1re = froid)"
    )
    a = ap.parse_args()
    if a.min_batch is not None:
        os.environ["GGML_OP_OFFLOAD_MIN_BATCH"] = str(a.min_batch)
    extra = ["--no-op-offload"] if a.no_op_offload else []
    proc = start_server(Path(a.model_dir), a.port, extra, a.ngl)
    base = f"http://127.0.0.1:{a.port}"
    dec = Decider(base, id_slot=0)
    http = httpx.Client(timeout=600)
    rows = []
    try:
        for rnd in range(a.rounds):
            for i, (msg, truth) in enumerate(TICKETS):
                d = dec.decide(INSTRUCTIONS, SCHEMA, msg)
                j, jms, jt = json_baseline(http, base, msg)
                agree = sum(1 for f in SCHEMA if j.get(f.name) == d.values[f.name])
                ok_d = sum(1 for k, v in truth.items() if d.values[k] == v)
                ok_j = sum(1 for k, v in truth.items() if j.get(k) == v)
                rows.append(
                    {
                        "round": rnd,
                        "ticket": i,
                        "dec_ms": d.ms,
                        "json_ms": jms,
                        "dec_req": d.requests,
                        "dec_prompt_tok": d.prompt_tokens,
                        "dec_server_ms": d.server_ms,
                        "dec_tokenize_ms": d.tokenize_ms,
                        "json_gen_tok": jt.get("predicted_n"),
                        "agree": agree,
                        "truth_n": len(truth),
                        "ok_dec": ok_d,
                        "ok_json": ok_j,
                        "dec": d.values,
                        "json": j,
                        "probs": {
                            k: round(max(v.values()), 2) for k, v in d.probs.items()
                        },
                        "coverage": {k: round(v, 2) for k, v in d.coverage.items()},
                    }
                )
                r = rows[-1]
                print(
                    f"r{rnd} t{i}: décision {d.ms:7.0f} ms (serveur {d.server_ms:.0f}, "
                    f"tokenize {d.tokenize_ms:.0f}, {d.requests} req, "
                    f"{d.prompt_tokens} tok prompt) | JSON {jms:7.0f} ms "
                    f"({jt.get('predicted_n')} tok gen) | accord {agree}/{len(SCHEMA)} | "
                    f"juste décision {ok_d}/{len(truth)} JSON {ok_j}/{len(truth)}"
                )
                print("      décision:", json.dumps(d.values, ensure_ascii=False))
                print("      p_max   :", json.dumps(r["probs"]))
                print("      couvert :", json.dumps(r["coverage"]))
                print("      JSON    :", json.dumps(j, ensure_ascii=False))
    finally:
        dec.close()
        proc.kill()
    warm = [r for r in rows if r["round"] > 0] or rows
    print("\n=== RÉSUMÉ ===")
    print(
        f"froid (r0 t0)  : décision {rows[0]['dec_ms']:.0f} ms | JSON {rows[0]['json_ms']:.0f} ms"
    )
    print(
        f"chaud (médiane): décision {statistics.median(r['dec_ms'] for r in warm):.0f} ms | "
        f"JSON {statistics.median(r['json_ms'] for r in warm):.0f} ms"
    )
    print(
        f"justesse (champs annotés, tous rounds) : décision "
        f"{sum(r['ok_dec'] for r in rows)}/{sum(r['truth_n'] for r in rows)} | JSON "
        f"{sum(r['ok_json'] for r in rows)}/{sum(r['truth_n'] for r in rows)}"
    )
    print(
        f"accord décision/JSON : {sum(r['agree'] for r in rows)}/{len(rows) * len(SCHEMA)}"
    )
    out = (
        Path("var") / f"bench_decision_{Path(a.model_dir).name}_{int(time.time())}.json"
    )
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[bench] détail :", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
