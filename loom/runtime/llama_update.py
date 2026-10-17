"""Veille des mises à jour llama.cpp : compare la build du binaire configuré à la build
désignée par la dernière release stable de llama.cpp (celle que `loom-setup` installe) et
suit l'état des PR dont dépend le binaire maison.

Rien n'est jamais installé : le résultat alimente un bandeau. Tant qu'une PR suivie est
ouverte, une mise à jour veut dire RECOMPILER (commande `[server] rebuild_hint`) ; quand
toutes sont mergées, le binaire officiel suffit (`uv run loom-setup`).

Coût réseau : 1 à 3 + len(track_prs) requêtes GitHub anonymes par jour (quota 60/h),
résultat mis en cache. Hors ligne : aucun bandeau, jamais d'exception."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

PR_URL = "https://api.github.com/repos/ggml-org/llama.cpp/pulls/{n}"
TTL_S = 24 * 3600
_HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "loom-update-check"}


def parse_build(version_line: str | None) -> int | None:
    """Numéro de build depuis `llama-server --version` : nouveau format
    « 0.5.0-dev (build 11364, commit …) » ou ancien « version: 9442 (…) »."""
    if not version_line:
        return None
    m = re.search(r"build (\d+)", version_line) or re.search(
        r"version:\s*(\d+)\b", version_line
    )
    return int(m.group(1)) if m else None


def binary_build(server_bin: str) -> int | None:
    """Build OFFICIELLE à comparer. Le numéro llama.cpp compte les commits : un build
    maison compte aussi ses commits de PR (11364 = base 11362 + 2) et masquerait un
    build officiel plus récent. `BUILD.txt` (écrit par le script de recompilation à
    côté du binaire) donne la vraie base : `upstream_build=N`. Sinon : `--version`."""
    from loom.setup import llama_release

    try:
        txt = (Path(server_bin).parent / "BUILD.txt").read_text(encoding="utf-8")
        m = re.search(r"^upstream_build=(\d+)\s*$", txt, re.MULTILINE)
        if m:
            return int(m.group(1))
    except OSError:
        pass
    return parse_build(llama_release.verify_binary(server_bin))


def latest_release(client) -> dict | None:
    """Build `bNNNN` désignée par la dernière release stable de llama.cpp : la MÊME source
    que `loom-setup`, donc le bandeau annonce exactement ce que le setup installe. Les
    nightlies sortent plusieurs fois par jour (bandeau permanent, assets parfois encore en
    cours d'envoi) ; la stable pointe une build complète. Lève RuntimeError hors ligne."""
    from loom.setup import llama_release

    rel = llama_release.fetch_latest_release(client)
    tag = rel.get("tag_name") or ""
    if not re.fullmatch(r"b\d+", tag):
        return None
    return {"tag": tag, "build": int(tag[1:]), "published_at": rel.get("published_at")}


def pr_states(client, prs: list[int]) -> list[dict]:
    out = []
    for n in prs:
        r = client.get(PR_URL.format(n=n), headers=_HEADERS, follow_redirects=True)
        if r.status_code != 200:
            raise RuntimeError(f"GitHub PR #{n} : HTTP {r.status_code}")
        d = r.json()
        out.append(
            {
                "number": d.get("number", n),
                "title": d.get("title", ""),
                "merged": bool(d.get("merged")),
                "state": d.get("state", ""),
                "merged_at": d.get("merged_at"),
                "url": d.get("html_url", ""),
            }
        )
    return out


def build_notice(
    current: int | None, latest: dict | None, prs: list[dict], rebuild_hint: str
) -> dict | None:
    """Bandeau à afficher, ou None. PUR : aucune entrée/sortie."""
    if current is None or latest is None:
        return None
    merged = [p for p in prs if p["merged"]]
    still_open = [p for p in prs if not p["merged"]]
    newer = latest["build"] > current
    if not newer and not merged:
        return None
    parts = []
    if newer:
        parts.append(
            f"llama.cpp {latest['tag']} est disponible (binaire actuel : b{current}, "
            f"{latest['build'] - current} builds d'écart)."
        )
    for p in merged:
        parts.append(f"PR #{p['number']} mergée ({p['title']}).")
    if not still_open:
        kind, command = "official", "uv run loom-setup"
        # « Plus besoin de recompiler » n'a de sens qu'en sortie de build maison.
        parts.append(
            "Le binaire officiel suffit désormais : plus besoin de recompiler."
            if merged
            else "Mise à jour : uv run loom-setup (installe à côté, l'ancien reste)."
        )
    elif newer:
        kind, command = "rebuild", rebuild_hint or ""
        opened = ", ".join(f"#{p['number']}" for p in still_open)
        verb = "encore ouverte" if len(still_open) == 1 else "encore ouvertes"
        parts.append(
            f"PR {opened} {verb} : la mise à jour passe par une recompilation."
        )
    else:
        kind, command = "info", ""
    key = f"{kind}:{latest['tag']}:{current}:" + ",".join(
        str(p["number"]) for p in merged
    )
    return {"kind": kind, "message": " ".join(parts), "command": command, "key": key}


def check(
    server_bin: str,
    track_prs: list[int],
    rebuild_hint: str,
    cache_path: str | Path,
    client=None,
    ttl_s: int = TTL_S,
) -> dict:
    """Résultat de la veille. Les données GitHub en cache servent pendant `ttl_s` ; le
    bandeau, lui, est toujours recalculé avec la build lue AU BINAIRE (recompilé ->
    bandeau à jour sans attendre). Ne lève jamais."""
    cache_path = Path(cache_path)
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - cache absent ou illisible
        cached = {}
    reuse = (
        not cached.get("error")
        and cached.get("latest") is not None
        and cached.get("track_prs") == list(track_prs)
        and time.time() - cached.get("checked_at", 0) < ttl_s
    )
    res = {
        "checked_at": cached["checked_at"] if reuse else time.time(),
        "server_bin": server_bin,
        "track_prs": list(track_prs),
        "current_build": binary_build(server_bin),
        "latest": cached.get("latest") if reuse else None,
        "prs": cached.get("prs", []) if reuse else [],
        "notice": None,
        "error": None,
    }
    if not reuse:
        own = client is None
        if own:
            import httpx

            client = httpx.Client(timeout=20)
        try:
            res["latest"] = latest_release(client)
            res["prs"] = pr_states(client, track_prs)
        except Exception as exc:  # noqa: BLE001 - hors ligne, quota : pas de bandeau
            res["error"] = f"{type(exc).__name__}: {exc}"[:200]
        finally:
            if own:
                client.close()
    if not res["error"]:
        res["notice"] = build_notice(
            res["current_build"], res["latest"], res["prs"], rebuild_hint
        )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    except Exception:  # noqa: BLE001 - cache best-effort
        pass
    return res
