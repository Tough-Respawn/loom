"""Store de plugins compatible Claude Code (marketplaces + install + cache) + CLI.

Loom héberge son PROPRE store, indépendant de ~/.claude : on installe n'importe quel plugin
Claude Code et le modèle local s'en sert hors-ligne. Tranche 1 : store + install + découverte ;
seuls les SKILLS sont consommés (par loom/skills.py). Agents/hooks/commands sont INVENTORIÉS
mais pas encore câblés.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_ROOT = "loom/plugins"


class PluginError(Exception):
    """Erreur métier du store de plugins (message montrable)."""


@dataclass
class Plugin:
    name: str
    marketplace: str
    path: Path  # dossier <version>/ dans le cache
    version: str = "unknown"
    description: str = ""
    skills: list[Path] = field(default_factory=list)
    agents: list[Path] = field(default_factory=list)
    hooks: list[Path] = field(default_factory=list)
    commands: list[Path] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plugins_root(root: str | Path | None = None) -> Path:
    """Racine du store (config ou défaut), créée avec ses sous-dossiers."""
    p = Path(root) if root else Path(_DEFAULT_ROOT)
    (p / "marketplaces").mkdir(parents=True, exist_ok=True)
    (p / "cache").mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path: str | Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_manifest(plugin_dir: Path) -> dict | None:
    """Lit .claude-plugin/plugin.json (None si absent/invalide)."""
    return _read_json(plugin_dir / ".claude-plugin" / "plugin.json", None)


def _scan_components(plugin_dir: Path) -> dict[str, list[Path]]:
    """Inventorie les 4 types de composants d'un plugin matérialisé."""

    def md_glob(sub: str, pattern: str) -> list[Path]:
        d = plugin_dir / sub
        return sorted(d.glob(pattern)) if d.is_dir() else []

    hooks_json = plugin_dir / "hooks" / "hooks.json"
    return {
        "skills": md_glob("skills", "*/SKILL.md"),
        "agents": md_glob("agents", "*.md"),
        "hooks": [hooks_json] if hooks_json.exists() else [],
        "commands": md_glob("commands", "*.md"),
    }


def _plugin_from_dir(name: str, marketplace: str, plugin_dir: Path) -> Plugin | None:
    """Construit un Plugin depuis un dossier matérialisé (None si pas de manifeste)."""
    man = _read_manifest(plugin_dir)
    if man is None:
        return None
    comp = _scan_components(plugin_dir)
    return Plugin(
        name=man.get("name") or name,
        marketplace=marketplace,
        path=plugin_dir,
        version=str(man.get("version") or "unknown"),
        description=(man.get("description") or "").strip(),
        skills=comp["skills"],
        agents=comp["agents"],
        hooks=comp["hooks"],
        commands=comp["commands"],
    )


def discover_plugins(root: str | Path | None = None) -> list[Plugin]:
    """Liste les plugins installés. Lit installed_plugins.json, et en repli scanne cache/
    pour les dossiers déposés à la main (fallback B) absents du JSON."""
    root = plugins_root(root)
    found: dict[str, Plugin] = {}  # clé installPath résolu -> Plugin (dédoublonnage)
    installed = _read_json(root / "installed_plugins.json", {"plugins": {}})
    for ref, entries in (installed.get("plugins") or {}).items():
        name = ref.split("@", 1)[0]
        marketplace = ref.split("@", 1)[1] if "@" in ref else "_local"
        for entry in entries or []:
            path = Path(entry.get("installPath", ""))
            if not path.is_dir():
                continue
            pl = _plugin_from_dir(name, marketplace, path)
            if pl:
                found[str(path.resolve())] = pl
    # Repli : tout cache/<mkt>/<plugin>/<version>/ valide non déjà vu.
    cache = root / "cache"
    if cache.is_dir():
        for mkt in sorted(p for p in cache.iterdir() if p.is_dir()):
            for plug in sorted(p for p in mkt.iterdir() if p.is_dir()):
                for ver in sorted(p for p in plug.iterdir() if p.is_dir()):
                    key = str(ver.resolve())
                    if key in found:
                        continue
                    pl = _plugin_from_dir(plug.name, mkt.name, ver)
                    if pl:
                        found[key] = pl
    return sorted(found.values(), key=lambda p: (p.marketplace, p.name))


def _looks_like_git(source: str) -> bool:
    return source.startswith(("http://", "https://", "git@")) or source.endswith(".git")


def _git_clone(
    url: str, dest: Path, ref: str | None = None, sha: str | None = None
) -> str:
    """Clone shallow `url` dans `dest`, checkout sha/ref si fourni. Renvoie le SHA HEAD.
    Lève PluginError sur échec (git absent, réseau, ref inconnue)."""

    def _run(args: list[str], cwd: Path | None = None) -> str:
        try:
            r = subprocess.run(
                args,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError as exc:
            raise PluginError("git introuvable (installe git)") from exc
        except subprocess.TimeoutExpired as exc:
            raise PluginError("git clone : délai dépassé") from exc
        if r.returncode != 0:
            raise PluginError(f"git a échoué : {(r.stderr or r.stdout)[:200]}")
        return r.stdout.strip()

    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", url, str(dest)])
    target = sha or ref
    if target:
        # un sha précis peut exiger un fetch non-shallow
        _run(["git", "fetch", "--depth", "1", "origin", target], cwd=dest)
        _run(["git", "checkout", target], cwd=dest)
    return _run(["git", "rev-parse", "HEAD"], cwd=dest)
