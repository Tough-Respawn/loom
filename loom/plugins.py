"""Store de plugins compatible Claude Code (marketplaces + install + cache) + CLI.

Loom héberge son PROPRE store, indépendant de ~/.claude : on installe n'importe quel plugin
Claude Code et le modèle local s'en sert hors-ligne. Tranche 1 : store + install + découverte ;
seuls les SKILLS sont consommés (par loom/skills.py). Agents/hooks/commands sont INVENTORIÉS
mais pas encore câblés.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
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

    # url/ref viennent d'un marketplace.json TIERS : on refuse toute valeur commençant par
    # '-' (smuggling de flag git) et on isole l'argument positionnel par '--'. Env durci :
    # pas de prompt interactif, pas de config système (clone non interactif, reproductible).
    def _safe(value: str, what: str) -> str:
        if value.startswith("-"):
            raise PluginError(f"{what} invalide (commence par '-') : {value[:60]}")
        return value

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"}

    def _run(args: list[str], cwd: Path | None = None) -> str:
        try:
            r = subprocess.run(
                args,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
        except FileNotFoundError as exc:
            raise PluginError("git introuvable (installe git)") from exc
        except subprocess.TimeoutExpired as exc:
            raise PluginError("git clone : délai dépassé") from exc
        if r.returncode != 0:
            raise PluginError(f"git a échoué : {(r.stderr or r.stdout).strip()[-200:]}")
        return r.stdout.strip()

    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "--", _safe(url, "url"), str(dest)])
    target = sha or ref
    if target:
        target = _safe(target, "ref")
        # un sha précis peut exiger un fetch non-shallow
        _run(["git", "fetch", "--depth", "1", "origin", target], cwd=dest)
        _run(["git", "checkout", target, "--"], cwd=dest)
    return _run(["git", "rev-parse", "HEAD"], cwd=dest)


# ---------------------------------------------------------------------------
# Task 2 : marketplace add/list, plugin install/add-local/remove
# ---------------------------------------------------------------------------


def marketplace_add(root: str | Path | None, source: str) -> str:
    """Ajoute une marketplace (git clone ou copie locale) sous marketplaces/<name>.
    Renvoie son nom. Lève PluginError sans .claude-plugin/marketplace.json."""
    root = plugins_root(root)
    tmp = Path(tempfile.mkdtemp(prefix="loom-mkt-"))
    try:
        if _looks_like_git(source):
            _git_clone(source, tmp / "repo")
            repo = tmp / "repo"
        else:
            repo = Path(source).expanduser().resolve()
            if not repo.is_dir():
                raise PluginError(f"chemin introuvable : {source}")
        data = _read_json(repo / ".claude-plugin" / "marketplace.json", None)
        if not isinstance(data, dict) or not data.get("name"):
            raise PluginError("source sans .claude-plugin/marketplace.json valide")
        name = str(data["name"])
        dest = root / "marketplaces" / name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(repo, dest)
        known = _read_json(
            root / "known_marketplaces.json", {"version": 1, "marketplaces": {}}
        )
        known.setdefault("marketplaces", {})[name] = {
            "source": source,
            "addedAt": _now_iso(),
        }
        _write_json(root / "known_marketplaces.json", known)
        return name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def marketplace_list(root: str | Path | None = None) -> list[dict]:
    root = plugins_root(root)
    known = _read_json(root / "known_marketplaces.json", {"marketplaces": {}})
    return [{"name": n, **v} for n, v in (known.get("marketplaces") or {}).items()]


def _find_entry(root: Path, plugin: str, marketplace: str | None) -> tuple[str, dict]:
    """Trouve (marketplace, entrée plugin) dans les marketplace.json. Lève si introuvable."""
    mkts = root / "marketplaces"
    names = (
        [marketplace]
        if marketplace
        else [d.name for d in mkts.iterdir() if d.is_dir()]
        if mkts.is_dir()
        else []
    )
    for mname in names:
        data = _read_json(mkts / mname / ".claude-plugin" / "marketplace.json", None)
        for entry in (data or {}).get("plugins", []):
            if entry.get("name") == plugin:
                return mname, entry
    raise PluginError(
        f"plugin '{plugin}' introuvable dans les marketplaces installées "
        f"(ajoute d'abord la marketplace via marketplace_add)"
    )


def _resolve_within(base: Path, rel: str, what: str) -> Path:
    """Résout `rel` SOUS `base` et refuse toute échappée (..). `source`/`path` viennent d'un
    marketplace.json TIERS : sans ce confinement, '../../windows' copierait un dossier hors
    de la marketplace dans le cache (lecture de fichiers arbitraires)."""
    base = base.resolve()
    target = (base / rel).resolve()
    if target != base and not str(target).startswith(str(base) + os.sep):
        raise PluginError(f"{what} sort du dossier autorisé : {rel!r}")
    return target


def _materialize(root: Path, mname: str, entry: dict, staging: Path) -> str:
    """Matérialise la source d'un plugin dans `staging`. Renvoie le sha git ('' si local)."""
    source = entry.get("source")
    if isinstance(source, str):  # chemin relatif au dépôt de la marketplace
        src = _resolve_within(root / "marketplaces" / mname, source, "source")
        if not src.is_dir():
            raise PluginError(f"source locale introuvable : {source}")
        shutil.copytree(src, staging)
        return ""
    if isinstance(source, dict):
        kind = source.get("source")
        url = source.get("url")
        if kind == "git":
            return _git_clone(url, staging, ref=source.get("ref"))
        if kind == "git-subdir":
            tmp = Path(tempfile.mkdtemp(prefix="loom-sub-"))
            try:
                sha = _git_clone(
                    url, tmp / "repo", ref=source.get("ref"), sha=source.get("sha")
                )
                sub = _resolve_within(tmp / "repo", source.get("path") or "", "path")
                if not sub.is_dir():
                    raise PluginError(f"sous-dossier absent : {source.get('path')}")
                shutil.copytree(sub, staging)
                return sha
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    raise PluginError(f"forme de source non gérée : {source!r}")


def _register_installed(root: Path, ref_key: str, dest: Path, version: str, sha: str):
    data = _read_json(root / "installed_plugins.json", {"version": 2, "plugins": {}})
    data.setdefault("plugins", {})[ref_key] = [
        {
            "scope": "user",
            "installPath": str(dest.resolve()),
            "version": version,
            "installedAt": _now_iso(),
            "lastUpdated": _now_iso(),
            "gitCommitSha": sha,
        }
    ]
    _write_json(root / "installed_plugins.json", data)


def plugin_install(root: str | Path | None, ref: str) -> dict:
    """Installe '<plugin>' ou '<plugin>@<marketplace>' : résout la source, matérialise sous
    cache/<mkt>/<plugin>/<version>/, enregistre. Renvoie {name, marketplace, version, path}."""
    root = plugins_root(root)
    plugin, _, marketplace = ref.partition("@")
    plugin = plugin.strip()
    marketplace = marketplace.strip() or None
    mname, entry = _find_entry(root, plugin, marketplace)
    staging = Path(tempfile.mkdtemp(prefix="loom-plug-"))
    try:
        sha = _materialize(root, mname, entry, staging / "p")
        man = _read_manifest(staging / "p")
        if man is None:
            raise PluginError("plugin sans .claude-plugin/plugin.json")
        version = str(man.get("version") or "unknown")
        dest = root / "cache" / mname / plugin / version
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging / "p"), str(dest))
        _register_installed(root, f"{plugin}@{mname}", dest, version, sha)
        return {
            "name": plugin,
            "marketplace": mname,
            "version": version,
            "path": str(dest),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def plugin_add_local(root: str | Path | None, path: str) -> dict:
    """Fallback B : copie un dossier plugin (avec .claude-plugin/plugin.json) sous
    cache/_local/<plugin>/<version>/ et l'enregistre."""
    root = plugins_root(root)
    src = Path(path).expanduser().resolve()
    man = _read_manifest(src)
    if man is None:
        raise PluginError(f"pas de .claude-plugin/plugin.json dans {path}")
    name = man.get("name") or src.name
    version = str(man.get("version") or "unknown")
    dest = root / "cache" / "_local" / name / version
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    _register_installed(root, f"{name}@_local", dest, version, "")
    return {
        "name": name,
        "marketplace": "_local",
        "version": version,
        "path": str(dest),
    }


def plugin_remove(root: str | Path | None, ref: str) -> None:
    """Retire un plugin (cache + installed_plugins.json). ref = '<plugin>' ou '<plugin>@<mkt>'."""
    root = plugins_root(root)
    data = _read_json(root / "installed_plugins.json", {"version": 2, "plugins": {}})
    plugins = data.get("plugins") or {}
    keys = [k for k in plugins if k == ref or k.split("@", 1)[0] == ref]
    if not keys:
        raise PluginError(f"plugin non installé : {ref}")
    for k in keys:
        for entry in plugins[k]:
            shutil.rmtree(Path(entry.get("installPath", "")), ignore_errors=True)
        del plugins[k]
    _write_json(root / "installed_plugins.json", data)
