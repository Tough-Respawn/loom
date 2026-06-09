# Store de plugins compatible Claude Code (Tranche 1) — Plan d'implémentation

> **Pour les agents :** SOUS-SKILL REQUIS — utilise superpowers:subagent-driven-development
> (recommandé) ou superpowers:executing-plans pour exécuter ce plan tâche par tâche. Les
> étapes utilisent des cases `- [ ]`.

**Goal :** Loom héberge son propre store de plugins au format Claude Code (marketplaces +
install + cache), installe n'importe quel plugin CC, et expose ses **skills** au modèle local
par déclenchement automatique (`use_skill`).

**Architecture :** un module `loom/plugins.py` (store + install + découverte + CLI) ; une
refonte de `loom/skills.py` (catalogue + chargement à la demande) ; deux modules de tools
(`use_skill`, install/marketplace) ; câblage web (catalogue dans le prompt, suppression de
l'activation manuelle). Tranche 1 ne consomme que les skills ; agents/hooks/commands sont
inventoriés.

**Tech Stack :** Python 3.13, `uv`, ruff, git (subprocess), Flask. Pas de pytest (smokes
`uv run python -c`). Spec : `docs/superpowers/specs/2026-06-09-loom-plugins-store-cc.md`.

**Contraintes projet :** branche `feat/harness-reflexion` · `uvx ruff` (pas `uv run ruff`) ·
hook autoflake retire les imports non utilisés (écrire import+usage ensemble) · commits
courts Conventional Commits, finir par `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## Structure des fichiers

| Fichier | Rôle | Action |
|---|---|---|
| `loom/plugins.py` | Store CC (root/JSON), `marketplace_add`, `plugin_install`, `plugin_add_local`, `plugin_remove`, `discover_plugins`, `_git_clone`, dataclass `Plugin`, CLI `main()` | Créer |
| `loom/plugins/.gitignore` | Ignorer le contenu installé (`marketplaces/`, `cache/`, `*.json`) | Créer |
| `loom/skills.py` | `Skill.base_dir`, `collect_skills`, `render_catalog`, `load_skill_body` ; suppr. `compose_system_prompt`/`load_skill`/`list_skills` | Refondre |
| `loom/tools/skills.py` | `make_use_skill(skills_provider)` | Créer |
| `loom/tools/plugins.py` | `make_add_marketplace`, `make_install_plugin`, `make_list_plugins` | Créer |
| `loom/permissions.py` | `use_skill`/`list_plugins` → READ_TOOLS ; `PLUGIN_TOOLS` gardé comme WRITE | Modifier |
| `loom/tools/base.py` | 4 nouveaux tools dans `AVAILABLE_TOOLS` | Modifier |
| `loom/tools/__init__.py` | `build_registry` enregistre les 4 tools (params `skills_dir`, `plugins_root`) | Modifier |
| `loom/conversation.py` | Retirer `active_skills`/`set_skills` | Modifier |
| `loom/web/app.py` | Catalogue dans le prompt ; suppr. route `/skills`, `active_skills`, imports | Modifier |
| `loom/web/__main__.py` | Calcule `plugins_root`, le passe à `create_app` + `tool_factory` | Modifier |
| `loom/web/templates/_skills.html` | Liste lecture seule, groupée par source | Modifier |
| `loom/loom.config.toml` | Ajoute les 4 tools dans `tools_enabled` ; section `[plugins]` | Modifier |
| `loom/skills/exemple/` | Supprimer (placeholder inutile) | Supprimer |

---

## Task 1 : `loom/plugins.py` — store, découverte, git (sans install)

**Files:**
- Create: `loom/plugins.py`
- Create: `loom/plugins/.gitignore`

- [ ] **Step 1 : Écrire le squelette du module (root, JSON, dataclass, scan, découverte, git)**

```python
# loom/plugins.py
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
```

- [ ] **Step 2 : Écrire `loom/plugins/.gitignore`**

```
# Contenu installé (tiers) : pas versionné. Le dossier existe pour le store.
marketplaces/
cache/
*.json
!.gitignore
```

- [ ] **Step 3 : Smoke — découverte sur une fixture cache déposée à la main (fallback B)**

Run :
```bash
uv run python -c "
import tempfile, json, os
from pathlib import Path
from loom.plugins import plugins_root, discover_plugins
root = Path(tempfile.mkdtemp())
pv = root/'cache'/'_local'/'demo'/'1.0.0'
(pv/'.claude-plugin').mkdir(parents=True)
(pv/'.claude-plugin'/'plugin.json').write_text(json.dumps({'name':'demo','version':'1.0.0','description':'d'}))
(pv/'skills'/'hello').mkdir(parents=True)
(pv/'skills'/'hello'/'SKILL.md').write_text('---\nname: hello\ndescription: salut\n---\ncorps')
plugins_root(root)
pl = discover_plugins(root)
assert len(pl)==1 and pl[0].name=='demo' and len(pl[0].skills)==1, pl
print('TASK1_OK', pl[0].name, len(pl[0].skills))
"
```
Expected : `TASK1_OK demo 1`

- [ ] **Step 4 : Lint + commit**

```bash
uvx ruff check loom/plugins.py && uvx ruff format loom/plugins.py
git add loom/plugins.py loom/plugins/.gitignore
git commit -m "feat(plugins): store CC (decouverte + git clone), tranche 1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2 : `loom/plugins.py` — install (marketplace, plugin, local)

**Files:**
- Modify: `loom/plugins.py` (ajouter les fonctions d'install, après `_git_clone`)

- [ ] **Step 1 : Ajouter `marketplace_add`, `marketplace_list`, install/local/remove**

```python
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
    return [
        {"name": n, **v} for n, v in (known.get("marketplaces") or {}).items()
    ]


def _find_entry(
    root: Path, plugin: str, marketplace: str | None
) -> tuple[str, dict]:
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


def _materialize(root: Path, mname: str, entry: dict, staging: Path) -> str:
    """Matérialise la source d'un plugin dans `staging`. Renvoie le sha git ('' si local)."""
    source = entry.get("source")
    if isinstance(source, str):  # chemin relatif au dépôt de la marketplace
        src = (root / "marketplaces" / mname / source).resolve()
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
                sub = (tmp / "repo" / (source.get("path") or "")).resolve()
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
    return {"name": name, "marketplace": "_local", "version": version, "path": str(dest)}


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
```

- [ ] **Step 2 : Smoke — fausse marketplace locale (source relative), install, découverte**

Run :
```bash
uv run python -c "
import tempfile, json
from pathlib import Path
from loom.plugins import marketplace_add, plugin_install, discover_plugins, plugin_add_local
store = Path(tempfile.mkdtemp())
# fausse marketplace locale
mkt = Path(tempfile.mkdtemp())
(mkt/'.claude-plugin').mkdir()
(mkt/'.claude-plugin'/'marketplace.json').write_text(json.dumps(
  {'name':'mymkt','plugins':[{'name':'demo','source':'./plugins/demo'}]}))
pd = mkt/'plugins'/'demo'
(pd/'.claude-plugin').mkdir(parents=True)
(pd/'.claude-plugin'/'plugin.json').write_text(json.dumps({'name':'demo','version':'2.1.0'}))
(pd/'skills'/'hello').mkdir(parents=True)
(pd/'skills'/'hello'/'SKILL.md').write_text('---\nname: hello\ndescription: salut\n---\ncorps')
name = marketplace_add(store, str(mkt)); assert name=='mymkt', name
info = plugin_install(store, 'demo'); assert info['version']=='2.1.0', info
assert (store/'cache'/'mymkt'/'demo'/'2.1.0'/'.claude-plugin'/'plugin.json').exists()
assert (store/'installed_plugins.json').exists()
pl = discover_plugins(store); assert len(pl)==1 and len(pl[0].skills)==1, pl
print('TASK2_OK', info['name'], info['version'], len(pl[0].skills))
"
```
Expected : `TASK2_OK demo 2.1.0 1`

- [ ] **Step 3 : Lint + commit**

```bash
uvx ruff check loom/plugins.py && uvx ruff format loom/plugins.py
git add loom/plugins.py && git commit -m "feat(plugins): install (marketplace/git/git-subdir/local) + remove

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3 : CLI `python -m loom.plugins`

**Files:**
- Modify: `loom/plugins.py` (ajouter `main()` + garde `__main__` à la fin)

- [ ] **Step 1 : Ajouter la CLI argparse**

```python
def _fmt_plugin(p: Plugin) -> str:
    return (
        f"{p.marketplace}/{p.name} ({p.version}) — "
        f"skills:{len(p.skills)} agents:{len(p.agents)} "
        f"hooks:{len(p.hooks)} commands:{len(p.commands)}"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m loom.plugins")
    ap.add_argument("--root", default=None, help="racine du store (défaut loom/plugins)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("marketplace").add_subparsers(dest="mcmd", required=True)
    mk.add_parser("add").add_argument("source")
    mk.add_parser("list")
    sub.add_parser("install").add_argument("ref")
    sub.add_parser("add-local").add_argument("path")
    sub.add_parser("remove").add_argument("ref")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "marketplace" and args.mcmd == "add":
            print("ajoutée :", marketplace_add(args.root, args.source))
        elif args.cmd == "marketplace" and args.mcmd == "list":
            for m in marketplace_list(args.root):
                print(f"{m['name']} <- {m.get('source', '')}")
        elif args.cmd == "install":
            info = plugin_install(args.root, args.ref)
            print(f"installé : {info['marketplace']}/{info['name']} ({info['version']})")
        elif args.cmd == "add-local":
            info = plugin_add_local(args.root, args.path)
            print(f"ajouté : _local/{info['name']} ({info['version']})")
        elif args.cmd == "remove":
            plugin_remove(args.root, args.ref)
            print("retiré :", args.ref)
        elif args.cmd == "list":
            for p in discover_plugins(args.root):
                print(_fmt_plugin(p))
    except PluginError as exc:
        print(f"erreur : {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 2 : Smoke — `list` sur store vide ne casse pas**

Run :
```bash
uv run python -m loom.plugins --root "$(python -c 'import tempfile;print(tempfile.mkdtemp())')" list; echo "EXIT=$?"
```
Expected : aucune ligne plugin, `EXIT=0`

- [ ] **Step 3 : Lint + commit**

```bash
uvx ruff check loom/plugins.py && uvx ruff format loom/plugins.py
git add loom/plugins.py && git commit -m "feat(plugins): CLI python -m loom.plugins

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4 : Refonte `loom/skills.py` — catalogue + chargement à la demande

**Files:**
- Modify: `loom/skills.py` (réécriture complète)

- [ ] **Step 1 : Réécrire `loom/skills.py`**

```python
# loom/skills.py
"""Skills : modules de connaissance markdown DÉCLENCHÉS PAR LE MODÈLE (façon Claude Code).

On annonce au modèle un CATALOGUE (nom : description) ; quand un skill est pertinent il
appelle l'outil use_skill(nom) qui renvoie le corps. Plus d'activation manuelle. Les skills
viennent du dossier local (loom/skills, non namespacés) ET des plugins installés (namespacés
`plugin:nom`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    body: str
    base_dir: str = ""  # dossier du SKILL.md (pour résoudre references/)


def _parse_skill_md(text: str, fallback_name: str) -> tuple[str, str, str]:
    """Parse frontmatter (name/description) + corps. Renvoie (name, description, body)."""
    name, description, body = fallback_name, "", text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            for line in front.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key, val = key.strip().lower(), val.strip()
                    if key == "name" and val:
                        name = val
                    elif key == "description":
                        description = val
    return name, description, body


def _load_skill_file(md: Path, namespace: str | None) -> Skill | None:
    try:
        text = md.read_text(encoding="utf-8")
    except OSError:
        return None
    name, desc, body = _parse_skill_md(text, md.parent.name)
    if namespace:
        name = f"{namespace}:{name}"
    return Skill(name=name, description=desc, body=body, base_dir=str(md.parent))


def _scan_dir(skills_dir: str | Path, namespace: str | None) -> list[Skill]:
    skills_dir = Path(skills_dir)
    out: list[Skill] = []
    if not skills_dir.exists():
        return out
    for sub in sorted(skills_dir.iterdir()):
        md = sub / "SKILL.md"
        if sub.is_dir() and md.exists():
            sk = _load_skill_file(md, namespace)
            if sk:
                out.append(sk)
    return out


def collect_skills(
    local_dir: str | Path, plugins_root_path: str | Path | None = None
) -> list[Skill]:
    """Agrège les skills locaux (non namespacés) + ceux des plugins installés
    (namespacés `plugin:nom`)."""
    skills = _scan_dir(local_dir, namespace=None)
    if plugins_root_path is not None:
        from loom.plugins import discover_plugins

        for plugin in discover_plugins(plugins_root_path):
            for md in plugin.skills:
                sk = _load_skill_file(md, namespace=plugin.name)
                if sk:
                    skills.append(sk)
    return skills


def render_catalog(skills: list[Skill]) -> str:
    """Bloc injecté au prompt système : la liste nom : description + comment déclencher."""
    if not skills:
        return ""
    lines = [
        "# Skills disponibles",
        "Quand l'un de ces skills correspond à la demande, APPELLE l'outil "
        "`use_skill(name)` pour charger ses instructions, puis suis-les. Ne devine pas "
        "leur contenu.",
    ]
    for s in skills:
        desc = " ".join(s.description.split())
        if len(desc) > 220:
            desc = desc[:217] + "…"
        lines.append(f"- {s.name} : {desc}")
    return "\n".join(lines)


def load_skill_body(skills: list[Skill], name: str) -> str | None:
    """Corps d'un skill par nom (préfixé du dossier de base pour lire references/)."""
    for s in skills:
        if s.name == name:
            head = f"Base directory for this skill: {s.base_dir}\n\n" if s.base_dir else ""
            return f"{head}{s.body}"
    return None
```

- [ ] **Step 2 : Smoke — collect/render/load + namespacing plugin**

Run :
```bash
uv run python -c "
import tempfile, json
from pathlib import Path
from loom.plugins import plugin_add_local
from loom.skills import collect_skills, render_catalog, load_skill_body
# skill local
local = Path(tempfile.mkdtemp())
(local/'natif').mkdir(); (local/'natif'/'SKILL.md').write_text('---\nname: natif\ndescription: skill local\n---\nCORPS_LOCAL')
# plugin local
store = Path(tempfile.mkdtemp())
src = Path(tempfile.mkdtemp())
(src/'.claude-plugin').mkdir(); (src/'.claude-plugin'/'plugin.json').write_text(json.dumps({'name':'myplug','version':'1.0'}))
(src/'skills'/'aide').mkdir(parents=True); (src/'skills'/'aide'/'SKILL.md').write_text('---\nname: aide\ndescription: skill plugin\n---\nCORPS_PLUGIN')
plugin_add_local(store, str(src))
sk = collect_skills(local, store)
names = sorted(s.name for s in sk)
assert names==['myplug:aide','natif'], names
cat = render_catalog(sk); assert 'use_skill' in cat and 'myplug:aide' in cat
body = load_skill_body(sk, 'myplug:aide'); assert 'CORPS_PLUGIN' in body and 'Base directory' in body
assert load_skill_body(sk, 'inconnu') is None
print('TASK4_OK', names)
"
```
Expected : `TASK4_OK ['myplug:aide', 'natif']`

- [ ] **Step 3 : Lint + commit**

```bash
uvx ruff check loom/skills.py && uvx ruff format loom/skills.py
git add loom/skills.py && git commit -m "refactor(skills): catalogue + use_skill (declenchement par le modele)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5 : Tools (`use_skill`, install/marketplace) + permissions + registre

**Files:**
- Create: `loom/tools/skills.py`
- Create: `loom/tools/plugins.py`
- Modify: `loom/permissions.py`
- Modify: `loom/tools/base.py` (AVAILABLE_TOOLS)
- Modify: `loom/tools/__init__.py` (build_registry)

- [ ] **Step 1 : `loom/tools/skills.py`**

```python
# loom/tools/skills.py
"""Outil use_skill : charge à la demande le corps d'un skill du catalogue (déclenché par le
modèle d'après les descriptions annoncées dans le prompt système)."""

from __future__ import annotations

from collections.abc import Callable

from loom.skills import Skill, load_skill_body
from loom.tools.base import ToolError, ToolSpec


def make_use_skill(skills_provider: Callable[[], list[Skill]]) -> ToolSpec:
    def run(args: dict) -> str:
        name = (args.get("name") or "").strip()
        if not name:
            raise ToolError("argument 'name' manquant (nom du skill à charger)")
        skills = skills_provider()
        body = load_skill_body(skills, name)
        if body is None:
            valid = ", ".join(s.name for s in skills) or "(aucun)"
            raise ToolError(f"skill inconnu '{name}'. Skills valides : {valid}")
        return body

    return ToolSpec(
        name="use_skill",
        description=(
            "Charge les instructions d'un skill listé dans « Skills disponibles » du prompt "
            "système. Appelle-le DÈS qu'un skill correspond à la demande, puis suis son "
            "contenu. Argument : name (le nom exact du catalogue, ex. 'plugin:skill')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Nom exact du skill au catalogue."}
            },
            "required": ["name"],
        },
        run=run,
    )
```

- [ ] **Step 2 : `loom/tools/plugins.py`**

```python
# loom/tools/plugins.py
"""Outils d'installation de plugins (appelables par le modèle, sous garde de permission) :
add_marketplace / install_plugin clonent du code tiers ; list_plugins est en lecture seule."""

from __future__ import annotations

from loom import plugins as store
from loom.tools.base import ToolError, ToolSpec


def make_add_marketplace(root) -> ToolSpec:
    def run(args: dict) -> str:
        source = (args.get("source") or "").strip()
        if not source:
            raise ToolError("argument 'source' manquant (URL git ou chemin local)")
        try:
            name = store.marketplace_add(root, source)
        except store.PluginError as exc:
            raise ToolError(str(exc)) from exc
        return f"Marketplace ajoutée : {name}"

    return ToolSpec(
        name="add_marketplace",
        description=(
            "Ajoute une marketplace de plugins Claude Code (URL git ou chemin local) au store "
            "de Loom. Prérequis pour install_plugin."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "URL git ou chemin local."}
            },
            "required": ["source"],
        },
        run=run,
    )


def make_install_plugin(root) -> ToolSpec:
    def run(args: dict) -> str:
        ref = (args.get("ref") or "").strip()
        if not ref:
            raise ToolError("argument 'ref' manquant ('<plugin>' ou '<plugin>@<marketplace>')")
        try:
            info = store.plugin_install(root, ref)
        except store.PluginError as exc:
            raise ToolError(str(exc)) from exc
        return (
            f"Plugin installé : {info['marketplace']}/{info['name']} ({info['version']}). "
            "Ses skills sont désormais au catalogue."
        )

    return ToolSpec(
        name="install_plugin",
        description=(
            "Installe un plugin depuis une marketplace ajoutée. ref = '<plugin>' ou "
            "'<plugin>@<marketplace>'. Rend ses skills disponibles via use_skill."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "<plugin> ou <plugin>@<marketplace>"}
            },
            "required": ["ref"],
        },
        run=run,
    )


def make_list_plugins(root) -> ToolSpec:
    def run(args: dict) -> str:
        plugins = store.discover_plugins(root)
        if not plugins:
            return "Aucun plugin installé."
        return "\n".join(
            f"{p.marketplace}/{p.name} ({p.version}) — skills:{len(p.skills)} "
            f"agents:{len(p.agents)} hooks:{len(p.hooks)} commands:{len(p.commands)}"
            for p in plugins
        )

    return ToolSpec(
        name="list_plugins",
        description="Liste les plugins installés et l'inventaire de leurs composants.",
        parameters={"type": "object", "properties": {}},
        run=run,
    )
```

- [ ] **Step 3 : Permissions — `loom/permissions.py`**

Ajouter `use_skill` et `list_plugins` à `READ_TOOLS` (dans le frozenset existant), et après
`SHELL_TOOLS` ajouter :
```python
# Outils d'INSTALLATION de plugins : clonent du code tiers + écrivent sur disque -> gardés
# comme les écritures (ask par défaut). Le modèle peut proposer/lancer l'install, pas sans
# accord en mode 'ask'.
PLUGIN_TOOLS = frozenset({"add_marketplace", "install_plugin"})
```
Puis dans `evaluate(...)`, juste avant le bloc `if tool_name in WRITE_TOOLS:`, ajouter :
```python
    if tool_name in PLUGIN_TOOLS:
        if cfg.mode == "deny_all":
            return Decision("deny", "mode deny_all")
        if cfg.mode == "allow":
            return Decision("allow")
        return Decision("ask")
```

- [ ] **Step 4 : AVAILABLE_TOOLS — `loom/tools/base.py`**

Dans la liste `AVAILABLE_TOOLS`, après l'entrée `manage_todos`, ajouter :
```python
    {"name": "use_skill", "label": "use_skill", "danger": False},
    {"name": "list_plugins", "label": "list_plugins", "danger": False},
    {"name": "add_marketplace", "label": "add_marketplace", "danger": True},
    {"name": "install_plugin", "label": "install_plugin", "danger": True},
```

- [ ] **Step 5 : Registre — `loom/tools/__init__.py`**

Ajouter à la signature de `build_registry` (après `active_model`) deux params :
```python
    skills_dir: str | None = None,
    plugins_root: str | None = None,
```
Puis, avant le `from loom.models_profile import load_profile` final, ajouter le câblage :
```python
    if "use_skill" in enabled and skills_dir is not None:
        from loom.skills import collect_skills
        from loom.tools.skills import make_use_skill

        specs.append(
            make_use_skill(lambda: collect_skills(skills_dir, plugins_root))
        )
    if plugins_root is not None:
        from loom.tools.plugins import (
            make_add_marketplace,
            make_install_plugin,
            make_list_plugins,
        )

        if "add_marketplace" in enabled:
            specs.append(make_add_marketplace(plugins_root))
        if "install_plugin" in enabled:
            specs.append(make_install_plugin(plugins_root))
        if "list_plugins" in enabled:
            specs.append(make_list_plugins(plugins_root))
```

- [ ] **Step 6 : Smoke — les tools se construisent et répondent**

Run :
```bash
uv run python -c "
import tempfile, json
from pathlib import Path
from loom.plugins import plugin_add_local
from loom.skills import collect_skills
from loom.tools.skills import make_use_skill
from loom.tools.plugins import make_list_plugins
from loom.permissions import evaluate, PermissionConfig
store = Path(tempfile.mkdtemp()); src = Path(tempfile.mkdtemp())
(src/'.claude-plugin').mkdir(); (src/'.claude-plugin'/'plugin.json').write_text(json.dumps({'name':'pp','version':'1'}))
(src/'skills'/'k').mkdir(parents=True); (src/'skills'/'k'/'SKILL.md').write_text('---\nname: k\ndescription: d\n---\nBODYZ')
plugin_add_local(store, str(src))
loc = Path(tempfile.mkdtemp())
us = make_use_skill(lambda: collect_skills(loc, store))
assert 'BODYZ' in us.run({'name':'pp:k'})
lp = make_list_plugins(store); assert 'pp' in lp.run({})
assert evaluate('use_skill', {}, PermissionConfig(mode='ask')).action=='allow'
assert evaluate('install_plugin', {}, PermissionConfig(mode='ask')).action=='ask'
print('TASK5_OK')
"
```
Expected : `TASK5_OK`

- [ ] **Step 7 : Lint + commit**

```bash
uvx ruff check loom/ && uvx ruff format loom/tools/skills.py loom/tools/plugins.py loom/permissions.py loom/tools/base.py loom/tools/__init__.py
git add loom/tools/skills.py loom/tools/plugins.py loom/permissions.py loom/tools/base.py loom/tools/__init__.py
git commit -m "feat(plugins): tools use_skill/install_plugin/add_marketplace/list_plugins + permissions

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6 : Câblage web + suppression de l'activation manuelle

**Files:**
- Modify: `loom/conversation.py`
- Modify: `loom/web/__main__.py`
- Modify: `loom/web/app.py`
- Modify: `loom/web/templates/_skills.html`
- Modify: `loom/loom.config.toml`

- [ ] **Step 1 : `loom/conversation.py` — retirer `active_skills`/`set_skills`**

Supprimer : le champ `active_skills: list[str] = field(default_factory=list)` ; la méthode
`set_skills` ; la clé `"active_skills": self.active_skills` dans `to_dict` ; la ligne
`active_skills=list(data.get("active_skills", []))` dans `from_dict`.

- [ ] **Step 2 : `loom/web/__main__.py` — calculer `plugins_root` et le diffuser**

Après le calcul de `data_root` (`data_root = Path(cfg.chat.history_path).resolve().parent`),
ajouter :
```python
    from loom.plugins import plugins_root as _plugins_root

    plugins_dir = str(_plugins_root(getattr(cfg.chat, "plugins_root", None)))
```
Dans la closure `make_registry(...)` (l'appel `build_registry(...)`), ajouter les arguments :
```python
            skills_dir=cfg.chat.skills_dir,
            plugins_root=plugins_dir,
```
Dans l'appel `create_app(...)`, ajouter :
```python
        plugins_dir=plugins_dir,
```

> Note : si `cfg.chat` n'a pas d'attribut `plugins_root`, `getattr(..., None)` retombe sur le
> défaut `loom/plugins`. (La clé config est ajoutée au Step 5 ; le code reste tolérant.)

- [ ] **Step 3 : `loom/web/app.py` — catalogue dans le prompt, suppression `/skills`**

a) Remplacer l'import (ligne ~22) :
```python
from loom.skills import collect_skills, render_catalog
```
b) `create_app(...)` : ajouter le paramètre `plugins_dir="loom/plugins",` (keyword-only, près
de `workspace_dir`).
c) Dans le dict de rendu d'index (~ligne 255), retirer `"active_skills": conv.active_skills,`
et remplacer `"skills": list_skills(skills_dir),` par
`"skills": collect_skills(skills_dir, plugins_dir),`.
d) Remplacer le bloc de composition (lignes ~333-336) :
```python
            skills = collect_skills(skills_dir, plugins_dir)
            catalog = render_catalog(skills)
            system_prompt = (
                f"{conv.system_prompt}\n\n{catalog}" if catalog else conv.system_prompt
            )
```
e) Supprimer entièrement la route `@app.post("/skills")` / `def skills_update()` (lignes
~467-475).

- [ ] **Step 4 : `loom/web/templates/_skills.html` — lecture seule, groupé par source**

Remplacer le contenu par une liste non interactive (les skills de plugin ont un nom
`plugin:skill` ; on groupe sur le préfixe) :
```html
<div id="skills-panel">
  {% for s in skills %}
    <div class="skill-ro" title="{{ s.description }}">{{ s.name }}</div>
  {% else %}
    <span class="muted">Aucun skill. Installe un plugin (install_plugin) ou ajoute
      loom/skills/&lt;nom&gt;/SKILL.md</span>
  {% endfor %}
</div>
```

- [ ] **Step 5 : `loom/loom.config.toml`**

Dans `tools_enabled`, ajouter `"use_skill", "list_plugins", "add_marketplace",
"install_plugin"`. Ajouter en fin de section `[chat]` (ou nouvelle clé lue par la config) :
```toml
    # Racine du store de plugins (défaut loom/plugins). Contenu installé git-ignoré.
    plugins_root = "loom/plugins"
```
> Si `cfg.chat` est typé strictement et ne lit pas les clés inconnues, ajouter `plugins_root`
> au parseur de config (`loom/config.py`, dataclass `chat`) ; sinon le `getattr` du Step 2
> suffit. Vérifier `loom/config.py` au moment de l'exécution.

- [ ] **Step 6 : Smoke — l'app se construit, le prompt porte le catalogue**

Run :
```bash
uv run python -c "
import inspect
from loom.web.app import create_app
assert 'plugins_dir' in inspect.signature(create_app).parameters
from loom.conversation import Conversation
c = Conversation(system_prompt='x')
assert not hasattr(c, 'active_skills'), 'active_skills doit etre retire'
from loom.skills import collect_skills, render_catalog
print('TASK6_OK')
"
```
Expected : `TASK6_OK`

- [ ] **Step 7 : Lint + commit**

```bash
uvx ruff check loom/ && uvx ruff format loom/conversation.py loom/web/app.py loom/web/__main__.py
git add loom/conversation.py loom/web/app.py loom/web/__main__.py loom/web/templates/_skills.html loom/loom.config.toml
git commit -m "feat(plugins): catalogue de skills dans le prompt + tools cables; retrait activation manuelle

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7 : Nettoyage — retirer le skill d'exemple inutile

**Files:**
- Delete: `loom/skills/exemple/`

- [ ] **Step 1 : Supprimer le placeholder**

```bash
git rm -r loom/skills/exemple
```

- [ ] **Step 2 : Smoke — collect_skills tolère un dossier local vide**

Run :
```bash
uv run python -c "from loom.skills import collect_skills; print('TASK7_OK', collect_skills('loom/skills', None))"
```
Expected : `TASK7_OK []`

- [ ] **Step 3 : Commit**

```bash
git commit -m "chore(skills): retire le skill exemple (tout passe par les plugins)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Vérification finale (manuelle, hors smoke auto)

- [ ] `python -m loom.plugins marketplace add <chemin-local-ou-url>` puis `install <plugin>` →
  `python -m loom.plugins list` montre le plugin et ses skills.
- [ ] Lancer la stack (par l'utilisateur), poser une question qui matche un skill installé →
  le modèle appelle `use_skill` et suit le contenu.
- [ ] Test git réel : ajouter une marketplace GitHub (ex. le repo defensive-prompt-injection)
  et installer un plugin avec `source` git/git-subdir.

## Notes d'auto-revue (writing-plans)

- Couverture spec : store+JSON (T1), install 3 formes + local (T2), CLI (T3), catalogue +
  use_skill (T4), tools + permissions + registre (T5), câblage web + suppression manuel (T6),
  nettoyage (T7). ✓
- Cohérence des noms : `plugins_root` (fonction et param), `collect_skills(local, root)`,
  `make_use_skill(provider)`, `PLUGIN_TOOLS`, refs `<plugin>@<marketplace>` partout. ✓
- Point à vérifier à l'exécution : `loom/config.py` lit-il une clé `plugins_root` sous
  `[chat]` ? Si la dataclass est stricte, l'ajouter (Step 5 le signale). Le `getattr` du
  Step 2 garde un défaut sûr en attendant.
- Déféré (specs suivantes) : moteur de hooks, agents→personas, commands/MCP.
