# Loom — Store de plugins compatible Claude Code (Tranche 1) — Design

**Date :** 2026-06-09
**Statut :** design validé (en attente de relecture utilisateur avant plan)
**Branche cible :** `feat/harness-reflexion`

## Objectif

Donner à Loom son **propre** store de plugins, réplique du mécanisme de Claude Code
(marketplaces + install + cache, **même format de plugin**), pour installer **n'importe
quel** plugin Claude Code et le rendre utilisable par **n'importe quel modèle local**,
hors-ligne. Loom **ne dépend pas** de l'installation de Claude Code (`~/.claude/` peut être
absent) : il héberge tout lui-même.

Tranche 1 = la **colonne vertébrale** :
1. Store local au format CC + installation (marketplace/git/local).
2. Consommation des **skills** des plugins selon le modèle Claude Code :
   **déclenchement par le modèle** (le modèle voit les `description`, décide, charge le corps
   à la demande) — PAS de cases à cocher manuelles.
3. Installation pilotable par le **modèle** (tool) ET par toi (CLI).

Les autres types de composants (hooks, agents, commands, MCP) sont **inventoriés mais pas
encore câblés** — chacun fera l'objet d'une spec ultérieure.

## Changement de modèle des skills (acté)

Le mécanisme actuel (`loom/skills/` + cases à cocher dans l'UI → corps injecté en permanence
dans le prompt système) est remplacé. Il ne sert plus à rien aujourd'hui et ne passe pas à
l'échelle de dizaines de skills de plugins.

**Nouveau modèle (façon Anthropic) :**
- Tous les skills disponibles (locaux **et** de plugins) sont annoncés au modèle sous forme
  de **catalogue** `nom: description` dans le prompt système.
- Quand un skill est pertinent pour la demande, **le modèle appelle le tool `use_skill(nom)`**
  qui renvoie le corps du `SKILL.md` (le modèle le suit ensuite). Le savoir n'est chargé
  qu'à la demande, et il vit naturellement dans l'historique (résultat de tool).
- Plus d'activation manuelle : c'est le modèle qui décide, sur la base de la `description`.

Conséquence : suppression de l'activation manuelle (`active_skills`, `set_skills`, route
`/skills`, cases du panneau). Le panneau Skills devient **informatif** (liste en lecture seule
des skills disponibles, groupés par source).

## Périmètre

**Dans la tranche 1 :**
- Store local au format CC sous `loom/plugins/`.
- Ajouter une marketplace (git clone ou chemin local), lire son `marketplace.json`.
- Installer un plugin en résolvant les 3 formes de `source` réelles : chemin relatif, `git`,
  `git-subdir`.
- **Fallback manuel (B)** : déposer/pointer un dossier plugin direct dans le cache, sans
  marketplace.
- Découverte : parser `.claude-plugin/plugin.json`, inventorier les 4 types de composants.
- Skills des plugins (+ locaux) exposés en **catalogue** + tool `use_skill` (déclenchement
  par le modèle, namespacés `plugin:skill`).
- Installation via **tool LLM** (sous garde de permission) **et** **CLI**
  (`python -m loom.plugins`).

**Hors tranche 1 (specs séparées) :**
- ① Moteur de **hooks** (PostToolUse : matcher + exécution). Rend un plugin *actif*.
  **Exécute du code tiers → nécessitera une porte de confiance.**
- ② **Agents** des plugins → personas dispatchables, en étendant `dispatch_agent`.
- ③ **commands** (slash) et **MCP** : substrat inexistant dans Loom → déférés.

## Architecture

### Emplacement du store

Racine : `loom/plugins/`, **au niveau de `loom/skills/`** (cohérence : skills et plugins au
même endroit, dans le package). Surchargeable par `[plugins].root` dans `loom.config.toml`
(défaut `loom/plugins`). Le contenu installé (cache/marketplaces) est git-ignoré
(`loom/plugins/.gitignore` : tout sauf un éventuel `.gitkeep`).

### Disposition (format Claude Code)

```
loom/plugins/
  known_marketplaces.json
  installed_plugins.json
  marketplaces/<marketplace>/.claude-plugin/marketplace.json   # + dépôt cloné
  cache/<marketplace>/<plugin>/<version>/
      .claude-plugin/plugin.json
      skills/<nom>/SKILL.md [+ references/…]
      agents/<nom>.md
      hooks/hooks.json [+ scripts]
      commands/…
```

**Note de compatibilité.** La compat « 100% » porte sur le **format de plugin** consommé
(`plugin.json`, `marketplace.json`, `skills/…`) — un plugin écrit pour Claude Code s'installe
et fonctionne dans Loom sans modification. Les fichiers de **comptabilité** de Loom
(`installed_plugins.json`, `known_marketplaces.json`) reprennent la forme de CC par
familiarité mais appartiennent à Loom.

### Formats lus / écrits

**`.claude-plugin/plugin.json`** (lu) : `{ name, version?, description?, author?, license? }`.
`version` peut valoir `"unknown"`. Seul `name` est requis.

**`.claude-plugin/marketplace.json`** (lu) :
`{ name, owner?, plugins: [ { name, description?, source } ] }`. Formes de `source` :
- **chemin relatif** : `"./plugins/x"` → plugin dans le dépôt de la marketplace.
- **git-subdir** : `{ "source":"git-subdir", "url", "path", "ref"?, "sha"? }`.
- **git** : `{ "source":"git", "url", "ref"? }` → plugin à la racine du dépôt cloné.

**`installed_plugins.json`** (écrit, forme CC v2) :
```json
{ "version": 2, "plugins": {
  "<plugin>@<marketplace>": [ { "scope":"user", "installPath":"<abs cache dir>",
    "version":"...", "installedAt":"<iso>", "lastUpdated":"<iso>",
    "gitCommitSha":"<sha|''>" } ] } }
```

**`known_marketplaces.json`** (écrit) :
```json
{ "version": 1, "marketplaces": {
  "<name>": { "source":"<git-url|local-path>", "addedAt":"<iso>" } } }
```

## Composants (code)

### `loom/plugins.py` (nouveau) — logique du store + CLI

Logique pure (résolution/parsing/scan) + I/O fichier + git encapsulé. Aucune dépendance
Flask. La **CLI vit dans ce même module** (`main()` argparse + `if __name__ == "__main__"`),
invoquée par `python -m loom.plugins` — pas de second fichier `loom/plugin.py` (éviter la
confusion d'import).

Fonctions :
- `plugins_root(cfg) -> Path` — racine du store (config ou défaut), créée si absente.
- `marketplace_add(root, source) -> str` — clone (git) ou copie (chemin local) dans un temp,
  lit `marketplace.json`, renomme d'après son `name`, installe sous `marketplaces/<name>/`,
  écrit `known_marketplaces.json`. Renvoie le nom. Erreur claire sans `marketplace.json`.
- `marketplace_list(root) -> list[dict]`.
- `plugin_install(root, ref) -> dict` — `ref` = `"<plugin>"` ou `"<plugin>@<marketplace>"` ;
  résout la `source` (3 formes), matérialise sous `cache/<marketplace>/<plugin>/<version>/`,
  met à jour `installed_plugins.json`.
- `plugin_add_local(root, path) -> dict` — **fallback B** : copie un dossier plugin (avec
  `.claude-plugin/plugin.json`) sous `cache/_local/<plugin>/<version>/`, marketplace `_local`.
- `plugin_remove(root, ref) -> None`.
- `discover_plugins(root) -> list[Plugin]` — lit `installed_plugins.json` ; pour chaque
  `installPath`, parse `plugin.json` et inventorie skills/agents/hooks/commands. **Fallback :**
  scanne aussi `cache/` pour les plugins valides absents du JSON (dépôt manuel).

Dataclasse :
```python
@dataclass
class Plugin:
    name: str
    marketplace: str
    path: Path                 # dossier <version>/
    version: str
    description: str
    skills: list[Path]         # chemins des SKILL.md
    agents: list[Path]
    hooks: list[Path]
    commands: list[Path]
```

CLI (argparse) : `marketplace add <git-url|chemin>` · `marketplace list` ·
`install <plugin[@marketplace]>` · `add-local <chemin>` · `list` · `remove <plugin[@mkt]>`.

Git encapsulé : `_git_clone(url, dest, ref=None, sha=None)` via `subprocess`
(`git clone --depth 1` puis `git checkout` si fourni) ; pour `git-subdir`, clone puis copie le
sous-dossier. Échec → message actionnable.

### `loom/skills.py` (refonte) — catalogue + chargement à la demande

- `Skill` gagne `base_dir: str` (dossier du `SKILL.md`).
- `collect_skills(local_dir, plugins_root) -> list[Skill]` — agrège (a) les skills locaux de
  `local_dir` (non namespacés) et (b) ceux des plugins découverts, **namespacés**
  `"<plugin>:<nom>"`.
- `render_catalog(skills) -> str` — bloc texte injecté au prompt système :
  `Skills disponibles (appelle use_skill(nom) quand l'un est pertinent) :` puis une ligne
  `- <nom> : <description>` par skill.
- `load_skill_body(skills, name) -> str | None` — corps du skill, préfixé d'une ligne
  `Base directory for this skill: <base_dir>` (pour que le modèle puisse `read_file` les
  `references/` via chemin absolu).
- Suppression de `compose_system_prompt`/`load_skill`/`set_skills` (modèle manuel obsolète).

**Budget de contexte.** Le catalogue n'embarque que `nom + description` (descriptions
tronquées ~à 1 ligne) — léger. Si le nombre de skills explose (dizaines de plugins), une
future itération introduira un tool `search_skills` au lieu de tout lister. Hors tranche 1.

### `loom/tools/skills.py` (nouveau) — tool `use_skill`

`make_use_skill(skills_provider) -> ToolSpec` : `use_skill(name)` renvoie
`load_skill_body(...)` (ou une erreur listant les noms valides). **Sans effet de bord** →
classé dans `READ_TOOLS` (auto-autorisé). `skills_provider` est un callable rendant la liste
courante de `Skill` (pour refléter les plugins fraîchement installés dans la session).

### `loom/tools/plugins.py` (nouveau) — tools d'installation

- `make_add_marketplace(root)` : `add_marketplace(source)` → `marketplace_add`.
- `make_install_plugin(root)` : `install_plugin(ref)` → `plugin_install`.
- `make_list_plugins(root)` : `list_plugins()` → inventaire lisible (sans effet de bord).

`add_marketplace`/`install_plugin` **clonent du code tiers + écrivent sur disque** → traités
par la politique de permission comme une action à effet de bord (catégorie `PLUGIN_TOOLS`,
gating identique aux écritures : `allow`/`ask`/`deny` selon le mode ; `ask` par défaut →
l'utilisateur confirme avant tout clone). `list_plugins` → `READ_TOOLS`.

### Permissions (`loom/permissions.py`)

- `use_skill`, `list_plugins` → ajoutés à `READ_TOOLS`.
- `PLUGIN_TOOLS = {"add_marketplace", "install_plugin"}` → évalués comme les `WRITE_TOOLS`
  (mode `deny_all`→deny, `allow`→allow, sinon `ask`).

### Câblage web (`__main__.py`, `app.py`, templates)

- `__main__.py` : calcule `plugins_root`, le passe à `create_app` et au `tool_factory`.
- `tool_factory` (`loom/tools/__init__.py` + `build_registry`) : enregistre `use_skill`,
  `add_marketplace`, `install_plugin`, `list_plugins` quand activés ; `use_skill` reçoit un
  `skills_provider` qui appelle `collect_skills(skills_dir, plugins_root)`.
- Système : le prompt est composé `CHAT_SYSTEM + "\n\n" + render_catalog(collect_skills(...))`.
- Suppression de la route `/skills`, de `set_skills`, du champ `active_skills` (conversation),
  et des cases du panneau.
- `_skills.html` : liste **lecture seule** des skills disponibles, groupés par source
  (locaux, puis par plugin).
- `AVAILABLE_TOOLS` (`loom/tools/base.py`) : déclare les 4 nouveaux tools (pour la config et
  l'UI des outils).

## Flux de données

```
(tool) install_plugin / (CLI) install ─► loom/plugins.py (git/copy ─► cache/ + JSON)
                                                  │
session/tour ─► collect_skills(local, plugins_root) ─► render_catalog ─► prompt système
                                                  │
                         le modèle voit les descriptions ─► décide ─► use_skill(nom)
                                                  │
                         load_skill_body (+ Base directory) ─► résultat de tool ─► le modèle suit
```

## Gestion d'erreurs

- `marketplace.json` / `plugin.json` absent ou invalide → élément **ignoré** + message clair
  (jamais de crash de la découverte).
- Échec git (binaire absent, réseau, ref inconnue) → message actionnable (tool result / exit).
- `use_skill` sur un nom inconnu → renvoie la liste des noms valides.
- Plugin sans `skills/` → 0 skill, l'inventaire montre les autres types.
- Collisions de noms → évitées par le namespacing `plugin:skill`.
- Dépôt manuel mal formé dans `cache/` → ignoré (pas de `plugin.json`).

## Sécurité

- **Lecture des skills** (markdown → contexte) : aucune exécution de code tiers.
- **Installation** (`add_marketplace`/`install_plugin`) : clone du code tiers + écriture
  disque → **garde de permission `ask`** par défaut (l'utilisateur confirme). Le modèle peut
  proposer/lancer l'install, mais pas sans accord en mode `ask`.
- L'**exécution** de code tiers n'arrive qu'à la **tranche hooks** : cette spec-là devra
  ajouter une porte de confiance dédiée (allowlist/confirm par plugin) avant de lancer un
  script de hook. À acter là-bas.
- Rappel : le contenu d'un `SKILL.md` tiers est **externe** ; les principes du skill
  `trust-boundary` (données ≠ instructions) s'y appliquent, mais le corps d'un skill est par
  nature une consigne — n'installer que des plugins de confiance.

## Tests (contrainte projet : pas de pytest sur Loom)

Smokes `uv run python -c …`, **sans réseau**, sur une fixture `tempfile` :
1. Fausse marketplace locale (`marketplace.json` + plugin `skills/demo/SKILL.md`,
   `source` = chemin relatif).
2. `marketplace_add(local_path)` puis `plugin_install("<plugin>")` → vérifier l'arborescence
   du cache + `installed_plugins.json`.
3. `discover_plugins` → plugin + skill inventoriés.
4. `collect_skills` → skill namespacé `demo:hello`, `base_dir` correct ; `render_catalog`
   contient sa description ; `load_skill_body` renvoie le corps + la ligne `Base directory`.
5. `use_skill` (via ToolSpec.run) sur le nom namespacé → renvoie le corps ; sur un nom
   inconnu → liste des noms valides.
6. `plugin_add_local(dir)` (fallback B) → découvert sous `_local`.
Les chemins **git / git-subdir** sont vérifiés à la main (vrai clone), hors smoke automatique.

## Décisions actées

- **C** (store Loom autonome, format CC) plutôt que lire `~/.claude/` : indépendance de CC.
- **B** comme repli manuel (dépôt local).
- Store sous **`loom/plugins/`** (au niveau de `loom/skills/`), pas dans `data/`.
- Skills **déclenchés par le modèle** (catalogue + `use_skill`), **plus** d'activation
  manuelle ; l'ancien mécanisme (cases/injection permanente) est supprimé.
- Installation exposée comme **tool LLM** (gardé en `ask`) **et** CLI.
- Skills locaux non namespacés ; skills de plugin namespacés `plugin:skill`.
