# Loom — Store de plugins compatible Claude Code (Tranche 1) — Design

**Date :** 2026-06-09
**Statut :** design validé (en attente de relecture utilisateur avant plan)
**Branche cible :** `feat/harness-reflexion`

## Objectif

Donner à Loom son **propre** store de plugins, réplique du mécanisme de Claude Code
(marketplaces + install + cache, **même format de plugin**), pour pouvoir installer
**n'importe quel** plugin Claude Code et le rendre utilisable par **n'importe quel modèle
local**, hors-ligne. Loom **ne dépend pas** de l'installation de Claude Code (`~/.claude/`
peut être absent) : il héberge tout lui-même.

Tranche 1 = la **colonne vertébrale** : store + installation + consommation des **skills**
(le seul type de composant dont Loom a déjà le substrat). Les autres types (hooks, agents,
commands, MCP) sont **inventoriés mais pas encore câblés** — chacun fera l'objet d'une spec
ultérieure.

## Périmètre

**Dans la tranche 1 :**
- Store local au format CC (`marketplaces/`, `cache/`, `installed_plugins.json`,
  `known_marketplaces.json`).
- Ajouter une marketplace (git clone ou chemin local) et lire son `marketplace.json`.
- Installer un plugin depuis une marketplace, en résolvant les 3 formes de `source` vues sur
  les marketplaces réelles : chemin relatif, `git`, `git-subdir`.
- **Fallback manuel (B)** : déposer/pointer un dossier plugin direct dans le cache, sans
  marketplace.
- Découverte : parser `.claude-plugin/plugin.json`, inventorier les 4 types de composants.
- Câbler les **skills** : les `skills/<nom>/SKILL.md` des plugins installés rejoignent la
  liste de skills de Loom, **namespacés** `plugin:skill`.
- Surface **CLI** : `python -m loom.plugins …`.

**Hors tranche 1 (specs séparées) :**
- ① Moteur de **hooks** (PostToolUse : matcher + exécution de commande/rappel). C'est ce qui
  rend un plugin *actif*. **Exécute du code tiers → nécessite une porte de confiance.**
- ② **Agents** des plugins → personas dispatchables (prompt système + outils restreints),
  en étendant `dispatch_agent`.
- ③ **commands** (slash) et **MCP** : substrat inexistant dans Loom → gros lift, valeur
  locale la plus faible, déférés.

## Architecture

### Emplacement du store

Racine : `<data_root>/plugins/`, où `data_root` est le dossier parent de
`cfg.chat.history_path` (déjà calculé dans `loom/web/__main__.py` :
`data_root = Path(cfg.chat.history_path).resolve().parent`, qui porte déjà `sessions/`).
Le store vit à côté des sessions. Optionnellement surchargé par `[plugins].root` dans
`loom.config.toml` (défaut = `data_root/plugins`).

### Disposition (format Claude Code)

```
<data_root>/plugins/
  known_marketplaces.json
  installed_plugins.json
  marketplaces/<marketplace>/.claude-plugin/marketplace.json   # + contenu du dépôt cloné
  cache/<marketplace>/<plugin>/<version>/
      .claude-plugin/plugin.json
      skills/<nom>/SKILL.md [+ references/…]
      agents/<nom>.md
      hooks/hooks.json [+ scripts]
      commands/…
```

**Note de compatibilité.** La compat « 100% » porte sur le **format de plugin** consommé
(`plugin.json`, `marketplace.json`, `skills/…`) — un plugin écrit pour Claude Code
s'installe et fonctionne dans Loom sans modification. Les fichiers de **comptabilité** de
Loom (`installed_plugins.json`, `known_marketplaces.json`) reprennent la forme de CC par
familiarité mais appartiennent à Loom ; ils ne sont pas partagés avec un éventuel CC présent
sur la machine.

### Formats lus / écrits

**`.claude-plugin/plugin.json`** (lu, format CC) :
```json
{ "name": "...", "version": "0.1.0", "description": "...",
  "author": { "name": "..." }, "license": "MIT" }
```
`version` peut valoir `"unknown"`. Seul `name` est requis ; le reste est optionnel.

**`.claude-plugin/marketplace.json`** (lu, format CC) :
```json
{ "name": "...", "owner": {...},
  "plugins": [ { "name": "...", "description": "...", "source": <SOURCE> } ] }
```
Formes de `<SOURCE>` à gérer (toutes observées sur la machine de l'utilisateur) :
- **chemin relatif** : `"./plugins/strategy-france"` → le plugin est dans le dépôt de la
  marketplace, à ce chemin.
- **git-subdir** : `{ "source": "git-subdir", "url": "...", "path": "plugins/x",
  "ref": "v1.5.5", "sha": "..." }` → cloner `url`, se placer sur `ref`/`sha`, plugin sous
  `path`.
- **git** : `{ "source": "git", "url": "...", "ref": "..."? }` → cloner le dépôt entier ;
  le plugin est à la racine.

**`installed_plugins.json`** (écrit par Loom, forme CC v2) :
```json
{ "version": 2, "plugins": {
  "<plugin>@<marketplace>": [ {
    "scope": "user",
    "installPath": "<abs cache dir>",
    "version": "0.1.0",
    "installedAt": "<iso>", "lastUpdated": "<iso>",
    "gitCommitSha": "<sha|''>" } ] } }
```

**`known_marketplaces.json`** (écrit par Loom) :
```json
{ "version": 1, "marketplaces": {
  "<name>": { "source": "<git-url|local-path>", "addedAt": "<iso>" } } }
```

## Composants (code)

### `loom/plugins.py` (nouveau)

Logique pure (résolution, parsing, scan) + I/O fichier + appels git encapsulés. Aucune
dépendance à Flask. Fonctions :

- `plugins_root(cfg) -> Path` — résout la racine du store (config ou défaut), crée si absent.
- `marketplace_add(root, source) -> str` — clone (git) ou copie (chemin local) la marketplace
  dans un dossier temporaire, lit `marketplace.json`, **renomme** le dossier d'après son
  `name`, l'installe sous `marketplaces/<name>/`, écrit `known_marketplaces.json`. Renvoie le
  nom. Lève une erreur claire si pas de `marketplace.json` valide.
- `marketplace_list(root) -> list[dict]` — depuis `known_marketplaces.json`.
- `plugin_install(root, ref) -> dict` — `ref` = `"<plugin>"` ou `"<plugin>@<marketplace>"`.
  Cherche l'entrée dans le(s) `marketplace.json`, résout la `source` (3 formes ci-dessus),
  matérialise le plugin sous `cache/<marketplace>/<plugin>/<version>/`, met à jour
  `installed_plugins.json`. `version` issue de `plugin.json` (sinon `"unknown"`).
- `plugin_add_local(root, path) -> dict` — **fallback B** : copie un dossier plugin
  (contenant `.claude-plugin/plugin.json`) sous `cache/_local/<plugin>/<version>/`, marketplace
  conventionnelle `_local`. Enregistre dans `installed_plugins.json`.
- `plugin_remove(root, ref) -> None` — retire du cache + de `installed_plugins.json`.
- `discover_plugins(root) -> list[Plugin]` — lit `installed_plugins.json` ; pour chaque
  `installPath`, parse `plugin.json` et **inventorie** : `skills` (liste de chemins
  `SKILL.md`), `agents`, `hooks`, `commands` (présence + comptes). **Fallback de découverte :**
  scanne aussi `cache/` pour les dossiers plugins valides absents de `installed_plugins.json`
  (dépôt manuel), en les rattachant. Renvoie des dataclasses `Plugin`.

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

**Git encapsulé** : un helper `_git_clone(url, dest, ref=None, sha=None)` via `subprocess`
(`git clone --depth 1`, puis `git checkout <sha|ref>` si fourni). Pour `git-subdir`, clone
puis copie le sous-dossier `path` vers la destination du cache. Échec réseau/git → message
actionnable (« git introuvable » / « clone échoué : … »).

### `loom/skills.py` (étendu)

- `Skill` gagne `base_dir: str = ""` (dossier du `SKILL.md`) pour résoudre `references/`.
- Nouvelle fonction `collect_skills(local_dir, plugins_root) -> list[Skill]` : agrège
  (a) les skills locaux de `local_dir` (comportement actuel, **non** namespacés, pour ne pas
  casser l'existant) et (b) ceux des plugins découverts, **namespacés** `"<plugin>:<nom>"`.
- `load_skill` accepte un nom namespacé.
- `compose_system_prompt` : pour chaque skill actif, préfixe
  `# Skill : <name>\nBase directory for this skill: <base_dir>\n\n<body>` afin que le modèle
  puisse `read_file` les `references/` via chemin absolu.

### CLI dans `loom/plugins.py` (`python -m loom.plugins`)

La surface CLI vit **dans le même module** `loom/plugins.py` (un `main()` argparse + garde
`if __name__ == "__main__"`), invoqué par `python -m loom.plugins`. Pas de second fichier
`loom/plugin.py` (singulier) — éviter la confusion d'import avec `plugins.py`.

Sous-commandes (argparse) :
- `marketplace add <git-url|chemin>`
- `marketplace list`
- `install <plugin[@marketplace]>`
- `add-local <chemin>`
- `list` (plugins installés + inventaire des composants : `skills:N agents:N hooks:N`)
- `remove <plugin[@marketplace]>`

Pas de **tool LLM** d'installation en tranche 1 : installer = cloner du code tiers, décision
de l'utilisateur, hors boucle agent.

### Câblage web (`loom/web/__main__.py` + `app.py` + `_skills.html`)

- `__main__.py` : calcule `plugins_root`, le passe à `create_app`.
- `app.py` : remplace les appels `list_skills(skills_dir)` par `collect_skills(skills_dir,
  plugins_root)` (route d'index + route `/skills`). `active_skills` (déjà persisté par
  conversation) stocke désormais des noms éventuellement namespacés.
- `_skills.html` : grouper l'affichage par plugin (skills locaux d'abord, puis un sous-titre
  par plugin). La case reste un `name=skill value="<nom namespacé>"`.

## Flux de données

```
CLI install ─► loom/plugins.py (git/copy ─► cache/ + installed_plugins.json)
                     │
web start ──► discover_plugins(root) ──► collect_skills(local, root)
                     │                          │
                     ▼                          ▼
            inventaire 4 types        liste Skills (locaux + plugin:skill)
                                              │
                            UI (panneau Skills groupé) ──cocher──► conversation.active_skills
                                              │
                            compose_system_prompt (+ Base directory) ──► prompt du modèle
```

## Gestion d'erreurs

- `marketplace.json` / `plugin.json` absent ou invalide → l'élément est **ignoré** avec un
  message clair (jamais de crash de la découverte).
- Échec git (binaire absent, réseau, ref inconnue) → message actionnable côté CLI, exit ≠ 0.
- Plugin sans `skills/` → 0 skill, l'inventaire montre les autres types.
- Collisions de noms de skills → évitées par le namespacing `plugin:skill`.
- Dépôt manuel mal formé dans `cache/` → ignoré par la découverte (pas de `plugin.json`).

## Sécurité

Tranche 1 ne fait que **lire** des skills (markdown injecté dans le prompt) : aucune
exécution de code tiers. Le risque d'exécution apparaît à la **tranche hooks** (scripts
shell lancés sur événement d'outil) — cette spec-là devra ajouter une **porte de confiance**
(confirmation/allowlist par plugin) avant tout lancement. À acter quand on y arrivera.

## Tests (contrainte projet : pas de pytest sur Loom)

Smokes `uv run python -c …`, **sans réseau**, sur une fixture locale créée en `tempfile` :
1. Créer une fausse marketplace locale (`marketplace.json` + un plugin `skills/demo/SKILL.md`,
   `source` = chemin relatif).
2. `marketplace_add(local_path)` puis `plugin_install("<plugin>")` → vérifier l'arborescence
   du cache + `installed_plugins.json`.
3. `discover_plugins` → le plugin et son skill sont inventoriés.
4. `collect_skills` → le skill apparaît namespacé `demo_plugin:demo`, `base_dir` pointe sur le
   bon dossier.
5. `plugin_add_local(dir)` (fallback B) → découvert sous marketplace `_local`.
Les chemins **git / git-subdir** sont vérifiés à la main (un vrai clone), hors smoke
automatique.

## Décisions actées

- **C** (store Loom autonome, format CC) plutôt que lire `~/.claude/` : Loom ne dépend pas de
  Claude Code.
- **B** comme repli manuel (dépôt local), pas comme mécanisme principal.
- Activation **par skill** (cases existantes), pas d'enable/disable par plugin en tranche 1.
- Surface **CLI** uniquement (pas de slash-commands dans Loom, pas de tool LLM d'install).
- Skills locaux **non** namespacés (compat existant) ; skills de plugin **namespacés**.
