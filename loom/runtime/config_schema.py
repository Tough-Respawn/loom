"""Introspection + édition de la config Loom pour la CONSOLE (UI).

Modèle à deux axes (voulu par l'utilisateur) :
  - COUCHE : `commun` (portable, tout OS/machine -> config/defaults.toml versionné) vs
    `systeme` (cette machine/cet OS -> config/local.toml gitignored).
  - NATURE : `fixe` (ne bouge pas), `override` (défaut commun surchargeable), `libre` (réglé
    au quotidien). Sert juste de badge visuel dans la console.

Le SPEC ci-dessous est la source unique de la surface éditable : chaque entrée porte son aide
CONDENSÉE (une ligne) -> le savoir des murs de commentaires devient une infobulle à la demande.
L'écriture passe par tomlkit (édite la valeur, PRÉSERVE commentaires et structure du fichier).
Les modèles (locaux par dossier, distants gérés à part) ne sont PAS ici : ils ont leur propre
gestionnaire. Cette console couvre les sections scalaires.
"""

from __future__ import annotations

from pathlib import Path

import tomlkit

# Libellés lisibles des sections (ordre d'affichage = ordre du dict).
SECTION_LABELS = {
    "server": "Serveur (llama.cpp)",
    "override": "Override GPU / CPU",
    "chat": "Chat / runtime",
    "tools": "Outils",
    "memory": "Mémoire",
    "web_search": "Recherche web",
    "permissions": "Permissions",
}

# type : int | float | bool | str | secret | select | list
# layer : commun | systeme      nature : fixe | override | libre
# applies : live (effet immédiat / prochain appel) | restart (relance llama/loom requise)
# editable défaut True ; options = valeurs d'un select.
SPEC: list[dict] = [
    # -- server --
    {
        "section": "server",
        "key": "context",
        "label": "Contexte serveur (tokens)",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "restart",
        "help": "Fenêtre de contexte du serveur local. Borne le KV en VRAM.",
    },
    {
        "section": "server",
        "key": "n_parallel",
        "label": "Slots parallèles",
        "layer": "commun",
        "nature": "fixe",
        "type": "int",
        "applies": "restart",
        "help": "Slots llama-server. Loom est mono-flux -> 1.",
    },
    {
        "section": "server",
        "key": "gpu_kv_headroom_mb",
        "label": "Marge VRAM (Mo)",
        "layer": "systeme",
        "nature": "override",
        "type": "int",
        "applies": "restart",
        "help": "Réserve VRAM hors couches offloadées (KV + buffers). Trop bas -> spill mémoire partagée.",
    },
    {
        "section": "server",
        "key": "port",
        "label": "Port du modèle local",
        "layer": "systeme",
        "nature": "override",
        "type": "int",
        "applies": "restart",
        "help": "Port du serveur llama-swap/llama-server.",
    },
    {
        "section": "server",
        "key": "bin",
        "label": "Binaire llama-server",
        "layer": "systeme",
        "nature": "override",
        "type": "str",
        "applies": "restart",
        "help": "Chemin de llama-server sur cette machine.",
    },
    {
        "section": "server",
        "key": "swap_bin",
        "label": "Binaire llama-swap",
        "layer": "systeme",
        "nature": "override",
        "type": "str",
        "applies": "restart",
        "help": "Chemin de llama-swap sur cette machine.",
    },
    # -- override --
    {
        "section": "override",
        "key": "n_gpu_layers",
        "label": "Forcer couches GPU",
        "layer": "systeme",
        "nature": "override",
        "type": "int",
        "applies": "restart",
        "help": "Force l'offload GPU (nb de couches). Vide = auto-détecté.",
    },
    {
        "section": "override",
        "key": "threads",
        "label": "Forcer threads CPU",
        "layer": "systeme",
        "nature": "override",
        "type": "int",
        "applies": "restart",
        "help": "Force le nombre de threads CPU. Vide = auto.",
    },
    # -- chat --
    {
        "section": "chat",
        "key": "default_model",
        "label": "Modèle par défaut",
        "layer": "commun",
        "nature": "libre",
        "type": "str",
        "applies": "restart",
        "help": "Modèle chargé au démarrage (le sélecteur hot-swap ensuite).",
    },
    {
        "section": "chat",
        "key": "max_tokens",
        "label": "Sortie max / tour",
        "layer": "commun",
        "nature": "libre",
        "type": "int",
        "applies": "live",
        "help": "Plafond de tokens générés par tour.",
    },
    {
        "section": "chat",
        "key": "request_timeout",
        "label": "Timeout requête (s)",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Délai max d'une requête au modèle.",
    },
    {
        "section": "chat",
        "key": "max_retries",
        "label": "Retries API",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Tentatives sur erreur transitoire.",
    },
    {
        "section": "chat",
        "key": "context_token_budget",
        "label": "Budget résumé (tokens)",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Budget du résumé inter-tours de l'historique.",
    },
    {
        "section": "chat",
        "key": "keep_recent_messages",
        "label": "Messages récents gardés",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Nb de messages récents laissés intacts au résumé.",
    },
    {
        "section": "chat",
        "key": "web_port",
        "label": "Port de l'UI",
        "layer": "systeme",
        "nature": "override",
        "type": "int",
        "applies": "restart",
        "help": "Port du serveur web Loom.",
    },
    {
        "section": "chat",
        "key": "identity_max_tokens",
        "label": "Budget identité",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Budget du bloc identité (SOUL/USER/MEMORY) injecté au prompt.",
    },
    {
        "section": "chat",
        "key": "keepwarm_enabled",
        "label": "Keep-warm modèle",
        "layer": "systeme",
        "nature": "libre",
        "type": "bool",
        "applies": "live",
        "help": "Ping d'entretien du modèle local (évite le cold-start Windows).",
    },
    {
        "section": "chat",
        "key": "keepwarm_interval",
        "label": "Intervalle keep-warm (s)",
        "layer": "systeme",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Période du ping d'entretien à l'idle.",
    },
    {
        "section": "chat",
        "key": "reflect_enabled",
        "label": "Auto-apprentissage",
        "layer": "commun",
        "nature": "libre",
        "type": "bool",
        "applies": "live",
        "help": "Étape reflect post-tour (mémoire + skills appris).",
    },
    {
        "section": "chat",
        "key": "reflect_min_actions",
        "label": "Reflect: actions min",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Nb minimal d'actions dans un tour pour déclencher reflect.",
    },
    # -- tools --
    {
        "section": "tools",
        "key": "workspace_dir",
        "label": "Racine des projets",
        "layer": "systeme",
        "nature": "libre",
        "type": "str",
        "applies": "live",
        "help": "Racine autorisée des outils sur cette machine (anti-traversal).",
    },
    {
        "section": "tools",
        "key": "read_file_max_bytes",
        "label": "read_file max (car.)",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Cap par appel read_file. Bas -> lecture par tranches, pas de débordement.",
    },
    {
        "section": "tools",
        "key": "shell_timeout",
        "label": "Timeout run_shell (s)",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Délai max d'une commande shell (npm install/build).",
    },
    # -- memory --
    {
        "section": "memory",
        "key": "recall_summarize",
        "label": "Résumer le recall",
        "layer": "commun",
        "nature": "libre",
        "type": "bool",
        "applies": "restart",
        "help": "Condense les souvenirs FTS en une note dense (petit modèle).",
    },
    {
        "section": "memory",
        "key": "recall_summarize_threshold",
        "label": "Seuil de résumé recall",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "restart",
        "help": "Nb de hits au-delà duquel on résume le recall.",
    },
    # -- web_search --
    {
        "section": "web_search",
        "key": "enabled",
        "label": "Recherche web active",
        "layer": "commun",
        "nature": "libre",
        "type": "bool",
        "applies": "live",
        "help": "Active l'outil de recherche web.",
    },
    {
        "section": "web_search",
        "key": "backend",
        "label": "Backend",
        "layer": "commun",
        "nature": "libre",
        "type": "select",
        "options": ["auto", "searxng", "tavily", "ddgs"],
        "applies": "live",
        "help": "Fournisseur de recherche.",
    },
    {
        "section": "web_search",
        "key": "searxng_url",
        "label": "URL SearXNG",
        "layer": "systeme",
        "nature": "override",
        "type": "str",
        "applies": "live",
        "help": "Instance SearXNG auto-hébergée (si backend searxng).",
    },
    {
        "section": "web_search",
        "key": "tavily_api_key",
        "label": "Clé Tavily",
        "layer": "systeme",
        "nature": "override",
        "type": "secret",
        "applies": "live",
        "help": "Clé API Tavily (si backend tavily).",
    },
    {
        "section": "web_search",
        "key": "max_results",
        "label": "Résultats max",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Nb de résultats par recherche.",
    },
    {
        "section": "web_search",
        "key": "fetch_pages",
        "label": "Scraper les pages",
        "layer": "commun",
        "nature": "libre",
        "type": "bool",
        "applies": "live",
        "help": "false = extraits seulement (rapide) ; true = contenu complet.",
    },
    {
        "section": "web_search",
        "key": "http_timeout",
        "label": "Timeout HTTP (s)",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Délai des requêtes de recherche/scraping.",
    },
    {
        "section": "web_search",
        "key": "max_chars_per_page",
        "label": "Car. max / page",
        "layer": "commun",
        "nature": "override",
        "type": "int",
        "applies": "live",
        "help": "Troncature du contenu scrapé par page.",
    },
    # -- permissions --
    {
        "section": "permissions",
        "key": "mode",
        "label": "Mode d'exécution",
        "layer": "commun",
        "nature": "libre",
        "type": "select",
        "options": ["allow", "ask", "allowlist", "deny_all"],
        "applies": "live",
        "help": "allow = outils sans confirmation ; ask = confirmation. Deny-list dure toujours active.",
    },
]


# Défauts du CODE (dataclasses) pour les params absents des DEUX fichiers -> la console montre
# la vraie valeur effective, pas une case vide. Miroir des defaults de config.py/web.py ;
# `None` = pas de valeur (ex. override auto-détecté) -> champ vide avec placeholder « auto ».
CODE_DEFAULTS = {
    ("server", "context"): 24576,
    ("server", "n_parallel"): 1,
    ("server", "gpu_kv_headroom_mb"): 1024,
    ("server", "port"): 8080,
    ("server", "bin"): "llama-server",
    ("server", "swap_bin"): "llama-swap",
    ("override", "n_gpu_layers"): None,
    ("override", "threads"): None,
    ("chat", "default_model"): "",
    ("chat", "max_tokens"): 2048,
    ("chat", "request_timeout"): 120,
    ("chat", "max_retries"): 6,
    ("chat", "context_token_budget"): 3000,
    ("chat", "keep_recent_messages"): 6,
    ("chat", "web_port"): 8000,
    ("chat", "identity_max_tokens"): 600,
    ("chat", "keepwarm_enabled"): True,
    ("chat", "keepwarm_interval"): 150,
    ("chat", "reflect_enabled"): True,
    ("chat", "reflect_min_actions"): 1,
    ("tools", "workspace_dir"): ".",
    ("tools", "read_file_max_bytes"): 40000,
    ("tools", "shell_timeout"): 180,
    ("memory", "recall_summarize"): True,
    ("memory", "recall_summarize_threshold"): 5,
    ("web_search", "enabled"): True,
    ("web_search", "backend"): "auto",
    ("web_search", "searxng_url"): "",
    ("web_search", "tavily_api_key"): "",
    ("web_search", "max_results"): 5,
    ("web_search", "fetch_pages"): True,
    ("web_search", "http_timeout"): 6,
    ("web_search", "max_chars_per_page"): 4000,
    ("permissions", "mode"): "ask",
}


# Paramètres AVANCÉS : repliés par défaut dans la console (rarement touchés). Garde la vue
# initiale légère -> l'essentiel visible, le reste sous « Réglages avancés ».
ADVANCED = {
    ("server", "n_parallel"),
    ("server", "port"),
    ("override", "threads"),
    ("chat", "request_timeout"),
    ("chat", "max_retries"),
    ("chat", "context_token_budget"),
    ("chat", "keep_recent_messages"),
    ("chat", "identity_max_tokens"),
    ("chat", "reflect_min_actions"),
    ("chat", "keepwarm_interval"),
    ("chat", "web_port"),
    ("tools", "read_file_max_bytes"),
    ("memory", "recall_summarize"),
    ("memory", "recall_summarize_threshold"),
    ("web_search", "backend"),
    ("web_search", "searxng_url"),
    ("web_search", "tavily_api_key"),
    ("web_search", "max_results"),
    ("web_search", "http_timeout"),
    ("web_search", "max_chars_per_page"),
}


def _read_toml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return tomlkit.parse(p.read_text(encoding="utf-8")).unwrap()
    except (OSError, ValueError):
        return {}


def _get(d: dict, section: str, key: str):
    sec = d.get(section)
    if isinstance(sec, dict) and key in sec:
        return sec[key]
    return None


def describe(defaults_path: str | Path, local_path: str | Path) -> dict:
    """Surface éditable groupée par section, avec valeur EFFECTIVE + provenance.

    Provenance (`source`) : `systeme` si la clé est posée dans local.toml, `commun` si elle
    vient de defaults.toml, `defaut` si aucune des deux (valeur par défaut du code). Permet à
    la console d'afficher « surchargé pour cette machine » vs « défaut commun »."""
    defaults = _read_toml(defaults_path)
    local = _read_toml(local_path)
    sections: dict[str, list] = {}
    for spec in SPEC:
        s, k = spec["section"], spec["key"]
        in_local = _get(local, s, k)
        in_def = _get(defaults, s, k)
        if in_local is not None:
            value, source = in_local, "systeme"
        elif in_def is not None:
            value, source = in_def, "commun"
        else:
            value, source = CODE_DEFAULTS.get((s, k)), "defaut"
        row = dict(spec)
        row["value"] = value
        row["source"] = source
        row["advanced"] = (s, k) in ADVANCED
        sections.setdefault(s, []).append(row)
    return {
        "sections": [
            {"section": s, "label": SECTION_LABELS.get(s, s), "params": sections[s]}
            for s in SECTION_LABELS
            if s in sections
        ]
    }


def _coerce(spec: dict, raw):
    """Convertit la valeur brute de l'UI selon le type déclaré. Renvoie None si vide (=reset)."""
    t = spec["type"]
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip() == "" and t in ("int", "float"):
        return None
    if t == "int":
        return int(raw)
    if t == "float":
        return float(raw)
    if t == "bool":
        return (
            bool(raw)
            if isinstance(raw, bool)
            else str(raw).lower() in ("1", "true", "on", "yes")
        )
    return str(raw)  # str | secret | select


def _spec_for(section: str, key: str) -> dict | None:
    for s in SPEC:
        if s["section"] == section and s["key"] == key:
            return s
    return None


def _target_path(spec: dict, defaults_path, local_path) -> Path:
    """Fichier cible selon la COUCHE : systeme -> local.toml, commun -> defaults.toml."""
    return Path(local_path if spec["layer"] == "systeme" else defaults_path)


def set_value(defaults_path, local_path, section: str, key: str, raw) -> dict:
    """Écrit une valeur dans le bon fichier (par couche), commentaires PRÉSERVÉS (tomlkit).
    Valeur vide sur un champ numérique -> retrait de la clé (retour au défaut). Renvoie
    {ok, source} après écriture."""
    spec = _spec_for(section, key)
    if not spec:
        return {"ok": False, "error": f"paramètre inconnu: {section}.{key}"}
    if spec.get("editable") is False:
        return {"ok": False, "error": "paramètre en lecture seule"}
    val = _coerce(spec, raw)
    target = _target_path(spec, defaults_path, local_path)
    if val is None:
        return reset_value(defaults_path, local_path, section, key)
    doc = (
        tomlkit.parse(target.read_text(encoding="utf-8"))
        if target.exists()
        else tomlkit.document()
    )
    if section not in doc:
        doc[section] = tomlkit.table()
    doc[section][key] = val
    target.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return {"ok": True, "source": spec["layer"]}


def reset_value(defaults_path, local_path, section: str, key: str) -> dict:
    """Retire la clé de son fichier de couche (revient au défaut inférieur). Commentaires
    préservés. No-op si la clé n'y est pas."""
    spec = _spec_for(section, key)
    if not spec:
        return {"ok": False, "error": f"paramètre inconnu: {section}.{key}"}
    target = _target_path(spec, defaults_path, local_path)
    if not target.exists():
        return {"ok": True, "source": "defaut"}
    doc = tomlkit.parse(target.read_text(encoding="utf-8"))
    if section in doc and key in doc[section]:
        del doc[section][key]
        target.write_text(tomlkit.dumps(doc), encoding="utf-8")
    return {"ok": True, "source": "defaut"}
