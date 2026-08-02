"""Mode permission : décide d'autoriser, demander ou refuser un appel d'outil.

Module PUR (aucune I/O) et 100% testable. Le cœur sécurité est la `DEFAULT_DENY` :
des motifs durs (regex) refusés même en mode `allow` et même si l'utilisateur
confirme. Le reste suit le `mode` configuré (`ask`, `allow`, `allowlist`,
`deny_all`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Les outils sans effet destructeur ne demandent pas de confirmation.
READ_TOOLS = frozenset(
    {
        "read_file",
        "read_image",
        # Les calculs purs sont équivalents aux lectures pour la permission.
        "calculate",
        "current_date",
        "find_files",
        "search_text",
        "list_dir",
        "web_search",
        "fetch_url",
        "check_page",
        "manage_todos",
        "use_skill",
        "list_plugins",
        # La mémoire interne reste dans le répertoire de données de Loom.
        "write_note",
        "read_note",
        "recall",
        "remember",
        # Rendre un résultat structuré en mémoire n'est pas un effet à confirmer.
        "submit_result",
    }
)
SHELL_TOOLS = frozenset({"run_shell", "serve_and_check", "monitor"})
# Outils d'INSTALLATION de plugins : clonent du code tiers + écrivent sur disque -> gardés
# comme les écritures (ask par défaut). Le modèle peut proposer/lancer l'install, pas sans
# accord en mode 'ask'.
PLUGIN_TOOLS = frozenset({"add_marketplace", "install_plugin"})
WRITE_TOOLS = frozenset(
    {
        "write_file",
        "append_file",
        "edit_file",
        "format_code",
    }
)


# Les motifs irréversibles restent refusés dans tous les modes.
DEFAULT_DENY: tuple[re.Pattern, ...] = (
    re.compile(r"\brm\s+(-[a-z]*\s+)*-?(rf|fr)\b", re.IGNORECASE),
    re.compile(
        r"\brm\s+(-[a-z]*r[a-z]*\s+.*-[a-z]*f|-[a-z]*f[a-z]*\s+.*-[a-z]*r)",
        re.IGNORECASE,
    ),
    re.compile(r"remove-item\b(?=.*\brecurse\b)(?=.*\bforce\b)", re.IGNORECASE),
    re.compile(r"\brmdir\s+/s\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/f\b", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\bmkfs", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r":\(\)\{", re.IGNORECASE),
    re.compile(r"git\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(
        r"git\s+clean\s+-[a-z]*f[a-z]*d|git\s+clean\s+-[a-z]*d[a-z]*f", re.IGNORECASE
    ),
)


# Refuser toute écriture résolue vers des dossiers système ou secrets critiques.
DEFAULT_PROTECTED_PATHS: tuple[re.Pattern, ...] = (
    re.compile(r"^[a-z]:/windows(/|$)", re.IGNORECASE),
    re.compile(r"^[a-z]:/program files( \(x86\))?(/|$)", re.IGNORECASE),
    re.compile(r"^/(bin|sbin|boot|dev|proc|sys|usr|lib|lib64|etc)(/|$)", re.IGNORECASE),
    re.compile(r"^/system(/|$)", re.IGNORECASE),
    re.compile(r"/\.ssh/", re.IGNORECASE),
    re.compile(r"/\.gnupg/", re.IGNORECASE),
)


@dataclass(frozen=True)
class Decision:
    action: str  # 'allow' | 'ask' | 'deny'
    reason: str = ""


@dataclass
class PermissionConfig:
    mode: str = "ask"  # 'ask' | 'allow' | 'allowlist' | 'deny_all'
    allow_commands: list[str] = field(default_factory=list)
    allow_paths: list[str] = field(default_factory=list)
    deny_commands: list[str] = field(default_factory=list)
    deny_paths: list[str] = field(default_factory=list)


def parse_permissions(data: dict) -> PermissionConfig:
    """Construit la config depuis `data['permissions']` (absent => défauts)."""
    p = data.get("permissions", {})
    return PermissionConfig(
        mode=p.get("mode", "ask"),
        allow_commands=list(p.get("allow_commands", [])),
        allow_paths=list(p.get("allow_paths", [])),
        deny_commands=list(p.get("deny_commands", [])),
        deny_paths=list(p.get("deny_paths", [])),
    )


def _is_hard_denied(command: str, deny_commands: list[str]) -> bool:
    """Vrai si la commande matche la deny-list dure ou une deny custom."""
    for pat in DEFAULT_DENY:
        if pat.search(command):
            return True
    low = command.lower()
    return any(extra.lower() in low for extra in deny_commands)


def is_protected_write_path(path: str, deny_paths: list[str]) -> bool:
    """Vrai si `path` (résolu, absolu) vise un emplacement protégé en écriture : un
    motif dur de DEFAULT_PROTECTED_PATHS, ou un fragment de la deny custom `deny_paths`.
    Séparateurs normalisés en '/' pour matcher pareil sous Windows et Unix."""
    norm = str(path).replace("\\", "/")
    for pat in DEFAULT_PROTECTED_PATHS:
        if pat.search(norm):
            return True
    low = norm.lower()
    return any(extra.replace("\\", "/").lower() in low for extra in deny_paths)


def _command_allowlisted(command: str, allow_commands: list[str]) -> bool:
    low = command.strip().lower()
    return any(low.startswith(a.strip().lower()) for a in allow_commands)


def _path_allowlisted(rel: str, allow_paths: list[str]) -> bool:
    norm = rel.replace("\\", "/").strip("/")
    for a in allow_paths:
        a_norm = a.replace("\\", "/").strip("/")
        if norm == a_norm or norm.startswith(a_norm + "/"):
            return True
    return False


def evaluate(tool_name: str, args: dict, cfg: PermissionConfig) -> Decision:
    """Décide allow/ask/deny pour un appel d'outil selon la politique."""
    if tool_name in READ_TOOLS:
        return Decision("allow")

    if tool_name in SHELL_TOOLS:
        # Lister/arrêter un monitor existant ne lance aucun nouveau code. Seul
        # start suit exactement la barrière de run_shell.
        if tool_name == "monitor" and args.get("action") in ("list", "stop"):
            return Decision("allow")
        command = (args.get("command") or "").strip()
        if _is_hard_denied(command, cfg.deny_commands):
            return Decision("deny", "commande interdite par la politique de sécurité")
        if cfg.mode == "deny_all":
            return Decision("deny", "mode deny_all")
        if cfg.mode == "allow":
            return Decision("allow")
        if cfg.mode == "allowlist":
            if _command_allowlisted(command, cfg.allow_commands):
                return Decision("allow")
            return Decision("ask", "commande non listée")
        return Decision("ask")  # mode 'ask'

    if tool_name in PLUGIN_TOOLS:
        if cfg.mode == "deny_all":
            return Decision("deny", "mode deny_all")
        if cfg.mode == "allow":
            return Decision("allow")
        return Decision("ask")

    if tool_name in WRITE_TOOLS:
        rel = (args.get("path") or "").strip()
        # Hors workspace, les chemins protégés restent le garde-fou non contournable.
        if is_protected_write_path(rel, cfg.deny_paths):
            return Decision(
                "deny", "chemin protégé par la politique de sécurité (deny_paths)"
            )
        if cfg.mode == "deny_all":
            return Decision("deny", "mode deny_all")
        if cfg.mode == "allow":
            return Decision("allow")
        if cfg.mode == "allowlist":
            if _path_allowlisted(rel, cfg.allow_paths):
                return Decision("allow")
            return Decision("ask", "chemin non listé")
        return Decision("ask")  # mode 'ask'

    if tool_name in ("dispatch_agent", "run_workflow"):
        # Le Python d'un workflow n'est pas confiné: l'approuver équivaut à approuver du shell.
        if cfg.mode == "deny_all":
            return Decision("deny", "mode deny_all")
        if cfg.mode == "allow":
            return Decision("allow")
        return Decision("ask")

    # Demander par prudence pour tout outil inconnu.
    if cfg.mode == "deny_all":
        return Decision("deny", "mode deny_all")
    return Decision("ask")
