# tests/test_permissions.py
from loom.permissions import (
    DEFAULT_DENY,
    Decision,
    PermissionConfig,
    evaluate,
    parse_permissions,
)


def _shell(cmd: str) -> dict:
    return {"command": cmd}


# --- deny-list dure : non contournable même en mode allow/allowlist ---


def test_rm_rf_denied_even_in_allowlist_with_command_allowed():
    cfg = PermissionConfig(mode="allowlist", allow_commands=["rm"], deny_commands=[])
    d = evaluate("run_shell", _shell("rm -rf /"), cfg)
    assert d.action == "deny"


def test_rm_fr_variant_denied():
    cfg = PermissionConfig(mode="allowlist", allow_commands=["rm"])
    assert evaluate("run_shell", _shell("rm -fr /tmp/x"), cfg).action == "deny"


def test_remove_item_recurse_force_denied_various_order_and_case():
    cfg = PermissionConfig(mode="allowlist", allow_commands=["Remove-Item"])
    for cmd in (
        "Remove-Item -Recurse -Force C:/x",
        "remove-item -force -recurse C:/x",
        "REMOVE-ITEM C:/x -Recurse -Force",
    ):
        assert evaluate("run_shell", _shell(cmd), cfg).action == "deny", cmd


def test_remove_item_without_both_flags_not_hard_denied():
    cfg = PermissionConfig(mode="allowlist", allow_commands=["Remove-Item"])
    # uniquement -Force, pas -Recurse -> pas dans la deny-list dure
    d = evaluate("run_shell", _shell("Remove-Item -Force a.txt"), cfg)
    assert d.action != "deny"


def test_various_hard_denied_patterns():
    cfg = PermissionConfig(mode="allow")
    for cmd in (
        "rmdir /s C:/x",
        "del /f a.txt",
        "format C:",
        "mkfs.ext4 /dev/sda",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown -h now",
        ":(){ :|:& };:",
        "git reset --hard HEAD~1",
        "git clean -fd",
    ):
        assert evaluate("run_shell", _shell(cmd), cfg).action == "deny", cmd


def test_bash_tool_name_also_hard_denied():
    cfg = PermissionConfig(mode="allow")
    assert evaluate("bash", _shell("rm -rf /"), cfg).action == "deny"


# --- mode allow : shell normal autorisé ---


def test_allow_mode_runs_normal_shell():
    cfg = PermissionConfig(mode="allow")
    assert evaluate("run_shell", _shell("ls -la"), cfg).action == "allow"


# --- mode deny_all ---


def test_deny_all_denies_shell_edit_write():
    cfg = PermissionConfig(mode="deny_all")
    assert evaluate("run_shell", _shell("ls"), cfg).action == "deny"
    assert evaluate("write_file", {"path": "a.txt"}, cfg).action == "deny"
    assert evaluate("edit_file", {"path": "a.txt"}, cfg).action == "deny"


# --- lecture toujours autorisée ---


def test_read_tools_always_allow():
    cfg = PermissionConfig(mode="deny_all")
    assert evaluate("read_file", {"path": "a.txt"}, cfg).action == "allow"
    assert evaluate("web_search", {"q": "x"}, cfg).action == "allow"
    assert evaluate("list_dir", {"path": "."}, cfg).action == "allow"


# --- write/edit : hors workspace -> deny ---


def test_write_outside_workspace_denied(tmp_path):
    cfg = PermissionConfig(mode="ask", workspace_root=str(tmp_path))
    d = evaluate("write_file", {"path": "../evade.txt"}, cfg)
    assert d.action == "deny"


def test_write_inside_workspace_ask_in_ask_mode(tmp_path):
    cfg = PermissionConfig(mode="ask", workspace_root=str(tmp_path))
    d = evaluate("write_file", {"path": "sub/ok.txt"}, cfg)
    assert d.action == "ask"


def test_write_allowlist_path_allowed(tmp_path):
    cfg = PermissionConfig(
        mode="allowlist", workspace_root=str(tmp_path), allow_paths=["src"]
    )
    assert evaluate("write_file", {"path": "src/a.py"}, cfg).action == "allow"
    # chemin non listé -> ask
    assert evaluate("write_file", {"path": "other/a.py"}, cfg).action == "ask"


def test_shell_allowlist_command_allowed():
    cfg = PermissionConfig(mode="allowlist", allow_commands=["git status", "ls"])
    assert evaluate("run_shell", _shell("git status"), cfg).action == "allow"
    assert evaluate("run_shell", _shell("npm install"), cfg).action == "ask"


# --- parse_permissions ---


def test_parse_defaults_when_section_absent():
    cfg = parse_permissions({})
    assert cfg.mode == "ask"
    assert cfg.workspace_root == "."
    assert cfg.allow_commands == []
    assert cfg.allow_paths == []
    assert cfg.deny_commands == []


def test_parse_reads_section():
    cfg = parse_permissions(
        {
            "permissions": {
                "mode": "allowlist",
                "workspace_root": "/w",
                "allow_commands": ["git status"],
                "allow_paths": ["src"],
                "deny_commands": ["curl"],
            }
        }
    )
    assert cfg.mode == "allowlist"
    assert cfg.workspace_root == "/w"
    assert cfg.allow_commands == ["git status"]
    assert cfg.allow_paths == ["src"]
    assert cfg.deny_commands == ["curl"]


def test_custom_deny_command_denied():
    cfg = PermissionConfig(mode="allow", deny_commands=["curl"])
    assert evaluate("run_shell", _shell("curl http://x"), cfg).action == "deny"


def test_decision_is_frozen_dataclass():
    d = Decision("allow")
    assert d.action == "allow"
    assert d.reason == ""


def test_default_deny_is_nonempty():
    assert len(DEFAULT_DENY) > 0
