"""Tool fallbacks and destructive-operation policy for the agent loop.

Extracted without changing existing transformations, patterns, or permission logic.
"""
from __future__ import annotations
import base64
import re
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------
# Bash/terminal fallback transforms
# ----------------------------------------------------------------------

def _sh_single(s: str) -> str:
    return str(s).replace("'", "'\\''")


def _fallback_error_command(msg: str) -> Dict[str, Any]:
    return {"command": f"echo {_sh_single('FALLBACK ERROR: ' + msg)} >&2; exit 1"}


def _bash_fallback_write(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("filePath") or args.get("path") or args.get("filename") or ""
    content = args.get("content")
    if content is None:
        content = args.get("text") or args.get("data") or ""
    if not path:
        return _fallback_error_command(
            "write called without filePath/path; cannot fall back to bash."
        )
    # Base64 avoids heredoc delimiters and shell expansion entirely.  A
    # model-supplied line equal to the old EOF marker could otherwise escape
    # the heredoc and run arbitrary shell commands.
    encoded = base64.b64encode(str(content).encode("utf-8")).decode("ascii")
    script = (
        "import base64,pathlib,sys;"
        "pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode(sys.argv[2]))"
    )
    return {
        "command": (
            f"python3 -c '{script}' "
            f"'{_sh_single(path)}' '{encoded}'"
        )
    }


def _bash_fallback_read(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("filePath") or args.get("path") or args.get("filename") or ""
    if not path:
        return _fallback_error_command(
            "read called without filePath/path; cannot fall back to bash."
        )
    return {"command": f"cat '{_sh_single(path)}'"}


def _bash_fallback_list(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path") or args.get("dir") or "."
    return {"command": f"ls -la '{_sh_single(path)}'"}


def _bash_fallback_edit(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("filePath") or args.get("path") or args.get("filename") or ""
    old_str = args.get("oldString") or args.get("old_text") or ""
    new_str = args.get("newString") or args.get("new_text") or ""
    if not path:
        return _fallback_error_command(
            "edit called without filePath/path; cannot fall back to bash."
        )
    if not old_str:
        return _fallback_error_command(
            "edit called without oldString; cannot fall back to bash."
        )
    old_b64 = base64.b64encode(old_str.encode("utf-8")).decode("ascii")
    new_b64 = base64.b64encode(new_str.encode("utf-8")).decode("ascii")
    script = (
        "import base64,pathlib,sys;"
        "p=pathlib.Path(sys.argv[1]);"
        "s=p.read_text();"
        "old=base64.b64decode(sys.argv[2]).decode();"
        "new=base64.b64decode(sys.argv[3]).decode();"
        "sys.exit(3) if old not in s else (p.write_text(s.replace(old,new,1)) or 0)"
    )
    return {
        "command": (
            f"python3 -c '{script}' "
            f"'{_sh_single(path)}' '{old_b64}' '{new_b64}'"
        )
    }


def _bash_fallback_glob(args: Dict[str, Any]) -> Dict[str, Any]:
    root = args.get("path") or args.get("dir") or "."
    pattern = args.get("pattern") or args.get("glob") or "*"
    return {
        "command": (
            f"find '{_sh_single(root)}' -name '{_sh_single(pattern)}' -type f"
        )
    }


def _bash_fallback_grep(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args.get("pattern") or args.get("query") or ""
    root = args.get("path") or args.get("dir") or "."
    if not pattern:
        return _fallback_error_command(
            "grep called without pattern; cannot fall back to bash."
        )
    return {
        "command": f"grep -rn -- '{_sh_single(pattern)}' '{_sh_single(root)}'"
    }


def _bash_fallback_patch(args: Dict[str, Any]) -> Dict[str, Any]:
    patch_file = (
        args.get("patchFile") or args.get("patch_file")
        or args.get("path") or ""
    )
    if not patch_file:
        return _fallback_error_command(
            "apply_patch called without patchFile/path; cannot fall back to bash."
        )
    return {"command": f"patch -p1 < '{_sh_single(patch_file)}'"}


def _bash_fallback_filesystem(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path") or args.get("dir") or "."
    return {"command": f"ls -la '{_sh_single(path)}'"}


TOOL_FALLBACKS: Dict[str, List[tuple]] = {
    "write": [("bash", _bash_fallback_write), ("terminal", _bash_fallback_write)],
    "edit": [("bash", _bash_fallback_edit), ("terminal", _bash_fallback_edit)],
    "apply_patch": [("bash", _bash_fallback_patch), ("terminal", _bash_fallback_patch)],
    "read": [("bash", _bash_fallback_read), ("terminal", _bash_fallback_read)],
    "list": [("bash", _bash_fallback_list), ("terminal", _bash_fallback_list)],
    "glob": [("bash", _bash_fallback_glob), ("terminal", _bash_fallback_glob)],
    "grep": [("bash", _bash_fallback_grep), ("terminal", _bash_fallback_grep)],
    "filesystem": [
        ("bash", _bash_fallback_filesystem),
        ("terminal", _bash_fallback_filesystem),
    ],
    "patch": [("bash", _bash_fallback_patch), ("terminal", _bash_fallback_patch)],
}


# ----------------------------------------------------------------------
# Destructive-call detection
# ----------------------------------------------------------------------
_DESTRUCTIVE_TOOL_TOKENS = (
    "delete", "remove", "unlink", "destroy", "purge", "wipe", "erase",
    "rmdir", "shred", "truncate", "drop",
)

_DESTRUCTIVE_ACTION_TOKENS = (
    "delete", "remove", "reset", "clean", "drop", "prune",
    "destroy", "purge", "wipe", "erase", "truncate",
    "force", "overwrite", "hard", "kill", "terminate",
)

_SHELL_TOOL_TOKENS = ("bash", "terminal", "shell", "exec", "process", "proc")
_SHELL_ARG_KEYS = frozenset({"command", "cmd", "script", "shell", "exec", "cmdline"})

_DESTRUCTIVE_SHELL_PATTERNS = (
    re.compile(r"\brm\b"),
    re.compile(r"\brmdir\b"),
    re.compile(r"\bunlink\b"),
    re.compile(r"\bshred\b"),
    re.compile(r"\btruncate\b"),
    re.compile(r"\bdd\b.*\bof="),
    re.compile(r"\bmkfs\b"),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\s+-[a-z]*f"),
    re.compile(r"\bgit\s+push\s+--force\b"),
    re.compile(r"\bkubectl\s+delete\b"),
    re.compile(r"\baws\s+s3\s+rm\b"),
    re.compile(r"\baws\s+s3api\s+delete"),
    re.compile(r"\baws\s+ec2\s+terminate"),
    re.compile(r"\baws\s+rds\s+delete"),
    re.compile(r"\bdocker\s+rm\b"),
    re.compile(r"\bdocker\s+rmi\b"),
    re.compile(r"\bdocker\s+system\s+prune\b"),
    re.compile(r"\bdocker\s+volume\s+rm\b"),
    re.compile(r"\bfind\b.*\s-delete\b"),
    re.compile(r"curl\b.*\|\s*(?:ba)?sh\b"),
    re.compile(r"wget\b.*\|\s*(?:ba)?sh\b"),
)

_DEMO_OFFER_PATTERNS = (
    re.compile(r"\bwould you like me to\b", re.I),
    re.compile(r"\bwant me to (run|show|launch|demo|open|execute)\b", re.I),
    re.compile(r"\bshall i (run|show|launch|demo|open|execute)\b", re.I),
    re.compile(r"\bi can (run|show|launch|demo|open|execute)\b", re.I),
    re.compile(r"\bready (for me )?to (run|show|launch|demo)\b", re.I),
)

_USER_DECLINED_PATTERNS = (
    re.compile(r"\bno\b\s*,?\s*(thanks|thank you|thx)\b", re.I),
    re.compile(r"\bdon'?t\b.*\b(run|launch|show)\b", re.I),
    re.compile(r"\bnot?\b.*\b(run|launch|show)\b", re.I),
    re.compile(r"\bjust\b.*\b(explain|tell me|describe)\b", re.I),
)


def _looks_destructive_call(name: str, args: Dict[str, Any]) -> bool:
    name_l = (name or "").lower()

    for tok in _DESTRUCTIVE_TOOL_TOKENS:
        if tok in name_l:
            return True

    action = ""
    if isinstance(args, dict):
        for k in ("action", "operation", "op", "verb"):
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                action = v.strip().lower()
                break
    if not action:
        for tok in _DESTRUCTIVE_ACTION_TOKENS:
            if tok in name_l:
                action = tok
                break
    if action:
        for tok in _DESTRUCTIVE_ACTION_TOKENS:
            if tok in action:
                return True

    if isinstance(args, dict):
        is_shell_tool = any(tok in name_l for tok in _SHELL_TOOL_TOKENS)
        for key, val in args.items():
            if not isinstance(val, str) or not val:
                continue
            if not is_shell_tool and str(key).lower() not in _SHELL_ARG_KEYS:
                continue
            for pat in _DESTRUCTIVE_SHELL_PATTERNS:
                if pat.search(val):
                    return True

    return False


def _mutation_target(name: str, args: Dict[str, Any]) -> Optional[str]:
    """Best-effort target identity for a mutating call.

    Used to scope verification clearing so a read of file A does not clear
    a pending verification for file B.
    """
    if not isinstance(args, dict):
        return None
    for key in ("path", "filePath", "file_path", "filename", "file", "target"):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    cmd = args.get("command")
    if isinstance(cmd, str) and cmd.strip():
        m = re.search(r">\s*([^\s|;&]+)", cmd)
        if m:
            return m.group(1).strip()
        return f"cmd::{cmd.strip()[:120]}"
    return None


def _read_target(name: str, args: Dict[str, Any]) -> Optional[str]:
    """Target identity for a read call, matching _mutation_target keys."""
    return _mutation_target(name, args)
