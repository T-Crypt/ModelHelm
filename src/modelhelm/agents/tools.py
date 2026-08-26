"""Policy-gated tools an autonomous agent can call to read/write files,
run shell commands, and interact with git within a scoped repository.

Safety model:
  - All filesystem access is resolved against ``self.repository`` and
    rejected with ``PathScopeError`` if the resolved path would escape it
    (e.g. via ``../`` traversal or an absolute path elsewhere).
  - State-changing operations (file writes, destructive shell commands,
    git commits) are gated through ``PolicyEngine.check()`` first:
      - "deny"  -> raises ToolDenied, operation never runs
      - "ask"   -> raises ToolNeedsApproval, operation never runs
      - "allow" -> operation proceeds
"""
import fnmatch
import re
import subprocess
from pathlib import Path

from modelhelm.policies.engine import PolicyEngine, PathScopeError
from modelhelm.git.inspector import GitInspector

# --- Phase 1 shell-command gating -------------------------------------------
#
# LIMITATION (deliberate, Phase 1): everything below is pattern matching over
# the raw command string, NOT a shell parser. It does not understand quoting,
# variable expansion, aliases, `sh -c "..."` nesting, base64-encoded payloads,
# or `;`/`&&` chaining into an equivalent-but-differently-spelled command. It
# exists to close the concrete bypasses that let a model route a gated
# operation (a commit, a push, a recursive delete) through `run_command` and
# skip the policy engine entirely. A later phase should replace it with real
# argv-level parsing (e.g. shlex + per-executable rules).

# Git subcommands that have a dedicated policy operation. When one of these
# appears inside a run_command string it MUST be routed to the same
# PolicyEngine operation the dedicated tool uses, or `run_command` becomes a
# universal bypass of the git gates. Word-boundary anchored so `gitcommit` or
# `mygit committer` do not match.
GIT_COMMIT_PATTERN = re.compile(r"\bgit\b[^\n;&|]*?\bcommit\b", re.IGNORECASE)
GIT_PUSH_PATTERN = re.compile(r"\bgit\b[^\n;&|]*?\bpush\b", re.IGNORECASE)
# A push carrying any of these flags is a *force* push, a separate (stricter)
# policy operation than a plain push.
FORCE_PUSH_FLAG_PATTERN = re.compile(
    r"(?:^|\s)(?:--force-with-lease(?:=\S*)?|--force|-f)(?=\s|$)", re.IGNORECASE
)

# Destructive-command blocklist. Case-insensitive regexes rather than plain
# substrings because flag order and spacing vary freely in real commands
# (`rm -rf`, `rm -fr`, `rm -r -f`, `rm  --recursive --force`, ...). Same Phase 1
# caveat as above: a blocklist, not a parser.
DESTRUCTIVE_PATTERNS = [
    # rm with BOTH a recurse flag and a force flag, in any order/spacing.
    re.compile(
        r"\brm\b(?=[^\n;&|]*(?:\s-\w*r|\s--recursive\b))"
        r"(?=[^\n;&|]*(?:\s-\w*f|\s--force\b))",
        re.IGNORECASE,
    ),
    # PowerShell Remove-Item with -Recurse and -Force (this machine is Windows).
    re.compile(
        r"\bRemove-Item\b(?=[^\n;&|]*\s-Recurse\b)(?=[^\n;&|]*\s-Force\b)",
        re.IGNORECASE,
    ),
    # PowerShell forced delete without recursion (the del /f equivalent).
    re.compile(r"\bRemove-Item\b[^\n;&|]*\s-Force\b", re.IGNORECASE),
    re.compile(r"\bdel\b[^\n;&|]*\s/f\b", re.IGNORECASE),
    re.compile(r"\bgit\b[^\n;&|]*\breset\b[^\n;&|]*\s--hard\b", re.IGNORECASE),
    re.compile(
        r"\bgit\b[^\n;&|]*\bclean\b(?=[^\n;&|]*(?:\s-\w*f|\s--force\b))", re.IGNORECASE
    ),
    re.compile(r"\bdd\b[^\n;&|]*\bif=", re.IGNORECASE),
    re.compile(r"\bmkfs(?:\.\w+)?\b", re.IGNORECASE),
    re.compile(r"\bchmod\b[^\n;&|]*\s-\w*R\w*\s+777\b"),
    # curl/wget piped straight into a shell interpreter.
    re.compile(
        r"\b(?:curl|wget|iwr|Invoke-WebRequest)\b[^\n]*\|[^\n]*\b(?:sh|bash|zsh|powershell|pwsh|iex|Invoke-Expression)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bformat\s", re.IGNORECASE),
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
]

# Files an agent must never read into the LM Studio conversation. Matched
# case-insensitively with fnmatch against both the file name and its
# repo-relative POSIX path. Phase 1: a short hardcoded list, not configurable.
SENSITIVE_READ_PATTERNS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "credentials*",
    ".git/config",
    "*/.git/config",
]


class ToolDenied(Exception):
    """Raised when a policy check returns "deny" for a requested operation."""


class ToolNeedsApproval(Exception):
    """Raised when a policy check returns "ask" for a requested operation.

    Carries the operation name and a human-readable detail so a caller
    (e.g. a CLI or approval UI) can surface what needs approval.
    """

    def __init__(self, operation: str, detail: str):
        self.operation = operation
        self.detail = detail
        super().__init__(f"{operation} requires approval: {detail}")


class AgentTools:
    """Tool surface exposed to an autonomous agent, scoped to one repository."""

    def __init__(self, repository: str, policy_engine: PolicyEngine):
        self.repository = Path(repository).resolve()
        self.policy_engine = policy_engine

    def _resolve_scoped_path(self, path: str) -> Path:
        """Resolve ``path`` relative to the repository root and verify it
        does not escape the repository (e.g. via ``../`` traversal).

        Raises PathScopeError if the resolved path is not the repository
        itself or a descendant of it.
        """
        resolved = (self.repository / path).resolve()
        if resolved != self.repository and self.repository not in resolved.parents:
            raise PathScopeError(f"path escapes repository scope: {path}")
        return resolved

    def _is_sensitive_path(self, target: Path) -> bool:
        """True if ``target`` looks like a credential/secret file that must not
        be read into the model conversation."""
        name = target.name.lower()
        try:
            relative = target.relative_to(self.repository).as_posix().lower()
        except ValueError:  # pragma: no cover - path is always scoped by caller
            relative = name
        return any(
            fnmatch.fnmatch(name, pattern.lower())
            or fnmatch.fnmatch(relative, pattern.lower())
            for pattern in SENSITIVE_READ_PATTERNS
        )

    def read_file(self, path: str) -> str:
        target = self._resolve_scoped_path(path)
        # Deny loudly rather than returning empty content: the model needs to
        # know why, or it will keep retrying the same read.
        if self._is_sensitive_path(target):
            raise ToolDenied(
                f"read_file denied: {path} matches a sensitive-file pattern "
                "(credentials, keys and .env files are never readable by the agent)"
            )
        return target.read_text()

    def write_file(self, path: str, content: str) -> str:
        verdict = self.policy_engine.check("file_write")
        if verdict == "deny":
            raise ToolDenied("file_write is denied by policy")

        target = self._resolve_scoped_path(path)

        if verdict == "ask":
            raise ToolNeedsApproval("file_write", f"write to {path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def list_directory(self, path: str = ".") -> list[str]:
        target = self._resolve_scoped_path(path)
        return sorted(p.name for p in target.iterdir())

    def _gate(self, operation: str, detail: str) -> None:
        """Apply one policy operation with the standard ask/deny/allow semantics."""
        verdict = self.policy_engine.check(operation)
        if verdict == "deny":
            raise ToolDenied(f"{operation} is denied by policy: {detail}")
        if verdict == "ask":
            raise ToolNeedsApproval(operation, detail)

    def run_command(self, command: str) -> dict:
        # A shell string can spell any gated operation the dedicated tools
        # gate, so route the recognisable ones through the SAME policy
        # operations first — otherwise `run_command("git commit -m x")` walks
        # straight past the git_commit gate. See the module-level note on the
        # limits of this pattern-matching approach.
        if GIT_PUSH_PATTERN.search(command):
            # Force-push is a stricter, separate policy operation; check it
            # instead of (not in addition to) the plain push gate.
            if FORCE_PUSH_FLAG_PATTERN.search(command):
                self._gate("force_push", command)
            else:
                self._gate("git_push", command)
        if GIT_COMMIT_PATTERN.search(command):
            self._gate("git_commit", command)

        if any(pattern.search(command) for pattern in DESTRUCTIVE_PATTERNS):
            verdict = self.policy_engine.check("destructive_commands")
            if verdict == "deny":
                raise ToolDenied(f"destructive command denied: {command}")
            if verdict == "ask":
                raise ToolNeedsApproval("destructive_commands", command)

        result = subprocess.run(
            command,
            shell=True,
            cwd=str(self.repository),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def git_diff(self) -> str:
        return GitInspector(str(self.repository)).diff_summary()

    def git_commit(self, message: str) -> str:
        verdict = self.policy_engine.check("git_commit")
        if verdict == "deny":
            raise ToolDenied("git_commit is denied by policy")
        if verdict == "ask":
            raise ToolNeedsApproval("git_commit", message)

        # git's own stderr/stdout is the only thing that distinguishes "nothing
        # to commit" from "no identity configured" from "pre-commit hook
        # rejected". Raise ToolDenied so the agent loop feeds the detail back to
        # the model as a tool result instead of killing the run.
        try:
            subprocess.run(
                ["git", "-C", str(self.repository), "add", "-A"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-C", str(self.repository), "commit", "-m", message],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            # "nothing to commit" is reported on stdout, not stderr.
            detail = (exc.stderr or "").strip() or (exc.stdout or "").strip()
            raise ToolDenied(f"git commit failed: {detail}") from exc
        return f"committed: {message}"


def get_tool_definitions() -> list[dict]:
    """Return OpenAI-style tool-calling JSON schemas for all agent tools,
    suitable for passing as ``chat_completion(tools=...)``.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file relative to the repository root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file relative to the repository root, creating it if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories at a path relative to the repository root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "."}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command inside the repository (e.g. run tests).",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": "Show a summary of uncommitted changes in the repository.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Stage and commit all changes with the given message.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        },
    ]
