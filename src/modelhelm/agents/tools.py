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
import subprocess
from pathlib import Path

from modelhelm.policies.engine import PolicyEngine, PathScopeError
from modelhelm.git.inspector import GitInspector

# Commands matched (case-insensitively) as substrings before a shell
# command is allowed to run without triggering the destructive_commands
# policy check.
DESTRUCTIVE_PATTERNS = ["rm -rf", "del /f", "format ", "drop database", "drop table"]


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

    def read_file(self, path: str) -> str:
        target = self._resolve_scoped_path(path)
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

    def run_command(self, command: str) -> dict:
        if any(pattern in command.lower() for pattern in DESTRUCTIVE_PATTERNS):
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

        subprocess.run(
            ["git", "-C", str(self.repository), "add", "-A"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-m", message],
            check=True,
            capture_output=True,
        )
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
