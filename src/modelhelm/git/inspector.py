import subprocess
from pydantic import BaseModel


class GitSnapshot(BaseModel):
    branch: str
    commit: str
    is_dirty: bool


class GitInspector:
    def __init__(self, repository: str):
        self.repository = repository

    def _run(self, args: list[str]) -> str:
        result = subprocess.run(
            ["git", "-C", self.repository, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def snapshot(self) -> GitSnapshot:
        branch = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        commit = self._run(["rev-parse", "HEAD"])
        status = self._run(["status", "--porcelain"])
        return GitSnapshot(branch=branch, commit=commit, is_dirty=bool(status))

    def diff_summary(self) -> str:
        return self._run(["diff", "--stat"])

    def files_changed_count(self) -> int:
        status = self._run(["status", "--porcelain"])
        if not status:
            return 0
        return len(status.splitlines())
