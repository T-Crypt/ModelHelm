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

    def _dirty_files(self) -> set[str]:
        """Repo-relative paths with uncommitted changes.

        Uses ``-z`` (NUL-separated, never quoted/escaped) so paths containing
        spaces or unicode parse correctly, and so the fixed-width ``XY `` status
        prefix can be sliced reliably -- plain ``--porcelain`` output would lose
        a leading-space status code to whitespace stripping. With ``-z`` a
        rename emits the new path and the old path as two separate records, so
        the old path is skipped explicitly.
        """
        result = subprocess.run(
            ["git", "-C", self.repository, "status", "--porcelain", "-z"],
            capture_output=True,
            text=True,
            check=True,
        )
        records = [r for r in result.stdout.split("\0") if r]
        files = set()
        skip_next = False
        for record in records:
            if skip_next:
                # Origin path of the preceding rename/copy entry.
                skip_next = False
                continue
            # Fixed layout: two status columns, a space, then the path. Safe to
            # slice because -z output is not whitespace-stripped.
            status, path = record[:2], record[3:]
            if status[:1] in ("R", "C"):
                skip_next = True
            if path:
                files.add(path)
        return files

    def dirty_files(self) -> set[str]:
        """Repo-relative paths with uncommitted changes (public baseline hook)."""
        return self._dirty_files()

    def files_changed_since(
        self, base_commit: str, base_dirty_files: set[str] | None = None
    ) -> int:
        """Count distinct files changed since ``base_commit``.

        A bare ``git status`` count is wrong in both directions: it credits the
        agent with dirt that predated the task, and it drops to zero the moment
        the agent commits. So union the still-dirty files with the files
        changed by any commits made since ``base_commit``, then subtract files
        that were already dirty at the start and were never committed since.

        ``base_dirty_files`` is the working-tree dirt captured before the task
        began; pass it to exclude pre-existing changes the agent did not make.
        """
        changed = self._dirty_files()
        committed: set[str] = set()
        head = self._run(["rev-parse", "HEAD"])
        if base_commit and head != base_commit:
            diff = self._run(["diff", "--name-only", base_commit, "HEAD"])
            committed = {line.strip() for line in diff.splitlines() if line.strip()}
            changed.update(committed)
        if base_dirty_files:
            # A file that was already dirty still counts if a commit during
            # this run touched it -- that commit is the agent's doing.
            changed -= base_dirty_files - committed
        return len(changed)
