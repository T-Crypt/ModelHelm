import json
import sqlite3
import uuid
from datetime import datetime, timezone

from modelhelm.tasks.models import DelegatedTask, PendingApproval, TaskResult, TaskStatus


class TaskStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    repository TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_results (
                    task_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    task_id TEXT PRIMARY KEY,
                    pending_json TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
                )
                """
            )

    def create_task(self, description: str, repository: str) -> DelegatedTask:
        task = DelegatedTask(
            task_id=str(uuid.uuid4()),
            description=description,
            repository=repository,
            status="pending",
            model=None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, description, repository, status, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task.task_id, task.description, task.repository, task.status, task.model, task.created_at),
            )
        return task

    def get_task(self, task_id: str) -> DelegatedTask | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT task_id, description, repository, status, model, created_at "
                "FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return DelegatedTask(
            task_id=row[0],
            description=row[1],
            repository=row[2],
            status=row[3],
            model=row[4],
            created_at=row[5],
        )

    def set_status(self, task_id: str, status: TaskStatus, model: str | None = None) -> None:
        with self._connect() as conn:
            if model is not None:
                conn.execute(
                    "UPDATE tasks SET status = ?, model = ? WHERE task_id = ?",
                    (status, model, task_id),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (status, task_id),
                )

    def save_result(self, task_id: str, result: TaskResult) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO task_results (task_id, result_json) VALUES (?, ?)",
                (task_id, result.model_dump_json()),
            )

    def get_result(self, task_id: str) -> TaskResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM task_results WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return TaskResult(**json.loads(row[0]))

    def save_pending_approval(self, task_id: str, pending: PendingApproval) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pending_approvals (task_id, pending_json) VALUES (?, ?)",
                (task_id, pending.model_dump_json()),
            )

    def get_pending_approval(self, task_id: str) -> PendingApproval | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pending_json FROM pending_approvals WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return PendingApproval(**json.loads(row[0]))

    def delete_pending_approval(self, task_id: str) -> None:
        """Drop the pending-approval record for ``task_id``.

        Must be called once an approval has been consumed: a stale record lets
        a repeated resume_task(approved=True) silently re-execute the already
        approved operation against its original (now outdated) arguments.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM pending_approvals WHERE task_id = ?", (task_id,))
