import sqlite3

from modelhelm.tasks.store import TaskStore
from modelhelm.tasks.models import TaskResult


PHASE_1_TASKS_SCHEMA = """
    CREATE TABLE tasks (
        task_id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        repository TEXT NOT NULL,
        status TEXT NOT NULL,
        model TEXT,
        created_at TEXT NOT NULL
    )
"""


def _write_phase_1_db(path, *, with_row=True):
    """Create a database with the pre-milestone tasks table (no task_class)."""
    conn = sqlite3.connect(str(path))
    conn.execute(PHASE_1_TASKS_SCHEMA)
    if with_row:
        conn.execute(
            "INSERT INTO tasks (task_id, description, repository, status, model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy-1", "add auth", "/repo", "completed", "qwen3-coder-30b-a3b",
             "2026-01-01T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()


def test_create_and_get_task(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")

    assert task.status == "pending"
    assert task.description == "add auth"

    fetched = store.get_task(task.task_id)
    assert fetched == task


def test_set_status_updates_task(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")

    store.set_status(task.task_id, "running", model="qwen3-coder-30b-a3b")
    fetched = store.get_task(task.task_id)

    assert fetched.status == "running"
    assert fetched.model == "qwen3-coder-30b-a3b"


def test_save_and_get_result(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")

    result = TaskResult(
        task_id=task.task_id,
        status="completed",
        model="qwen3-coder-30b-a3b",
        runtime="lm-studio",
        duration_seconds=184.0,
        files_changed=8,
        tests_run=32,
        tests_passed=32,
        tests_failed=0,
        iterations=3,
        estimated_cloud_tokens_saved=18400,
        review_recommended=True,
        task_class="implementation",
        summary="Implemented auth.",
    )
    store.save_result(task.task_id, result)

    fetched = store.get_result(task.task_id)
    assert fetched == result


def test_create_task_starts_with_no_task_class(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")
    assert task.task_class is None


def test_set_status_with_task_class_persists_it(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")

    store.set_status(task.task_id, "running", model="qwen3-coder-30b-a3b", task_class="implementation")
    fetched = store.get_task(task.task_id)

    assert fetched.task_class == "implementation"


def test_set_status_without_task_class_leaves_existing_value(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")
    store.set_status(task.task_id, "running", model="x", task_class="implementation")

    store.set_status(task.task_id, "completed")  # no task_class passed
    fetched = store.get_task(task.task_id)

    assert fetched.task_class == "implementation"


def test_get_task_returns_none_for_unknown_id(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    assert store.get_task("does-not-exist") is None


def test_save_and_get_pending_approval(tmp_path):
    from modelhelm.tasks.models import PendingApproval

    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")

    pending = PendingApproval(
        operation="git_commit",
        detail="add notes",
        tool_call_id="call_1",
        messages=[{"role": "user", "content": "commit notes"}],
    )
    store.save_pending_approval(task.task_id, pending)

    fetched = store.get_pending_approval(task.task_id)
    assert fetched == pending


def test_get_pending_approval_returns_none_when_absent(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")
    assert store.get_pending_approval(task.task_id) is None


def test_delete_pending_approval_removes_record(tmp_path):
    """A consumed approval must be deleted, or a repeated resume replays it."""
    from modelhelm.tasks.models import PendingApproval

    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")
    store.save_pending_approval(
        task.task_id,
        PendingApproval(
            operation="git_commit",
            detail="add notes",
            tool_call_id="call_1",
            messages=[{"role": "user", "content": "commit notes"}],
        ),
    )
    assert store.get_pending_approval(task.task_id) is not None

    store.delete_pending_approval(task.task_id)

    assert store.get_pending_approval(task.task_id) is None


def test_delete_pending_approval_is_idempotent(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")
    store.delete_pending_approval(task.task_id)  # no record: must not raise
    assert store.get_pending_approval(task.task_id) is None


# --- I1: a Phase-1-shape database must migrate, not crash -------------------

def test_phase_1_database_without_task_class_column_is_migrated(tmp_path):
    """CREATE TABLE IF NOT EXISTS never touches an existing table, so a
    Phase-1 tasks table used to survive un-migrated and then fail every read
    with a bare "sqlite3.OperationalError: no such column: task_class"."""
    db = tmp_path / "phase1.db"
    _write_phase_1_db(db)

    TaskStore(str(db))  # opening must perform the migration

    columns = {row[1] for row in sqlite3.connect(str(db)).execute("PRAGMA table_info(tasks)")}
    assert "task_class" in columns


def test_migrated_phase_1_rows_are_readable_and_preserved(tmp_path):
    db = tmp_path / "phase1.db"
    _write_phase_1_db(db)

    store = TaskStore(str(db))
    legacy = store.get_task("legacy-1")

    # The pre-existing row survives intact and simply has no class yet.
    assert legacy is not None
    assert legacy.description == "add auth"
    assert legacy.status == "completed"
    assert legacy.model == "qwen3-coder-30b-a3b"
    assert legacy.task_class is None


def test_writes_work_against_a_migrated_phase_1_database(tmp_path):
    db = tmp_path / "phase1.db"
    _write_phase_1_db(db)
    store = TaskStore(str(db))

    task = store.create_task(description="add auth", repository="/repo")
    store.set_status(task.task_id, "running", model="m", task_class="security")

    assert store.get_task(task.task_id).task_class == "security"


def test_migration_is_idempotent_across_reopens(tmp_path):
    """The migration runs on every open, so it must be safe to re-run."""
    db = tmp_path / "phase1.db"
    _write_phase_1_db(db)

    TaskStore(str(db))
    TaskStore(str(db))
    store = TaskStore(str(db))

    assert store.get_task("legacy-1") is not None
    columns = [row[1] for row in sqlite3.connect(str(db)).execute("PRAGMA table_info(tasks)")]
    assert columns.count("task_class") == 1
