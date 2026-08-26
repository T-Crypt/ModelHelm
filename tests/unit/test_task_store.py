from modelhelm.tasks.store import TaskStore
from modelhelm.tasks.models import TaskResult


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
        summary="Implemented auth.",
    )
    store.save_result(task.task_id, result)

    fetched = store.get_result(task.task_id)
    assert fetched == result


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
