import pytest
from modelhelm.config.settings import Settings, SafetyPolicy, AgentConfig
from modelhelm.mcp.server import create_server
from modelhelm.tasks.store import TaskStore

class FakeLMStudioClient:
    async def list_models(self):
        from modelhelm.runtimes.lmstudio import LMStudioModel
        return [
            LMStudioModel(id="qwen3-coder-30b-a3b", state="loaded", max_context_length=262144, capabilities=["tool_use"]),
        ]
    async def chat_completion(self, model, messages, tools=None):
        return {"choices": [{"message": {"role": "assistant", "content": "done", "tool_calls": None}}]}

class FakeLlmfitClient:
    def recommend(self):
        from modelhelm.models.llmfit_client import LlmfitModel
        return [LlmfitModel(name="Qwen/Qwen3-Coder-30B-A3B", capability_ids=["tool_use"], fit_level="Excellent", score=98.5, context_length=262144, estimated_tps=62.3)]

@pytest.fixture
def server(tmp_path):
    settings = Settings(safety=SafetyPolicy(), agent=AgentConfig(max_iterations=2, test_before_completion=False))
    store = TaskStore(str(tmp_path / "tasks.db"))
    return create_server(
        settings=settings,
        task_store=store,
        lmstudio_client=FakeLMStudioClient(),
        llmfit_client=FakeLlmfitClient(),
    )

@pytest.mark.asyncio
async def test_get_status(server):
    status = await server.tools["get_status"]()
    assert status["default_runtime"] == "lm-studio"

@pytest.mark.asyncio
async def test_list_models(server):
    models = await server.tools["list_models"]()
    assert models[0]["name"] == "qwen3-coder-30b-a3b"

@pytest.mark.asyncio
async def test_delegate_task_and_get_status(server, tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    (tmp_path / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)

    result = await server.tools["delegate_task"](description="implement a no-op change", repository=str(tmp_path))
    # "implement" matches the implementation class -> local, same as before this milestone.
    assert result["status"] == "completed"
    assert result["task_class"] == "implementation"

    task_status = await server.tools["get_task_status"](task_id=result["task_id"])
    assert task_status["status"] == "completed"
    assert task_status["task_class"] == "implementation"

@pytest.mark.asyncio
async def test_get_task_status_unknown_returns_error(server):
    result = await server.tools["get_task_status"](task_id="nonexistent")
    assert "error" in result

@pytest.mark.asyncio
async def test_cancel_task(server, tmp_path):
    task = server.task_store.create_task(description="x", repository=str(tmp_path))
    result = await server.tools["cancel_task"](task_id=task.task_id)
    assert result["status"] == "cancelled"

@pytest.mark.asyncio
async def test_resume_task_rejected_cancels(server, tmp_path):
    task = server.task_store.create_task(description="x", repository=str(tmp_path))
    result = await server.tools["resume_task"](task_id=task.task_id, approved=False)
    assert result["status"] == "cancelled"

@pytest.mark.asyncio
async def test_resume_task_no_pending_approval_returns_error(server, tmp_path):
    task = server.task_store.create_task(description="x", repository=str(tmp_path))
    result = await server.tools["resume_task"](task_id=task.task_id, approved=True)
    assert "error" in result

@pytest.mark.asyncio
async def test_resume_task_approved_executes_pending_commit_and_continues(server, tmp_path):
    import subprocess
    from modelhelm.tasks.models import PendingApproval

    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    (tmp_path / "notes.txt").write_text("draft")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)
    (tmp_path / "notes.txt").write_text("updated")

    task = server.task_store.create_task(description="commit notes", repository=str(tmp_path))
    # resume_task threads task.task_class into agent.run(), which requires a str;
    # this fixture bypasses delegate_task, so the class must be set explicitly.
    server.task_store.set_status(
        task.task_id, "pending_approval", model="qwen3-coder-30b-a3b", task_class="testing"
    )
    pending = PendingApproval(
        operation="git_commit",
        detail="add notes",
        tool_call_id="call_1",
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "commit notes"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "git_commit", "arguments": '{"message": "add notes"}'}}
                ],
            },
            # Task 10's agent loop always appends a placeholder tool-role reply
            # for the gated call before building PendingApproval (so every
            # tool_call_id in the assistant turn has a response) — resume_task
            # must replace this placeholder's content, not append after it.
            {"role": "tool", "tool_call_id": "call_1", "content": "not executed: run paused pending approval"},
        ],
    )
    server.task_store.save_pending_approval(task.task_id, pending)

    result = await server.tools["resume_task"](task_id=task.task_id, approved=True)

    assert result["status"] == "completed"
    log = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "-1", "--pretty=%s"],
        check=True, capture_output=True, text=True,
    )
    assert log.stdout.strip() == "add notes"

@pytest.mark.asyncio
async def test_resume_task_approved_executes_pending_file_write(tmp_path):
    """A file_write approval must dispatch on the *tool* name (write_file), not
    on the policy operation name (file_write) — the two namespaces coincide only
    for git_commit, so a git_commit-only test cannot catch that confusion.
    """
    import subprocess
    from modelhelm.tasks.models import PendingApproval

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)

    # file_write="ask" is what makes AgentTools.write_file raise ToolNeedsApproval
    # in the first place, so the resumed run must rely on the one-off elevation.
    settings = Settings(
        safety=SafetyPolicy(file_write="ask"),
        agent=AgentConfig(max_iterations=2, test_before_completion=False),
    )
    store = TaskStore(str(tmp_path / "tasks.db"))
    server = create_server(
        settings=settings,
        task_store=store,
        lmstudio_client=FakeLMStudioClient(),
        llmfit_client=FakeLlmfitClient(),
    )

    task = store.create_task(description="write notes", repository=str(repo))
    store.set_status(
        task.task_id, "pending_approval", model="qwen3-coder-30b-a3b", task_class="testing"
    )
    store.save_pending_approval(
        task.task_id,
        PendingApproval(
            operation="file_write",
            detail="write to notes.txt",
            tool_call_id="call_1",
            messages=[
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "write notes"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path": "notes.txt", "content": "hello"}',
                            },
                        }
                    ],
                },
                # Task 10's placeholder reply for the gated call.
                {"role": "tool", "tool_call_id": "call_1", "content": "not executed: run paused pending approval"},
            ],
        ),
    )

    result = await server.tools["resume_task"](task_id=task.task_id, approved=True)

    assert result["status"] == "completed"
    assert (repo / "notes.txt").read_text() == "hello"


# --- C3: a consumed approval must not be replayable -------------------------

@pytest.mark.asyncio
async def test_second_resume_does_not_replay_the_approved_operation(server, tmp_path):
    """Replaying a stale approval re-executes the original (now outdated)
    arguments, silently reverting whatever a human changed in between."""
    import subprocess
    from modelhelm.tasks.models import PendingApproval

    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    (tmp_path / "notes.txt").write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True, capture_output=True)

    task = server.task_store.create_task(description="write notes", repository=str(tmp_path))
    server.task_store.set_status(
        task.task_id, "pending_approval", model="qwen3-coder-30b-a3b", task_class="testing"
    )
    server.task_store.save_pending_approval(
        task.task_id,
        PendingApproval(
            operation="file_write",
            detail="write to notes.txt",
            tool_call_id="call_1",
            messages=[
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "write notes"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path": "notes.txt", "content": "agent version"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "not executed"},
            ],
        ),
    )

    first = await server.tools["resume_task"](task_id=task.task_id, approved=True)
    assert first["status"] == "completed"
    assert (tmp_path / "notes.txt").read_text() == "agent version"

    # A human edits the file after the approved write landed.
    (tmp_path / "notes.txt").write_text("human edit")

    second = await server.tools["resume_task"](task_id=task.task_id, approved=True)

    assert "error" in second
    # The stale approval must NOT have overwritten the human's edit.
    assert (tmp_path / "notes.txt").read_text() == "human edit"
    assert server.task_store.get_pending_approval(task.task_id) is None


# --- I3: MCP tools return structured failures, never raw exceptions ---------

@pytest.mark.asyncio
async def test_delegate_task_returns_failed_dict_on_lmstudio_timeout(tmp_path):
    """An LM Studio timeout previously propagated out of the MCP tool, leaving
    the task stuck in "running" with no result persisted."""
    import httpx
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "f.txt").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True)

    class TimingOutClient(FakeLMStudioClient):
        async def chat_completion(self, model, messages, tools=None):
            raise httpx.TimeoutException("timed out talking to LM Studio")

    store = TaskStore(str(tmp_path / "tasks.db"))
    server = create_server(
        settings=Settings(agent=AgentConfig(max_iterations=2, test_before_completion=False)),
        task_store=store,
        lmstudio_client=TimingOutClient(),
        llmfit_client=FakeLlmfitClient(),
    )

    result = await server.tools["delegate_task"](description="implement x", repository=str(repo))

    assert result["status"] == "failed"
    assert "TimeoutException" in result["error"]
    # The task must not be left stranded in "running".
    assert store.get_task(result["task_id"]).status == "failed"


@pytest.mark.asyncio
async def test_delegate_task_returns_failed_dict_on_bad_repository(server):
    """A repository path that does not exist must surface as a structured
    failure rather than an unhandled subprocess/git error."""
    result = await server.tools["delegate_task"](
        description="implement x", repository="/definitely/not/a/repo"
    )

    assert result["status"] == "failed"
    assert "error" in result


@pytest.mark.asyncio
async def test_resume_task_returns_failed_dict_when_execution_raises(server, tmp_path):
    """The approved tool call itself can fail (here: a write outside the repo
    scope) -- resume_task must not leak the exception."""
    from modelhelm.tasks.models import PendingApproval

    task = server.task_store.create_task(description="write", repository=str(tmp_path))
    server.task_store.set_status(
        task.task_id, "pending_approval", model="qwen3-coder-30b-a3b", task_class="testing"
    )
    server.task_store.save_pending_approval(
        task.task_id,
        PendingApproval(
            operation="file_write",
            detail="write outside",
            tool_call_id="call_1",
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path": "../escape.txt", "content": "x"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "not executed"},
            ],
        ),
    )

    result = await server.tools["resume_task"](task_id=task.task_id, approved=True)

    assert result["status"] == "failed"
    assert "PathScopeError" in result["error"]


# --- Phase 2: the classification gate ---------------------------------------

@pytest.mark.asyncio
async def test_classify_task_previews_without_side_effects(server):
    result = await server.tools["classify_task"](description="design the system architecture")
    assert result["task_class"] == "architecture"
    assert result["disposition"] == "claude"
    # Pure preview: nothing was persisted.
    all_tasks_db_path = server.task_store.db_path
    import sqlite3
    conn = sqlite3.connect(all_tasks_db_path)
    count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_delegate_task_short_circuits_for_claude_default_class(tmp_path):
    # Spy on FakeLlmfitClient.recommend(): TaskRouter.select_model() always
    # calls registry.refresh(), which always calls llmfit_client.recommend().
    # If recommend() is never invoked, the router (and therefore the
    # registry and LM Studio) was never touched -- proving the short-circuit
    # happens before any of Phase 1's model-selection machinery runs.
    class SpyLlmfitClient(FakeLlmfitClient):
        def __init__(self):
            self.recommend_calls = 0

        def recommend(self):
            self.recommend_calls += 1
            return super().recommend()

    llmfit_spy = SpyLlmfitClient()
    store = TaskStore(str(tmp_path / "tasks.db"))
    server = create_server(
        settings=Settings(agent=AgentConfig(max_iterations=2, test_before_completion=False)),
        task_store=store,
        lmstudio_client=FakeLMStudioClient(),
        llmfit_client=llmfit_spy,
    )

    result = await server.tools["delegate_task"](
        description="design the caching architecture", repository=str(tmp_path)
    )

    assert result["status"] == "escalation_recommended"
    assert result["task_class"] == "architecture"
    assert result["model"] == "none"
    assert result["runtime"] == "none"
    assert "recommend Claude" in result["summary"]
    assert llmfit_spy.recommend_calls == 0


@pytest.mark.asyncio
async def test_delegate_task_persists_escalation_result(server, tmp_path):
    result = await server.tools["delegate_task"](
        description="review this security vulnerability", repository=str(tmp_path)
    )
    assert result["status"] == "escalation_recommended"

    task_status = await server.tools["get_task_status"](task_id=result["task_id"])
    assert task_status["status"] == "escalation_recommended"
    # "security" (via the "security" keyword) beats "final_review" (via
    # "review this") because security is earlier in DEFAULT_TASK_CLASSES
    # table order.
    assert task_status["task_class"] == "security"


@pytest.mark.asyncio
async def test_ambiguous_description_escalates(server, tmp_path):
    result = await server.tools["delegate_task"](
        description="do the thing with the stuff", repository=str(tmp_path)
    )
    assert result["status"] == "escalation_recommended"
    assert result["task_class"] == "ambiguous"
