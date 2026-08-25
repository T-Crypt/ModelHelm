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

    result = await server.tools["delegate_task"](description="no-op", repository=str(tmp_path))
    assert result["status"] == "completed"

    task_status = await server.tools["get_task_status"](task_id=result["task_id"])
    assert task_status["status"] == "completed"

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
    server.task_store.set_status(task.task_id, "pending_approval", model="qwen3-coder-30b-a3b")
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
