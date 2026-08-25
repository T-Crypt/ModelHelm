import pytest
from modelhelm.agents.local_agent import LocalAgent
from modelhelm.agents.tools import AgentTools
from modelhelm.policies.engine import PolicyEngine
from modelhelm.config.settings import SafetyPolicy, AgentConfig
from modelhelm.git.inspector import GitInspector
import subprocess


def _init_repo(path):
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)


class FakeLMStudioClient:
    """Replays a scripted sequence of assistant messages."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def chat_completion(self, model, messages, tools=None):
        message = self.script[self.calls]
        self.calls += 1
        return {"choices": [{"message": message}]}


@pytest.mark.asyncio
async def test_run_completes_when_model_signals_done_with_no_tool_calls(tmp_path):
    _init_repo(tmp_path)
    fake_client = FakeLMStudioClient([
        {"role": "assistant", "content": "Task complete, no changes needed.", "tool_calls": None},
    ])
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy()))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=8, test_before_completion=False),
    )

    result, pending = await agent.run(task_id="t1", description="no-op task", model="qwen3-coder-30b-a3b")

    assert result.status == "completed"
    assert result.iterations == 1
    assert result.model == "qwen3-coder-30b-a3b"
    assert result.runtime == "lm-studio"
    assert pending is None


@pytest.mark.asyncio
async def test_run_executes_tool_call_then_completes(tmp_path):
    _init_repo(tmp_path)
    fake_client = FakeLMStudioClient([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "notes.txt", "content": "hi"}',
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    ])
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(file_write="allow")))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=8, test_before_completion=False),
    )

    result, pending = await agent.run(task_id="t2", description="write a note", model="qwen3-coder-30b-a3b")

    assert result.status == "completed"
    assert (tmp_path / "notes.txt").read_text() == "hi"
    assert result.files_changed == 1
    assert pending is None


@pytest.mark.asyncio
async def test_run_stops_at_max_iterations_with_escalation(tmp_path):
    _init_repo(tmp_path)
    endless_tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_x", "function": {"name": "git_diff", "arguments": "{}"}}
        ],
    }
    fake_client = FakeLMStudioClient([endless_tool_call] * 3)
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy()))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=3, test_before_completion=False),
    )

    result, pending = await agent.run(task_id="t3", description="loop forever", model="qwen3-coder-30b-a3b")

    assert result.status == "escalation_recommended"
    assert result.iterations == 3
    assert pending is None


@pytest.mark.asyncio
async def test_run_pauses_on_needs_approval(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "notes.txt").write_text("draft")
    fake_client = FakeLMStudioClient([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "git_commit", "arguments": '{"message": "add notes"}'}}
            ],
        },
    ])
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(git_commit="ask")))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=8, test_before_completion=False),
    )

    result, pending = await agent.run(task_id="t4", description="commit notes", model="qwen3-coder-30b-a3b")

    assert result.status == "pending_approval"
    assert "git_commit" in result.summary
    assert pending is not None
    assert pending.operation == "git_commit"
    assert pending.detail == "add notes"
    assert pending.tool_call_id == "call_1"
    assert pending.messages[-1]["tool_calls"][0]["function"]["name"] == "git_commit"


@pytest.mark.asyncio
async def test_resume_messages_continues_prior_conversation(tmp_path):
    """Simulates what resume_task (Task 11) does: it has already executed the
    approved tool call and appended the tool result itself, then hands the
    extended conversation back to run() to continue from there."""
    _init_repo(tmp_path)
    fake_client = FakeLMStudioClient([
        {"role": "assistant", "content": "Committed successfully.", "tool_calls": None},
    ])
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(git_commit="ask")))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=8, test_before_completion=False),
    )
    prior_messages_with_approved_result = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "commit notes"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "git_commit", "arguments": '{"message": "add notes"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "committed: add notes"},
    ]

    result, pending = await agent.run(
        task_id="t5", description="commit notes", model="qwen3-coder-30b-a3b",
        resume_messages=prior_messages_with_approved_result,
    )

    assert result.status == "completed"
    assert pending is None
