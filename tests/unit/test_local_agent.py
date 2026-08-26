import pytest
from modelhelm.agents.local_agent import LocalAgent, NOT_EXECUTED_MESSAGE
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
        # Snapshot of the conversation as of each turn, so tests can assert on
        # what the model would actually have seen.
        self.received = []

    async def chat_completion(self, model, messages, tools=None):
        self.received.append(list(messages))
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

    result, pending = await agent.run(
        task_id="t1", description="no-op task", model="qwen3-coder-30b-a3b", task_class="testing",
    )

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

    result, pending = await agent.run(
        task_id="t2", description="write a note", model="qwen3-coder-30b-a3b", task_class="testing",
    )

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

    result, pending = await agent.run(
        task_id="t3", description="loop forever", model="qwen3-coder-30b-a3b", task_class="testing",
    )

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

    result, pending = await agent.run(
        task_id="t4", description="commit notes", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    assert result.status == "pending_approval"
    assert "git_commit" in result.summary
    assert pending is not None
    assert pending.operation == "git_commit"
    assert pending.detail == "add notes"
    assert pending.tool_call_id == "call_1"
    # The assistant turn that requested the gated call is retained, and the
    # gated call itself carries a placeholder tool result so the conversation
    # has no orphaned tool_call_id.
    assistant_message = pending.messages[-2]
    assert assistant_message["tool_calls"][0]["function"]["name"] == "git_commit"
    assert pending.messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": NOT_EXECUTED_MESSAGE,
    }


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
        task_id="t5", description="commit notes", model="qwen3-coder-30b-a3b", task_class="testing",
        resume_messages=prior_messages_with_approved_result,
    )

    assert result.status == "completed"
    assert pending is None


@pytest.mark.asyncio
async def test_files_changed_counts_work_that_was_committed_during_the_run(tmp_path):
    """After a successful git_commit the tree is clean again, so a raw
    `git status` count reported 0 files changed — which also wrongly suppressed
    review_recommended on exactly the runs that committed something."""
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
                },
                {
                    "id": "call_2",
                    "function": {"name": "git_commit", "arguments": '{"message": "add notes"}'},
                },
            ],
        },
        {"role": "assistant", "content": "Committed.", "tool_calls": None},
    ])
    tools = AgentTools(
        str(tmp_path),
        PolicyEngine(SafetyPolicy(file_write="allow", git_commit="allow")),
    )
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=8, test_before_completion=False),
    )

    result, pending = await agent.run(
        task_id="t12", description="write and commit a note", model="qwen3-coder-30b-a3b",
        task_class="testing",
    )

    assert result.status == "completed"
    # The tree is clean post-commit, but the run really did change a file.
    assert GitInspector(str(tmp_path)).files_changed_count() == 0
    assert result.files_changed >= 1
    assert result.review_recommended is True


@pytest.mark.asyncio
async def test_files_changed_excludes_dirt_that_predates_the_task(tmp_path):
    """Changes already sitting in the working tree must not be billed to the
    agent."""
    _init_repo(tmp_path)
    (tmp_path / "preexisting.txt").write_text("not the agent's doing")

    fake_client = FakeLMStudioClient([
        {"role": "assistant", "content": "Nothing to do.", "tool_calls": None},
    ])
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy()))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=8, test_before_completion=False),
    )

    result, _ = await agent.run(
        task_id="t13", description="no-op", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    # The agent changed nothing, so the pre-existing dirt must not be billed to
    # it -- and review_recommended must not be triggered by it either.
    assert result.files_changed == 0
    assert result.review_recommended is False


@pytest.mark.asyncio
async def test_files_changed_counts_agent_work_alongside_preexisting_dirt(tmp_path):
    """Pre-existing dirt is excluded, but the agent's own new file still counts."""
    _init_repo(tmp_path)
    (tmp_path / "preexisting.txt").write_text("not the agent's doing")

    fake_client = FakeLMStudioClient([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "agent.txt", "content": "mine"}',
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

    result, _ = await agent.run(
        task_id="t14", description="write a note", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    assert result.files_changed == 1


def _make_agent(tmp_path, script, policy=None, max_iterations=8):
    """Build a LocalAgent over ``tmp_path`` driven by a scripted fake client.

    Returns ``(agent, fake_client)`` so tests can inspect the messages the
    client received on later turns.
    """
    fake_client = FakeLMStudioClient(script)
    tools = AgentTools(str(tmp_path), PolicyEngine(policy or SafetyPolicy()))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=max_iterations, test_before_completion=False),
    )
    return agent, fake_client


def _tool_messages(messages):
    return [m for m in messages if m.get("role") == "tool"]


@pytest.mark.asyncio
async def test_unknown_tool_name_is_reported_to_model_not_raised(tmp_path):
    _init_repo(tmp_path)
    agent, fake_client = _make_agent(tmp_path, [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "summon_daemon", "arguments": "{}"}}
            ],
        },
        {"role": "assistant", "content": "Sorry, I made that tool up.", "tool_calls": None},
    ])

    result, pending = await agent.run(
        task_id="t6", description="use a fake tool", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    assert result.status == "completed"
    assert pending is None
    # The second turn must have seen the error as this call's tool result.
    second_turn_messages = fake_client.received[1]
    error_message = _tool_messages(second_turn_messages)[-1]
    assert error_message["tool_call_id"] == "call_1"
    assert error_message["content"].startswith("ERROR:")
    assert "summon_daemon" in error_message["content"]


@pytest.mark.asyncio
async def test_malformed_json_arguments_are_reported_to_model_not_raised(tmp_path):
    _init_repo(tmp_path)
    agent, fake_client = _make_agent(tmp_path, [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": '}}
            ],
        },
        {"role": "assistant", "content": "Retrying with valid JSON next time.", "tool_calls": None},
    ])

    result, pending = await agent.run(
        task_id="t7", description="send broken args", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    assert result.status == "completed"
    assert pending is None
    error_message = _tool_messages(fake_client.received[1])[-1]
    assert error_message["tool_call_id"] == "call_1"
    assert error_message["content"].startswith("ERROR:")


@pytest.mark.asyncio
async def test_missing_required_argument_is_reported_to_model_not_raised(tmp_path):
    _init_repo(tmp_path)
    agent, fake_client = _make_agent(tmp_path, [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "assistant", "content": "I forgot the path.", "tool_calls": None},
    ])

    result, pending = await agent.run(
        task_id="t8", description="omit an argument", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    assert result.status == "completed"
    assert pending is None
    error_message = _tool_messages(fake_client.received[1])[-1]
    assert error_message["content"].startswith("ERROR:")


@pytest.mark.asyncio
async def test_read_file_on_missing_path_is_reported_to_model_not_raised(tmp_path):
    """Reading a path that does not exist is normal exploration behavior and
    must let the agent self-correct, not abort the whole run."""
    _init_repo(tmp_path)
    agent, fake_client = _make_agent(tmp_path, [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": '{"path": "nope.txt"}'},
                }
            ],
        },
        {"role": "assistant", "content": "That file does not exist.", "tool_calls": None},
    ])

    result, pending = await agent.run(
        task_id="t9", description="read a missing file", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    assert result.status == "completed"
    assert pending is None
    error_message = _tool_messages(fake_client.received[1])[-1]
    assert error_message["tool_call_id"] == "call_1"
    assert "ERROR:" in error_message["content"]
    assert "nope.txt" in error_message["content"]


@pytest.mark.asyncio
async def test_pause_in_multi_call_batch_leaves_no_orphan_tool_call_ids(tmp_path):
    """When the FIRST of two calls needs approval, the second call still needs a
    tool-role response or the resumed conversation is malformed."""
    _init_repo(tmp_path)
    agent, _ = _make_agent(
        tmp_path,
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "git_commit", "arguments": '{"message": "add notes"}'},
                    },
                    {"id": "call_2", "function": {"name": "git_diff", "arguments": "{}"}},
                ],
            },
        ],
        policy=SafetyPolicy(git_commit="ask"),
    )

    result, pending = await agent.run(
        task_id="t10", description="commit then diff", model="qwen3-coder-30b-a3b", task_class="testing",
    )

    assert result.status == "pending_approval"
    assert pending is not None
    assert pending.tool_call_id == "call_1"

    responded_ids = {m["tool_call_id"] for m in _tool_messages(pending.messages)}
    assert responded_ids == {"call_1", "call_2"}
    by_id = {m["tool_call_id"]: m["content"] for m in _tool_messages(pending.messages)}
    assert by_id["call_1"] == NOT_EXECUTED_MESSAGE
    assert by_id["call_2"] == NOT_EXECUTED_MESSAGE


@pytest.mark.asyncio
async def test_calls_before_pause_in_batch_still_execute(tmp_path):
    """A call preceding the approval-gated one runs for real and gets its
    genuine result recorded."""
    _init_repo(tmp_path)
    agent, _ = _make_agent(
        tmp_path,
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "git_commit", "arguments": '{"message": "wip"}'},
                    },
                    {"id": "call_3", "function": {"name": "git_diff", "arguments": "{}"}},
                ],
            },
        ],
        policy=SafetyPolicy(git_commit="ask"),
    )

    result, pending = await agent.run(
        task_id="t11", description="read then commit then diff", model="qwen3-coder-30b-a3b",
        task_class="testing",
    )

    assert result.status == "pending_approval"
    assert pending.tool_call_id == "call_2"
    by_id = {m["tool_call_id"]: m["content"] for m in _tool_messages(pending.messages)}
    assert set(by_id) == {"call_1", "call_2", "call_3"}
    assert by_id["call_1"].strip() == "hello"
    assert by_id["call_2"] == NOT_EXECUTED_MESSAGE
    assert by_id["call_3"] == NOT_EXECUTED_MESSAGE


@pytest.mark.asyncio
async def test_task_class_is_threaded_onto_the_result(tmp_path):
    _init_repo(tmp_path)
    fake_client = FakeLMStudioClient([
        {"role": "assistant", "content": "done", "tool_calls": None},
    ])
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy()))
    agent = LocalAgent(
        lmstudio_client=fake_client,
        tools=tools,
        git_inspector=GitInspector(str(tmp_path)),
        agent_config=AgentConfig(max_iterations=8, test_before_completion=False),
    )

    result, _ = await agent.run(
        task_id="t-class", description="find the bug", model="qwen3-coder-30b-a3b",
        task_class="debugging",
    )

    assert result.task_class == "debugging"
