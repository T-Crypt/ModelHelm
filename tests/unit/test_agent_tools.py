import subprocess
import pytest
from modelhelm.agents.tools import AgentTools, ToolDenied, ToolNeedsApproval, get_tool_definitions
from modelhelm.policies.engine import PolicyEngine, PathScopeError
from modelhelm.config.settings import SafetyPolicy

def _init_repo(path):
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)

def test_write_and_read_file(tmp_path):
    _init_repo(tmp_path)
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(file_write="allow")))

    tools.write_file("notes.txt", "hello world")
    assert tools.read_file("notes.txt") == "hello world"

def test_write_file_denied_raises(tmp_path):
    _init_repo(tmp_path)
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(file_write="deny")))

    with pytest.raises(ToolDenied):
        tools.write_file("notes.txt", "hello")

def test_write_file_outside_repo_raises_path_scope_error(tmp_path):
    _init_repo(tmp_path)
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(file_write="allow")))

    with pytest.raises(PathScopeError):
        tools.write_file("../outside.txt", "escape attempt")

def test_git_commit_ask_raises_needs_approval(tmp_path):
    _init_repo(tmp_path)
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(git_commit="ask")))

    with pytest.raises(ToolNeedsApproval) as exc_info:
        tools.git_commit("my change")
    assert exc_info.value.operation == "git_commit"

def test_run_command_destructive_pattern_denied(tmp_path):
    _init_repo(tmp_path)
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy(destructive_commands="deny")))

    with pytest.raises(ToolDenied):
        tools.run_command("rm -rf /")

def test_run_command_normal_command_runs(tmp_path):
    _init_repo(tmp_path)
    tools = AgentTools(str(tmp_path), PolicyEngine(SafetyPolicy()))

    result = tools.run_command("git status --porcelain")
    assert result["returncode"] == 0

def test_get_tool_definitions_returns_six_tools():
    defs = get_tool_definitions()
    names = {d["function"]["name"] for d in defs}
    assert names == {
        "read_file", "write_file", "list_directory",
        "run_command", "git_diff", "git_commit",
    }
