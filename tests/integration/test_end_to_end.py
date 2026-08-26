import subprocess
import httpx
import pytest
from modelhelm.config.settings import Settings, SafetyPolicy, AgentConfig, LMStudioConfig
from modelhelm.mcp.server import create_server
from modelhelm.runtimes.lmstudio import LMStudioClient
from modelhelm.models.llmfit_client import LlmfitClient
from modelhelm.tasks.store import TaskStore

LM_STUDIO_ENDPOINT = "http://localhost:1234"


def _lm_studio_available() -> bool:
    try:
        httpx.get(f"{LM_STUDIO_ENDPOINT}/api/v0/models", timeout=2.0)
        return True
    except Exception:
        return False


def _init_repo(path):
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "T"], check=True)
    (path / "README.md").write_text("# Scratch repo\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)


@pytest.mark.skipif(not _lm_studio_available(), reason="LM Studio not reachable at localhost:1234")
@pytest.mark.asyncio
async def test_delegate_task_end_to_end(tmp_path):
    _init_repo(tmp_path)

    settings = Settings(
        lm_studio=LMStudioConfig(endpoint=LM_STUDIO_ENDPOINT),
        safety=SafetyPolicy(file_write="allow", git_commit="ask"),
        agent=AgentConfig(max_iterations=5, test_before_completion=False),
    )
    server = create_server(
        settings=settings,
        task_store=TaskStore(str(tmp_path / "tasks.db")),
        lmstudio_client=LMStudioClient(endpoint=LM_STUDIO_ENDPOINT),
        llmfit_client=LlmfitClient(),
    )

    result = await server.tools["delegate_task"](
        description="Create a file named hello.txt containing the text 'hello from modelhelm'. Do not commit.",
        repository=str(tmp_path),
    )

    assert result["status"] in ("completed", "pending_approval", "escalation_recommended")
    assert result["model"]
    assert result["runtime"] == "lm-studio"
