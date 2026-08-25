# ModelHelm Phase 1: MCP Server + Local Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a working Claude Code → ModelHelm (MCP) → LM Studio/Qwen3-Coder delegation loop: Claude calls `delegate_task()`, ModelHelm selects a model via llmfit, runs a self-driving agent loop (edit/test/fix) against LM Studio with a policy-gated commit/push approval step, and returns a structured result.

**Architecture:** A Python package (`src/modelhelm/`) exposes an MCP server. `delegate_task` flows through a thin router (llmfit + LM Studio state) into a tool-calling agent loop (`agents/local_agent.py`) that calls `read_file`/`write_file`/`list_directory`/`run_command`/`git_diff`, each gated by an ALLOW/DENY/ASK policy engine. Task state persists in SQLite so `get_task_status`/`cancel_task`/`resume_task` work across MCP calls. All execution is direct subprocess on the host, scoped to the target repo path — no containerization.

**Tech Stack:** Python ≥3.11 (3.13.15 confirmed available at `C:\Users\Trevin\AppData\Local\Programs\Python\Python313\python.exe`), `mcp` (official Python SDK, FastMCP-style `@mcp.tool()` decorators), `pydantic`, `httpx`, `asyncio`, stdlib `sqlite3`, git via subprocess, `pytest` + `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md`

## Global Constraints

- Python ≥3.11 required (confirmed interpreter: Python 3.13.15).
- No containerization — direct subprocess execution scoped to the target repository path.
- `file_write`/`file_delete` are hard-scoped to the resolved `repository` path regardless of policy setting — never write outside the target repo.
- Default safety policy: `file_write: allow`, `file_delete: allow`, `git_commit: ask`, `git_push: ask`, `force_push: deny`, `destructive_commands: deny`, `production_changes: deny`.
- `git_commit`/`git_push` under `ask` MUST pause the loop (`status: pending_approval`) and return control to Claude — never auto-approve, never silently skip.
- Result contract fields and names are fixed by spec Section 8 — do not rename.
- LM Studio endpoint: `http://localhost:1234` (OpenAI-compatible API), model `qwen3-coder-30b-a3b` (tool-calling capable).
- `llmfit` binary: resolve via PATH by default; `modelhelm.yaml` may override `llmfit.binary_path` for environments (like this dev machine) where scoop shims aren't on PATH in non-interactive shells.
- Structured logging only — no `print()`/bare stdout in library code; MCP tool return values are Pydantic models, not ad hoc dicts.
- Use approved verb-noun style for internal helper function names where natural (matches user's global PowerShell conventions in spirit — for Python, use clear verb_noun snake_case: `get_task`, `set_status`, not vague names).

---

## File Structure

```text
ModelHelm/
├── src/
│   └── modelhelm/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py          # Task 1
│       ├── runtimes/
│       │   ├── __init__.py
│       │   └── lmstudio.py          # Task 2
│       ├── models/
│       │   ├── __init__.py
│       │   ├── llmfit_client.py     # Task 3
│       │   └── registry.py          # Task 4
│       ├── policies/
│       │   ├── __init__.py
│       │   └── engine.py            # Task 5
│       ├── git/
│       │   ├── __init__.py
│       │   └── inspector.py         # Task 6
│       ├── tasks/
│       │   ├── __init__.py
│       │   ├── models.py            # Task 7 (Task/Result Pydantic schemas)
│       │   └── store.py             # Task 7
│       ├── routing/
│       │   ├── __init__.py
│       │   └── router.py            # Task 8
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── tools.py             # Task 9
│       │   └── local_agent.py       # Task 10
│       └── mcp/
│           ├── __init__.py
│           └── server.py            # Task 11
├── tests/
│   ├── unit/
│   │   ├── test_settings.py
│   │   ├── test_lmstudio.py
│   │   ├── test_llmfit_client.py
│   │   ├── test_registry.py
│   │   ├── test_policy_engine.py
│   │   ├── test_git_inspector.py
│   │   ├── test_task_store.py
│   │   ├── test_router.py
│   │   ├── test_agent_tools.py
│   │   ├── test_local_agent.py
│   │   └── test_mcp_server.py
│   ├── integration/
│   │   └── test_end_to_end.py       # Task 12
│   └── fixtures/
│       ├── llmfit_recommend.json
│       └── llmfit_list.json
├── modelhelm.yaml                   # Task 1
├── pyproject.toml                   # Task 0
├── README.md                        # Task 13 (update)
└── LICENSE                          # already present
```

---

### Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/modelhelm/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: an installable package `modelhelm` importable from `src/modelhelm`, pytest discoverable.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "modelhelm"
version = "0.1.0"
description = "Model-agnostic AI coding orchestration: local/cloud delegation via MCP."
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.2.0",
    "pydantic>=2.6",
    "httpx>=0.27",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/modelhelm/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Create `.gitignore`**

```text
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
venv/
*.db
.env
```

- [ ] **Step 4: Create a venv and install in editable+dev mode**

Run (PowerShell, since python.exe resolves there on this machine):
```powershell
& "C:\Users\Trevin\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```
Expected: package installs cleanly, `pytest` available in `.venv\Scripts\`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/modelhelm/__init__.py .gitignore
git commit -m "chore: scaffold modelhelm package"
```

---

### Task 1: Config Loading

**Files:**
- Create: `src/modelhelm/config/settings.py`
- Create: `modelhelm.yaml`
- Test: `tests/unit/test_settings.py`

**Interfaces:**
- Produces:
  - `class SafetyPolicy(BaseModel)` — fields `file_write, file_delete, git_commit, git_push, force_push, destructive_commands, production_changes: Literal["allow","deny","ask"]`
  - `class AgentConfig(BaseModel)` — fields `max_iterations: int`, `test_before_completion: bool`
  - `class LMStudioConfig(BaseModel)` — field `endpoint: str`
  - `class Settings(BaseModel)` — fields `default_runtime: str`, `lm_studio: LMStudioConfig`, `llmfit_binary_path: str | None`, `prefer_local: bool`, `safety: SafetyPolicy`, `agent: AgentConfig`
  - `def load_settings(path: str | None = None) -> Settings` — reads `modelhelm.yaml` from `path` or CWD; returns `Settings()` defaults if the file doesn't exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_settings.py
import textwrap
from modelhelm.config.settings import load_settings, Settings

def test_load_settings_from_file(tmp_path):
    config_path = tmp_path / "modelhelm.yaml"
    config_path.write_text(textwrap.dedent("""
        modelhelm:
          default_runtime: lm-studio
        runtimes:
          lm-studio:
            endpoint: http://localhost:1234
        llmfit:
          binary_path: null
        routing:
          prefer_local: true
        safety:
          file_write: allow
          file_delete: allow
          git_commit: ask
          git_push: ask
          force_push: deny
          destructive_commands: deny
          production_changes: deny
        agent:
          max_iterations: 8
          test_before_completion: true
    """))
    settings = load_settings(str(config_path))
    assert settings.default_runtime == "lm-studio"
    assert settings.lm_studio.endpoint == "http://localhost:1234"
    assert settings.safety.git_commit == "ask"
    assert settings.safety.force_push == "deny"
    assert settings.agent.max_iterations == 8

def test_load_settings_missing_file_returns_defaults(tmp_path):
    settings = load_settings(str(tmp_path / "does-not-exist.yaml"))
    assert isinstance(settings, Settings)
    assert settings.default_runtime == "lm-studio"
    assert settings.safety.force_push == "deny"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.config.settings'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/config/settings.py
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel

PolicyState = Literal["allow", "deny", "ask"]

class SafetyPolicy(BaseModel):
    file_write: PolicyState = "allow"
    file_delete: PolicyState = "allow"
    git_commit: PolicyState = "ask"
    git_push: PolicyState = "ask"
    force_push: PolicyState = "deny"
    destructive_commands: PolicyState = "deny"
    production_changes: PolicyState = "deny"

class AgentConfig(BaseModel):
    max_iterations: int = 8
    test_before_completion: bool = True

class LMStudioConfig(BaseModel):
    endpoint: str = "http://localhost:1234"

class Settings(BaseModel):
    default_runtime: str = "lm-studio"
    lm_studio: LMStudioConfig = LMStudioConfig()
    llmfit_binary_path: str | None = None
    prefer_local: bool = True
    safety: SafetyPolicy = SafetyPolicy()
    agent: AgentConfig = AgentConfig()

def load_settings(path: str | None = None) -> Settings:
    config_path = Path(path) if path else Path.cwd() / "modelhelm.yaml"
    if not config_path.exists():
        return Settings()

    raw = yaml.safe_load(config_path.read_text()) or {}
    return Settings(
        default_runtime=raw.get("modelhelm", {}).get("default_runtime", "lm-studio"),
        lm_studio=LMStudioConfig(
            endpoint=raw.get("runtimes", {}).get("lm-studio", {}).get(
                "endpoint", "http://localhost:1234"
            )
        ),
        llmfit_binary_path=raw.get("llmfit", {}).get("binary_path"),
        prefer_local=raw.get("routing", {}).get("prefer_local", True),
        safety=SafetyPolicy(**raw.get("safety", {})),
        agent=AgentConfig(**raw.get("agent", {})),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_settings.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Create the real `modelhelm.yaml` at repo root**

```yaml
modelhelm:
  default_runtime: lm-studio

runtimes:
  lm-studio:
    endpoint: http://localhost:1234

llmfit:
  binary_path: null

routing:
  prefer_local: true

safety:
  file_write: allow
  file_delete: allow
  git_commit: ask
  git_push: ask
  force_push: deny
  destructive_commands: deny
  production_changes: deny

agent:
  max_iterations: 8
  test_before_completion: true
```

- [ ] **Step 6: Commit**

```bash
git add src/modelhelm/config/ modelhelm.yaml tests/unit/test_settings.py
git commit -m "feat: add config loading for modelhelm.yaml"
```

---

### Task 2: LM Studio Runtime Client

**Files:**
- Create: `src/modelhelm/runtimes/lmstudio.py`
- Test: `tests/unit/test_lmstudio.py`

**Interfaces:**
- Consumes: `LMStudioConfig.endpoint` (Task 1)
- Produces:
  - `class LMStudioModel(BaseModel)` — fields `id: str`, `state: Literal["loaded","not-loaded"]`, `max_context_length: int`, `capabilities: list[str]`
  - `class LMStudioClient` — constructor `__init__(self, endpoint: str)`
    - `async def list_models(self) -> list[LMStudioModel]`
    - `async def chat_completion(self, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict` — returns raw OpenAI-shaped response dict (caller extracts `choices[0].message`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_lmstudio.py
import httpx
import pytest
from modelhelm.runtimes.lmstudio import LMStudioClient, LMStudioModel

@pytest.mark.asyncio
async def test_list_models(monkeypatch):
    payload = {
        "data": [
            {
                "id": "qwen3-coder-30b-a3b",
                "state": "loaded",
                "max_context_length": 262144,
                "capabilities": ["tool_use"],
            },
            {
                "id": "text-embedding-nomic-embed-text-v1.5",
                "state": "not-loaded",
                "max_context_length": 2048,
                "capabilities": [],
            },
        ]
    }

    async def mock_get(self, url, *args, **kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    client = LMStudioClient(endpoint="http://localhost:1234")
    models = await client.list_models()

    assert len(models) == 2
    assert models[0] == LMStudioModel(
        id="qwen3-coder-30b-a3b",
        state="loaded",
        max_context_length=262144,
        capabilities=["tool_use"],
    )

@pytest.mark.asyncio
async def test_chat_completion(monkeypatch):
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}]
    }

    async def mock_post(self, url, *args, **kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    client = LMStudioClient(endpoint="http://localhost:1234")
    result = await client.chat_completion(
        model="qwen3-coder-30b-a3b",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result["choices"][0]["message"]["content"] == "hello"

@pytest.mark.asyncio
async def test_chat_completion_raises_on_timeout(monkeypatch):
    async def mock_post(self, url, *args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    client = LMStudioClient(endpoint="http://localhost:1234")
    with pytest.raises(httpx.TimeoutException):
        await client.chat_completion(model="x", messages=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_lmstudio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.runtimes.lmstudio'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/runtimes/lmstudio.py
from typing import Literal
import httpx
from pydantic import BaseModel

class LMStudioModel(BaseModel):
    id: str
    state: Literal["loaded", "not-loaded"]
    max_context_length: int
    capabilities: list[str] = []

class LMStudioClient:
    def __init__(self, endpoint: str, timeout: float = 120.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    async def list_models(self) -> list[LMStudioModel]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.endpoint}/api/v0/models")
            response.raise_for_status()
            data = response.json()
        return [LMStudioModel(**model) for model in data["data"]]

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        body = {"model": model, "messages": messages}
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.endpoint}/v1/chat/completions", json=body
            )
            response.raise_for_status()
            return response.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_lmstudio.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/runtimes/ tests/unit/test_lmstudio.py
git commit -m "feat: add LM Studio runtime client"
```

---

### Task 3: llmfit CLI Client

**Files:**
- Create: `src/modelhelm/models/llmfit_client.py`
- Create: `tests/fixtures/llmfit_recommend.json`
- Create: `tests/fixtures/llmfit_list.json`
- Test: `tests/unit/test_llmfit_client.py`

**Interfaces:**
- Consumes: `Settings.llmfit_binary_path` (Task 1)
- Produces:
  - `class LlmfitModel(BaseModel)` — fields `name: str`, `capability_ids: list[str]`, `fit_level: str`, `score: float`, `context_length: int`, `estimated_tps: float | None`
  - `class LlmfitClient` — constructor `__init__(self, binary_path: str | None = None)`
    - `def recommend(self) -> list[LlmfitModel]` — runs `llmfit recommend --json`, parses `{"models": [...]}`
    - `def list_models(self) -> list[LlmfitModel]` — runs `llmfit list --json`, parses bare array `[...]`, tolerates missing `fit_level`/`score`/`estimated_tps` (default `"Unknown"`, `0.0`, `None`)
  - `class LlmfitError(Exception)` — raised on nonzero exit code or JSON parse failure

**Note:** Real captured samples (trimmed to a few entries each) go in the fixtures — copy the shapes observed during brainstorming: `recommend --json` → `{"models": [{...with fit_level, score, capability_ids, context_length, estimated_tps...}]}`; `list --json` → bare `[{...with capabilities (not capability_ids), context_length, no score/fit_level...}]`.

- [ ] **Step 1: Create fixture files**

```json
// tests/fixtures/llmfit_recommend.json
{
  "models": [
    {
      "name": "Qwen/Qwen3-Coder-30B-A3B",
      "capability_ids": ["tool_use"],
      "fit_level": "Excellent",
      "score": 98.5,
      "context_length": 262144,
      "estimated_tps": 62.3
    },
    {
      "name": "groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit",
      "capability_ids": ["vision", "tool_use"],
      "fit_level": "Good",
      "score": 100.0,
      "context_length": 262144,
      "estimated_tps": 40.5
    }
  ]
}
```

```json
// tests/fixtures/llmfit_list.json
[
  {
    "name": "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
    "capabilities": ["tool_use"],
    "context_length": 32768
  },
  {
    "name": "AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16",
    "capabilities": ["tool_use"],
    "context_length": 262144
  }
]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/test_llmfit_client.py
import json
import subprocess
from pathlib import Path
import pytest
from modelhelm.models.llmfit_client import LlmfitClient, LlmfitError, LlmfitModel

FIXTURES = Path(__file__).parent.parent / "fixtures"

def test_recommend_parses_models_key(monkeypatch):
    fixture = (FIXTURES / "llmfit_recommend.json").read_text()

    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=fixture, stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    models = client.recommend()

    assert len(models) == 2
    assert models[0] == LlmfitModel(
        name="Qwen/Qwen3-Coder-30B-A3B",
        capability_ids=["tool_use"],
        fit_level="Excellent",
        score=98.5,
        context_length=262144,
        estimated_tps=62.3,
    )

def test_list_models_parses_bare_array_with_defaults(monkeypatch):
    fixture = (FIXTURES / "llmfit_list.json").read_text()

    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=fixture, stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    models = client.list_models()

    assert len(models) == 2
    assert models[0].name == "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
    assert models[0].capability_ids == ["tool_use"]
    assert models[0].fit_level == "Unknown"
    assert models[0].score == 0.0
    assert models[0].estimated_tps is None

def test_recommend_raises_on_nonzero_exit(monkeypatch):
    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="hardware detection failed")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    with pytest.raises(LlmfitError, match="hardware detection failed"):
        client.recommend()

def test_recommend_raises_on_invalid_json(monkeypatch):
    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="not json", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    with pytest.raises(LlmfitError):
        client.recommend()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_llmfit_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.models.llmfit_client'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/modelhelm/models/llmfit_client.py
import json
import shutil
import subprocess
from pydantic import BaseModel

class LlmfitModel(BaseModel):
    name: str
    capability_ids: list[str] = []
    fit_level: str = "Unknown"
    score: float = 0.0
    context_length: int = 0
    estimated_tps: float | None = None

class LlmfitError(Exception):
    pass

class LlmfitClient:
    def __init__(self, binary_path: str | None = None):
        self.binary_path = binary_path or shutil.which("llmfit") or "llmfit"

    def _run(self, args: list[str]) -> str:
        result = subprocess.run(
            [self.binary_path, *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise LlmfitError(result.stderr.strip() or "llmfit exited nonzero")
        return result.stdout

    def recommend(self) -> list[LlmfitModel]:
        stdout = self._run(["recommend", "--json"])
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LlmfitError(f"invalid JSON from llmfit recommend: {exc}") from exc
        return [LlmfitModel(**entry) for entry in payload.get("models", [])]

    def list_models(self) -> list[LlmfitModel]:
        stdout = self._run(["list", "--json"])
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LlmfitError(f"invalid JSON from llmfit list: {exc}") from exc

        models = []
        for entry in payload:
            models.append(
                LlmfitModel(
                    name=entry["name"],
                    capability_ids=entry.get("capabilities", []),
                    context_length=entry.get("context_length", 0),
                )
            )
        return models
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_llmfit_client.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add src/modelhelm/models/llmfit_client.py tests/unit/test_llmfit_client.py tests/fixtures/
git commit -m "feat: add llmfit CLI client with recommend/list parsing"
```

---

### Task 4: Model Registry

**Files:**
- Create: `src/modelhelm/models/registry.py`
- Test: `tests/unit/test_registry.py`

**Interfaces:**
- Consumes:
  - `LMStudioClient.list_models() -> list[LMStudioModel]` (Task 2)
  - `LlmfitClient.recommend() -> list[LlmfitModel]` (Task 3)
- Produces:
  - `class RegistryEntry(BaseModel)` — fields `name: str`, `runtime: str`, `available: bool`, `loaded: bool`, `context_length: int`, `capabilities: list[str]`, `fit_score: float | None`
  - `class ModelRegistry` — constructor `__init__(self, lmstudio_client: LMStudioClient, llmfit_client: LlmfitClient)`
    - `async def refresh(self) -> list[RegistryEntry]` — cross-references LM Studio's available models against llmfit's fit scores by matching on model name substring (case-insensitive); LM Studio entries with no llmfit match get `fit_score=None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_registry.py
import pytest
from modelhelm.models.registry import ModelRegistry, RegistryEntry
from modelhelm.runtimes.lmstudio import LMStudioModel
from modelhelm.models.llmfit_client import LlmfitModel

class FakeLMStudioClient:
    async def list_models(self):
        return [
            LMStudioModel(
                id="qwen3-coder-30b-a3b",
                state="loaded",
                max_context_length=262144,
                capabilities=["tool_use"],
            ),
            LMStudioModel(
                id="text-embedding-nomic-embed-text-v1.5",
                state="not-loaded",
                max_context_length=2048,
                capabilities=[],
            ),
        ]

class FakeLlmfitClient:
    def recommend(self):
        return [
            LlmfitModel(
                name="Qwen/Qwen3-Coder-30B-A3B",
                capability_ids=["tool_use"],
                fit_level="Excellent",
                score=98.5,
                context_length=262144,
                estimated_tps=62.3,
            )
        ]

@pytest.mark.asyncio
async def test_refresh_cross_references_by_name():
    registry = ModelRegistry(
        lmstudio_client=FakeLMStudioClient(), llmfit_client=FakeLlmfitClient()
    )
    entries = await registry.refresh()

    assert len(entries) == 2
    coder = next(e for e in entries if e.name == "qwen3-coder-30b-a3b")
    assert coder.available is True
    assert coder.loaded is True
    assert coder.fit_score == 98.5

    embed = next(e for e in entries if e.name == "text-embedding-nomic-embed-text-v1.5")
    assert embed.fit_score is None
    assert embed.loaded is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.models.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/models/registry.py
from pydantic import BaseModel

class RegistryEntry(BaseModel):
    name: str
    runtime: str
    available: bool
    loaded: bool
    context_length: int
    capabilities: list[str]
    fit_score: float | None = None

def _names_match(lmstudio_name: str, llmfit_name: str) -> bool:
    normalized_lmstudio = lmstudio_name.lower().replace("-", "").replace("_", "").replace(".", "")
    normalized_llmfit = llmfit_name.lower().replace("-", "").replace("_", "").replace(".", "").replace("/", "")
    return normalized_lmstudio in normalized_llmfit or normalized_llmfit.endswith(normalized_lmstudio)

class ModelRegistry:
    def __init__(self, lmstudio_client, llmfit_client):
        self.lmstudio_client = lmstudio_client
        self.llmfit_client = llmfit_client

    async def refresh(self) -> list[RegistryEntry]:
        lmstudio_models = await self.lmstudio_client.list_models()
        try:
            llmfit_models = self.llmfit_client.recommend()
        except Exception:
            llmfit_models = []

        entries = []
        for model in lmstudio_models:
            fit_score = None
            for llmfit_model in llmfit_models:
                if _names_match(model.id, llmfit_model.name):
                    fit_score = llmfit_model.score
                    break

            entries.append(
                RegistryEntry(
                    name=model.id,
                    runtime="lm-studio",
                    available=True,
                    loaded=model.state == "loaded",
                    context_length=model.max_context_length,
                    capabilities=model.capabilities,
                    fit_score=fit_score,
                )
            )
        return entries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_registry.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/models/registry.py tests/unit/test_registry.py
git commit -m "feat: add model registry cross-referencing LM Studio and llmfit"
```

---

### Task 5: Policy Engine

**Files:**
- Create: `src/modelhelm/policies/engine.py`
- Test: `tests/unit/test_policy_engine.py`

**Interfaces:**
- Consumes: `SafetyPolicy` (Task 1)
- Produces:
  - `OperationType = Literal["file_write","file_delete","git_commit","git_push","force_push","destructive_commands","production_changes"]`
  - `class PolicyEngine` — constructor `__init__(self, policy: SafetyPolicy)`
    - `def check(self, operation: OperationType) -> Literal["allow","deny","ask"]`
  - `class PathScopeError(Exception)` — raised by callers (not this module) when a write target resolves outside the repo; `PolicyEngine` itself is scope-agnostic — Task 9 enforces path scoping using this exception.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_policy_engine.py
from modelhelm.policies.engine import PolicyEngine
from modelhelm.config.settings import SafetyPolicy

def test_check_returns_configured_state_for_each_operation():
    policy = SafetyPolicy(
        file_write="allow",
        file_delete="allow",
        git_commit="ask",
        git_push="ask",
        force_push="deny",
        destructive_commands="deny",
        production_changes="deny",
    )
    engine = PolicyEngine(policy)

    assert engine.check("file_write") == "allow"
    assert engine.check("git_commit") == "ask"
    assert engine.check("force_push") == "deny"

def test_check_unknown_operation_raises():
    engine = PolicyEngine(SafetyPolicy())
    import pytest
    with pytest.raises(ValueError, match="unknown operation"):
        engine.check("delete_universe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_policy_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.policies.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/policies/engine.py
from typing import Literal
from modelhelm.config.settings import SafetyPolicy

OperationType = Literal[
    "file_write",
    "file_delete",
    "git_commit",
    "git_push",
    "force_push",
    "destructive_commands",
    "production_changes",
]

class PathScopeError(Exception):
    pass

class PolicyEngine:
    def __init__(self, policy: SafetyPolicy):
        self.policy = policy

    def check(self, operation: str) -> Literal["allow", "deny", "ask"]:
        if not hasattr(self.policy, operation):
            raise ValueError(f"unknown operation: {operation}")
        return getattr(self.policy, operation)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_policy_engine.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/policies/ tests/unit/test_policy_engine.py
git commit -m "feat: add ALLOW/DENY/ASK policy engine"
```

---

### Task 6: Git Inspector

**Files:**
- Create: `src/modelhelm/git/inspector.py`
- Test: `tests/unit/test_git_inspector.py`

**Interfaces:**
- Produces:
  - `class GitSnapshot(BaseModel)` — fields `branch: str`, `commit: str`, `is_dirty: bool`
  - `class GitInspector` — constructor `__init__(self, repository: str)`
    - `def snapshot(self) -> GitSnapshot`
    - `def diff_summary(self) -> str` — output of `git diff --stat`
    - `def files_changed_count(self) -> int` — count of changed files from `git diff --stat` / `git status --porcelain`

All methods run `git` via `subprocess.run(cwd=self.repository, ...)`.

- [ ] **Step 1: Write the failing test**

Uses a real temp git repo (no mocking needed — git itself is the fixture).

```python
# tests/unit/test_git_inspector.py
import subprocess
from modelhelm.git.inspector import GitInspector

def _init_repo(path):
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "test@test.com"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    (path / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "init"], check=True, capture_output=True)

def test_snapshot_clean_repo(tmp_path):
    _init_repo(tmp_path)
    inspector = GitInspector(str(tmp_path))
    snapshot = inspector.snapshot()

    assert snapshot.branch == "main"
    assert len(snapshot.commit) == 40
    assert snapshot.is_dirty is False

def test_snapshot_dirty_after_edit(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "README.md").write_text("changed\n")
    inspector = GitInspector(str(tmp_path))

    assert inspector.snapshot().is_dirty is True
    assert inspector.files_changed_count() == 1
    assert "README.md" in inspector.diff_summary()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_git_inspector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.git.inspector'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/git/inspector.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_git_inspector.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/git/ tests/unit/test_git_inspector.py
git commit -m "feat: add git state inspector"
```

---

### Task 7: Task Schemas + SQLite Store

**Files:**
- Create: `src/modelhelm/tasks/models.py`
- Create: `src/modelhelm/tasks/store.py`
- Test: `tests/unit/test_task_store.py`

**Interfaces:**
- Produces (`tasks/models.py`):
  - `TaskStatus = Literal["pending","running","pending_approval","completed","escalation_recommended","failed","cancelled"]`
  - `class DelegatedTask(BaseModel)` — fields `task_id: str`, `description: str`, `repository: str`, `status: TaskStatus`, `model: str | None`, `created_at: str` (ISO8601)
  - `class TaskResult(BaseModel)` — fields exactly matching spec Section 8: `task_id: str`, `status: TaskStatus`, `model: str`, `runtime: str`, `duration_seconds: float`, `files_changed: int`, `tests_run: int`, `tests_passed: int`, `tests_failed: int`, `iterations: int`, `estimated_cloud_tokens_saved: int`, `review_recommended: bool`, `summary: str`
  - `class PendingApproval(BaseModel)` — fields `operation: str`, `detail: str`, `tool_call_id: str`, `messages: list[dict]` (the full conversation so far, including the assistant message with the pending tool call) — this is what makes `resume_task` able to continue the *same* conversation instead of starting a fresh one.
- Produces (`tasks/store.py`):
  - `class TaskStore` — constructor `__init__(self, db_path: str)` (creates schema if missing)
    - `def create_task(self, description: str, repository: str) -> DelegatedTask`
    - `def get_task(self, task_id: str) -> DelegatedTask | None`
    - `def set_status(self, task_id: str, status: TaskStatus, model: str | None = None) -> None`
    - `def save_result(self, task_id: str, result: TaskResult) -> None`
    - `def get_result(self, task_id: str) -> TaskResult | None`
    - `def save_pending_approval(self, task_id: str, pending: PendingApproval) -> None`
    - `def get_pending_approval(self, task_id: str) -> PendingApproval | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_task_store.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_task_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.tasks.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/tasks/models.py
from typing import Literal
from pydantic import BaseModel

TaskStatus = Literal[
    "pending",
    "running",
    "pending_approval",
    "completed",
    "escalation_recommended",
    "failed",
    "cancelled",
]

class DelegatedTask(BaseModel):
    task_id: str
    description: str
    repository: str
    status: TaskStatus
    model: str | None = None
    created_at: str

class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    model: str
    runtime: str
    duration_seconds: float
    files_changed: int
    tests_run: int
    tests_passed: int
    tests_failed: int
    iterations: int
    estimated_cloud_tokens_saved: int
    review_recommended: bool
    summary: str

class PendingApproval(BaseModel):
    operation: str
    detail: str
    tool_call_id: str
    messages: list[dict]
```

```python
# src/modelhelm/tasks/store.py
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from modelhelm.tasks.models import DelegatedTask, TaskResult, PendingApproval

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
            task_id=row[0], description=row[1], repository=row[2],
            status=row[3], model=row[4], created_at=row[5],
        )

    def set_status(self, task_id: str, status: str, model: str | None = None) -> None:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_task_store.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/tasks/ tests/unit/test_task_store.py
git commit -m "feat: add task schemas and SQLite-backed task store"
```

---

### Task 8: Task Router

**Files:**
- Create: `src/modelhelm/routing/router.py`
- Test: `tests/unit/test_router.py`

**Interfaces:**
- Consumes: `ModelRegistry.refresh() -> list[RegistryEntry]` (Task 4)
- Produces:
  - `class NoSuitableModelError(Exception)`
  - `class TaskRouter` — constructor `__init__(self, registry: ModelRegistry)`
    - `async def select_model(self, task_description: str) -> str` — returns the `RegistryEntry.name` of the best candidate: filters to `available=True` and `"tool_use" in capabilities`, then picks highest `fit_score` (entries with `fit_score=None` sort last), raises `NoSuitableModelError` if none qualify.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_router.py
import pytest
from modelhelm.routing.router import TaskRouter, NoSuitableModelError
from modelhelm.models.registry import RegistryEntry

class FakeRegistry:
    def __init__(self, entries):
        self._entries = entries

    async def refresh(self):
        return self._entries

@pytest.mark.asyncio
async def test_select_model_picks_highest_fit_score_with_tool_use():
    registry = FakeRegistry([
        RegistryEntry(
            name="qwen3-coder-30b-a3b", runtime="lm-studio", available=True,
            loaded=True, context_length=262144, capabilities=["tool_use"], fit_score=98.5,
        ),
        RegistryEntry(
            name="qwen2.5-coder-14b-instruct", runtime="lm-studio", available=True,
            loaded=False, context_length=131072, capabilities=["tool_use"], fit_score=70.0,
        ),
    ])
    router = TaskRouter(registry)
    selected = await router.select_model("implement a REST client")
    assert selected == "qwen3-coder-30b-a3b"

@pytest.mark.asyncio
async def test_select_model_excludes_non_tool_use():
    registry = FakeRegistry([
        RegistryEntry(
            name="text-embedding-nomic", runtime="lm-studio", available=True,
            loaded=False, context_length=2048, capabilities=[], fit_score=99.0,
        ),
    ])
    router = TaskRouter(registry)
    with pytest.raises(NoSuitableModelError):
        await router.select_model("implement a REST client")

@pytest.mark.asyncio
async def test_select_model_treats_missing_fit_score_as_lowest():
    registry = FakeRegistry([
        RegistryEntry(
            name="no-fit-data", runtime="lm-studio", available=True,
            loaded=True, context_length=131072, capabilities=["tool_use"], fit_score=None,
        ),
        RegistryEntry(
            name="has-fit-data", runtime="lm-studio", available=True,
            loaded=True, context_length=131072, capabilities=["tool_use"], fit_score=1.0,
        ),
    ])
    router = TaskRouter(registry)
    selected = await router.select_model("task")
    assert selected == "has-fit-data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.routing.router'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/routing/router.py
from modelhelm.models.registry import RegistryEntry

class NoSuitableModelError(Exception):
    pass

class TaskRouter:
    def __init__(self, registry):
        self.registry = registry

    async def select_model(self, task_description: str) -> str:
        entries = await self.registry.refresh()
        candidates = [
            e for e in entries if e.available and "tool_use" in e.capabilities
        ]
        if not candidates:
            raise NoSuitableModelError(
                f"no tool-use-capable model available for task: {task_description!r}"
            )
        candidates.sort(key=lambda e: e.fit_score if e.fit_score is not None else -1, reverse=True)
        return candidates[0].name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_router.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/routing/ tests/unit/test_router.py
git commit -m "feat: add task router with tool-use + fit-score model selection"
```

---

### Task 9: Agent Tools (policy-gated file/exec operations)

**Files:**
- Create: `src/modelhelm/agents/tools.py`
- Test: `tests/unit/test_agent_tools.py`

**Interfaces:**
- Consumes: `PolicyEngine.check()` (Task 5), `PathScopeError` (Task 5)
- Produces:
  - `class ToolDenied(Exception)` — raised when policy check returns `"deny"`
  - `class ToolNeedsApproval(Exception)` — raised when policy check returns `"ask"`; carries `.operation: str` and `.detail: str`
  - `class AgentTools` — constructor `__init__(self, repository: str, policy_engine: PolicyEngine)`
    - `def read_file(self, path: str) -> str`
    - `def write_file(self, path: str, content: str) -> str` — returns confirmation message; enforces path stays within `self.repository` (raises `PathScopeError` otherwise), checks `policy_engine.check("file_write")` first
    - `def list_directory(self, path: str = ".") -> list[str]`
    - `def run_command(self, command: str) -> dict` — checks `destructive_commands` policy only if command matches a hardcoded destructive pattern list (`rm -rf`, `del /f`, `format`, `drop database`, `drop table`); returns `{"stdout": str, "stderr": str, "returncode": int}`; runs with `cwd=self.repository`, `timeout=120`
    - `def git_diff(self) -> str` — delegates to `GitInspector(self.repository).diff_summary()`
    - `def git_commit(self, message: str) -> str` — checks `policy_engine.check("git_commit")`; on `"ask"` raises `ToolNeedsApproval`; on `"allow"` runs `git add -A && git commit -m <message>`
    - `def get_tool_definitions() -> list[dict]` (module-level function) — returns the OpenAI-style tool-calling JSON schemas for all six tools above, for use in `chat_completion(tools=...)`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_agent_tools.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_agent_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.agents.tools'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/agents/tools.py
import subprocess
from pathlib import Path
from modelhelm.policies.engine import PolicyEngine, PathScopeError
from modelhelm.git.inspector import GitInspector

DESTRUCTIVE_PATTERNS = ["rm -rf", "del /f", "format ", "drop database", "drop table"]

class ToolDenied(Exception):
    pass

class ToolNeedsApproval(Exception):
    def __init__(self, operation: str, detail: str):
        self.operation = operation
        self.detail = detail
        super().__init__(f"{operation} requires approval: {detail}")

class AgentTools:
    def __init__(self, repository: str, policy_engine: PolicyEngine):
        self.repository = Path(repository).resolve()
        self.policy_engine = policy_engine

    def _resolve_scoped_path(self, path: str) -> Path:
        resolved = (self.repository / path).resolve()
        if self.repository not in resolved.parents and resolved != self.repository:
            raise PathScopeError(f"path escapes repository scope: {path}")
        return resolved

    def read_file(self, path: str) -> str:
        target = self._resolve_scoped_path(path)
        return target.read_text()

    def write_file(self, path: str, content: str) -> str:
        verdict = self.policy_engine.check("file_write")
        if verdict == "deny":
            raise ToolDenied("file_write is denied by policy")
        target = self._resolve_scoped_path(path)
        if verdict == "ask":
            raise ToolNeedsApproval("file_write", f"write to {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def list_directory(self, path: str = ".") -> list[str]:
        target = self._resolve_scoped_path(path)
        return sorted(p.name for p in target.iterdir())

    def run_command(self, command: str) -> dict:
        if any(pattern in command.lower() for pattern in DESTRUCTIVE_PATTERNS):
            verdict = self.policy_engine.check("destructive_commands")
            if verdict == "deny":
                raise ToolDenied(f"destructive command denied: {command}")
            if verdict == "ask":
                raise ToolNeedsApproval("destructive_commands", command)

        result = subprocess.run(
            command, shell=True, cwd=str(self.repository),
            capture_output=True, text=True, timeout=120,
        )
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

    def git_diff(self) -> str:
        return GitInspector(str(self.repository)).diff_summary()

    def git_commit(self, message: str) -> str:
        verdict = self.policy_engine.check("git_commit")
        if verdict == "deny":
            raise ToolDenied("git_commit is denied by policy")
        if verdict == "ask":
            raise ToolNeedsApproval("git_commit", message)

        subprocess.run(["git", "-C", str(self.repository), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-m", message],
            check=True, capture_output=True,
        )
        return f"committed: {message}"

def get_tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file relative to the repository root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file relative to the repository root, creating it if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories at a path relative to the repository root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "."}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command inside the repository (e.g. run tests).",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": "Show a summary of uncommitted changes in the repository.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Stage and commit all changes with the given message.",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        },
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_agent_tools.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/agents/tools.py tests/unit/test_agent_tools.py
git commit -m "feat: add policy-gated agent tools (file/exec/git)"
```

---

### Task 10: Local Agent Loop

**Files:**
- Create: `src/modelhelm/agents/local_agent.py`
- Test: `tests/unit/test_local_agent.py`

**Interfaces:**
- Consumes:
  - `LMStudioClient.chat_completion()` (Task 2)
  - `AgentTools` + `get_tool_definitions()` + `ToolNeedsApproval` + `ToolDenied` (Task 9)
  - `GitInspector` (Task 6)
  - `TaskResult` (Task 7)
  - `AgentConfig` (Task 1)
- Produces:
  - `class LocalAgent` — constructor `__init__(self, lmstudio_client, tools: AgentTools, git_inspector: GitInspector, agent_config: AgentConfig)`
    - `async def run(self, task_id: str, description: str, model: str, resume_messages: list[dict] | None = None) -> tuple[TaskResult, PendingApproval | None]` — drives the loop described in spec Section 6. If `resume_messages` is given, the loop continues that conversation instead of starting fresh from `description` (used by `resume_task` in Task 11). On `ToolNeedsApproval`, returns `(TaskResult(status="pending_approval", ...), PendingApproval(...))` where `PendingApproval.messages` is the full conversation up to and including the assistant message that requested the pending tool call — this is what lets `resume_task` continue the *same* conversation rather than starting over (a fresh restart would just hit the same approval gate again, since the model has no memory of already being denied). On any other terminal status the second tuple element is `None`.

To keep this task self-contained and testable without a real LLM, `LocalAgent.run` takes the chat loop's tool-call decisions from `lmstudio_client.chat_completion`, which is fully mocked in this task's tests (real end-to-end behavior is verified in Task 12's integration test).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_local_agent.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_local_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.agents.local_agent'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/agents/local_agent.py
import json
import time
from modelhelm.agents.tools import get_tool_definitions, ToolDenied, ToolNeedsApproval
from modelhelm.tasks.models import TaskResult, PendingApproval

TOOL_DISPATCH = {
    "read_file": lambda tools, args: tools.read_file(args["path"]),
    "write_file": lambda tools, args: tools.write_file(args["path"], args["content"]),
    "list_directory": lambda tools, args: tools.list_directory(args.get("path", ".")),
    "run_command": lambda tools, args: tools.run_command(args["command"]),
    "git_diff": lambda tools, args: tools.git_diff(),
    "git_commit": lambda tools, args: tools.git_commit(args["message"]),
}

SYSTEM_PROMPT = (
    "You are a coding agent. Use the available tools to inspect the "
    "repository, make changes, and run tests. When finished, respond "
    "with a final message and no tool calls."
)

class LocalAgent:
    def __init__(self, lmstudio_client, tools, git_inspector, agent_config):
        self.lmstudio_client = lmstudio_client
        self.tools = tools
        self.git_inspector = git_inspector
        self.agent_config = agent_config

    async def run(
        self,
        task_id: str,
        description: str,
        model: str,
        resume_messages: list[dict] | None = None,
    ) -> tuple[TaskResult, PendingApproval | None]:
        start_time = time.monotonic()

        if resume_messages is not None:
            messages = list(resume_messages)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ]

        iterations = 0
        for iterations in range(1, self.agent_config.max_iterations + 1):
            response = await self.lmstudio_client.chat_completion(
                model=model, messages=messages, tools=get_tool_definitions()
            )
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                result = self._build_result(
                    task_id, "completed", model, start_time, iterations,
                    message.get("content") or "Task completed.",
                )
                return result, None

            messages.append(message)
            for call in tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                try:
                    tool_result = TOOL_DISPATCH[name](self.tools, args)
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": str(tool_result)}
                    )
                except ToolNeedsApproval as exc:
                    result = self._build_result(
                        task_id, "pending_approval", model, start_time, iterations,
                        f"Paused: {exc.operation} requires approval ({exc.detail}).",
                    )
                    pending = PendingApproval(
                        operation=exc.operation,
                        detail=exc.detail,
                        tool_call_id=call["id"],
                        messages=messages,
                    )
                    return result, pending
                except ToolDenied as exc:
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": f"DENIED: {exc}"}
                    )

        result = self._build_result(
            task_id, "escalation_recommended", model, start_time, iterations,
            f"Reached max_iterations ({self.agent_config.max_iterations}) without completion.",
        )
        return result, None

    def _build_result(self, task_id, status, model, start_time, iterations, summary) -> TaskResult:
        duration = time.monotonic() - start_time
        files_changed = self.git_inspector.files_changed_count()
        return TaskResult(
            task_id=task_id,
            status=status,
            model=model,
            runtime="lm-studio",
            duration_seconds=round(duration, 2),
            files_changed=files_changed,
            tests_run=0,
            tests_passed=0,
            tests_failed=0,
            iterations=iterations,
            estimated_cloud_tokens_saved=files_changed * 500,
            review_recommended=status != "completed" or files_changed > 0,
            summary=summary,
        )
```

**Resume mechanics note:** when `resume_task` (Task 11) re-invokes `run` with `resume_messages` set, the *last* message in that list is the assistant message containing the pending tool call. On resume, the policy for the pending operation has been elevated to `"allow"` for that one call (Task 11 handles this by re-executing the tool call directly — not by relying on the model to re-request it, since a model replaying the same `tool_calls` message would otherwise hit `ToolNeedsApproval` again through the normal dispatch path). Concretely, Task 11's `resume_task` executes the approved tool call itself using `PendingApproval.tool_call_id`, appends the resulting `{"role": "tool", ...}` message, and only then calls `agent.run(..., resume_messages=messages_with_tool_result_appended)` so the loop continues from a state where the model already sees its tool call succeeded.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_local_agent.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/agents/local_agent.py tests/unit/test_local_agent.py
git commit -m "feat: add local agent execution loop with escalation and approval pausing"
```

---

### Task 11: MCP Server

**Files:**
- Create: `src/modelhelm/mcp/server.py`
- Test: `tests/unit/test_mcp_server.py`

**Interfaces:**
- Consumes: `Settings`/`load_settings` (Task 1), `LMStudioClient` (Task 2), `LlmfitClient` (Task 3), `ModelRegistry` (Task 4), `PolicyEngine` (Task 5), `GitInspector` (Task 6), `TaskStore`/`DelegatedTask`/`TaskResult` (Task 7), `TaskRouter`/`NoSuitableModelError` (Task 8), `AgentTools` (Task 9), `LocalAgent` (Task 10)
- Produces: an MCP server instance (module-level `mcp = FastMCP("modelhelm")`) with tools:
  - `get_status() -> dict` — `{"lm_studio_reachable": bool, "llmfit_available": bool, "default_runtime": str}`
  - `list_models() -> list[dict]` — serialized `RegistryEntry` list
  - `recommend_model(task_description: str) -> str` — the selected model name (delegates to `TaskRouter.select_model`)
  - `delegate_task(description: str, repository: str) -> dict` — creates a task, runs `LocalAgent.run`, persists result (and `PendingApproval` if any), returns serialized `TaskResult`
  - `get_task_status(task_id: str) -> dict` — serialized `DelegatedTask`, 404-style `{"error": "not found"}` dict if missing
  - `cancel_task(task_id: str) -> dict` — sets status to `"cancelled"`, returns updated task
  - `resume_task(task_id: str, approved: bool) -> dict` — Phase 1 semantics: looks up the task's `PendingApproval` via `task_store.get_pending_approval(task_id)`; if missing, returns `{"error": "no pending approval for this task"}`. If `approved=False`, sets status `"cancelled"` and returns the task. If `approved=True`: directly executes the approved operation via `AgentTools` (bypassing the policy engine for this one call, since the human already approved it out-of-band), appends a `{"role": "tool", "tool_call_id": ..., "content": ...}` message to `pending.messages`, then calls `LocalAgent.run(..., resume_messages=<extended messages>)` to continue the same conversation. This correctly unblocks exactly the one pending step; if the model requests another gated operation afterward, a new `PendingApproval` is saved and `resume_task` must be called again (multi-step approval chains are handled by repeated resume calls, not a single call — documented as a Phase 1 limitation, not a bug, since each step still gets independent human sign-off).

This task wires everything together behind module-level singletons constructed from `load_settings()`, matching the standard MCP Python SDK `FastMCP` pattern (`from mcp.server.fastmcp import FastMCP`, `@mcp.tool()` decorators, `mcp.run()` in `if __name__ == "__main__"`).

- [ ] **Step 1: Write the failing test**

Server logic is tested by calling the underlying tool functions directly (not through the MCP transport — that's exercised in Task 12's integration test), using dependency injection via a `create_server(settings, task_store=None)` factory so tests can inject fakes.

```python
# tests/unit/test_mcp_server.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/unit/test_mcp_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.mcp.server'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/mcp/server.py
import json
import shutil
from mcp.server.fastmcp import FastMCP
from modelhelm.config.settings import Settings, load_settings
from modelhelm.runtimes.lmstudio import LMStudioClient
from modelhelm.models.llmfit_client import LlmfitClient, LlmfitError
from modelhelm.models.registry import ModelRegistry
from modelhelm.policies.engine import PolicyEngine
from modelhelm.git.inspector import GitInspector
from modelhelm.tasks.store import TaskStore
from modelhelm.routing.router import TaskRouter, NoSuitableModelError
from modelhelm.agents.tools import AgentTools
from modelhelm.agents.local_agent import LocalAgent, TOOL_DISPATCH

class ModelHelmServer:
    """Thin holder so tests can call tool functions directly without an MCP transport."""
    def __init__(self, mcp: FastMCP, task_store: TaskStore, tools: dict):
        self.mcp = mcp
        self.task_store = task_store
        self.tools = tools

def create_server(
    settings: Settings,
    task_store: TaskStore,
    lmstudio_client,
    llmfit_client,
) -> ModelHelmServer:
    mcp = FastMCP("modelhelm")
    registry = ModelRegistry(lmstudio_client=lmstudio_client, llmfit_client=llmfit_client)
    router = TaskRouter(registry)

    async def get_status() -> dict:
        try:
            await lmstudio_client.list_models()
            lm_reachable = True
        except Exception:
            lm_reachable = False
        try:
            llmfit_client.recommend()
            llmfit_ok = True
        except (LlmfitError, Exception):
            llmfit_ok = False
        return {
            "lm_studio_reachable": lm_reachable,
            "llmfit_available": llmfit_ok,
            "default_runtime": settings.default_runtime,
        }

    async def list_models() -> list[dict]:
        entries = await registry.refresh()
        return [entry.model_dump() for entry in entries]

    async def recommend_model(task_description: str) -> str:
        return await router.select_model(task_description)

    async def delegate_task(description: str, repository: str) -> dict:
        task = task_store.create_task(description=description, repository=repository)
        try:
            model = await router.select_model(description)
        except NoSuitableModelError as exc:
            task_store.set_status(task.task_id, "failed")
            return {"task_id": task.task_id, "status": "failed", "summary": str(exc)}

        task_store.set_status(task.task_id, "running", model=model)
        policy_engine = PolicyEngine(settings.safety)
        agent_tools = AgentTools(repository, policy_engine)
        git_inspector = GitInspector(repository)
        agent = LocalAgent(
            lmstudio_client=lmstudio_client,
            tools=agent_tools,
            git_inspector=git_inspector,
            agent_config=settings.agent,
        )
        result, pending = await agent.run(task_id=task.task_id, description=description, model=model)
        task_store.set_status(task.task_id, result.status, model=model)
        task_store.save_result(task.task_id, result)
        if pending is not None:
            task_store.save_pending_approval(task.task_id, pending)
        return result.model_dump()

    async def get_task_status(task_id: str) -> dict:
        task = task_store.get_task(task_id)
        if task is None:
            return {"error": "not found"}
        return task.model_dump()

    async def cancel_task(task_id: str) -> dict:
        task = task_store.get_task(task_id)
        if task is None:
            return {"error": "not found"}
        task_store.set_status(task_id, "cancelled")
        return task_store.get_task(task_id).model_dump()

    async def resume_task(task_id: str, approved: bool) -> dict:
        task = task_store.get_task(task_id)
        if task is None:
            return {"error": "not found"}

        pending = task_store.get_pending_approval(task_id)
        if pending is None:
            return {"error": "no pending approval for this task"}

        if not approved:
            task_store.set_status(task_id, "cancelled")
            return task_store.get_task(task_id).model_dump()

        # The human already approved this exact operation out-of-band, so execute
        # it directly with an elevated one-off policy rather than routing back
        # through the normal ask-gated AgentTools methods (which would just raise
        # ToolNeedsApproval again).
        elevated_policy = settings.safety.model_copy(update={pending.operation: "allow"})
        agent_tools = AgentTools(task.repository, PolicyEngine(elevated_policy))
        tool_result = TOOL_DISPATCH[pending.operation](
            agent_tools, json.loads(pending.messages[-1]["tool_calls"][0]["function"]["arguments"])
        )
        extended_messages = pending.messages + [
            {"role": "tool", "tool_call_id": pending.tool_call_id, "content": str(tool_result)}
        ]

        git_inspector = GitInspector(task.repository)
        agent = LocalAgent(
            lmstudio_client=lmstudio_client,
            tools=AgentTools(task.repository, PolicyEngine(settings.safety)),
            git_inspector=git_inspector,
            agent_config=settings.agent,
        )
        result, new_pending = await agent.run(
            task_id=task.task_id, description=task.description, model=task.model,
            resume_messages=extended_messages,
        )
        task_store.set_status(task.task_id, result.status, model=task.model)
        task_store.save_result(task.task_id, result)
        if new_pending is not None:
            task_store.save_pending_approval(task.task_id, new_pending)
        return result.model_dump()

    tools = {
        "get_status": get_status,
        "list_models": list_models,
        "recommend_model": recommend_model,
        "delegate_task": delegate_task,
        "get_task_status": get_task_status,
        "cancel_task": cancel_task,
        "resume_task": resume_task,
    }
    for name, fn in tools.items():
        mcp.tool(name=name)(fn)

    return ModelHelmServer(mcp=mcp, task_store=task_store, tools=tools)

def main() -> None:
    settings = load_settings()
    server = create_server(
        settings=settings,
        task_store=TaskStore("modelhelm_tasks.db"),
        lmstudio_client=LMStudioClient(endpoint=settings.lm_studio.endpoint),
        llmfit_client=LlmfitClient(binary_path=settings.llmfit_binary_path or shutil.which("llmfit")),
    )
    server.mcp.run()

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/unit/test_mcp_server.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

Run: `.venv\Scripts\pytest.exe tests/unit/ -v`
Expected: PASS (all tests across all Task 1-11 test files)

- [ ] **Step 6: Commit**

```bash
git add src/modelhelm/mcp/ tests/unit/test_mcp_server.py
git commit -m "feat: add MCP server wiring all components together"
```

---

### Task 12: Integration Test (real LM Studio, scratch repo)

**Files:**
- Create: `tests/integration/test_end_to_end.py`

**Interfaces:**
- Consumes: everything from Tasks 1-11.
- Produces: no new production code — a real end-to-end pytest that is skipped when LM Studio is unreachable.

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_end_to_end.py
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
```

- [ ] **Step 2: Run against real LM Studio to confirm it works**

Run: `.venv\Scripts\pytest.exe tests/integration/ -v -s`
Expected: PASS if LM Studio is running with a tool-use model loaded; SKIPPED otherwise. Manually inspect `tmp_path` behavior isn't retained after the test (pytest tmp_path is ephemeral) — for a manual sanity check, temporarily point `repository` at a real scratch folder outside `tmp_path` and inspect the result.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "test: add end-to-end integration test against real LM Studio"
```

---

### Task 13: README + Run Instructions

**Files:**
- Modify: `README.md`

**Interfaces:**
- None (documentation only).

- [ ] **Step 1: Update README.md**

```markdown
# ModelHelm

Intelligent orchestration for local and frontier AI.

ModelHelm is an MCP server that lets Claude Code delegate coding tasks to
locally hosted models (via LM Studio) while Claude handles planning,
architecture, and final review. Model selection is informed by
[llmfit](https://github.com/) hardware/model fit scoring.

## Status

Phase 1: MCP server + local agent loop. See
`docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md` for
scope and `docs/ModelHelm-Spec.md` for the full project vision.

## Requirements

- Python >= 3.11
- [LM Studio](https://lmstudio.ai/) running locally with a tool-use-capable
  model loaded (developed against `qwen3-coder-30b-a3b`)
- [llmfit](https://github.com/) installed and resolvable (via PATH, or set
  `llmfit.binary_path` in `modelhelm.yaml`)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Running tests

```powershell
.\.venv\Scripts\pytest.exe tests/unit/ -v
.\.venv\Scripts\pytest.exe tests/integration/ -v   # requires LM Studio running
```

## Running the MCP server

```powershell
.\.venv\Scripts\python.exe -m modelhelm.mcp.server
```

Configure via `modelhelm.yaml` at the repository root (see that file for
the full default configuration, including the safety policy).
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with Phase 1 status and run instructions"
```

---

## Self-Review Notes (completed during plan authoring)

- **Spec coverage:** Every Phase 1 in-scope item from the design doc (Section 3) maps to a task: MCP tools → Task 11; LM Studio client → Task 2; llmfit integration → Task 3; router → Task 8; agent loop → Task 10; policy engine → Task 5; git inspection → Task 6; task store → Task 7; result contract → Task 7 (`TaskResult`) verified against Task 10/11 usage. `resume_task` (spec Section 6) → Task 11.
- **Placeholder scan:** No TBD/TODO markers; every step has runnable code.
- **Type consistency:** `TaskResult` fields defined in Task 7 are used identically in Task 10 (`_build_result`) and Task 11 (`.model_dump()`); `RegistryEntry` fields defined in Task 4 match usage in Task 8's router; `PolicyEngine.check()` operation names match `SafetyPolicy` field names 1:1 (Task 1 ↔ Task 5); `AgentTools` methods (Task 9) match `TOOL_DISPATCH` keys and `get_tool_definitions()` names exactly (Task 10).
- **Known Phase 1 limitation, documented not hidden:** `resume_task`'s "elevate policy for one resume call" semantics only unblock a single pending step per resume — multi-step approval chains require repeated resume calls, one per gated step (Task 11).
- **Bug caught and fixed during self-review:** the first draft of `resume_task` re-invoked `LocalAgent.run` from scratch with the same policy, which would have immediately hit the same `ToolNeedsApproval` again (the model has no memory of being denied, so it would just re-request the same gated call). Fixed by adding `PendingApproval` (Task 7, persisted alongside tasks) that captures the full conversation up to the pending tool call; `resume_task` now executes that exact tool call directly with a one-off elevated policy, appends the real tool result to the conversation, and resumes `LocalAgent.run` via a new `resume_messages` parameter (Task 10) so the model sees its call actually succeeded.
