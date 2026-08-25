# ModelHelm Phase 1 — MCP Server + Local Agent Loop

Status: Approved for planning
Date: 2026-08-25
Parent spec: `docs/ModelHelm-Spec.md` (project-wide vision, named "LocalForge" therein — product name is **ModelHelm**)

## 1. Purpose

Deliver the smallest end-to-end slice of ModelHelm that satisfies the parent
spec's Section 31 success criteria: Claude Code describes a task, ModelHelm
selects a local model via llmfit, drives an autonomous agent loop against it
through LM Studio, and returns a structured result for Claude to review —
without manually copying prompts between tools.

This is Phase 1 of a multi-phase build. It intentionally excludes: full
multi-signal routing intelligence, `.ai/` context management, cost/telemetry
analytics, and multi-runtime abstraction (Ollama/llama.cpp). Those are
follow-up specs layered on top of this foundation.

## 2. Environment (confirmed during brainstorming)

- LM Studio running locally, OpenAI-compatible API at `http://localhost:1234`,
  with `qwen3-coder-30b-a3b` (tool-calling capable, 262K context) loaded.
- `llmfit` installed via scoop (`C:\Users\Trevin\scoop\shims\llmfit.exe`,
  not on PATH in non-interactive shells — must be invoked by resolved path
  or after ensuring PATH includes the scoop shims dir). Supports
  `recommend --json`, `list --json`, `--tool-use` filter is not a valid flag
  on `recommend` (confirmed via `--help`); use `list`/`fit` filters instead
  where needed and confirm exact flags per subcommand during implementation.
- Repo: `T-Crypt/ModelHelm` on GitHub, cloned and restructured so
  `E:\ModelHelm` is the git working tree root.

## 3. Scope

### In scope (Phase 1)
- MCP server exposing: `get_status`, `list_models`, `recommend_model`,
  `delegate_task`, `get_task_status`, `cancel_task`, `resume_task`.
- LM Studio runtime client (list models, chat completions with tool calling,
  streaming, error/timeout handling).
- llmfit CLI integration (subprocess, `--json` output, normalized into the
  model registry shape).
- Thin task router: default-local classification, model selection from
  llmfit recommendation + LM Studio available/loaded models.
- Self-driving local agent loop: repo inspection, file edit, command exec,
  test run, iterative fix, capped iterations, escalation on stall/failure.
- Policy engine: ALLOW/DENY/ASK, Phase 1 default policy (see Section 7).
- Git state inspection (pre/post task branch/commit/dirty/diff summary).
- SQLite-backed task store for status/cancel/resume across calls.
- Structured result contract matching parent spec Section 19.

### Out of scope (later phases)
- `get_project_context` / `update_project_context` — needs `.ai/` design.
- `get_usage` and real cost/token analytics — needs telemetry design.
- Multi-signal routing (VRAM/latency/performance scoring) — Phase 1 uses a
  simple default-local + llmfit-fit heuristic only.
- Ollama/llama.cpp runtime abstraction — LM Studio only.
- Multi-agent pipelines, parallel execution.
- Containerized/sandboxed execution — Phase 1 runs subprocesses directly on
  the host, scoped to the target repository path.

## 4. Architecture

```text
Claude Code
    │ MCP
    ▼
modelhelm.mcp.server
    │
    ├── delegate_task() ──► routing.router ──► models.registry (llmfit)
    │                              │
    │                              ▼
    │                        agents.local_agent
    │                              │
    │                    ┌─────────┼─────────┐
    │                    ▼         ▼         ▼
    │              runtimes.lmstudio  policies.engine  git.inspector
    │                    │
    │                    ▼
    │              LM Studio (qwen3-coder-30b-a3b)
    │
    └── get_status/list_models/get_task_status/cancel_task/resume_task
              │
              ▼
         tasks.store (SQLite)
```

Execution model: direct subprocess calls on the host (file I/O, git, test
runners), no containerization. Safety comes from the policy engine plus git
being the reversibility net for file edits, not OS-level sandboxing.

## 5. Components

| Module | Responsibility |
|---|---|
| `mcp/server.py` | MCP server; tool registration and request/response marshaling |
| `runtimes/lmstudio.py` | LM Studio OpenAI-compatible client: models, chat completion w/ tool calling, streaming |
| `models/llmfit_client.py` | Subprocess wrapper for `llmfit list/recommend --json`; parses and normalizes output |
| `models/registry.py` | In-memory normalized model registry (spec Section 11 shape), refreshed per call |
| `routing/router.py` | Task classification + model selection (Phase 1: simple heuristic, not full Section 8 scoring) |
| `agents/local_agent.py` | Agent execution loop (Section 16): plan → edit → test → fix → result |
| `agents/tools.py` | Tool-calling functions exposed to the model: `read_file`, `write_file`, `list_directory`, `run_command`, `git_diff` |
| `policies/engine.py` | ALLOW/DENY/ASK evaluation for every write/exec/git operation |
| `git/inspector.py` | Pre/post task git state and diff summary |
| `tasks/store.py` | SQLite-backed task records: status, results, pending-approval state |
| `config/settings.py` | Loads `modelhelm.yaml` (runtime endpoints, policy overrides, iteration caps) |

## 6. Agent Loop

```text
delegate_task(description, repository, ...)
    │
    ▼
router.select_model(task) ──► llmfit recommend + LM Studio available models
    │
    ▼
local_agent.run(task, model)
    │
    ├─ git.inspector.snapshot() [pre-state]
    ├─ loop (max_iterations, default 8):
    │     ├─ prompt model with tool definitions
    │     ├─ for each tool call:
    │     │     ├─ policies.engine.check(operation) → ALLOW / DENY / ASK
    │     │     ├─ ALLOW → execute, return result to model
    │     │     ├─ DENY  → return refusal to model, log
    │     │     └─ ASK   → set task status = pending_approval, STOP loop,
    │     │                return control to Claude via delegate_task response
    │     ├─ if test_before_completion: run tests
    │     └─ if tests pass and model signals done → break
    │     └─ if no tool calls for 2 consecutive turns → break (stalled)
    ├─ git.inspector.snapshot() [post-state, diff summary]
    └─ build Result (Section 19 contract)
```

Escalation conditions (Section 21), all resulting in
`status: "escalation_recommended"`: iterations exhausted with failing tests,
model stalls (no tool calls, 2 consecutive turns), or a DENY is hit on an
operation the model cannot work around.

`resume_task(task_id, approved: bool)` — new Phase 1 tool, not in the
parent spec's initial list but required by the approval flow: lets Claude
approve or reject a pending `git_commit`/`git_push` step and continue the
loop from where it paused.

## 7. Safety Policy (Phase 1 default)

```yaml
safety:
  file_write: allow          # scoped to target repository path only
  file_delete: allow          # scoped to target repository path only
  git_commit: ask
  git_push: ask
  force_push: deny
  destructive_commands: deny  # rm -rf, disk/db operations, etc.
  production_changes: deny    # no prod config paths in Phase 1
```

Rationale: file edits are cheap to reverse via git, so the loop can move
autonomously within the working tree. Anything that leaves the working tree
(a commit, a push) or is inherently destructive stops the loop and hands
control back to Claude Code, which relays the approval decision to the user.
This matches "Claude is the master for now" — ModelHelm executes, Claude
supervises and is the human-facing approval relay.

`file_write`/`file_delete` are hard-scoped to the resolved `repository` path
regardless of policy setting — the agent cannot write outside the target
repo even under `allow`.

## 8. Result Contract

Matches parent spec Section 19 exactly:

```json
{
  "task_id": "uuid",
  "status": "completed | escalation_recommended | pending_approval | failed",
  "model": "qwen3-coder-30b-a3b",
  "runtime": "lm-studio",
  "duration_seconds": 184,
  "files_changed": 8,
  "tests_run": 32,
  "tests_passed": 32,
  "tests_failed": 0,
  "iterations": 3,
  "estimated_cloud_tokens_saved": 18400,
  "review_recommended": true,
  "summary": "Implemented Proxmox API client and tests."
}
```

`estimated_cloud_tokens_saved` in Phase 1 is a rough heuristic based on
repo/diff size, not real token accounting — flagged as an approximation
until the Section 20 cost-analytics phase.

## 9. Configuration

`modelhelm.yaml` at repo root, minimal Phase 1 shape:

```yaml
modelhelm:
  default_runtime: lm-studio

runtimes:
  lm-studio:
    endpoint: http://localhost:1234

llmfit:
  binary_path: null   # null = resolve via PATH; override if not on PATH

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

## 10. Testing

- **Unit tests** (`tests/unit/`): each module tested in isolation.
  - `runtimes/lmstudio.py` — mocked `httpx` responses.
  - `models/llmfit_client.py` — mocked subprocess output, using fixtures
    captured from real `llmfit recommend --json` / `list --json` runs.
  - `policies/engine.py` — full ALLOW/DENY/ASK matrix, pure logic.
  - `tasks/store.py` — against a temp SQLite DB.
  - `routing/router.py` — mocked registry + llmfit client.
- **Integration test** (`tests/integration/`): end-to-end run against a
  scratch git repo fixture, using the real LM Studio endpoint. Skipped
  automatically if LM Studio isn't reachable at test time (not part of the
  default fast suite; run manually or in a dedicated CI job with LM Studio
  available).
- `pytest` + `pytest-asyncio`. No coverage gate in Phase 1.

## 11. Tech Stack

- Python ≥3.11
- MCP Python SDK
- Pydantic (task/result schemas, config validation)
- httpx (async LM Studio client)
- asyncio
- Git via subprocess (not GitPython — consistent with direct-subprocess
  execution model)
- SQLite via stdlib `sqlite3`
- pytest, pytest-asyncio

FastAPI is **not** included in Phase 1 — no independent HTTP surface is
needed; MCP handles the Claude-facing transport.

## 12. Repository Structure (Phase 1 slice of parent Section 29)

```text
ModelHelm/
├── src/
│   └── modelhelm/
│       ├── mcp/
│       ├── agents/
│       ├── models/
│       ├── routing/
│       ├── runtimes/
│       ├── policies/
│       ├── git/
│       ├── tasks/
│       └── config/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── ModelHelm-Spec.md
│   └── superpowers/specs/
├── modelhelm.yaml
├── pyproject.toml
├── README.md
└── LICENSE
```

## 13. Success Criteria (Phase 1)

Mirrors parent spec Section 31, scoped to what Phase 1 can actually prove:

1. Claude Code connects to the ModelHelm MCP server.
2. Claude calls `delegate_task()` with a real implementation task against a
   test repository.
3. ModelHelm selects `qwen3-coder-30b-a3b` via llmfit + LM Studio state.
4. The local agent loop inspects the repo, edits files, runs tests, and
   iterates on failures — without any manual prompt copying.
5. A `git_commit` step correctly pauses the loop with `pending_approval`,
   and `resume_task()` correctly continues it after approval.
6. ModelHelm returns a structured result matching the Section 19 contract.
7. Claude Code can read and act on that result.

## 14. Open Items Carried Forward

- Confirm exact `llmfit` subcommand flags during implementation (`--help`
  output for `list`/`fit`/`recommend` should be re-checked per subcommand;
  `--tool-use` is valid on some subcommands, not on `recommend`).
- `resume_task` is a Phase 1 addition beyond the parent spec's initial MCP
  tool list — should be folded back into the parent spec's Section 7 tool
  list in a future revision.
- Cost/token-saved estimation is a placeholder heuristic pending Phase-N
  telemetry design (parent spec Section 20).
