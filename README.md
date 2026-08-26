# ModelHelm

**Intelligent orchestration for local and frontier AI.**

ModelHelm is an **MCP-native AI model orchestration platform** that allows AI agents to intelligently delegate work across local and cloud-hosted models.

ModelHelm is designed around a simple principle:

> **Use the right model for the task, not the same model for every task.**

A frontier model such as Claude can remain the high-level supervisor responsible for planning, architecture, ambiguity, and final review, while ModelHelm delegates appropriate implementation and execution workloads to capable local models running on the user's hardware.

Model selection can incorporate hardware capabilities, model capabilities, context requirements, performance, cost, and policy through integrations such as [llmfit](https://github.com/AlexsJones/llmfit).

---

## What ModelHelm Is

ModelHelm is **more than an MCP server**.

The MCP server is the primary interface through which AI agents interact with ModelHelm, while the underlying platform provides the orchestration and execution layer.

```text
                         AI Agent
                       Claude Code
                           │
                           │ MCP
                           ▼
                  ┌──────────────────┐
                  │    ModelHelm     │
                  │    MCP Server    │
                  └────────┬─────────┘
                           │
                  ┌────────▼────────┐
                  │  ModelHelm Core  │
                  │                  │
                  │ Task Routing     │
                  │ Model Selection  │
                  │ Context          │
                  │ Policy           │
                  │ Task Execution   │
                  │ Telemetry        │
                  └────────┬─────────┘
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
              llmfit   LM Studio   Ollama
                 │         │         │
                 └─────────┼─────────┘
                           ▼
                     Local Models
```

The initial implementation focuses on Claude Code + MCP + LM Studio, but the architecture is intentionally designed to remain **agent-, model-, and runtime-agnostic**.

---

# Project Vision

The long-term goal of ModelHelm is to become an **intelligent execution layer for AI agents**.

Instead of an AI agent performing every operation itself, it can delegate work to the most appropriate available model.

For example:

```text
User
 │
 ▼
Claude
 │
 ├── Architecture ───────────────► Claude
 │
 ├── Repository exploration ────► Local model
 │
 ├── Implementation ────────────► Qwen
 │
 ├── Test generation ───────────► Local model
 │
 ├── Debugging ─────────────────► Local model
 │
 └── Final review ──────────────► Claude
```

ModelHelm becomes the layer responsible for deciding:

- **Which model should perform the task?**
- **Where should that model run?**
- **What context does it need?**
- **What tools should it have access to?**
- **What policies apply?**
- **When should the task be escalated?**
- **How much local execution can replace cloud execution?**

---

# Core Architecture

ModelHelm is intended to consist of several independent layers.

## ModelHelm Core

The core orchestration engine is responsible for:

- Task classification
- Model selection
- Task delegation
- Agent execution
- Context management
- Runtime management
- Safety policies
- Task state
- Telemetry and metrics
- Escalation

The core should not depend on MCP.

This allows the same orchestration engine to eventually support multiple interfaces.

## ModelHelm MCP

MCP provides the AI-facing interface.

Tools implemented in Phase 1:

```text
get_status()
list_models()
recommend_model()
delegate_task()
get_task_status()
cancel_task()
resume_task()
```

Added in Phase 2:

```text
classify_task()
```

`classify_task(description)` previews how a description would be classified — returning its task class, disposition (`local` or `claude`), and the keyword that matched — without creating a task or running anything.

Planned for later phases:

```text
get_project_context()
update_project_context()
get_usage()
```

Claude Code is the initial consumer, but MCP should not be treated as a Claude-specific protocol.

## Model Runtimes

ModelHelm should eventually support multiple execution backends:

```text
LM Studio          (Phase 1 — implemented)
Ollama
llama.cpp
OpenAI-compatible endpoints
Cloud providers
```

The orchestration layer should remain independent of the runtime.

---

# Intelligent Model Routing

ModelHelm should eventually route tasks based on multiple signals rather than simply selecting the largest available model.

Potential routing inputs include:

| Signal | Example |
|---|---|
| Task type | Coding, debugging, architecture |
| Complexity | Low / medium / high |
| Model capability | Coding, reasoning, tools |
| Hardware | GPU, VRAM, RAM, CPU |
| Context | Required context window |
| Runtime | LM Studio, Ollama, cloud |
| Latency | Interactive vs batch |
| Cost | Local vs cloud |
| Reliability | Historical task success |
| Risk | Safe vs sensitive operation |
| Policy | User-defined routing rules |

Example:

```text
Task:
Implement a Python REST API client

             │
             ▼
       ModelHelm Router
             │
             ▼
          llmfit
             │
             ▼
      Qwen3-Coder-30B-A3B
             │
             ▼
         LM Studio
```

A more difficult architectural task could instead remain with Claude.

Phase 1 ships a thin version of this router: tasks are classified default-local, and the best available tool-use-capable model is selected by `llmfit` fit score. Phase 2 (see roadmap below) extends this into full multi-signal routing.

---

# Local-First Execution

ModelHelm is designed around **local-first execution where practical**.

Local models are particularly valuable for high-volume operations such as:

- Repository exploration
- Code search
- Implementation
- Refactoring
- Test generation
- Test execution
- Log analysis
- Routine debugging
- Documentation
- Context maintenance

Frontier models remain valuable for:

- Architecture
- Complex reasoning
- Ambiguous requirements
- High-risk changes
- Security decisions
- Final review
- Escalation

The goal is not to eliminate cloud models.

The goal is to **avoid spending expensive frontier-model inference on tasks that capable local models can perform successfully**.

---

# Context Management

A major part of the long-term project is independent project context.

ModelHelm should eventually maintain project-local AI state such as:

```text
.ai/
├── project.md
├── architecture.md
├── conventions.md
├── decisions.md
├── environment.md
├── current-task.md
├── discoveries.md
├── failures.md
├── completed.md
└── state.json
```

This allows ModelHelm to provide models with the context relevant to the current task without repeatedly sending an entire conversation or repository.

Context management should eventually support:

- Relevant-context selection
- Project memory
- Task summaries
- Architectural decisions
- Failed approaches
- Environment information
- Context compaction
- Stale-context detection

---

# Agent Execution

ModelHelm supports autonomous local task execution.

A delegated task follows a loop such as:

```text
Receive task
     │
     ▼
Inspect repository
     │
     ▼
Understand existing implementation
     │
     ▼
Implement change
     │
     ▼
Run tests
     │
     ▼
Analyze failures
     │
     ▼
Fix problems
     │
     ▼
Run tests again
     │
     ▼
Return structured result
```

ModelHelm places configurable limits around this loop (`max_iterations` in `modelhelm.yaml`) rather than allowing agents to execute indefinitely.

---

# Escalation

Local execution should not be forced when it is failing.

ModelHelm recognizes conditions such as:

- Repeated failed attempts (iteration cap reached)
- Model stalls (no tool calls for consecutive turns)
- Persistent test failures
- Ambiguous requirements
- Architecture changes
- Security-sensitive decisions
- Insufficient context
- Model capability limitations

and escalates the task back to the supervising agent or another model.

```text
Local Model
     │
     ├── Attempt 1 ──► Failed
     ├── Attempt 2 ──► Failed
     └── Attempt 3 ──► Failed
                       │
                       ▼
                  Escalation
                       │
                       ▼
                    Claude
```

---

# Safety

ModelHelm is designed for controlled automation.

The system supports configurable policies:

```text
ALLOW
ASK
DENY
```

Shipped Phase 1 defaults:

```text
File write                → ALLOW  (scoped to the target repository)
File delete                → ALLOW  (scoped to the target repository)
Git commit                 → ASK
Git push                   → ASK
Force push                 → DENY
Destructive commands       → DENY
Production changes         → DENY
```

`run_command` routes `git commit`/`git push`/`--force` flags through the same policy gates as the dedicated tools, and a hardened pattern list blocks common destructive shell commands (`rm -rf` variants, `git reset --hard`, `dd`, `mkfs`, pipe-to-shell, etc.) across bash and PowerShell. Reading sensitive files (`.env`, `.pem`, `.key`, `credentials*`, `.git/config`) is denied outright. ModelHelm never silently performs a destructive or high-impact operation — every write stays hard-scoped to the target repository regardless of policy setting.

---

# Observability and Cost Optimization

ModelHelm should eventually measure the effectiveness of local delegation.

Potential metrics include:

```text
Tasks delegated locally
Tasks completed successfully
Tasks escalated
Local inference time
Model used
Runtime used
Agent iterations
Tests passed/failed
Estimated cloud tokens avoided
Estimated cloud cost avoided
Local task success rate
```

This allows ModelHelm to answer an important question:

> **How much development work can this machine perform locally without sacrificing quality?**

Phase 1's `TaskResult` contract already reports `duration_seconds`, `files_changed`, `iterations`, and `estimated_cloud_tokens_saved` (a size-based heuristic, not real token accounting) per task. Real usage analytics land in a later phase.

---

# Project Roadmap

## Phase 1 — MCP Server + Local Agent Loop ✅ Complete

The first milestone established the basic working architecture, end to end:

```text
Claude Code
    │
    │ MCP
    ▼
ModelHelm
    │
    ▼
LM Studio
    │
    ▼
Qwen3-Coder-30B-A3B
    │
    ▼
Repository
```

Phase 1 proved that ModelHelm can:

- Expose an MCP interface
- Receive delegated tasks
- Select a model via `llmfit` + LM Studio state
- Run a local coding agent loop with tool calling
- Allow the local model to inspect and modify a repository
- Pause for human approval on git commit/push and resume correctly
- Enforce a policy engine that cannot be bypassed via shell commands
- Return structured results to the calling agent

This was validated with a real end-to-end integration test against a running LM Studio instance with `qwen3-coder-30b-a3b` loaded — not just mocks — and went through a full security review (bypass of the policy engine via `run_command`, a weak destructive-command blocklist, and a stale-approval replay bug were all found and fixed before merge).

See:

- `docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md`
- `docs/ModelHelm-Spec.md`

### Phase 1 Requirements

- Python >= 3.11
- [LM Studio](https://lmstudio.ai/) running locally
- A tool-use-capable model loaded in LM Studio
- Developed and tested with `qwen3-coder-30b-a3b`
- [llmfit](https://github.com/AlexsJones/llmfit) installed and resolvable

---

# Phase 1 Setup

Create a virtual environment and install ModelHelm:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Running Tests

Unit tests:

```powershell
.\.venv\Scripts\pytest.exe tests/unit/ -v
```

Integration tests:

```powershell
.\.venv\Scripts\pytest.exe tests/integration/ -v
```

> Integration tests require LM Studio to be running with a compatible model loaded.

## Running the MCP Server

```powershell
.\.venv\Scripts\python.exe -m modelhelm.mcp.server
```

Configure ModelHelm through:

```text
modelhelm.yaml
```

The repository configuration contains the default settings, including the safety policy.

---

# Phase 2 — Intelligent Model Routing

**In progress** — Milestone 1 (task-aware routing) is complete.

## Milestone 1 — Task Classification ✅ Complete

Every task is now classified before delegation. Classification is keyword-based and runs first, so a task classified as architecture, security, high-risk, or final review never reaches the router, the registry, or LM Studio — `delegate_task` short-circuits and returns `escalation_recommended`, recommending Claude handle it directly.

Each task class carries a disposition:

- `local` — exploration, implementation, refactoring, testing, debugging, documentation, context
- `claude` — architecture, security, high_risk, final_review

A description matching both a `claude` and a `local` class resolves to the `claude` class, and a description matching nothing falls back to the `ambiguous` class, which always escalates. Both behaviors are deliberate and fail safe: escalating a borderline task costs tokens, whereas running a security or high-risk change on a local model costs correctness.

The resolved class is persisted as `task_class` on `DelegatedTask` and returned as `task_class` on `TaskResult`, so the routing decision is auditable after the fact. Use `classify_task()` to preview a classification without creating a task.

The keyword table can be replaced in `modelhelm.yaml`:

```yaml
classification:
  classes:
    - name: security
      disposition: claude
      keywords: ["security", "auth", "credential", "vulnerability", "encrypt"]
```

When a `classification:` section is present it replaces the built-in table wholesale rather than merging into it, so a custom table must list every class you want matched. Order matters — classification is first-match-wins in list order, so list `claude`-disposition classes first to preserve the fail-safe. `ambiguous` cannot be redefined: it is the internal fallback and always escalates, and declaring a class by that name is rejected at startup, as is any class with an empty keyword (which would match every description).

Remaining Phase 2 capabilities:

- Model capability detection
- Hardware-aware routing
- Deeper `llmfit` integration (beyond fit-score selection)
- Model ranking
- Runtime selection
- Configurable routing policies
- Local vs cloud decision making

Example:

```text
Task
 │
 ▼
ModelHelm
 │
 ├── Complexity
 ├── Context
 ├── Hardware
 ├── Model capability
 ├── Cost
 └── Policy
 │
 ▼
Best execution target
```

---

# Phase 3 — Runtime Abstraction

Expand beyond LM Studio.

Planned runtimes:

```text
LM Studio        (Phase 1 — implemented)
Ollama
llama.cpp
OpenAI-compatible endpoints
Cloud providers
```

ModelHelm should be able to switch execution backends without changing the agent-facing interface.

---

# Phase 4 — Persistent Context

Introduce the `.ai/` project context system.

Capabilities:

- Project memory
- Context retrieval
- Task state
- Architecture records
- Decision records
- Failure history
- Automatic summaries
- Context compaction

---

# Phase 5 — Multi-Agent Orchestration

Support multiple specialized agents:

```text
                 Task
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      Coder     Tester    Investigator
        │          │          │
        └──────────┼──────────┘
                   ▼
                Reviewer
```

Different models can be assigned to different roles.

---

# Phase 6 — Adaptive Routing

Use historical execution data to improve routing decisions.

ModelHelm could learn that:

```text
Qwen:
Python implementation → excellent

Qwen:
Simple refactoring → excellent

Local reasoning model:
Log analysis → excellent

Claude:
Complex architecture → excellent
```

Routing becomes based not only on static model capabilities, but observed performance.

---

# Phase 7 — Distributed Model Execution

Long-term, ModelHelm could treat multiple computers as AI workers.

```text
                         ModelHelm
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
         Workstation       Server           NAS
         RTX 4090          GPU Node        CPU Node
             │               │               │
           Qwen            Large LLM        Small LLM
```

The orchestration layer could select the best available worker for each task.

---

# Long-Term Vision

ModelHelm ultimately aims to become a **general-purpose model orchestration layer for AI agents**.

The final architecture could look like:

```text
                         AI Agents
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
      Claude Code         IDEs          Custom Agents
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                           MCP
                            │
                    ┌───────▼───────┐
                    │   ModelHelm   │
                    │               │
                    │ Orchestration │
                    │ Routing       │
                    │ Context       │
                    │ Policy        │
                    │ Telemetry     │
                    └───────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          Local          Remote          Cloud
          Models         Models          Models
```

The important abstraction is:

> **Agents decide what needs to be accomplished. ModelHelm decides how and where the work should be executed.**

---

# Project Principles

ModelHelm should remain:

### Agent-agnostic

Do not design the core around Claude Code.

### Model-agnostic

Do not assume Qwen is the only useful local model.

### Runtime-agnostic

LM Studio is the starting point, not the architectural boundary.

### Local-first

Prefer local inference when it provides sufficient capability.

### Policy-driven

Users control what the system is allowed to execute.

### Observable

Delegation decisions and results should be inspectable.

### Extensible

New models, runtimes, agents, and routing strategies should be addable without rewriting the core.

### MCP-native

MCP should be the primary AI-agent integration mechanism.

---

# Current Status

**Phase 1 — MCP Server + Local Agent Loop: Complete**

```text
Claude Code
     │
     ▼
ModelHelm MCP
     │
     ▼
LM Studio
     │
     ▼
Qwen3-Coder-30B-A3B
     │
     ▼
Local Repository
```

The full delegation chain — MCP tool call → router → policy-gated agent loop → LM Studio tool calling → repository edit → human-approved commit — is implemented, tested (109 unit tests + a real end-to-end integration test), and security-reviewed.

**Phase 2 — Intelligent Model Routing: Milestone 1 (task-aware routing) complete.** Tasks are classified before delegation, and architecture, security, high-risk and final-review work escalates to Claude instead of reaching a local model.

For the detailed technical requirements and architecture, see:

```text
docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md
docs/ModelHelm-Spec.md
```
