# LocalForge — Local/Cloud AI Coding Orchestration Platform

## 1. Project Summary

**LocalForge** is a model-agnostic AI coding orchestration platform that combines frontier cloud models such as Claude with locally hosted coding and reasoning models through LM Studio, Ollama, or compatible runtimes.

The core idea is:

> Use frontier models for high-value planning, architecture, ambiguity resolution, and final review while delegating routine implementation, investigation, testing, and context maintenance to capable local models.

The system should automatically determine where work should execute based on task complexity, available hardware, model capabilities, context requirements, cost, and user-defined policy.

### Primary goal

Reduce cloud-model token consumption without sacrificing the quality of software development.

### Initial target environment

- Windows/Linux workstation
- NVIDIA RTX 4090-class GPU
- LM Studio
- Qwen3-Coder-30B-A3B
- Claude Code
- llmfit
- Git repositories
- MCP-compatible tooling

---

# 2. Problem Statement

Current AI coding workflows often send nearly every operation to a premium cloud model:

```text
User
  ↓
Claude
  ↓
Read files
  ↓
Reason
  ↓
Edit code
  ↓
Run tests
  ↓
Debug
  ↓
Repeat
```

This is powerful but expensive and inefficient.

Many coding operations do not require a frontier model:

- repository exploration
- straightforward implementation
- refactoring
- test generation
- test execution
- log analysis
- documentation updates
- repetitive edits
- code search
- context summarization
- simple debugging

LocalForge should move these workloads to local models whenever they are capable of producing an acceptable result.

The resulting workflow becomes:

```text
                    User
                      │
                      ▼
                 Claude Code
                      │
             Planning / Supervision
                      │
                      ▼
              LocalForge MCP
                      │
             ┌────────┴────────┐
             │                 │
        Local execution    Cloud execution
             │                 │
          llmfit              Claude
             │
        Model selection
             │
         LM Studio
             │
       Qwen / other LLM
```

---

# 3. Core Design Principle

LocalForge should not attempt to replace Claude.

It should make Claude more efficient.

### Claude should primarily handle

- requirements interpretation
- architectural decisions
- difficult reasoning
- task decomposition
- ambiguity resolution
- high-risk changes
- final review
- escalation decisions

### Local models should primarily handle

- repository exploration
- implementation
- repetitive edits
- test generation
- test execution
- debugging
- log analysis
- documentation
- context maintenance
- routine refactoring

The system must support the reverse arrangement when a local model is demonstrably better suited to a task.

---

# 4. Objectives

## MVP Objectives

- [ ] Provide an MCP server that Claude Code can communicate with.
- [ ] Detect available local AI runtimes.
- [ ] Discover installed models.
- [ ] Integrate llmfit model recommendations.
- [ ] Route tasks to LM Studio.
- [ ] Execute coding tasks against a repository.
- [ ] Return structured task results to Claude.
- [ ] Track token/cost estimates.
- [ ] Maintain project-local AI context.
- [ ] Support explicit user approval for risky operations.
- [ ] Log all delegated tasks and model decisions.

## Future Objectives

- [ ] Automatic model loading/unloading.
- [ ] Multi-model task pipelines.
- [ ] Parallel local agents.
- [ ] Model performance benchmarking.
- [ ] Automatic context compaction.
- [ ] Persistent project memory.
- [ ] Model quality scoring.
- [ ] Human approval checkpoints.
- [ ] Web UI for task visualization.
- [ ] Remote workers.
- [ ] Multiple GPU support.
- [ ] Cloud-model failover.
- [ ] Cost optimization analytics.

---

# 5. Non-Goals

LocalForge should initially NOT attempt to:

- train models
- create a new inference engine
- replace LM Studio/Ollama
- replace Claude Code
- build a general-purpose chatbot
- automatically execute destructive commands
- silently modify production systems
- guarantee that a local model is equivalent to a frontier model

The project is an **orchestration and delegation layer**.

---

# 6. High-Level Architecture

```text
┌─────────────────────────────────────────────────────┐
│                     USER                            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                   CLAUDE CODE                       │
│                                                     │
│  Architecture • Planning • Review • Escalation      │
└──────────────────────┬──────────────────────────────┘
                       │ MCP
                       ▼
┌─────────────────────────────────────────────────────┐
│                 LOCALFORGE                          │
│                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Task Router │  │ Context      │  │ Policy     │ │
│  │             │  │ Manager      │  │ Engine     │ │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                │        │
│         └─────────────────┼────────────────┘        │
│                           │                         │
│                    ┌──────▼──────┐                  │
│                    │ Model       │                  │
│                    │ Registry    │                  │
│                    └──────┬──────┘                  │
└───────────────────────────┼─────────────────────────┘
                            │
                 ┌──────────┼──────────┐
                 │          │          │
                 ▼          ▼          ▼
              llmfit    LM Studio    Ollama
                 │          │          │
                 │       Local LLMs    │
                 │          │          │
                 └──────────┼──────────┘
                            ▼
                    ┌───────────────┐
                    │ Repository    │
                    │ Tools / Tests │
                    └───────────────┘
```

---

# 7. Core Components

## 7.1 MCP Server

LocalForge exposes its functionality to Claude Code through MCP.

Initial tools:

```text
localforge.get_status()
localforge.list_models()
localforge.recommend_model()
localforge.delegate_task()
localforge.get_task_status()
localforge.cancel_task()
localforge.get_project_context()
localforge.update_project_context()
localforge.get_usage()
```

The MCP interface should remain runtime-agnostic.

---

# 8. Task Router

The Task Router is the core intelligence layer.

Every delegated task should be represented internally as:

```json
{
  "task_id": "uuid",
  "description": "Implement Proxmox API client",
  "repository": "/path/to/repository",
  "complexity": "high",
  "risk": "medium",
  "context_required": "high",
  "language": "python",
  "requires_tools": true,
  "requires_network": false,
  "preferred_runtime": "lm-studio"
}
```

The router evaluates:

- task complexity
- task type
- programming language
- context requirements
- required tool capabilities
- model tool-calling support
- available VRAM
- available RAM
- model context length
- model performance
- latency
- estimated cloud cost
- user policy

---

# 9. Task Classification

Initial task classes:

| Class | Examples | Default |
|---|---|---|
| Exploration | Find files, inspect architecture | Local |
| Implementation | Write normal application code | Local |
| Refactoring | Mechanical/code-quality changes | Local |
| Testing | Create/run/fix tests | Local |
| Debugging | Analyze errors/logs | Local |
| Documentation | README/API/docs | Local |
| Context | Summarize/update project memory | Local |
| Architecture | Major system design | Claude |
| Security | High-impact security decisions | Claude |
| Ambiguous | Requirements unclear | Claude |
| High-risk | Production/destructive changes | Claude + approval |
| Final review | Review significant implementation | Claude |

This should be configurable.

---

# 10. Model Selection

LocalForge should integrate with llmfit rather than duplicate hardware/model compatibility logic.

Model selection should consider:

```text
Model capability
      +
Hardware compatibility
      +
Available VRAM
      +
Context length
      +
Tool calling
      +
Task type
      +
Performance
      +
Latency
      +
User policy
```

Example:

```text
Task:
"Implement a REST API client and tests."

llmfit:
  Qwen3-Coder-30B-A3B
  Fit: Excellent
  Context: 256K
  Tool calling: Yes
  Runtime: LM Studio

Router:
  Select Qwen3-Coder-30B-A3B
```

---

# 11. Model Registry

LocalForge should maintain a normalized model registry.

Example:

```json
{
  "name": "qwen3-coder-30b-a3b",
  "runtime": "lm-studio",
  "endpoint": "http://localhost:1234",
  "capabilities": [
    "coding",
    "tools",
    "reasoning"
  ],
  "context_length": 262144,
  "quantization": "Q4_K_M",
  "status": "available"
}
```

The registry should be refreshed dynamically.

---

# 12. LM Studio Integration

LM Studio is the first-class local runtime for MVP.

Required capabilities:

- discover available models
- determine loaded model
- send chat/completion requests
- support tool calling where available
- stream responses
- retrieve model status
- load models
- unload models
- handle errors/timeouts

The implementation should use LM Studio's supported API rather than relying on UI automation.

---

# 13. Claude Integration

Claude Code remains the primary high-level interface.

LocalForge should integrate through MCP rather than attempting to emulate Claude's internal behavior.

Example:

```text
Claude:
"I need to implement the Proxmox API layer."

Claude → localforge.delegate_task()

LocalForge:
1. Analyze task
2. Query model registry
3. Ask llmfit for recommendation
4. Select Qwen
5. Execute local agent
6. Run tests
7. Produce structured result

LocalForge → Claude:
"Task completed.
Files changed: 7
Tests: 24 passed
Warnings: 1
Review recommended: yes"
```

Claude can then review or continue.

---

# 14. Context Management

Context management is a primary feature rather than an afterthought.

Every repository can optionally contain:

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

## Context responsibilities

LocalForge should:

- identify relevant context
- inject only relevant context into tasks
- summarize completed work
- update project state
- record architectural decisions
- record failed approaches
- prevent stale context from being repeatedly injected

---

# 15. Context Pipeline

```text
Repository
    │
    ▼
File discovery
    │
    ▼
Relevant context selection
    │
    ▼
Task context assembly
    │
    ▼
Local model
    │
    ▼
Result extraction
    │
    ▼
Context summarization
    │
    ▼
.ai/ state update
```

The goal is to avoid sending the entire repository and conversation history to every model.

---

# 16. Agent Execution

A delegated local task should operate inside a controlled agent loop:

```text
Receive task
   ↓
Inspect repository
   ↓
Plan implementation
   ↓
Modify files
   ↓
Run tests
   ↓
Analyze failures
   ↓
Fix
   ↓
Run tests again
   ↓
Produce result
```

Maximum iterations should be configurable.

Example:

```yaml
agent:
  max_iterations: 8
  test_before_completion: true
  require_clean_git_diff: false
  require_tests: true
```

---

# 17. Safety Model

LocalForge must be conservative with filesystem and system operations.

Never automatically perform destructive operations such as:

- `rm -rf`
- disk formatting
- database destruction
- force pushes
- production configuration changes
- credential modification
- infrastructure destruction

unless explicitly approved.

The system should support:

```text
ALLOW
DENY
ASK
```

policy states.

Example:

```yaml
policies:
  file_write: allow
  git_commit: ask
  git_push: ask
  force_push: deny
  production_changes: ask
  destructive_commands: deny
```

---

# 18. Git Integration

LocalForge should understand Git state.

Before execution:

```text
branch
commit
working tree status
```

After execution:

```text
files changed
diff summary
tests
commit status
```

The agent should never silently force-push.

---

# 19. Result Contract

Every delegated task should return structured metadata.

Example:

```json
{
  "task_id": "123",
  "status": "completed",
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

---

# 20. Cost Optimization

One of LocalForge's primary metrics is cloud-token avoidance.

The system should estimate:

```text
Local execution
vs
Claude execution
```

Metrics:

- local inference time
- estimated Claude input tokens
- estimated Claude output tokens
- estimated cloud cost
- local task duration
- model used
- success/failure
- escalation rate

Dashboard example:

```text
Today

Local tasks:             47
Claude tasks:             9
Local inference:          2h 14m
Estimated tokens saved:  182K
Estimated cloud cost avoided: $X.XX
Local success rate:       91%
Escalations:               6
```

Exact cost calculations should be configurable by provider/model pricing.

---

# 21. Escalation

A local model should not be forced to solve a task indefinitely.

Escalate when:

- repeated attempts fail
- tests continue failing
- requirements become ambiguous
- architecture changes are required
- security-sensitive decisions arise
- context exceeds local model capabilities
- model confidence is low
- user policy requires cloud review

Example:

```text
Local Qwen
   ↓
Attempt 1: failed
   ↓
Attempt 2: failed
   ↓
Attempt 3: failed
   ↓
Escalate
   ↓
Claude
   ↓
Review / correction
```

---

# 22. Confidence and Quality

LocalForge should track task quality rather than assuming the largest model is always best.

Possible signals:

```text
Tests passed
Diff size
Repeated edits
Tool errors
Agent iterations
Compilation success
Lint success
Reviewer score
Escalation frequency
```

This allows the system to learn:

```text
Qwen 30B:
Python implementation → excellent

Qwen 30B:
Complex architecture → mediocre

Small model:
README edits → excellent
```

Future versions can automatically improve routing policies.

---

# 23. Multi-Agent Mode

Future versions may allow:

```text
             Task
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
   Coder    Tester   Investigator
      │        │        │
      └────────┼────────┘
               ▼
            Reviewer
```

All agents can run locally where hardware permits.

The controller coordinates them.

---

# 24. Parallel Execution

Independent tasks should be executable concurrently when GPU/CPU resources permit.

Example:

```text
Task
 ├── API implementation
 ├── Documentation
 └── Unit tests
```

LocalForge determines whether:

```text
parallel = beneficial
```

or

```text
parallel = resource constrained
```

---

# 25. Hardware Awareness

LocalForge should continuously know:

```text
GPU
VRAM
RAM
CPU
GPU utilization
VRAM utilization
loaded model
available models
```

This information feeds routing decisions.

Example:

```text
RTX 4090
24 GB VRAM
18 GB currently used

Qwen 30B:
AVAILABLE

70B model:
NOT RECOMMENDED
```

---

# 26. Runtime Abstraction

MVP:

```text
LM Studio
```

Next:

```text
Ollama
llama.cpp
OpenAI-compatible endpoints
```

Architecture:

```text
                 LocalForge
                     │
              Runtime Interface
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   LM Studio       Ollama       llama.cpp
```

The router should not care which runtime executes the model.

---

# 27. Configuration

Example:

```yaml
localforge:
  default_runtime: lm-studio
  auto_route: true
  auto_load_models: false

routing:
  prefer_local: true
  cloud_escalation: true
  minimum_local_confidence: 0.75

models:
  coding:
    preferred:
      - qwen3-coder-30b-a3b

  reasoning:
    preferred: []

safety:
  destructive_operations: deny
  git_push: ask
  production_changes: ask

context:
  enabled: true
  directory: .ai

agent:
  max_iterations: 8
  run_tests: true
```

---

# 28. Suggested Technology Stack

## Backend

Python is recommended for the first implementation.

Reasons:

- excellent AI ecosystem
- MCP support
- easy subprocess management
- easy HTTP integration
- good async support
- straightforward cross-platform development

Potential stack:

```text
Python
FastAPI
MCP SDK
Pydantic
httpx
asyncio
GitPython or Git CLI
SQLite
```

## Frontend

Not required for MVP.

Future:

```text
Next.js
React
Tailwind
shadcn/ui
```

## Storage

MVP:

```text
SQLite
```

Project state:

```text
.ai/*.md
```

---

# 29. Repository Structure

Proposed:

```text
localforge/
├── src/
│   └── localforge/
│       ├── api/
│       ├── agents/
│       ├── context/
│       ├── models/
│       ├── routing/
│       ├── runtimes/
│       ├── policies/
│       ├── git/
│       ├── tasks/
│       ├── telemetry/
│       └── mcp/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/
│
├── examples/
│
├── pyproject.toml
├── README.md
├── LICENSE
└── CONTRIBUTING.md
```

---

# 30. MVP Milestones

## Phase 1 — Local Runtime

- [ ] Create Python project.
- [ ] Implement LM Studio client.
- [ ] Detect endpoint.
- [ ] List models.
- [ ] Send prompt.
- [ ] Stream responses.
- [ ] Handle failures.

## Phase 2 — Model Intelligence

- [ ] Integrate llmfit.
- [ ] Normalize model metadata.
- [x] Implement task classification.
- [ ] Implement routing engine.
- [ ] Add configurable routing policies.

## Phase 3 — Agent

- [ ] Repository inspection.
- [ ] File editing.
- [ ] Command execution.
- [ ] Test execution.
- [ ] Iterative correction.
- [ ] Structured results.

## Phase 4 — MCP

- [ ] Implement MCP server.
- [ ] Expose delegation tools.
- [ ] Connect Claude Code.
- [ ] Test Claude → LocalForge → Qwen workflow.

## Phase 5 — Context

- [ ] `.ai/` project structure.
- [ ] Context selection.
- [ ] Task summaries.
- [ ] Decision tracking.
- [ ] Persistent state.

## Phase 6 — Safety

- [ ] Command policy engine.
- [ ] Approval workflow.
- [ ] Git safety.
- [ ] Destructive operation protection.

## Phase 7 — Optimization

- [ ] Token savings tracking.
- [ ] Local execution metrics.
- [ ] Model performance metrics.
- [ ] Escalation analytics.

---

# 31. MVP Success Criteria

The MVP is successful if the following workflow works reliably:

```text
1. User opens Claude Code.
2. User describes a development task.
3. Claude determines the task can be delegated.
4. Claude calls LocalForge through MCP.
5. LocalForge analyzes the task.
6. LocalForge asks llmfit for model suitability.
7. LocalForge selects Qwen3-Coder.
8. LocalForge sends the task to LM Studio.
9. Qwen inspects the repository.
10. Qwen modifies the code.
11. Qwen runs tests.
12. Qwen fixes failures.
13. LocalForge returns structured results.
14. Claude reviews the result.
15. Claude continues or accepts the implementation.
```

The complete process should happen without manually copying prompts between Claude and LM Studio.

---

# 32. Example User Experience

User:

```text
Add authentication to this application.
```

Claude:

```text
This can be delegated to LocalForge.
```

LocalForge:

```text
Task analysis:
Type: implementation
Complexity: high
Risk: medium
Context: high

Recommended model:
Qwen3-Coder-30B-A3B

Runtime:
LM Studio

Action:
Delegate
```

Qwen:

```text
Inspecting repository...
Found FastAPI application.
Found existing user model.
Implementing authentication...
Running tests...
```

Result:

```text
Task completed.

Model:
Qwen3-Coder-30B-A3B

Files changed:
9

Tests:
41 passed

Iterations:
3

Estimated Claude tokens avoided:
~22,000

Review:
Recommended
```

Claude then reviews the result.

---

# 33. Future Vision

LocalForge could eventually become a distributed AI execution platform.

```text
                         Claude
                            │
                            ▼
                     LocalForge Cloud
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          Desktop         Server         NAS
          RTX 4090        GPU Node       CPU Node
             │              │              │
           Qwen           DeepSeek       Small LLM
```

Tasks are automatically assigned to the best available execution environment.

The local workstation becomes an AI worker.

Additional machines become workers.

Cloud models become escalation/review workers.

---

# 34. Differentiation

LocalForge should differentiate itself from traditional AI coding assistants through:

1. **Model-agnostic orchestration**
2. **Automatic local/cloud routing**
3. **Hardware-aware model selection**
4. **llmfit integration**
5. **Persistent project context**
6. **MCP-native architecture**
7. **Transparent delegation**
8. **Cloud token/cost optimization**
9. **Local-first execution**
10. **Automatic escalation**
11. **Runtime abstraction**
12. **Safety and policy controls**

The key product concept is:

> **Don't ask which AI model is best. Ask which model should perform this task.**

---

# 35. Initial Project Definition

## Name

**LocalForge**

## Tagline

**Local intelligence. Frontier supervision.**

## One-sentence description

LocalForge is an MCP-native AI orchestration layer that intelligently delegates software development tasks between local LLMs and frontier cloud models based on capability, hardware, context, cost, and risk.

## MVP

Claude Code + MCP + LocalForge + llmfit + LM Studio + Qwen3-Coder.

## Primary demonstration

A user gives Claude Code a substantial software-development task. Claude delegates implementation to a local Qwen model running on an RTX 4090. The local agent edits the repository, runs tests, fixes failures, and returns a structured result to Claude for final review.

## Core metric

**Percentage of development workload completed locally while maintaining acceptable task quality.**
