ModelHelm

Intelligent orchestration for local and frontier AI.

ModelHelm is an MCP-native AI model orchestration platform that allows AI agents to intelligently delegate work across local and cloud-hosted models.

ModelHelm is designed around a simple principle:

Use the right model for the task, not the same model for every task.

A frontier model such as Claude can remain the high-level supervisor responsible for planning, architecture, ambiguity, and final review, while ModelHelm delegates appropriate implementation and execution workloads to capable local models running on the user’s hardware.

Model selection can incorporate hardware capabilities, model capabilities, context requirements, performance, cost, and policy through integrations such as llmfit.

⸻

What ModelHelm Is

ModelHelm is more than an MCP server.

The MCP server is the primary interface through which AI agents interact with ModelHelm, while the underlying platform provides the orchestration and execution layer.

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

The initial implementation focuses on Claude Code + MCP + LM Studio, but the architecture is intentionally designed to remain agent-, model-, and runtime-agnostic.

⸻

Project Vision

The long-term goal of ModelHelm is to become an intelligent execution layer for AI agents.

Instead of an AI agent performing every operation itself, it can delegate work to the most appropriate available model.

For example:

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

ModelHelm becomes the layer responsible for deciding:

* Which model should perform the task?
* Where should that model run?
* What context does it need?
* What tools should it have access to?
* What policies apply?
* When should the task be escalated?
* How much local execution can replace cloud execution?

⸻

Core Architecture

ModelHelm is intended to consist of several independent layers.

ModelHelm Core

The core orchestration engine is responsible for:

* Task classification
* Model selection
* Task delegation
* Agent execution
* Context management
* Runtime management
* Safety policies
* Task state
* Telemetry and metrics
* Escalation

The core should not depend on MCP.

This allows the same orchestration engine to eventually support multiple interfaces.

ModelHelm MCP

MCP provides the AI-facing interface.

Potential tools include:

modelhelm.get_status()
modelhelm.list_models()
modelhelm.recommend_model()
modelhelm.delegate_task()
modelhelm.get_task_status()
modelhelm.cancel_task()
modelhelm.get_project_context()
modelhelm.update_project_context()
modelhelm.get_usage()

Claude Code is the initial consumer, but MCP should not be treated as a Claude-specific protocol.

Model Runtimes

ModelHelm should eventually support multiple execution backends:

LM Studio
Ollama
llama.cpp
OpenAI-compatible endpoints
Cloud providers

The orchestration layer should remain independent of the runtime.

⸻

Intelligent Model Routing

ModelHelm should eventually route tasks based on multiple signals rather than simply selecting the largest available model.

Potential routing inputs include:

Signal	Example
Task type	Coding, debugging, architecture
Complexity	Low / medium / high
Model capability	Coding, reasoning, tools
Hardware	GPU, VRAM, RAM, CPU
Context	Required context window
Runtime	LM Studio, Ollama, cloud
Latency	Interactive vs batch
Cost	Local vs cloud
Reliability	Historical task success
Risk	Safe vs sensitive operation
Policy	User-defined routing rules

Example:

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

A more difficult architectural task could instead remain with Claude.

⸻

Local-First Execution

ModelHelm is designed around local-first execution where practical.

Local models are particularly valuable for high-volume operations such as:

* Repository exploration
* Code search
* Implementation
* Refactoring
* Test generation
* Test execution
* Log analysis
* Routine debugging
* Documentation
* Context maintenance

Frontier models remain valuable for:

* Architecture
* Complex reasoning
* Ambiguous requirements
* High-risk changes
* Security decisions
* Final review
* Escalation

The goal is not to eliminate cloud models.

The goal is to avoid spending expensive frontier-model inference on tasks that capable local models can perform successfully.

⸻

Context Management

A major part of the long-term project is independent project context.

ModelHelm should eventually maintain project-local AI state such as:

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

This allows ModelHelm to provide models with the context relevant to the current task without repeatedly sending an entire conversation or repository.

Context management should eventually support:

* Relevant-context selection
* Project memory
* Task summaries
* Architectural decisions
* Failed approaches
* Environment information
* Context compaction
* Stale-context detection

⸻

Agent Execution

ModelHelm is intended to support autonomous local task execution.

A delegated task may follow a loop such as:

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

ModelHelm should place configurable limits around this loop rather than allowing agents to execute indefinitely.

⸻

Escalation

Local execution should not be forced when it is failing.

ModelHelm should eventually recognize conditions such as:

* Repeated failed attempts
* Persistent test failures
* Ambiguous requirements
* Architecture changes
* Security-sensitive decisions
* Insufficient context
* Model capability limitations

and escalate the task back to the supervising agent or another model.

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

⸻

Safety

ModelHelm should be designed for controlled automation.

The system should support configurable policies such as:

ALLOW
ASK
DENY

Examples:

File modification       → ALLOW
Git commit               → ASK
Git push                 → ASK
Force push               → DENY
Production changes       → ASK
Destructive operations   → DENY
Credential changes       → DENY

ModelHelm should never silently perform destructive or high-impact operations.

⸻

Observability and Cost Optimization

ModelHelm should eventually measure the effectiveness of local delegation.

Potential metrics include:

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

This allows ModelHelm to answer an important question:

How much development work can this machine perform locally without sacrificing quality?

⸻

Project Roadmap

Phase 1 — MCP Server + Local Agent Loop

Current phase

The first milestone establishes the basic working architecture:

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

Phase 1 focuses on proving that ModelHelm can successfully:

* Expose an MCP interface
* Receive delegated tasks
* Communicate with LM Studio
* Run a local coding agent loop
* Allow the local model to inspect and modify a repository
* Execute tests
* Return results to the calling agent

See:

* docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md
* docs/ModelHelm-Spec.md

Phase 1 Requirements

* Python >= 3.11
* LM Studio running locally
* A tool-use-capable model loaded in LM Studio
* Developed and tested with qwen3-coder-30b-a3b
* llmfit installed and resolvable

⸻

Phase 1 Setup

Create a virtual environment and install ModelHelm:

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

Running Tests

Unit tests:

.\.venv\Scripts\pytest.exe tests/unit/ -v

Integration tests:

.\.venv\Scripts\pytest.exe tests/integration/ -v

Integration tests require LM Studio to be running with a compatible model loaded.

Running the MCP Server

.\.venv\Scripts\python.exe -m modelhelm.mcp.server

Configure ModelHelm through:

modelhelm.yaml

The repository configuration contains the default settings, including the safety policy.

⸻

Phase 2 — Intelligent Model Routing

Planned capabilities:

* Task classification
* Model capability detection
* Hardware-aware routing
* llmfit integration
* Model ranking
* Runtime selection
* Configurable routing policies
* Local vs cloud decision making

Example:

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

⸻

Phase 3 — Runtime Abstraction

Expand beyond LM Studio.

Planned runtimes:

LM Studio
Ollama
llama.cpp
OpenAI-compatible endpoints
Cloud providers

ModelHelm should be able to switch execution backends without changing the agent-facing interface.

⸻

Phase 4 — Persistent Context

Introduce the .ai/ project context system.

Capabilities:

* Project memory
* Context retrieval
* Task state
* Architecture records
* Decision records
* Failure history
* Automatic summaries
* Context compaction

⸻

Phase 5 — Multi-Agent Orchestration

Support multiple specialized agents:

                 Task
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      Coder     Tester    Investigator
        │          │          │
        └──────────┼──────────┘
                   ▼
                Reviewer

Different models can be assigned to different roles.

⸻

Phase 6 — Adaptive Routing

Use historical execution data to improve routing decisions.

ModelHelm could learn that:

Qwen:
Python implementation → excellent
Qwen:
Simple refactoring → excellent
Local reasoning model:
Log analysis → excellent
Claude:
Complex architecture → excellent

Routing becomes based not only on static model capabilities, but observed performance.

⸻

Phase 7 — Distributed Model Execution

Long-term, ModelHelm could treat multiple computers as AI workers.

                         ModelHelm
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
         Workstation       Server           NAS
         RTX 4090          GPU Node        CPU Node
             │               │               │
           Qwen            Large LLM        Small LLM

The orchestration layer could select the best available worker for each task.

⸻

Long-Term Vision

ModelHelm ultimately aims to become a general-purpose model orchestration layer for AI agents.

The final architecture could look like:

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

The important abstraction is:

Agents decide what needs to be accomplished. ModelHelm decides how and where the work should be executed.

⸻

Project Principles

ModelHelm should remain:

Agent-agnostic

Do not design the core around Claude Code.

Model-agnostic

Do not assume Qwen is the only useful local model.

Runtime-agnostic

LM Studio is the starting point, not the architectural boundary.

Local-first

Prefer local inference when it provides sufficient capability.

Policy-driven

Users control what the system is allowed to execute.

Observable

Delegation decisions and results should be inspectable.

Extensible

New models, runtimes, agents, and routing strategies should be addable without rewriting the core.

MCP-native

MCP should be the primary AI-agent integration mechanism.

⸻

Current Status

Phase 1 — MCP Server + Local Agent Loop

The initial proof of concept is working with:

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

The immediate objective is to stabilize this workflow before expanding into automatic routing, persistent context, additional runtimes, and multi-model orchestration.

For the detailed technical requirements and future architecture, see:

docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md
docs/ModelHelm-Spec.md