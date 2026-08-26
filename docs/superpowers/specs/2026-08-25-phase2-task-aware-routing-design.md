# ModelHelm Phase 2, Milestone 1 — Task-Aware Routing

Status: Approved for planning
Date: 2026-08-25
Parent spec: `docs/ModelHelm-Spec.md` Section 9 (Task Classification), Section 27 (Configuration)
Prior work: `docs/superpowers/specs/2026-08-25-phase1-mcp-agent-loop-design.md` (Phase 1, merged to `main`)

## 1. Purpose

Phase 1's `TaskRouter` is task-blind: it filters to tool-use-capable models and
picks the highest llmfit fit score, with no awareness of what kind of work is
being delegated. This means an architecture-level task and a "find this
function" task are routed identically — through the exact same local-agent
loop — even though the parent spec's Section 9 says architecture, security,
high-risk, and ambiguous tasks should never reach a local model at all.

This milestone closes that gap: `delegate_task` gains a classification step
that decides, before any model selection or local execution happens, whether
a task belongs to a class that defaults to local execution or to Claude.
Claude-default classes short-circuit immediately with a structured
escalation result — no wasted local inference, no router/registry calls.

This is the first of three planned Phase 2 milestones. It intentionally
excludes: richer multi-signal scoring within the "local" path (VRAM,
context-length fit, latency — still Phase 1's fit-score sort), and any
active local-vs-cloud negotiation beyond the static class-default table.
Both are separate, later milestones.

## 2. Scope

### In scope
- `TaskClassifier`: keyword/heuristic classification of a task description
  into one of 12 task classes (per spec Section 9), each with a
  local/claude default disposition.
- Config-driven class table: `modelhelm.yaml` gains a `classification:`
  section; users can override or extend the default keyword lists and
  dispositions.
- `delegate_task` short-circuits to `status: "escalation_recommended"` for
  claude-default classes, before calling `TaskRouter`, `ModelRegistry`, or
  `LocalAgent`.
- New MCP tool `classify_task(description: str) -> dict` — lets Claude
  preview a classification without delegating.
- `TaskResult` gains a `task_class` field.
- Fail-safe default: no keyword match → `ambiguous` → `claude`.

### Out of scope (later Phase 2 milestones)
- Multi-signal scoring depth (VRAM headroom, context-length fit, estimated
  tps/latency) within the "local" path — `TaskRouter.select_model()`'s
  fit-score-only sort is unchanged in this milestone.
- Active local-vs-cloud escalation logic beyond the static per-class
  default (e.g. retry-then-escalate, confidence-based escalation).
- LLM-based classification (asking a model to classify) — heuristic/keyword
  only, per explicit decision during brainstorming.
- Runtime abstraction (Ollama/llama.cpp) — LM Studio only, unchanged from
  Phase 1.

## 3. Task Class Table

Twelve classes, per parent spec Section 9, each with a default disposition
and a seed keyword/phrase list. All matching is case-insensitive substring
matching against the task description.

| Class | Default | Seed keywords/phrases |
|---|---|---|
| `exploration` | local | find, explore, search, inspect, locate, where is |
| `implementation` | local | implement, add, build, create |
| `refactoring` | local | refactor, rename, clean up, restructure |
| `testing` | local | test, write tests, add coverage, unit test |
| `debugging` | local | debug, fix bug, investigate error, why is, broken |
| `documentation` | local | document, readme, docstring, add comments |
| `context` | local | summarize, update context, memory |
| `architecture` | claude | architecture, design the system, redesign, system design |
| `security` | claude | security, auth, credential, vulnerability, encrypt |
| `high_risk` | claude | production, delete, drop table, force push, migrate database |
| `final_review` | claude | review this, final review, review the implementation |
| `ambiguous` | claude | (fallback only — never keyword-matched directly) |

**Matching algorithm:** scan the description for each class's keywords in
table order (local classes first, then claude classes, `ambiguous` last as
pure fallback); first class with a keyword hit wins. If a description
matches keywords from multiple classes, the first match in table order
wins — this means claude-default classes are checked after local-default
ones, so a description like "refactor the security module" matches
`refactoring` (local) before `security` (claude) is checked. This ordering
tradeoff is accepted for Milestone 1 simplicity; a future milestone may
introduce match-count or specificity-based tie-breaking if this proves
too coarse in practice.

If zero classes match: classify as `ambiguous`, disposition `claude`.

## 4. Configuration

`modelhelm.yaml` gains a `classification:` section:

```yaml
classification:
  classes:
    - name: exploration
      disposition: local
      keywords: [find, explore, search, inspect, locate, "where is"]
    - name: implementation
      disposition: local
      keywords: [implement, add, build, create]
    - name: refactoring
      disposition: local
      keywords: [refactor, rename, "clean up", restructure]
    - name: testing
      disposition: local
      keywords: [test, "write tests", "add coverage", "unit test"]
    - name: debugging
      disposition: local
      keywords: [debug, "fix bug", "investigate error", "why is", broken]
    - name: documentation
      disposition: local
      keywords: [document, readme, docstring, "add comments"]
    - name: context
      disposition: local
      keywords: [summarize, "update context", memory]
    - name: architecture
      disposition: claude
      keywords: [architecture, "design the system", redesign, "system design"]
    - name: security
      disposition: claude
      keywords: [security, auth, credential, vulnerability, encrypt]
    - name: high_risk
      disposition: claude
      keywords: [production, delete, "drop table", "force push", "migrate database"]
    - name: final_review
      disposition: claude
      keywords: ["review this", "final review", "review the implementation"]
```

`ambiguous` is not user-configurable — it is always the fallback when no
listed class matches, and always defaults to `claude`.

If `classification.classes` is present in the user's `modelhelm.yaml`, it
**replaces** the default table entirely (not merges) — matching the
existing pattern where `safety:` overrides are whole-section, not
per-key-merged, keeping config semantics consistent across the file. If
absent, the built-in default table (above) is used.

## 5. Architecture

```text
delegate_task(description, repository)
    │
    ▼
TaskClassifier.classify(description) -> ClassificationResult
    │
    ├─ disposition == "claude"
    │     │
    │     ▼
    │  create_task() [status="escalation_recommended" immediately]
    │     │
    │     ▼
    │  return {task_id, status: "escalation_recommended",
    │          task_class, summary} — NO router/registry/LocalAgent call
    │
    └─ disposition == "local"
          │
          ▼
       [existing Phase 1 flow: TaskRouter.select_model() -> LocalAgent.run()]
          │
          ▼
       TaskResult now includes task_class field
```

`TaskClassifier` lives in a new module, `src/modelhelm/classification/classifier.py`,
independent of `TaskRouter` — classification answers "should this be
delegated at all", routing answers "which model handles it". Keeping them
separate modules means the escalation short-circuit in `delegate_task` never
constructs a `ModelRegistry` or calls `llmfit`/LM Studio for a task that was
never going to run locally.

## 6. Interfaces

```python
# src/modelhelm/classification/classifier.py

class TaskClass(BaseModel):
    name: str
    disposition: Literal["local", "claude"]
    keywords: list[str]

class ClassificationResult(BaseModel):
    task_class: str          # e.g. "architecture", "ambiguous"
    disposition: Literal["local", "claude"]
    matched_keyword: str | None  # None when ambiguous fallback fired

class TaskClassifier:
    def __init__(self, classes: list[TaskClass]): ...
    def classify(self, description: str) -> ClassificationResult: ...

def load_classifier(settings: Settings) -> TaskClassifier:
    """Builds a TaskClassifier from settings.classification.classes,
    or the built-in default table if unset."""
```

`Settings` (Task 1's `config/settings.py`) gains:

```python
class ClassificationConfig(BaseModel):
    classes: list[TaskClass] = DEFAULT_TASK_CLASSES  # built-in table

class Settings(BaseModel):
    # ...existing fields...
    classification: ClassificationConfig = ClassificationConfig()
```

`TaskResult` (Task 7's `tasks/models.py`) gains:

```python
class TaskResult(BaseModel):
    # ...existing fields...
    task_class: str  # new, required — always populated, even for local runs
```

**Persistence and resume.** `LocalAgent` has no knowledge of classification —
it stays a pure tool-calling loop, per the Section 5 architecture split.
This means `task_class` must be computed once by `delegate_task` and
threaded through two paths that both eventually build a `TaskResult`:

1. **First delegation** (`delegate_task`): classify → pass `task_class` into
   `LocalAgent.run(..., task_class=classification.task_class)` → `run()`
   forwards it to every `_build_result()` call so it lands on every
   `TaskResult` this run can produce (`completed`, `pending_approval`,
   `escalation_recommended`).
2. **Resume** (`resume_task`): no re-classification happens — the task was
   already classified once, and re-running the classifier on resume could
   theoretically disagree with the original classification (the persisted
   conversation already reflects the original one). `DelegatedTask` (Task
   7's `tasks/models.py`) therefore gains a `task_class: str | None = None`
   field, persisted alongside `model` when `delegate_task` first calls
   `task_store.set_status`. `resume_task` reads `task.task_class` back and
   passes it into `agent.run(..., task_class=task.task_class)` the same way.

`TaskStore.create_task` (Task 7's `tasks/store.py`) gains a `task_class`
parameter (nullable, since it's set right after classification, not at
creation), and the `tasks` table gains a `task_class TEXT` column.
`set_status` gains an optional `task_class` parameter alongside its
existing optional `model` parameter, following the same pattern.

`LocalAgent.run()` and `_build_result()` (Task 10's `agents/local_agent.py`)
both gain a `task_class: str` parameter — required, not optional, so a
caller can never accidentally construct a `TaskResult` missing it.

MCP server (`mcp/server.py`) gains one new tool and modifies `delegate_task`:

```python
async def classify_task(description: str) -> dict:
    """Preview classification without delegating."""
    result = classifier.classify(description)
    return result.model_dump()

async def delegate_task(description: str, repository: str) -> dict:
    classification = classifier.classify(description)
    task = task_store.create_task(description=description, repository=repository)

    if classification.disposition == "claude":
        task_store.set_status(task.task_id, "escalation_recommended")
        result = TaskResult(
            task_id=task.task_id,
            status="escalation_recommended",
            model="none",
            runtime="none",
            duration_seconds=0.0,
            files_changed=0,
            tests_run=0, tests_passed=0, tests_failed=0,
            iterations=0,
            estimated_cloud_tokens_saved=0,
            review_recommended=True,
            task_class=classification.task_class,
            summary=(
                f"Classified as {classification.task_class} — "
                f"recommend Claude handle this directly."
            ),
        )
        task_store.save_result(task.task_id, result)
        return result.model_dump()

    # Existing Phase 1 flow, with task_class threaded through:
    try:
        model = await router.select_model(description)
    except NoSuitableModelError as exc:
        task_store.set_status(task.task_id, "failed")
        return {"task_id": task.task_id, "status": "failed", "summary": str(exc)}

    task_store.set_status(task.task_id, "running", model=model, task_class=classification.task_class)
    try:
        # ...construct policy_engine, agent_tools, git_inspector, agent (unchanged)...
        result, pending = await agent.run(
            task_id=task.task_id, description=description, model=model,
            task_class=classification.task_class,
        )
        task_store.set_status(task.task_id, result.status, model=model)
        task_store.save_result(task.task_id, result)
        if pending is not None:
            task_store.save_pending_approval(task.task_id, pending)
        return result.model_dump()
    except Exception as exc:
        # ...unchanged failure handling...
```

`model="none"` / `runtime="none"` on the escalation-recommended path: no
model or runtime was ever selected, since the router/registry are never
called. This is a deliberate, explicit sentinel rather than leaving the
fields empty or omitting them — `TaskResult`'s fields are otherwise all
required (per Phase 1's spec), so this milestone keeps that invariant
rather than making them optional.

`resume_task` (unchanged control flow, one new line): after fetching
`task = task_store.get_task(task_id)`, it now also has `task.task_class`
available (persisted at first delegation) and passes it through the same
way: `agent.run(..., task_class=task.task_class)`. No re-classification
happens on resume — the original classification is authoritative for the
life of the task, since the persisted conversation already reflects it.

## 7. Testing

- **Classifier unit tests** (`tests/unit/test_classifier.py`): table-driven
  — one test per class asserting a representative description classifies
  correctly; a test for the ambiguous fallback (nonsense description, no
  keyword hits); a test for the local-before-claude ordering tradeoff
  (e.g. "refactor the security module" → `refactoring`, not `security`,
  documenting the accepted coarseness from Section 3).
- **Config tests** (`tests/unit/test_settings.py` additions): a custom
  `modelhelm.yaml` with a `classification.classes` override produces a
  `TaskClassifier` using only the custom table (replace, not merge).
- **`delegate_task` short-circuit tests** (`tests/unit/test_mcp_server.py`
  additions): a claude-disposition description results in
  `status: "escalation_recommended"`, `task_class` set correctly, and
  (via a fake/spy registry) confirms `ModelRegistry.refresh()` and
  `TaskRouter.select_model()` are never called. A local-disposition
  description proceeds through the existing flow unchanged, with
  `task_class` now present on the result.
- **`classify_task` tool test**: calling it directly returns a
  classification without any task being created in the store (pure
  preview, no side effects).

## 8. Success Criteria

1. `classify_task("refactor the auth module")` returns
   `{"task_class": "refactoring", "disposition": "local", ...}` without
   touching the task store.
2. `delegate_task("design a new caching architecture", repo)` returns
   `status: "escalation_recommended"`, `task_class: "architecture"`,
   immediately — no LM Studio call, no llmfit call, verified via
   spy/mock assertion in tests.
3. `delegate_task("add unit tests for the parser", repo)` proceeds through
   the full Phase 1 local-agent flow unchanged, with `task_class: "testing"`
   on the final result.
4. A user-supplied `modelhelm.yaml` with a custom `classification.classes`
   list is honored — the built-in table is not silently merged in.
5. All existing Phase 1 tests (109) continue to pass unmodified except
   where they must be updated for the new required `task_class` field on
   `TaskResult` construction.

## 9. Open Items Carried Forward

- The local-before-claude keyword ordering tradeoff (Section 3) may prove
  too coarse once used against real task descriptions — revisit if the
  next Phase 2 milestone's scoring work surfaces misclassifications.
- LLM-based classification was explicitly rejected for this milestone
  (adds a local inference round-trip and a new failure mode) but remains
  a candidate for a future milestone if keyword matching proves too rigid.
- This milestone does not touch escalation *during* local execution
  (Phase 1's existing `escalation_recommended` on iteration exhaustion is
  unrelated and unchanged) — only the pre-execution classification gate.
