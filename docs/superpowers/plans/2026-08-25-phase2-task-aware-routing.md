# ModelHelm Phase 2, Milestone 1: Task-Aware Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a keyword-based `TaskClassifier` that runs before model selection in `delegate_task`, so tasks whose class defaults to Claude (architecture, security, high-risk, final-review, ambiguous) short-circuit to `escalation_recommended` without ever calling the router, registry, or local agent loop.

**Architecture:** A new `src/modelhelm/classification/classifier.py` module, independent of `TaskRouter`, does config-driven keyword matching to produce a `ClassificationResult`. `delegate_task` classifies first; on a `claude` disposition it persists an escalation `TaskResult` directly. On `local` disposition it proceeds through the unchanged Phase 1 flow, with `task_class` threaded through `LocalAgent.run()` (as a new required parameter) so every `TaskResult` that run can produce carries it, and persisted on `DelegatedTask` so a later `resume_task` can recover it without re-classifying.

**Tech Stack:** Same as Phase 1 — Python ≥3.11, `mcp` 2.x, `pydantic`, `httpx`, stdlib `sqlite3`, `pytest` + `pytest-asyncio`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-25-phase2-task-aware-routing-design.md`

## Global Constraints

- `task_class` is a required field on `TaskResult` and a required parameter on `LocalAgent.run()`/`_build_result()` — never optional, never defaulted silently. Every existing call site that constructs a `TaskResult` or calls `agent.run()` must be updated, not worked around.
- Classification is pure keyword/heuristic matching against a case-insensitive substring scan — no LLM call, no network I/O, no async needed for `TaskClassifier.classify()`.
- Matching order is: local-default classes first (in the Section 3 table order), then claude-default classes, with `ambiguous` as the pure fallback when nothing matches. This ordering is a known, tested tradeoff (see spec Section 3) — do not "fix" it by reordering during implementation without flagging it as a scope change.
- `modelhelm.yaml`'s `classification.classes`, when present, **replaces** the built-in default table entirely — it is not merged key-by-key, matching the existing whole-section-override pattern used by `safety:`.
- `ambiguous` is never user-configurable and always defaults to `claude` — it must not appear in `modelhelm.yaml`'s class list; it is the classifier's internal fallback.
- The `tasks` table gains a `task_class TEXT` column (nullable — a task starts without one before classification completes, though in practice classification happens synchronously before `create_task` in this milestone's flow).
- No containerization, no new runtime dependencies — this milestone only touches routing/classification logic and the existing SQLite schema.

---

## File Structure

```text
ModelHelm/
├── src/
│   └── modelhelm/
│       ├── classification/
│       │   ├── __init__.py          # Task 1
│       │   └── classifier.py        # Task 1 (TaskClass, ClassificationResult, TaskClassifier, load_classifier, DEFAULT_TASK_CLASSES)
│       ├── config/
│       │   └── settings.py          # Task 2 (modified: ClassificationConfig, Settings.classification)
│       ├── tasks/
│       │   ├── models.py            # Task 3 (modified: DelegatedTask.task_class, TaskResult.task_class)
│       │   └── store.py             # Task 3 (modified: create_task/set_status gain task_class, schema migration)
│       ├── agents/
│       │   └── local_agent.py       # Task 4 (modified: run()/_build_result() gain task_class param)
│       └── mcp/
│           └── server.py            # Task 5 (modified: delegate_task classifies + short-circuits; resume_task threads task_class; new classify_task tool)
├── tests/
│   └── unit/
│       ├── test_classifier.py       # Task 1
│       ├── test_settings.py         # Task 2 (modified)
│       ├── test_task_store.py       # Task 3 (modified)
│       ├── test_local_agent.py      # Task 4 (modified: all 14 agent.run() call sites)
│       └── test_mcp_server.py       # Task 5 (modified: all delegate_task/resume_task call sites + PendingApproval fixtures)
├── modelhelm.yaml                   # Task 2 (add classification: section)
```

---

### Task 1: TaskClassifier Module

**Files:**
- Create: `src/modelhelm/classification/__init__.py`
- Create: `src/modelhelm/classification/classifier.py`
- Test: `tests/unit/test_classifier.py`

**Interfaces:**
- Produces:
  - `class TaskClass(BaseModel)` — fields `name: str`, `disposition: Literal["local", "claude"]`, `keywords: list[str]`
  - `class ClassificationResult(BaseModel)` — fields `task_class: str`, `disposition: Literal["local", "claude"]`, `matched_keyword: str | None`
  - `DEFAULT_TASK_CLASSES: list[TaskClass]` — the 12-class built-in table from spec Section 3 (11 configurable + `ambiguous` fallback, though `ambiguous` is not stored as a `TaskClass` entry — see implementation below)
  - `class TaskClassifier` — constructor `__init__(self, classes: list[TaskClass])`
    - `def classify(self, description: str) -> ClassificationResult` — case-insensitive substring scan in list order; first class with a keyword hit wins; no match → `ClassificationResult(task_class="ambiguous", disposition="claude", matched_keyword=None)`

This task has no dependency on Settings yet — `load_classifier(settings)` is deferred to Task 2, once `Settings.classification` exists.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_classifier.py
from modelhelm.classification.classifier import (
    TaskClass, ClassificationResult, TaskClassifier, DEFAULT_TASK_CLASSES,
)

def test_default_table_has_twelve_entries_including_ambiguous_fallback():
    # 11 keyword-matchable classes; ambiguous is the classifier's internal
    # fallback and is not itself an entry in DEFAULT_TASK_CLASSES.
    assert len(DEFAULT_TASK_CLASSES) == 11
    names = {c.name for c in DEFAULT_TASK_CLASSES}
    assert "ambiguous" not in names
    assert names == {
        "exploration", "implementation", "refactoring", "testing",
        "debugging", "documentation", "context", "architecture",
        "security", "high_risk", "final_review",
    }

def test_classifies_exploration_as_local():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("find the function that handles login")
    assert result == ClassificationResult(
        task_class="exploration", disposition="local", matched_keyword="find"
    )

def test_classifies_architecture_as_claude():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("design the system architecture for caching")
    assert result.task_class == "architecture"
    assert result.disposition == "claude"

def test_classifies_security_as_claude():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("review the auth credential handling for vulnerabilities")
    assert result.task_class == "security"
    assert result.disposition == "claude"

def test_classifies_high_risk_as_claude():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("force push to fix the production branch")
    assert result.task_class == "high_risk"
    assert result.disposition == "claude"

def test_unmatched_description_falls_back_to_ambiguous_claude():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("xyzzy plugh frobnicate")
    assert result == ClassificationResult(
        task_class="ambiguous", disposition="claude", matched_keyword=None
    )

def test_local_before_claude_ordering_tradeoff_is_documented_behavior():
    # A description matching both a local-default and claude-default class
    # resolves to whichever class comes first in table order (local classes
    # precede claude classes). This is an accepted Milestone-1 tradeoff, not
    # a bug -- this test documents and locks in the current behavior.
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("refactor the security module")
    assert result.task_class == "refactoring"
    assert result.disposition == "local"

def test_classify_is_case_insensitive():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("FIND the login handler")
    assert result.task_class == "exploration"

def test_custom_class_list_is_used_verbatim():
    custom = [
        TaskClass(name="only_class", disposition="claude", keywords=["banana"]),
    ]
    classifier = TaskClassifier(custom)
    result = classifier.classify("find the login handler")  # would match exploration in defaults
    assert result.task_class == "ambiguous"  # not in custom list, no match
    result2 = classifier.classify("peel the banana")
    assert result2.task_class == "only_class"
    assert result2.disposition == "claude"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/unit/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'modelhelm.classification.classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/modelhelm/classification/__init__.py
```//empty

```python
# src/modelhelm/classification/classifier.py
from typing import Literal
from pydantic import BaseModel

class TaskClass(BaseModel):
    name: str
    disposition: Literal["local", "claude"]
    keywords: list[str]

class ClassificationResult(BaseModel):
    task_class: str
    disposition: Literal["local", "claude"]
    matched_keyword: str | None = None

# Table order matters: local-default classes are listed before claude-default
# classes, so a description matching keywords from both resolves to the local
# class. See spec Section 3 for the accepted tradeoff.
DEFAULT_TASK_CLASSES: list[TaskClass] = [
    TaskClass(name="exploration", disposition="local", keywords=[
        "find", "explore", "search", "inspect", "locate", "where is",
    ]),
    TaskClass(name="implementation", disposition="local", keywords=[
        "implement", "add", "build", "create",
    ]),
    TaskClass(name="refactoring", disposition="local", keywords=[
        "refactor", "rename", "clean up", "restructure",
    ]),
    TaskClass(name="testing", disposition="local", keywords=[
        "test", "write tests", "add coverage", "unit test",
    ]),
    TaskClass(name="debugging", disposition="local", keywords=[
        "debug", "fix bug", "investigate error", "why is", "broken",
    ]),
    TaskClass(name="documentation", disposition="local", keywords=[
        "document", "readme", "docstring", "add comments",
    ]),
    TaskClass(name="context", disposition="local", keywords=[
        "summarize", "update context", "memory",
    ]),
    TaskClass(name="architecture", disposition="claude", keywords=[
        "architecture", "design the system", "redesign", "system design",
    ]),
    TaskClass(name="security", disposition="claude", keywords=[
        "security", "auth", "credential", "vulnerability", "encrypt",
    ]),
    TaskClass(name="high_risk", disposition="claude", keywords=[
        "production", "delete", "drop table", "force push", "migrate database",
    ]),
    TaskClass(name="final_review", disposition="claude", keywords=[
        "review this", "final review", "review the implementation",
    ]),
]

class TaskClassifier:
    def __init__(self, classes: list[TaskClass]):
        self.classes = classes

    def classify(self, description: str) -> ClassificationResult:
        lowered = description.lower()
        for task_class in self.classes:
            for keyword in task_class.keywords:
                if keyword.lower() in lowered:
                    return ClassificationResult(
                        task_class=task_class.name,
                        disposition=task_class.disposition,
                        matched_keyword=keyword,
                    )
        return ClassificationResult(
            task_class="ambiguous", disposition="claude", matched_keyword=None
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/unit/test_classifier.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/classification/ tests/unit/test_classifier.py
git commit -m "feat: add keyword-based TaskClassifier"
```

---

### Task 2: Settings and modelhelm.yaml Integration

**Files:**
- Modify: `src/modelhelm/config/settings.py`
- Modify: `modelhelm.yaml`
- Test: `tests/unit/test_settings.py`

**Interfaces:**
- Consumes: `TaskClass`, `DEFAULT_TASK_CLASSES` (Task 1)
- Produces:
  - `class ClassificationConfig(BaseModel)` — field `classes: list[TaskClass] = DEFAULT_TASK_CLASSES`
  - `Settings.classification: ClassificationConfig = ClassificationConfig()` — new field
  - `def load_classifier(settings: Settings) -> TaskClassifier` — new function in `classification/classifier.py` (added in this task, not Task 1, since it depends on `Settings`); returns `TaskClassifier(settings.classification.classes)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_settings.py -- ADD to existing file, do not replace
import textwrap
from modelhelm.config.settings import load_settings, Settings
from modelhelm.classification.classifier import DEFAULT_TASK_CLASSES, load_classifier

def test_load_settings_defaults_to_builtin_classification_table():
    settings = Settings()
    assert settings.classification.classes == DEFAULT_TASK_CLASSES

def test_load_settings_with_custom_classification_replaces_default_table(tmp_path):
    config_path = tmp_path / "modelhelm.yaml"
    config_path.write_text(textwrap.dedent("""
        classification:
          classes:
            - name: only_class
              disposition: claude
              keywords: [banana]
    """))
    settings = load_settings(str(config_path))
    assert len(settings.classification.classes) == 1
    assert settings.classification.classes[0].name == "only_class"

def test_load_settings_without_classification_section_uses_default_table(tmp_path):
    config_path = tmp_path / "modelhelm.yaml"
    config_path.write_text(textwrap.dedent("""
        modelhelm:
          default_runtime: lm-studio
    """))
    settings = load_settings(str(config_path))
    assert settings.classification.classes == DEFAULT_TASK_CLASSES

def test_load_classifier_builds_from_settings():
    settings = Settings()
    classifier = load_classifier(settings)
    result = classifier.classify("find the login handler")
    assert result.task_class == "exploration"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/unit/test_settings.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'classification'` (and `ImportError` for `load_classifier`)

- [ ] **Step 3: Write minimal implementation**

Modify `src/modelhelm/config/settings.py`:

```python
# Add near the top, after existing imports:
from modelhelm.classification.classifier import TaskClass, DEFAULT_TASK_CLASSES

# Add new config class, alongside AgentConfig/LMStudioConfig:
class ClassificationConfig(BaseModel):
    classes: list[TaskClass] = DEFAULT_TASK_CLASSES

# Modify Settings to add the field:
class Settings(BaseModel):
    default_runtime: str = "lm-studio"
    lm_studio: LMStudioConfig = LMStudioConfig()
    llmfit_binary_path: str | None = None
    prefer_local: bool = True
    safety: SafetyPolicy = SafetyPolicy()
    agent: AgentConfig = AgentConfig()
    classification: ClassificationConfig = ClassificationConfig()

# Modify load_settings to parse the new section:
def load_settings(path: str | None = None) -> Settings:
    config_path = Path(path) if path else Path.cwd() / "modelhelm.yaml"
    if not config_path.exists():
        return Settings()

    raw = yaml.safe_load(config_path.read_text()) or {}
    classification_raw = raw.get("classification", {}).get("classes")
    classification = (
        ClassificationConfig(classes=[TaskClass(**c) for c in classification_raw])
        if classification_raw is not None
        else ClassificationConfig()
    )
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
        classification=classification,
    )
```

Add `load_classifier` to `src/modelhelm/classification/classifier.py` (append to the existing file from Task 1):

```python
# Append to src/modelhelm/classification/classifier.py:

def load_classifier(settings) -> TaskClassifier:
    """Builds a TaskClassifier from settings.classification.classes."""
    return TaskClassifier(settings.classification.classes)
```

Note: `settings` is left untyped here (no `Settings` import) to avoid a circular import, since `settings.py` already imports from `classification/classifier.py`. This mirrors how Task 1's module has no dependency on `config/settings.py`.

Add to `modelhelm.yaml` (append a new top-level section; do not remove existing sections):

```yaml
# Task classification table used by delegate_task's routing gate. Classes
# listed here REPLACE the built-in default table entirely (not merged).
# Omit this section to use the built-in 11-class table.
# The "ambiguous" fallback (disposition: claude) is not configurable and
# is not listed here -- it fires automatically when no class matches.
#
# classification:
#   classes:
#     - name: exploration
#       disposition: local
#       keywords: [find, explore, search, inspect, locate, "where is"]
#     - name: implementation
#       disposition: local
#       keywords: [implement, add, build, create]
#     - name: refactoring
#       disposition: local
#       keywords: [refactor, rename, "clean up", restructure]
#     - name: testing
#       disposition: local
#       keywords: [test, "write tests", "add coverage", "unit test"]
#     - name: debugging
#       disposition: local
#       keywords: [debug, "fix bug", "investigate error", "why is", broken]
#     - name: documentation
#       disposition: local
#       keywords: [document, readme, docstring, "add comments"]
#     - name: context
#       disposition: local
#       keywords: [summarize, "update context", memory]
#     - name: architecture
#       disposition: claude
#       keywords: [architecture, "design the system", redesign, "system design"]
#     - name: security
#       disposition: claude
#       keywords: [security, auth, credential, vulnerability, encrypt]
#     - name: high_risk
#       disposition: claude
#       keywords: [production, delete, "drop table", "force push", "migrate database"]
#     - name: final_review
#       disposition: claude
#       keywords: ["review this", "final review", "review the implementation"]
```

(Commented out so the shipped default continues to use the in-code
`DEFAULT_TASK_CLASSES` table — this matches how `modelhelm.yaml` documents
optional overrides elsewhere. If the repo's existing `modelhelm.yaml`
convention is to ship all sections active rather than commented, uncomment
this block instead — check the current file's style before finalizing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/unit/test_settings.py -v`
Expected: PASS (all settings tests, existing + 4 new)

- [ ] **Step 5: Run the full existing suite to check for import-cycle or regression issues**

Run: `.venv\Scripts\pytest.exe tests/unit/ -v`
Expected: PASS for everything except tests that will be touched in Tasks 3-5 (those may still be green at this point since `task_class` isn't required anywhere yet — this step is a sanity check, not a gate).

- [ ] **Step 6: Commit**

```bash
git add src/modelhelm/config/settings.py src/modelhelm/classification/classifier.py modelhelm.yaml tests/unit/test_settings.py
git commit -m "feat: wire TaskClassifier into Settings and modelhelm.yaml"
```

---

### Task 3: Persist task_class on DelegatedTask and TaskResult

**Files:**
- Modify: `src/modelhelm/tasks/models.py`
- Modify: `src/modelhelm/tasks/store.py`
- Test: `tests/unit/test_task_store.py`

**Interfaces:**
- Produces:
  - `DelegatedTask.task_class: str | None = None` — new field
  - `TaskResult.task_class: str` — new **required** field (no default)
  - `TaskStore.create_task(self, description: str, repository: str) -> DelegatedTask` — signature unchanged; `task_class` is not set at creation time (it's `None` until classification completes and `set_status` is called)
  - `TaskStore.set_status(self, task_id: str, status: str, model: str | None = None, task_class: str | None = None) -> None` — gains a new optional `task_class` parameter; when provided, updates the column; when omitted, leaves the existing value unchanged (same pattern as the existing `model` parameter)

**Migration note:** `_init_schema` uses `CREATE TABLE IF NOT EXISTS`, which does not add columns to an already-existing table. Since this is Phase 2 pre-release code with no production deployments, the schema is simply rewritten to include `task_class` from the start — no `ALTER TABLE` migration path is needed. Any existing local `.db` files a developer has from Phase 1 testing should be deleted (they are gitignored scratch files, not tracked data).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_task_store.py -- ADD to existing file, do not replace

def test_create_task_starts_with_no_task_class(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")
    assert task.task_class is None

def test_set_status_with_task_class_persists_it(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")

    store.set_status(task.task_id, "running", model="qwen3-coder-30b-a3b", task_class="implementation")
    fetched = store.get_task(task.task_id)

    assert fetched.task_class == "implementation"

def test_set_status_without_task_class_leaves_existing_value(tmp_path):
    store = TaskStore(str(tmp_path / "tasks.db"))
    task = store.create_task(description="add auth", repository="/repo")
    store.set_status(task.task_id, "running", model="x", task_class="implementation")

    store.set_status(task.task_id, "completed")  # no task_class passed
    fetched = store.get_task(task.task_id)

    assert fetched.task_class == "implementation"
```

Also update the existing `test_save_and_get_result` test (it constructs a
`TaskResult` directly and will fail Pydantic validation once `task_class`
becomes required):

```python
# In the existing test_save_and_get_result test, add task_class to the
# TaskResult(...) constructor call:
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
        task_class="implementation",  # ADD THIS LINE
        summary="Implemented auth.",
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/unit/test_task_store.py -v`
Expected: FAIL — new tests fail with `AttributeError` (no `task_class` column/attribute); `test_save_and_get_result` fails with a Pydantic `ValidationError` once `TaskResult.task_class` is made required in Step 3 (temporarily green until then, since the model doesn't have the field yet — this is expected TDD sequencing, not a bug)

- [ ] **Step 3: Write minimal implementation**

Modify `src/modelhelm/tasks/models.py`:

```python
class DelegatedTask(BaseModel):
    task_id: str
    description: str
    repository: str
    status: TaskStatus
    model: str | None = None
    task_class: str | None = None  # ADD THIS FIELD
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
    task_class: str  # ADD THIS FIELD -- required, no default
    summary: str
```

Modify `src/modelhelm/tasks/store.py`:

```python
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
                    task_class TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            # ...task_results and pending_approvals tables unchanged...

    def create_task(self, description: str, repository: str) -> DelegatedTask:
        task = DelegatedTask(
            task_id=str(uuid.uuid4()),
            description=description,
            repository=repository,
            status="pending",
            model=None,
            task_class=None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, description, repository, status, model, task_class, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task.task_id, task.description, task.repository, task.status,
                 task.model, task.task_class, task.created_at),
            )
        return task

    def get_task(self, task_id: str) -> DelegatedTask | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT task_id, description, repository, status, model, task_class, created_at "
                "FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return DelegatedTask(
            task_id=row[0],
            description=row[1],
            repository=row[2],
            status=row[3],
            model=row[4],
            task_class=row[5],
            created_at=row[6],
        )

    def set_status(self, task_id: str, status: str, model: str | None = None, task_class: str | None = None) -> None:
        with self._connect() as conn:
            if model is not None and task_class is not None:
                conn.execute(
                    "UPDATE tasks SET status = ?, model = ?, task_class = ? WHERE task_id = ?",
                    (status, model, task_class, task_id),
                )
            elif model is not None:
                conn.execute(
                    "UPDATE tasks SET status = ?, model = ? WHERE task_id = ?",
                    (status, model, task_id),
                )
            elif task_class is not None:
                conn.execute(
                    "UPDATE tasks SET status = ?, task_class = ? WHERE task_id = ?",
                    (status, task_class, task_id),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE task_id = ?",
                    (status, task_id),
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/unit/test_task_store.py -v`
Expected: PASS (all existing + 3 new tests)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/tasks/models.py src/modelhelm/tasks/store.py tests/unit/test_task_store.py
git commit -m "feat: persist task_class on DelegatedTask and require it on TaskResult"
```

---

### Task 4: Thread task_class Through LocalAgent

**Files:**
- Modify: `src/modelhelm/agents/local_agent.py`
- Modify: `tests/unit/test_local_agent.py`

**Interfaces:**
- Consumes: `TaskResult.task_class` (Task 3, now required)
- Produces:
  - `LocalAgent.run(self, task_id: str, description: str, model: str, task_class: str, resume_messages: list[dict] | None = None) -> tuple[TaskResult, PendingApproval | None]` — `task_class` inserted as a required positional-or-keyword parameter after `model` and before the optional `resume_messages`, so all three required params stay grouped before the one optional param.
  - `LocalAgent._build_result(self, task_id, status, model, task_class, start_time, iterations, summary) -> TaskResult` — gains `task_class`, threaded straight into the `TaskResult(...)` constructor call.

This is the highest-blast-radius task in this plan: every one of the 14
existing `agent.run(...)` call sites in `tests/unit/test_local_agent.py`
must add `task_class="..."` (any placeholder string is fine for tests that
don't assert on it, e.g. `task_class="testing"`).

- [ ] **Step 1: Update all 14 call sites in the test file**

For each `await agent.run(task_id=..., description=..., model=...)` call in
`tests/unit/test_local_agent.py`, add `task_class="testing"` (or a more
specific class name where it improves test readability) as a keyword
argument. Example transformation:

```python
# Before:
result, pending = await agent.run(task_id="t1", description="no-op task", model="qwen3-coder-30b-a3b")

# After:
result, pending = await agent.run(
    task_id="t1", description="no-op task", model="qwen3-coder-30b-a3b", task_class="testing",
)
```

Apply this to all 14 sites. Additionally, add one new test asserting
`task_class` lands on the result:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/unit/test_local_agent.py -v`
Expected: FAIL — `TypeError: LocalAgent.run() got an unexpected keyword argument 'task_class'` on every updated call site, and the new test fails the same way.

- [ ] **Step 3: Write minimal implementation**

Modify `src/modelhelm/agents/local_agent.py`:

```python
    async def run(
        self,
        task_id: str,
        description: str,
        model: str,
        task_class: str,
        resume_messages: list[dict] | None = None,
    ) -> tuple[TaskResult, PendingApproval | None]:
        """Drive the model/tool loop until completion, approval pause, or
        iteration exhaustion.

        ``task_class`` is opaque to this loop -- it is produced by
        classification upstream (delegate_task) and carried through purely so
        every TaskResult this run can produce reports which class the task
        was classified as. LocalAgent makes no routing decisions based on it.

        Returns ``(result, pending)``. ``pending`` is non-None only for
        ``status == "pending_approval"``, in which case it carries the
        conversation state needed to resume this same conversation later.
        """
        start_time = time.monotonic()
        # ...unchanged baseline-snapshot logic...

        # ...unchanged resume_messages / messages setup...

        iterations = 0
        for iterations in range(1, self.agent_config.max_iterations + 1):
            # ...unchanged chat_completion call...

            if not tool_calls:
                result = self._build_result(
                    task_id, "completed", model, task_class, start_time, iterations,
                    message.get("content") or "Task completed.",
                )
                return result, None

            # ...unchanged tool-call loop body...

            if approval_pause is not None:
                exc, pending_call_id = approval_pause
                result = self._build_result(
                    task_id, "pending_approval", model, task_class, start_time, iterations,
                    f"Paused: {exc.operation} requires approval ({exc.detail}).",
                )
                pending = PendingApproval(
                    operation=exc.operation,
                    detail=exc.detail,
                    tool_call_id=pending_call_id,
                    messages=messages,
                )
                return result, pending

        result = self._build_result(
            task_id, "escalation_recommended", model, task_class, start_time, iterations,
            f"Reached max_iterations ({self.agent_config.max_iterations}) without completion.",
        )
        return result, None

    def _build_result(self, task_id, status, model, task_class, start_time, iterations, summary) -> TaskResult:
        duration = time.monotonic() - start_time
        # ...unchanged files_changed computation...
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
            task_class=task_class,
            summary=summary,
        )
```

Only the signatures and the `task_class` threading shown above change —
every other line of `run()`/`_build_result()`'s existing logic (baseline
snapshot, resume handling, tool-call loop, escalation) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/unit/test_local_agent.py -v`
Expected: PASS (all existing tests + 1 new = 12 tests, per Phase 1's final count)

- [ ] **Step 5: Commit**

```bash
git add src/modelhelm/agents/local_agent.py tests/unit/test_local_agent.py
git commit -m "feat: thread task_class through LocalAgent.run() and _build_result()"
```

---

### Task 5: Wire Classification into the MCP Server

**Files:**
- Modify: `src/modelhelm/mcp/server.py`
- Modify: `tests/unit/test_mcp_server.py`

**Interfaces:**
- Consumes: `load_classifier`, `ClassificationResult` (Task 2), `TaskStore.set_status(..., task_class=...)` (Task 3), `LocalAgent.run(..., task_class=...)` (Task 4)
- Produces:
  - New MCP tool `classify_task(description: str) -> dict` — returns `classifier.classify(description).model_dump()`; no task store interaction, pure preview.
  - `delegate_task(description: str, repository: str) -> dict` — modified: classifies first; on `disposition == "claude"`, persists and returns an `escalation_recommended` `TaskResult` without calling `router.select_model()`, `ModelRegistry`, or `LocalAgent`; on `disposition == "local"`, proceeds through the existing flow with `task_class` threaded into `set_status` and `agent.run()`.
  - `resume_task(task_id: str, approved: bool) -> dict` — modified: after fetching `task = task_store.get_task(task_id)`, passes `task_class=task.task_class` into `agent.run(...)`. No re-classification.

**Every existing test in `tests/unit/test_mcp_server.py` that calls
`delegate_task` with a description that happens to match a claude-default
keyword must be checked** — e.g. a description containing "test" is fine
(classifies `testing` → local), but watch for accidental matches against
words like "review", "delete", "production", "security" in existing
fixture descriptions. Reading the current file (already done during
planning): existing descriptions are `"no-op"`, `"x"`, `"commit notes"`,
`"write notes"` — none of these match a claude-default keyword, so they
will classify as `ambiguous` (no keyword match at all) → **claude** →
escalation. This changes the behavior of several existing tests and each
must be updated (see Step 1).

- [ ] **Step 1: Update existing tests and write new ones**

The existing tests `test_delegate_task_and_get_status`,
`test_delegate_task_returns_failed_dict_on_lmstudio_timeout`, and
`test_delegate_task_returns_failed_dict_on_bad_repository` all use
descriptions that will now classify as `ambiguous` → `claude` →
`escalation_recommended`, which breaks their existing assertions (they
expect `"completed"` or `"failed"`, and expect the router/agent to actually
run). Fix by changing their description strings to unambiguously match a
**local**-default class, preserving each test's original intent:

```python
# In test_delegate_task_and_get_status, change:
#   result = await server.tools["delegate_task"](description="no-op", repository=str(tmp_path))
# to:
    result = await server.tools["delegate_task"](description="implement a no-op change", repository=str(tmp_path))
    # "implement" matches the implementation class -> local, same as before this milestone.
    assert result["status"] == "completed"
    assert result["task_class"] == "implementation"

    task_status = await server.tools["get_task_status"](task_id=result["task_id"])
    assert task_status["status"] == "completed"
    assert task_status["task_class"] == "implementation"
```

```python
# In test_delegate_task_returns_failed_dict_on_lmstudio_timeout, change:
#   result = await server.tools["delegate_task"](description="x", repository=str(repo))
# to:
    result = await server.tools["delegate_task"](description="implement x", repository=str(repo))
```

```python
# In test_delegate_task_returns_failed_dict_on_bad_repository, change:
#   result = await server.tools["delegate_task"](description="x", repository="/definitely/not/a/repo")
# to:
    result = await server.tools["delegate_task"](
        description="implement x", repository="/definitely/not/a/repo"
    )
```

Add new tests for the classification gate itself:

```python
@pytest.mark.asyncio
async def test_classify_task_previews_without_side_effects(server):
    result = await server.tools["classify_task"](description="design the system architecture")
    assert result["task_class"] == "architecture"
    assert result["disposition"] == "claude"
    # Pure preview: nothing was persisted.
    all_tasks_db_path = server.task_store.db_path
    import sqlite3
    conn = sqlite3.connect(all_tasks_db_path)
    count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert count == 0

@pytest.mark.asyncio
async def test_delegate_task_short_circuits_for_claude_default_class(tmp_path):
    # Spy on FakeLlmfitClient.recommend(): TaskRouter.select_model() always
    # calls registry.refresh(), which always calls llmfit_client.recommend().
    # If recommend() is never invoked, the router (and therefore the
    # registry and LM Studio) was never touched -- proving the short-circuit
    # happens before any of Phase 1's model-selection machinery runs.
    class SpyLlmfitClient(FakeLlmfitClient):
        def __init__(self):
            self.recommend_calls = 0

        def recommend(self):
            self.recommend_calls += 1
            return super().recommend()

    llmfit_spy = SpyLlmfitClient()
    store = TaskStore(str(tmp_path / "tasks.db"))
    server = create_server(
        settings=Settings(agent=AgentConfig(max_iterations=2, test_before_completion=False)),
        task_store=store,
        lmstudio_client=FakeLMStudioClient(),
        llmfit_client=llmfit_spy,
    )

    result = await server.tools["delegate_task"](
        description="design the caching architecture", repository=str(tmp_path)
    )

    assert result["status"] == "escalation_recommended"
    assert result["task_class"] == "architecture"
    assert result["model"] == "none"
    assert result["runtime"] == "none"
    assert "recommend Claude" in result["summary"]
    assert llmfit_spy.recommend_calls == 0

@pytest.mark.asyncio
async def test_delegate_task_persists_escalation_result(server, tmp_path):
    result = await server.tools["delegate_task"](
        description="review this security vulnerability", repository=str(tmp_path)
    )
    assert result["status"] == "escalation_recommended"

    task_status = await server.tools["get_task_status"](task_id=result["task_id"])
    assert task_status["status"] == "escalation_recommended"
    # "security" (via the "security" keyword) beats "final_review" (via
    # "review this") because security is earlier in DEFAULT_TASK_CLASSES
    # table order. Verified by simulating the match during plan authoring.
    assert task_status["task_class"] == "security"

@pytest.mark.asyncio
async def test_ambiguous_description_escalates(server, tmp_path):
    result = await server.tools["delegate_task"](
        description="do the thing with the stuff", repository=str(tmp_path)
    )
    assert result["status"] == "escalation_recommended"
    assert result["task_class"] == "ambiguous"
```

Update `test_resume_task_approved_executes_pending_commit_and_continues` and
`test_resume_task_approved_executes_pending_file_write` and
`test_second_resume_does_not_replay_the_approved_operation` and
`test_resume_task_returns_failed_dict_when_execution_raises`: each calls
`server.task_store.set_status(task.task_id, "pending_approval", model="qwen3-coder-30b-a3b")`
directly (bypassing `delegate_task`) to set up fixture state. Since
`task.task_class` will be `None` in these fixtures (never set), and
`resume_task` now passes `task_class=task.task_class` into `agent.run()`
which requires a `str` not `None`, either:
(a) update each fixture's `set_status` call to also pass `task_class="testing"`, or
(b) confirm during implementation whether `resume_task` needs a fallback for
`task.task_class is None` (a task created before this milestone's DB
migration, or via direct store manipulation as these fixtures do).

Prefer (a) — update every fixture's `set_status` call to include
`task_class="testing"` — since a `None` task_class reaching `LocalAgent.run()`
should be a real `TypeError` in production (it would mean `delegate_task`
failed to persist classification, which is itself a bug worth surfacing
loudly rather than silently defaulting).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/unit/test_mcp_server.py -v`
Expected: FAIL — `classify_task` tests fail with `KeyError: 'classify_task'`
(tool doesn't exist yet); `delegate_task` short-circuit tests fail because
`delegate_task` doesn't classify yet; resume tests fail with `TypeError`
once `task_class` becomes a required `str` param on `agent.run()`.

- [ ] **Step 3: Write minimal implementation**

Modify `src/modelhelm/mcp/server.py`:

```python
# Add import:
from modelhelm.classification.classifier import load_classifier

# Inside create_server, after `router = TaskRouter(registry)`:
    classifier = load_classifier(settings)

# New tool function, alongside the others:
    async def classify_task(description: str) -> dict:
        return classifier.classify(description).model_dump()

# Modify delegate_task:
    async def delegate_task(description: str, repository: str) -> dict:
        classification = classifier.classify(description)
        task = task_store.create_task(description=description, repository=repository)

        if classification.disposition == "claude":
            task_store.set_status(
                task.task_id, "escalation_recommended", task_class=classification.task_class
            )
            result = TaskResult(
                task_id=task.task_id,
                status="escalation_recommended",
                model="none",
                runtime="none",
                duration_seconds=0.0,
                files_changed=0,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
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

        try:
            model = await router.select_model(description)
        except NoSuitableModelError as exc:
            task_store.set_status(task.task_id, "failed", task_class=classification.task_class)
            return {"task_id": task.task_id, "status": "failed", "summary": str(exc)}

        task_store.set_status(
            task.task_id, "running", model=model, task_class=classification.task_class
        )
        try:
            policy_engine = PolicyEngine(settings.safety)
            agent_tools = AgentTools(repository, policy_engine)
            git_inspector = GitInspector(repository)
            agent = LocalAgent(
                lmstudio_client=lmstudio_client,
                tools=agent_tools,
                git_inspector=git_inspector,
                agent_config=settings.agent,
            )
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
            task_store.set_status(task.task_id, "failed", model=model)
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

# Modify resume_task -- only the agent.run() call changes:
    async def resume_task(task_id: str, approved: bool) -> dict:
        task = task_store.get_task(task_id)
        if task is None:
            return {"error": "not found"}

        if not approved:
            task_store.set_status(task_id, "cancelled")
            return task_store.get_task(task_id).model_dump()

        pending = task_store.get_pending_approval(task_id)
        if pending is None:
            return {"error": "no pending approval for this task"}

        try:
            pending_call = next(
                call
                for message in pending.messages
                if message.get("role") == "assistant" and message.get("tool_calls")
                for call in message["tool_calls"]
                if call["id"] == pending.tool_call_id
            )
            elevated_policy = settings.safety.model_copy(update={pending.operation: "allow"})
            agent_tools = AgentTools(task.repository, PolicyEngine(elevated_policy))
            tool_result = TOOL_DISPATCH[pending_call["function"]["name"]](
                agent_tools, json.loads(pending_call["function"]["arguments"])
            )
            extended_messages = [
                {**m, "content": str(tool_result)}
                if m.get("role") == "tool" and m.get("tool_call_id") == pending.tool_call_id
                else m
                for m in pending.messages
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
                task_class=task.task_class,
                resume_messages=extended_messages,
            )
            task_store.set_status(task.task_id, result.status, model=task.model)
            task_store.save_result(task.task_id, result)
            task_store.delete_pending_approval(task.task_id)
            if new_pending is not None:
                task_store.save_pending_approval(task.task_id, new_pending)
            return result.model_dump()
        except Exception as exc:
            task_store.set_status(task.task_id, "failed", model=task.model)
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

# Register the new tool alongside the existing seven:
    tools = {
        "get_status": get_status,
        "list_models": list_models,
        "recommend_model": recommend_model,
        "classify_task": classify_task,
        "delegate_task": delegate_task,
        "get_task_status": get_task_status,
        "cancel_task": cancel_task,
        "resume_task": resume_task,
    }
```

Also add the `TaskResult` import at the top of the file if not already
present (it is used directly in `delegate_task` now):

```python
from modelhelm.tasks.models import TaskResult
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/unit/test_mcp_server.py -v`
Expected: PASS (all existing + 4 new tests)

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

Run: `.venv\Scripts\pytest.exe tests/unit/ -v`
Expected: PASS (all tests across Tasks 1-5 and unmodified Phase 1 tests — 109 Phase 1 tests plus this milestone's additions, minus/plus any count changes from Steps 1-4 above)

- [ ] **Step 6: Commit**

```bash
git add src/modelhelm/mcp/server.py tests/unit/test_mcp_server.py
git commit -m "feat: classify tasks before delegation; add classify_task MCP tool"
```

---

### Task 6: Integration Test Update

**Files:**
- Modify: `tests/integration/test_end_to_end.py`

**Interfaces:**
- None new — this task only ensures the existing real-LM-Studio integration test still passes given `delegate_task`'s new classification gate.

The existing integration test's task description is:
`"Create a file named hello.txt containing the text 'hello from modelhelm'. Do not commit."`
This contains "create", which matches the `implementation` class's keyword
`"create"` (verified by simulating the classifier's matching logic against
this exact string during plan authoring — it resolves to
`("implementation", "create")`, the first and only match in table order).
Disposition is `local`, so this test's behavior is unchanged by this
milestone — **no edit to the description string is required.**

- [ ] **Step 1: Confirm classification with the real classifier (not just the simulation used during planning)**

Once Task 1 is merged, run the real `TaskClassifier` against the
integration test's exact description string (e.g. in a Python REPL or a
throwaway script) and confirm it still returns `task_class="implementation"`,
`disposition="local"`. This re-confirms the plan-time simulation against
the actual shipped `DEFAULT_TASK_CLASSES` table, in case Task 1's
implementation introduced any drift from the spec's keyword list. If it
does not match, adjust the description string minimally to restore a local
classification, preserving the test's original intent (a real local
delegation, not an escalation) — but this is not expected to be needed.

- [ ] **Step 2: Run the integration test if LM Studio is reachable**

Run: `.venv\Scripts\pytest.exe tests/integration/ -v -s`
Expected: PASS if LM Studio is running with a tool-use model loaded;
SKIPPED otherwise (same as Phase 1 — this is not a new requirement).

- [ ] **Step 3: Commit (only if the description needed changing)**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "test: confirm integration test task description classifies as local"
```

If no changes were needed in Step 1, skip this commit — there is nothing
to commit, and an empty commit should not be created.

---

## Self-Review Notes (completed during plan authoring)

- **Spec coverage:** Every in-scope item from the design doc (Section 2)
  maps to a task: `TaskClassifier` → Task 1; config-driven class table →
  Task 2; `delegate_task` short-circuit and `TaskResult.task_class` →
  Tasks 3 and 5; `classify_task` MCP tool → Task 5; `task_class` on
  `TaskResult` → Task 3; persistence/resume threading (the gap found
  during plan-writing prep, now folded into the spec's Section 6) → Tasks
  3, 4, and 5 together.
- **Placeholder scan:** No TBD/TODO markers. Task 6 has a conditional
  commit (only if a change is needed) — this is not a placeholder, it is
  an explicit instruction to avoid an empty commit, matching the
  "never commit unless there's something to commit" discipline.
- **Type consistency:** `ClassificationResult` (Task 1) fields match usage
  in Task 5's `delegate_task`/`classify_task` exactly. `LocalAgent.run()`'s
  new `task_class: str` parameter (Task 4) matches both call sites in
  Task 5 (`delegate_task` passes `classification.task_class`, `resume_task`
  passes `task.task_class`). `TaskStore.set_status`'s new optional
  `task_class` parameter (Task 3) matches all three call shapes used in
  Task 5 (`task_class` only, `model` only, both, neither).
- **Known Milestone-1 limitation, documented not hidden:** the
  local-before-claude keyword-matching ordering tradeoff (spec Section 3,
  locked in by Task 1's `test_local_before_claude_ordering_tradeoff_is_documented_behavior`
  test) may misclassify descriptions that mention both a local and a
  claude-default keyword. Explicitly flagged in the spec's Section 9 for
  revisiting in a later milestone.
- **Cross-task blast radius called out explicitly:** Task 4's note that all
  14 existing `agent.run()` call sites need updating, and Task 5's note
  that 3 existing `delegate_task` tests and 4 existing `resume_task`
  fixture-setup calls need updating, are both flagged in-line rather than
  discovered mid-implementation — this was found by reading the actual
  current test files during plan authoring (see the tool calls preceding
  this plan's file structure), not guessed.
