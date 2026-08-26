from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel

from modelhelm.classification.classifier import TaskClass, DEFAULT_TASK_CLASSES

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
    # Not yet implemented in Phase 1: the agent loop does not run tests and
    # reports tests_run/passed/failed as 0. Test execution is deferred to a
    # later phase; defaulting to False avoids promising behavior that does not
    # exist.
    test_before_completion: bool = False

class LMStudioConfig(BaseModel):
    endpoint: str = "http://localhost:1234"

class ClassificationConfig(BaseModel):
    classes: list[TaskClass] = DEFAULT_TASK_CLASSES

class Settings(BaseModel):
    default_runtime: str = "lm-studio"
    lm_studio: LMStudioConfig = LMStudioConfig()
    llmfit_binary_path: str | None = None
    prefer_local: bool = True
    safety: SafetyPolicy = SafetyPolicy()
    agent: AgentConfig = AgentConfig()
    classification: ClassificationConfig = ClassificationConfig()

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
