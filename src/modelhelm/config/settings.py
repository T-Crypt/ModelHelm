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
