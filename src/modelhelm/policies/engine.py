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
