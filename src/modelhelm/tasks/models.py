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
    task_class: str | None = None
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
    task_class: str
    summary: str


class PendingApproval(BaseModel):
    operation: str
    detail: str
    tool_call_id: str
    messages: list[dict]
