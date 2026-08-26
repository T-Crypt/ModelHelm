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

def load_classifier(settings) -> TaskClassifier:
    """Builds a TaskClassifier from settings.classification.classes.

    `settings` is left untyped here (no `Settings` import) to avoid a
    circular import, since `config/settings.py` already imports from this
    module.
    """
    return TaskClassifier(settings.classification.classes)
