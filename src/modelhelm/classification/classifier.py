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

# Table order matters, because classify() is first-match-wins: claude-default
# classes are listed BEFORE local-default classes, so a description matching
# keywords from both resolves to the claude class.
#
# This is deliberately the fail-safe direction. Realistic task descriptions
# almost always lead with a common local-matching verb ("Add OAuth2
# authentication...", "Create a migration to drop the audit_log table"), so
# ordering local first made the claude classes effectively unreachable for
# exactly the security/architecture/high-risk work that must never be handed
# to a local model. Escalating a borderline task to Claude costs tokens;
# silently running a security change locally costs correctness.
DEFAULT_TASK_CLASSES: list[TaskClass] = [
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
]

#: Reserved for the classifier's internal no-match fallback, which always
#: escalates. It is not an entry in DEFAULT_TASK_CLASSES and may not be
#: declared by a user-supplied table -- see TaskClassifier.__init__.
FALLBACK_CLASS_NAME = "ambiguous"


class TaskClassifier:
    def __init__(self, classes: list[TaskClass]):
        # Validate here rather than in load_classifier or Settings: __init__ is
        # the single chokepoint that both DEFAULT_TASK_CLASSES and any custom
        # table from modelhelm.yaml must pass through, so a bad table cannot
        # reach classify() by any route.
        for task_class in classes:
            # "ambiguous" is the fail-safe: anything matching nothing else
            # escalates to Claude. A user-declared class of the same name would
            # shadow it -- and with disposition: local would silently route
            # every unrecognized description to a local model, which is exactly
            # the case we are least confident about.
            if task_class.name.strip().lower() == FALLBACK_CLASS_NAME:
                raise ValueError(
                    f"'{FALLBACK_CLASS_NAME}' is reserved for the classifier's "
                    "internal fallback and cannot be redefined; remove it from "
                    "the classification.classes list in modelhelm.yaml"
                )
            # classify() tests `keyword in description`, and "" is a substring
            # of every string -- one empty keyword would make its class match
            # everything and render every class after it unreachable.
            if any(keyword == "" for keyword in task_class.keywords):
                raise ValueError(
                    f"task class '{task_class.name}' has an empty keyword, "
                    "which would match every description; remove it"
                )
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
            task_class=FALLBACK_CLASS_NAME, disposition="claude", matched_keyword=None
        )

def load_classifier(settings) -> TaskClassifier:
    """Builds a TaskClassifier from settings.classification.classes.

    `settings` is left untyped here (no `Settings` import) to avoid a
    circular import, since `config/settings.py` already imports from this
    module.
    """
    return TaskClassifier(settings.classification.classes)
