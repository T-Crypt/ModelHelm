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
