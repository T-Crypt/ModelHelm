import pytest

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

def test_claude_before_local_ordering_prioritizes_safety():
    # A description matching both a claude-default and a local-default class
    # resolves to the claude class, because claude classes precede local ones
    # in DEFAULT_TASK_CLASSES and classify() is first-match-wins.
    #
    # This is the fail-safe direction and is deliberate: escalating a
    # borderline task to Claude costs tokens, whereas running a security or
    # high-risk change on a local model costs correctness. "refactor the
    # security module" is security work that happens to be phrased as a
    # refactor, so it must escalate rather than route local.
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("refactor the security module")
    assert result.task_class == "security"
    assert result.disposition == "claude"


# --- C1: claude keywords must beat leading local verbs in realistic phrasing --
# Realistic task descriptions almost always open with a common local-matching
# verb (add, create, build, fix, implement). Before the reorder these all
# misrouted to a local model; each case below is one the final review
# demonstrated failing.

def test_claude_keyword_wins_over_local_verb_in_realistic_security_description():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("Add OAuth2 authentication to the login endpoint")
    # Previously: "add" -> implementation/local. Now: "auth" -> security/claude.
    assert result.task_class == "security"
    assert result.disposition == "claude"
    assert result.matched_keyword == "auth"


def test_claude_keyword_wins_over_local_verb_in_realistic_high_risk_description():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("Add a script to delete stale user records")
    # Previously: "add" -> implementation/local. Now: "delete" -> high_risk/claude.
    assert result.task_class == "high_risk"
    assert result.disposition == "claude"
    assert result.matched_keyword == "delete"


def test_claude_keyword_wins_over_local_verb_in_realistic_vulnerability_description():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("Fix the SQL injection vulnerability in the search handler")
    # Previously "search" (exploration/local) was reached first. Now the whole
    # claude group is checked before exploration, so "vulnerability" wins.
    assert result.task_class == "security"
    assert result.disposition == "claude"
    assert result.matched_keyword == "vulnerability"


def test_claude_keyword_wins_over_local_verb_in_realistic_architecture_description():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("Implement the redesign of the payment flow")
    assert result.task_class == "architecture"
    assert result.disposition == "claude"


def test_purely_local_description_still_routes_local_after_reorder():
    """The reorder must not drag ordinary local work into escalation: a
    description matching no claude keyword is unaffected by the group swap."""
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify(
        "Create a file named hello.txt containing the text Hello ModelHelm"
    )
    assert result.task_class == "implementation"
    assert result.disposition == "local"


def test_claude_disposition_classes_precede_local_ones_in_default_table():
    """Locks in the group ordering itself, so a future edit that appends a
    claude class after the local block fails loudly rather than silently
    re-opening the C1 misrouting."""
    dispositions = [c.disposition for c in DEFAULT_TASK_CLASSES]
    assert dispositions == ["claude"] * 4 + ["local"] * 7

def test_classify_is_case_insensitive():
    classifier = TaskClassifier(DEFAULT_TASK_CLASSES)
    result = classifier.classify("FIND the login handler")
    assert result.task_class == "exploration"

# --- I2: the fail-safe fallback must not be overridable ---------------------

def test_class_named_ambiguous_is_rejected():
    """A user-declared 'ambiguous' class would shadow the internal fallback;
    with disposition: local it would silently route every unrecognized
    description to a local model."""
    with pytest.raises(ValueError, match="reserved"):
        TaskClassifier([
            TaskClass(name="ambiguous", disposition="local", keywords=["whatever"]),
        ])


def test_class_named_ambiguous_is_rejected_case_insensitively():
    with pytest.raises(ValueError, match="reserved"):
        TaskClassifier([
            TaskClass(name="AMBIGUOUS", disposition="local", keywords=["whatever"]),
        ])


def test_class_named_ambiguous_is_rejected_even_with_claude_disposition():
    """Rejection is about ownership of the name, not the disposition: a
    shadowing class could still be edited to local later."""
    with pytest.raises(ValueError, match="reserved"):
        TaskClassifier([
            TaskClass(name="ambiguous", disposition="claude", keywords=["whatever"]),
        ])


def test_empty_string_keyword_is_rejected():
    """classify() tests `keyword in description`, and "" matches every string,
    so one empty keyword makes its class un-avoidable."""
    with pytest.raises(ValueError, match="empty keyword"):
        TaskClassifier([
            TaskClass(name="catch_all", disposition="local", keywords=[""]),
        ])


def test_empty_string_keyword_is_rejected_among_valid_keywords():
    with pytest.raises(ValueError, match="empty keyword"):
        TaskClassifier([
            TaskClass(name="catch_all", disposition="local", keywords=["banana", ""]),
        ])


def test_valid_custom_table_still_constructs():
    """The I2 guards must reject only the two unsafe shapes, nothing else."""
    classifier = TaskClassifier([
        TaskClass(name="only_class", disposition="claude", keywords=["banana"]),
    ])
    assert classifier.classify("peel the banana").task_class == "only_class"


def test_default_table_passes_validation():
    TaskClassifier(DEFAULT_TASK_CLASSES)  # must not raise


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
