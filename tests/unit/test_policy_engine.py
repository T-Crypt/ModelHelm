from modelhelm.policies.engine import PolicyEngine
from modelhelm.config.settings import SafetyPolicy


def test_check_returns_configured_state_for_each_operation():
    policy = SafetyPolicy(
        file_write="allow",
        file_delete="deny",
        git_commit="ask",
        git_push="allow",
        force_push="deny",
        destructive_commands="ask",
        production_changes="deny",
    )
    engine = PolicyEngine(policy)

    # Test all 7 operations with distinct, meaningful values
    assert engine.check("file_write") == "allow"
    assert engine.check("file_delete") == "deny"
    assert engine.check("git_commit") == "ask"
    assert engine.check("git_push") == "allow"
    assert engine.check("force_push") == "deny"
    assert engine.check("destructive_commands") == "ask"
    assert engine.check("production_changes") == "deny"


def test_check_unknown_operation_raises():
    engine = PolicyEngine(SafetyPolicy())
    import pytest
    with pytest.raises(ValueError, match="unknown operation"):
        engine.check("delete_universe")
