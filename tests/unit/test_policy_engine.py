from modelhelm.policies.engine import PolicyEngine
from modelhelm.config.settings import SafetyPolicy


def test_check_returns_configured_state_for_each_operation():
    policy = SafetyPolicy(
        file_write="allow",
        file_delete="allow",
        git_commit="ask",
        git_push="ask",
        force_push="deny",
        destructive_commands="deny",
        production_changes="deny",
    )
    engine = PolicyEngine(policy)

    assert engine.check("file_write") == "allow"
    assert engine.check("git_commit") == "ask"
    assert engine.check("force_push") == "deny"


def test_check_unknown_operation_raises():
    engine = PolicyEngine(SafetyPolicy())
    import pytest
    with pytest.raises(ValueError, match="unknown operation"):
        engine.check("delete_universe")
