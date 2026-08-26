import textwrap
from pathlib import Path
from modelhelm.config.settings import load_settings, Settings
from modelhelm.classification.classifier import DEFAULT_TASK_CLASSES, load_classifier

def test_load_settings_from_file(tmp_path):
    config_path = tmp_path / "modelhelm.yaml"
    config_path.write_text(textwrap.dedent("""
        modelhelm:
          default_runtime: lm-studio
        runtimes:
          lm-studio:
            endpoint: http://localhost:1234
        llmfit:
          binary_path: null
        routing:
          prefer_local: true
        safety:
          file_write: allow
          file_delete: allow
          git_commit: ask
          git_push: ask
          force_push: deny
          destructive_commands: deny
          production_changes: deny
        agent:
          max_iterations: 8
          test_before_completion: true
    """))
    settings = load_settings(str(config_path))
    assert settings.default_runtime == "lm-studio"
    assert settings.lm_studio.endpoint == "http://localhost:1234"
    assert settings.safety.git_commit == "ask"
    assert settings.safety.force_push == "deny"
    assert settings.agent.max_iterations == 8

def test_load_settings_missing_file_returns_defaults(tmp_path):
    settings = load_settings(str(tmp_path / "does-not-exist.yaml"))
    assert isinstance(settings, Settings)
    assert settings.default_runtime == "lm-studio"
    assert settings.safety.force_push == "deny"

def test_test_before_completion_defaults_to_false():
    """Test execution is not implemented in Phase 1, so the config must not
    default to promising it."""
    assert Settings().agent.test_before_completion is False

def test_shipped_config_does_not_enable_test_before_completion():
    repo_config = Path(__file__).resolve().parents[2] / "modelhelm.yaml"
    settings = load_settings(str(repo_config))
    assert settings.agent.test_before_completion is False

def test_load_settings_defaults_to_builtin_classification_table():
    settings = Settings()
    assert settings.classification.classes == DEFAULT_TASK_CLASSES

def test_load_settings_with_custom_classification_replaces_default_table(tmp_path):
    config_path = tmp_path / "modelhelm.yaml"
    config_path.write_text(textwrap.dedent("""
        classification:
          classes:
            - name: only_class
              disposition: claude
              keywords: [banana]
    """))
    settings = load_settings(str(config_path))
    assert len(settings.classification.classes) == 1
    assert settings.classification.classes[0].name == "only_class"

def test_load_settings_without_classification_section_uses_default_table(tmp_path):
    config_path = tmp_path / "modelhelm.yaml"
    config_path.write_text(textwrap.dedent("""
        modelhelm:
          default_runtime: lm-studio
    """))
    settings = load_settings(str(config_path))
    assert settings.classification.classes == DEFAULT_TASK_CLASSES

def test_load_classifier_builds_from_settings():
    settings = Settings()
    classifier = load_classifier(settings)
    result = classifier.classify("find the login handler")
    assert result.task_class == "exploration"


# --- C1: the shipped modelhelm.yaml must not undo the safe ordering ---------
# A present classification: section REPLACES the built-in table wholesale, so
# the repo config is what actually governs a real run. It shipped in the old
# local-first order, which silently defeated the in-code reorder; no test
# caught it because every other test builds Settings() directly.

REPO_CONFIG = Path(__file__).resolve().parents[2] / "modelhelm.yaml"


def test_repo_config_lists_claude_classes_before_local_ones():
    classes = load_settings(str(REPO_CONFIG)).classification.classes
    dispositions = [c.disposition for c in classes]
    assert dispositions == sorted(dispositions, key=lambda d: d != "claude"), (
        "claude-disposition classes must precede local ones in modelhelm.yaml"
    )


def test_repo_config_matches_the_builtin_default_table():
    """The shipped config is meant to be the defaults written out explicitly;
    drift between the two is how the C1 misrouting survived the code fix."""
    assert load_settings(str(REPO_CONFIG)).classification.classes == DEFAULT_TASK_CLASSES


def test_repo_config_escalates_realistic_security_description():
    classifier = load_classifier(load_settings(str(REPO_CONFIG)))
    result = classifier.classify("Add OAuth2 authentication to the login endpoint")
    assert result.task_class == "security"
    assert result.disposition == "claude"
