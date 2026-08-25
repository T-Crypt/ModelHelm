import textwrap
from modelhelm.config.settings import load_settings, Settings

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
