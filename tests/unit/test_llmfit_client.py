import json
import subprocess
from pathlib import Path
import pytest
from modelhelm.models.llmfit_client import LlmfitClient, LlmfitError, LlmfitModel

FIXTURES = Path(__file__).parent.parent / "fixtures"

def test_recommend_parses_models_key(monkeypatch):
    fixture = (FIXTURES / "llmfit_recommend.json").read_text()

    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=fixture, stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    models = client.recommend()

    assert len(models) == 2
    assert models[0] == LlmfitModel(
        name="Qwen/Qwen3-Coder-30B-A3B",
        capability_ids=["tool_use"],
        fit_level="Excellent",
        score=98.5,
        context_length=262144,
        estimated_tps=62.3,
    )

def test_list_models_parses_bare_array_with_defaults(monkeypatch):
    fixture = (FIXTURES / "llmfit_list.json").read_text()

    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=fixture, stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    models = client.list_models()

    assert len(models) == 2
    assert models[0].name == "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
    assert models[0].capability_ids == ["tool_use"]
    assert models[0].fit_level == "Unknown"
    assert models[0].score == 0.0
    assert models[0].estimated_tps is None

def test_recommend_raises_on_nonzero_exit(monkeypatch):
    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="hardware detection failed")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    with pytest.raises(LlmfitError, match="hardware detection failed"):
        client.recommend()

def test_recommend_raises_on_invalid_json(monkeypatch):
    def mock_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="not json", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    client = LlmfitClient(binary_path="llmfit")
    with pytest.raises(LlmfitError):
        client.recommend()
