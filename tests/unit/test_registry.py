import pytest
from modelhelm.models.registry import ModelRegistry, RegistryEntry
from modelhelm.runtimes.lmstudio import LMStudioModel
from modelhelm.models.llmfit_client import LlmfitModel

class FakeLMStudioClient:
    async def list_models(self):
        return [
            LMStudioModel(
                id="qwen3-coder-30b-a3b",
                state="loaded",
                max_context_length=262144,
                capabilities=["tool_use"],
            ),
            LMStudioModel(
                id="text-embedding-nomic-embed-text-v1.5",
                state="not-loaded",
                max_context_length=2048,
                capabilities=[],
            ),
        ]

class FakeLlmfitClient:
    def recommend(self):
        return [
            LlmfitModel(
                name="Qwen/Qwen3-Coder-30B-A3B",
                capability_ids=["tool_use"],
                fit_level="Excellent",
                score=98.5,
                context_length=262144,
                estimated_tps=62.3,
            )
        ]

@pytest.mark.asyncio
async def test_refresh_cross_references_by_name():
    registry = ModelRegistry(
        lmstudio_client=FakeLMStudioClient(), llmfit_client=FakeLlmfitClient()
    )
    entries = await registry.refresh()

    assert len(entries) == 2
    coder = next(e for e in entries if e.name == "qwen3-coder-30b-a3b")
    assert coder.available is True
    assert coder.loaded is True
    assert coder.fit_score == 98.5

    embed = next(e for e in entries if e.name == "text-embedding-nomic-embed-text-v1.5")
    assert embed.fit_score is None
    assert embed.loaded is False
