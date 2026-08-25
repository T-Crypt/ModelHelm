import pytest
from modelhelm.routing.router import TaskRouter, NoSuitableModelError
from modelhelm.models.registry import RegistryEntry

class FakeRegistry:
    def __init__(self, entries):
        self._entries = entries

    async def refresh(self):
        return self._entries

@pytest.mark.asyncio
async def test_select_model_picks_highest_fit_score_with_tool_use():
    registry = FakeRegistry([
        RegistryEntry(
            name="qwen3-coder-30b-a3b", runtime="lm-studio", available=True,
            loaded=True, context_length=262144, capabilities=["tool_use"], fit_score=98.5,
        ),
        RegistryEntry(
            name="qwen2.5-coder-14b-instruct", runtime="lm-studio", available=True,
            loaded=False, context_length=131072, capabilities=["tool_use"], fit_score=70.0,
        ),
    ])
    router = TaskRouter(registry)
    selected = await router.select_model("implement a REST client")
    assert selected == "qwen3-coder-30b-a3b"

@pytest.mark.asyncio
async def test_select_model_excludes_non_tool_use():
    registry = FakeRegistry([
        RegistryEntry(
            name="text-embedding-nomic", runtime="lm-studio", available=True,
            loaded=False, context_length=2048, capabilities=[], fit_score=99.0,
        ),
    ])
    router = TaskRouter(registry)
    with pytest.raises(NoSuitableModelError):
        await router.select_model("implement a REST client")

@pytest.mark.asyncio
async def test_select_model_treats_missing_fit_score_as_lowest():
    registry = FakeRegistry([
        RegistryEntry(
            name="no-fit-data", runtime="lm-studio", available=True,
            loaded=True, context_length=131072, capabilities=["tool_use"], fit_score=None,
        ),
        RegistryEntry(
            name="has-fit-data", runtime="lm-studio", available=True,
            loaded=True, context_length=131072, capabilities=["tool_use"], fit_score=1.0,
        ),
    ])
    router = TaskRouter(registry)
    selected = await router.select_model("task")
    assert selected == "has-fit-data"
