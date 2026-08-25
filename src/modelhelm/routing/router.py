from modelhelm.models.registry import RegistryEntry

class NoSuitableModelError(Exception):
    pass

class TaskRouter:
    def __init__(self, registry):
        self.registry = registry

    async def select_model(self, task_description: str) -> str:
        entries = await self.registry.refresh()
        candidates = [
            e for e in entries if e.available and "tool_use" in e.capabilities
        ]
        if not candidates:
            raise NoSuitableModelError(
                f"no tool-use-capable model available for task: {task_description!r}"
            )
        candidates.sort(key=lambda e: e.fit_score if e.fit_score is not None else -1, reverse=True)
        return candidates[0].name
