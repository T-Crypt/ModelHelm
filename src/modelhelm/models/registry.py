from pydantic import BaseModel

class RegistryEntry(BaseModel):
    name: str
    runtime: str
    available: bool
    loaded: bool
    context_length: int
    capabilities: list[str]
    fit_score: float | None = None

def _names_match(lmstudio_name: str, llmfit_name: str) -> bool:
    normalized_lmstudio = lmstudio_name.lower().replace("-", "").replace("_", "").replace(".", "")
    normalized_llmfit = llmfit_name.lower().replace("-", "").replace("_", "").replace(".", "").replace("/", "")
    return normalized_lmstudio in normalized_llmfit or normalized_llmfit.endswith(normalized_lmstudio)

class ModelRegistry:
    def __init__(self, lmstudio_client, llmfit_client):
        self.lmstudio_client = lmstudio_client
        self.llmfit_client = llmfit_client

    async def refresh(self) -> list[RegistryEntry]:
        lmstudio_models = await self.lmstudio_client.list_models()
        try:
            llmfit_models = self.llmfit_client.recommend()
        except Exception:
            llmfit_models = []

        entries = []
        for model in lmstudio_models:
            fit_score = None
            for llmfit_model in llmfit_models:
                if _names_match(model.id, llmfit_model.name):
                    fit_score = llmfit_model.score
                    break

            entries.append(
                RegistryEntry(
                    name=model.id,
                    runtime="lm-studio",
                    available=True,
                    loaded=model.state == "loaded",
                    context_length=model.max_context_length,
                    capabilities=model.capabilities,
                    fit_score=fit_score,
                )
            )
        return entries
