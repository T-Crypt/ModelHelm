from typing import Literal
import httpx
from pydantic import BaseModel

class LMStudioModel(BaseModel):
    id: str
    state: Literal["loaded", "not-loaded"]
    max_context_length: int
    capabilities: list[str] = []

class LMStudioClient:
    def __init__(self, endpoint: str, timeout: float = 120.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    async def list_models(self) -> list[LMStudioModel]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.endpoint}/api/v0/models")
            response.raise_for_status()
            data = response.json()
        return [LMStudioModel(**model) for model in data["data"]]

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        body = {"model": model, "messages": messages}
        if tools:
            body["tools"] = tools

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.endpoint}/v1/chat/completions", json=body
            )
            response.raise_for_status()
            return response.json()
