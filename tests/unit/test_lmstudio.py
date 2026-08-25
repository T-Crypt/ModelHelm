import httpx
import pytest
from modelhelm.runtimes.lmstudio import LMStudioClient, LMStudioModel

@pytest.mark.asyncio
async def test_list_models(monkeypatch):
    payload = {
        "data": [
            {
                "id": "qwen3-coder-30b-a3b",
                "state": "loaded",
                "max_context_length": 262144,
                "capabilities": ["tool_use"],
            },
            {
                "id": "text-embedding-nomic-embed-text-v1.5",
                "state": "not-loaded",
                "max_context_length": 2048,
                "capabilities": [],
            },
        ]
    }

    async def mock_get(self, url, *args, **kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    client = LMStudioClient(endpoint="http://localhost:1234")
    models = await client.list_models()

    assert len(models) == 2
    assert models[0] == LMStudioModel(
        id="qwen3-coder-30b-a3b",
        state="loaded",
        max_context_length=262144,
        capabilities=["tool_use"],
    )

@pytest.mark.asyncio
async def test_chat_completion(monkeypatch):
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}]
    }

    async def mock_post(self, url, *args, **kwargs):
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    client = LMStudioClient(endpoint="http://localhost:1234")
    result = await client.chat_completion(
        model="qwen3-coder-30b-a3b",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result["choices"][0]["message"]["content"] == "hello"

@pytest.mark.asyncio
async def test_chat_completion_raises_on_timeout(monkeypatch):
    async def mock_post(self, url, *args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    client = LMStudioClient(endpoint="http://localhost:1234")
    with pytest.raises(httpx.TimeoutException):
        await client.chat_completion(model="x", messages=[])
