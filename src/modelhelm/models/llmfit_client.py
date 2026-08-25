import json
import shutil
import subprocess
from pydantic import BaseModel

class LlmfitModel(BaseModel):
    name: str
    capability_ids: list[str] = []
    fit_level: str = "Unknown"
    score: float = 0.0
    context_length: int = 0
    estimated_tps: float | None = None

class LlmfitError(Exception):
    pass

class LlmfitClient:
    def __init__(self, binary_path: str | None = None):
        self.binary_path = binary_path or shutil.which("llmfit") or "llmfit"

    def _run(self, args: list[str]) -> str:
        result = subprocess.run(
            [self.binary_path, *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise LlmfitError(result.stderr.strip() or "llmfit exited nonzero")
        return result.stdout

    def recommend(self) -> list[LlmfitModel]:
        stdout = self._run(["recommend", "--json"])
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LlmfitError(f"invalid JSON from llmfit recommend: {exc}") from exc
        return [LlmfitModel(**entry) for entry in payload.get("models", [])]

    def list_models(self) -> list[LlmfitModel]:
        stdout = self._run(["list", "--json"])
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise LlmfitError(f"invalid JSON from llmfit list: {exc}") from exc

        models = []
        for entry in payload:
            models.append(
                LlmfitModel(
                    name=entry["name"],
                    capability_ids=entry.get("capabilities", []),
                    context_length=entry.get("context_length", 0),
                )
            )
        return models
