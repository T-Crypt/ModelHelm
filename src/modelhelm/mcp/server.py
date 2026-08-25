"""MCP server exposing ModelHelm's delegation surface as MCP tools.

This module is the integration point: it constructs the model registry,
router, policy engine, task store and agent loop from a ``Settings`` object
and publishes seven tools over MCP.

``create_server`` is a factory rather than module-level state so tests (and
alternative front ends) can inject fakes for the two external dependencies
that need a live process -- LM Studio and the ``llmfit`` binary. It returns a
``ModelHelmServer`` whose ``tools`` dict holds the raw async functions, so
they can be exercised directly without standing up an MCP transport.

Note on the SDK: the installed MCP Python SDK is 2.x, where the server class
formerly called ``FastMCP`` was renamed to ``MCPServer``. The decorator/run
surface used here (``@server.tool()``, ``server.run()``) is unchanged.
"""
import json
import shutil

from mcp.server.mcpserver import MCPServer

from modelhelm.config.settings import Settings, load_settings
from modelhelm.runtimes.lmstudio import LMStudioClient
from modelhelm.models.llmfit_client import LlmfitClient, LlmfitError
from modelhelm.models.registry import ModelRegistry
from modelhelm.policies.engine import PolicyEngine
from modelhelm.git.inspector import GitInspector
from modelhelm.tasks.store import TaskStore
from modelhelm.routing.router import TaskRouter, NoSuitableModelError
from modelhelm.agents.tools import AgentTools
from modelhelm.agents.local_agent import LocalAgent, TOOL_DISPATCH


class ModelHelmServer:
    """Thin holder so tests can call tool functions directly without an MCP transport."""

    def __init__(self, mcp: MCPServer, task_store: TaskStore, tools: dict):
        self.mcp = mcp
        self.task_store = task_store
        self.tools = tools


def create_server(
    settings: Settings,
    task_store: TaskStore,
    lmstudio_client,
    llmfit_client,
) -> ModelHelmServer:
    mcp = MCPServer("modelhelm")
    registry = ModelRegistry(lmstudio_client=lmstudio_client, llmfit_client=llmfit_client)
    router = TaskRouter(registry)

    async def get_status() -> dict:
        try:
            await lmstudio_client.list_models()
            lm_reachable = True
        except Exception:
            lm_reachable = False
        try:
            llmfit_client.recommend()
            llmfit_ok = True
        except (LlmfitError, Exception):
            llmfit_ok = False
        return {
            "lm_studio_reachable": lm_reachable,
            "llmfit_available": llmfit_ok,
            "default_runtime": settings.default_runtime,
        }

    async def list_models() -> list[dict]:
        entries = await registry.refresh()
        return [entry.model_dump() for entry in entries]

    async def recommend_model(task_description: str) -> str:
        return await router.select_model(task_description)

    async def delegate_task(description: str, repository: str) -> dict:
        task = task_store.create_task(description=description, repository=repository)
        try:
            model = await router.select_model(description)
        except NoSuitableModelError as exc:
            task_store.set_status(task.task_id, "failed")
            return {"task_id": task.task_id, "status": "failed", "summary": str(exc)}

        task_store.set_status(task.task_id, "running", model=model)
        policy_engine = PolicyEngine(settings.safety)
        agent_tools = AgentTools(repository, policy_engine)
        git_inspector = GitInspector(repository)
        agent = LocalAgent(
            lmstudio_client=lmstudio_client,
            tools=agent_tools,
            git_inspector=git_inspector,
            agent_config=settings.agent,
        )
        result, pending = await agent.run(task_id=task.task_id, description=description, model=model)
        task_store.set_status(task.task_id, result.status, model=model)
        task_store.save_result(task.task_id, result)
        if pending is not None:
            task_store.save_pending_approval(task.task_id, pending)
        return result.model_dump()

    async def get_task_status(task_id: str) -> dict:
        task = task_store.get_task(task_id)
        if task is None:
            return {"error": "not found"}
        return task.model_dump()

    async def cancel_task(task_id: str) -> dict:
        task = task_store.get_task(task_id)
        if task is None:
            return {"error": "not found"}
        task_store.set_status(task_id, "cancelled")
        return task_store.get_task(task_id).model_dump()

    async def resume_task(task_id: str, approved: bool) -> dict:
        task = task_store.get_task(task_id)
        if task is None:
            return {"error": "not found"}

        # Rejection is handled before the pending-approval lookup: declining is
        # a cancellation of the task either way, so it must not depend on an
        # approval record still being present. The "no pending approval" error
        # is only meaningful on the approve path, where there has to be a
        # specific gated call to execute.
        if not approved:
            task_store.set_status(task_id, "cancelled")
            return task_store.get_task(task_id).model_dump()

        pending = task_store.get_pending_approval(task_id)
        if pending is None:
            return {"error": "no pending approval for this task"}

        # The human already approved this exact operation out-of-band, so execute
        # it directly with an elevated one-off policy rather than routing back
        # through the normal ask-gated AgentTools methods (which would just raise
        # ToolNeedsApproval again).
        #
        # pending.messages does NOT necessarily end with the assistant message
        # that requested the gated call (Task 10's agent loop appends a
        # NOT_EXECUTED_MESSAGE placeholder tool-role reply for the gated call,
        # and for any later calls in the same batch, so every tool_call_id in
        # that assistant turn already has a tool-role response — required for
        # the conversation to be valid to resend). So: find the assistant
        # message whose tool_calls include pending.tool_call_id to recover the
        # call's arguments, and find-and-replace (never append) the matching
        # placeholder tool-role message with the real result.
        pending_call = next(
            call
            for message in pending.messages
            if message.get("role") == "assistant" and message.get("tool_calls")
            for call in message["tool_calls"]
            if call["id"] == pending.tool_call_id
        )
        elevated_policy = settings.safety.model_copy(update={pending.operation: "allow"})
        agent_tools = AgentTools(task.repository, PolicyEngine(elevated_policy))
        tool_result = TOOL_DISPATCH[pending.operation](
            agent_tools, json.loads(pending_call["function"]["arguments"])
        )
        extended_messages = [
            {**m, "content": str(tool_result)}
            if m.get("role") == "tool" and m.get("tool_call_id") == pending.tool_call_id
            else m
            for m in pending.messages
        ]

        git_inspector = GitInspector(task.repository)
        agent = LocalAgent(
            lmstudio_client=lmstudio_client,
            tools=AgentTools(task.repository, PolicyEngine(settings.safety)),
            git_inspector=git_inspector,
            agent_config=settings.agent,
        )
        result, new_pending = await agent.run(
            task_id=task.task_id, description=task.description, model=task.model,
            resume_messages=extended_messages,
        )
        task_store.set_status(task.task_id, result.status, model=task.model)
        task_store.save_result(task.task_id, result)
        if new_pending is not None:
            task_store.save_pending_approval(task.task_id, new_pending)
        return result.model_dump()

    tools = {
        "get_status": get_status,
        "list_models": list_models,
        "recommend_model": recommend_model,
        "delegate_task": delegate_task,
        "get_task_status": get_task_status,
        "cancel_task": cancel_task,
        "resume_task": resume_task,
    }
    for name, fn in tools.items():
        mcp.tool(name=name)(fn)

    return ModelHelmServer(mcp=mcp, task_store=task_store, tools=tools)


def main() -> None:
    settings = load_settings()
    server = create_server(
        settings=settings,
        task_store=TaskStore("modelhelm_tasks.db"),
        lmstudio_client=LMStudioClient(endpoint=settings.lm_studio.endpoint),
        llmfit_client=LlmfitClient(binary_path=settings.llmfit_binary_path or shutil.which("llmfit")),
    )
    server.mcp.run()


if __name__ == "__main__":
    main()
