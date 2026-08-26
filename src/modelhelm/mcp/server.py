"""MCP server exposing ModelHelm's delegation surface as MCP tools.

This module is the integration point: it constructs the model registry,
router, policy engine, task store and agent loop from a ``Settings`` object
and publishes eight tools over MCP.

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

from modelhelm.classification.classifier import load_classifier
from modelhelm.config.settings import Settings, load_settings
from modelhelm.tasks.models import TaskResult
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
    classifier = load_classifier(settings)

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

    async def classify_task(description: str) -> dict:
        """Preview a description's task class without creating or touching a task."""
        return classifier.classify(description).model_dump()

    async def delegate_task(description: str, repository: str) -> dict:
        # Classification happens before anything else: a claude-disposition task
        # must never reach the router, the registry, or LM Studio, so the
        # short-circuit below has to precede router.select_model().
        classification = classifier.classify(description)
        task = task_store.create_task(description=description, repository=repository)

        if classification.disposition == "claude":
            task_store.set_status(
                task.task_id, "escalation_recommended", task_class=classification.task_class
            )
            result = TaskResult(
                task_id=task.task_id,
                status="escalation_recommended",
                model="none",
                runtime="none",
                duration_seconds=0.0,
                files_changed=0,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                iterations=0,
                estimated_cloud_tokens_saved=0,
                review_recommended=True,
                task_class=classification.task_class,
                summary=(
                    f"Classified as {classification.task_class} — "
                    f"recommend Claude handle this directly."
                ),
            )
            task_store.save_result(task.task_id, result)
            return result.model_dump()

        try:
            model = await router.select_model(description)
        except NoSuitableModelError as exc:
            task_store.set_status(task.task_id, "failed", task_class=classification.task_class)
            return {"task_id": task.task_id, "status": "failed", "summary": str(exc)}

        task_store.set_status(
            task.task_id, "running", model=model, task_class=classification.task_class
        )
        # Safety net: anything the agent loop can raise (PathScopeError,
        # ToolDenied, subprocess failures, an httpx timeout talking to LM
        # Studio) must become a structured failure. Letting it propagate out of
        # an MCP tool leaves the task stuck in "running" with no result ever
        # persisted.
        try:
            policy_engine = PolicyEngine(settings.safety)
            agent_tools = AgentTools(repository, policy_engine)
            git_inspector = GitInspector(repository)
            agent = LocalAgent(
                lmstudio_client=lmstudio_client,
                tools=agent_tools,
                git_inspector=git_inspector,
                agent_config=settings.agent,
            )
            result, pending = await agent.run(
                task_id=task.task_id, description=description, model=model,
                task_class=classification.task_class,
            )
            task_store.set_status(task.task_id, result.status, model=model)
            task_store.save_result(task.task_id, result)
            if pending is not None:
                task_store.save_pending_approval(task.task_id, pending)
            return result.model_dump()
        except Exception as exc:
            task_store.set_status(task.task_id, "failed", model=model)
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

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

        # Validate task_class BEFORE executing the approved call. There is no
        # re-classification on resume -- the class persisted at delegation time
        # is authoritative -- so a None here means the task predates this
        # milestone's schema or was created outside delegate_task. agent.run()
        # would eventually reject it when constructing TaskResult, but only
        # AFTER the approved side effect (a file write, a commit) had already
        # landed; the resulting ValidationError was then swallowed by the broad
        # except below, and delete_pending_approval was never reached, leaving
        # the approval replayable against now-stale arguments.
        #
        # The approval record is deliberately left in place: nothing was
        # executed, so there is no consumed approval to retire, and preserving
        # it lets a legitimate retry succeed once the missing task_class is
        # repaired rather than forcing the human to re-approve from scratch.
        if task.task_class is None:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": "task_class is missing on this task; cannot resume",
            }

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
        try:
            pending_call = next(
                call
                for message in pending.messages
                if message.get("role") == "assistant" and message.get("tool_calls")
                for call in message["tool_calls"]
                if call["id"] == pending.tool_call_id
            )
        # Two distinct namespaces meet here and must not be conflated:
        # ``pending.operation`` is a *policy* operation name (a SafetyPolicy
        # field: file_write, git_commit, destructive_commands, ...) and is what
        # gets elevated to "allow"; ``pending_call["function"]["name"]`` is the
        # *tool* name (write_file, git_commit, run_command, ...) and is what
        # keys TOOL_DISPATCH. They coincide only for git_commit.
            elevated_policy = settings.safety.model_copy(update={pending.operation: "allow"})
            agent_tools = AgentTools(task.repository, PolicyEngine(elevated_policy))
            tool_result = TOOL_DISPATCH[pending_call["function"]["name"]](
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
            # No re-classification on resume: the class persisted at delegation
            # time is authoritative. It is guaranteed non-None by the early
            # guard above, which rejects the resume before any side effect.
            result, new_pending = await agent.run(
                task_id=task.task_id, description=task.description, model=task.model,
                task_class=task.task_class,
                resume_messages=extended_messages,
            )
            task_store.set_status(task.task_id, result.status, model=task.model)
            task_store.save_result(task.task_id, result)
            # This approval has now been consumed. Drop it BEFORE saving any new
            # one, or a second resume_task(approved=True) would silently replay
            # the same operation against its original, now-stale arguments —
            # potentially reverting edits a human made in between.
            task_store.delete_pending_approval(task.task_id)
            if new_pending is not None:
                task_store.save_pending_approval(task.task_id, new_pending)
            return result.model_dump()
        except Exception as exc:
            task_store.set_status(task.task_id, "failed", model=task.model)
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    tools = {
        "get_status": get_status,
        "list_models": list_models,
        "recommend_model": recommend_model,
        "classify_task": classify_task,
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
