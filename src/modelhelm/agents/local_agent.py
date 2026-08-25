"""Autonomous agent loop driving a local LLM through policy-gated tools.

The loop is deliberately synchronous in shape: ask the model what to do,
execute any tool calls it requests, feed the results back, repeat until the
model answers with no tool calls or ``max_iterations`` is exhausted.

Approval pausing is the subtle part. When a tool call trips
``ToolNeedsApproval``, the loop cannot simply abort and later re-run from
scratch: the model has no memory of having been gated, so a fresh run would
request the same tool call and hit the same gate forever. Instead ``run``
returns a ``PendingApproval`` carrying the *entire* conversation up to and
including the assistant message that requested the pending call. The caller
(``resume_task``, Task 11) executes the approved call itself, appends the
resulting tool message, and hands the extended list back via
``resume_messages`` so the loop continues the same conversation.
"""
import json
import time

from modelhelm.agents.tools import get_tool_definitions, ToolDenied, ToolNeedsApproval
from modelhelm.tasks.models import TaskResult, PendingApproval

TOOL_DISPATCH = {
    "read_file": lambda tools, args: tools.read_file(args["path"]),
    "write_file": lambda tools, args: tools.write_file(args["path"], args["content"]),
    "list_directory": lambda tools, args: tools.list_directory(args.get("path", ".")),
    "run_command": lambda tools, args: tools.run_command(args["command"]),
    "git_diff": lambda tools, args: tools.git_diff(),
    "git_commit": lambda tools, args: tools.git_commit(args["message"]),
}

SYSTEM_PROMPT = (
    "You are a coding agent. Use the available tools to inspect the "
    "repository, make changes, and run tests. When finished, respond "
    "with a final message and no tool calls."
)


class LocalAgent:
    def __init__(self, lmstudio_client, tools, git_inspector, agent_config):
        self.lmstudio_client = lmstudio_client
        self.tools = tools
        self.git_inspector = git_inspector
        self.agent_config = agent_config

    async def run(
        self,
        task_id: str,
        description: str,
        model: str,
        resume_messages: list[dict] | None = None,
    ) -> tuple[TaskResult, PendingApproval | None]:
        """Drive the model/tool loop until completion, approval pause, or
        iteration exhaustion.

        Returns ``(result, pending)``. ``pending`` is non-None only for
        ``status == "pending_approval"``, in which case it carries the
        conversation state needed to resume this same conversation later.
        """
        start_time = time.monotonic()

        if resume_messages is not None:
            # Copy so the caller's list is not mutated as the loop appends.
            messages = list(resume_messages)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ]

        iterations = 0
        for iterations in range(1, self.agent_config.max_iterations + 1):
            response = await self.lmstudio_client.chat_completion(
                model=model, messages=messages, tools=get_tool_definitions()
            )
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                result = self._build_result(
                    task_id, "completed", model, start_time, iterations,
                    message.get("content") or "Task completed.",
                )
                return result, None

            messages.append(message)
            for call in tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                try:
                    tool_result = TOOL_DISPATCH[name](self.tools, args)
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": str(tool_result)}
                    )
                except ToolNeedsApproval as exc:
                    # Pause here. ``messages`` ends with the assistant message
                    # holding this tool call, which is exactly the state the
                    # resumer needs to append the approved tool result onto.
                    result = self._build_result(
                        task_id, "pending_approval", model, start_time, iterations,
                        f"Paused: {exc.operation} requires approval ({exc.detail}).",
                    )
                    pending = PendingApproval(
                        operation=exc.operation,
                        detail=exc.detail,
                        tool_call_id=call["id"],
                        messages=messages,
                    )
                    return result, pending
                except ToolDenied as exc:
                    # Denial is not fatal: tell the model so it can adapt.
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": f"DENIED: {exc}"}
                    )

        result = self._build_result(
            task_id, "escalation_recommended", model, start_time, iterations,
            f"Reached max_iterations ({self.agent_config.max_iterations}) without completion.",
        )
        return result, None

    def _build_result(self, task_id, status, model, start_time, iterations, summary) -> TaskResult:
        duration = time.monotonic() - start_time
        files_changed = self.git_inspector.files_changed_count()
        return TaskResult(
            task_id=task_id,
            status=status,
            model=model,
            runtime="lm-studio",
            duration_seconds=round(duration, 2),
            files_changed=files_changed,
            tests_run=0,
            tests_passed=0,
            tests_failed=0,
            iterations=iterations,
            estimated_cloud_tokens_saved=files_changed * 500,
            review_recommended=status != "completed" or files_changed > 0,
            summary=summary,
        )
