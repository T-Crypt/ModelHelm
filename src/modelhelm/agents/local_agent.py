"""Autonomous agent loop driving a local LLM through policy-gated tools.

The loop is deliberately synchronous in shape: ask the model what to do,
execute any tool calls it requests, feed the results back, repeat until the
model answers with no tool calls or ``max_iterations`` is exhausted.

Approval pausing is the subtle part. When a tool call trips
``ToolNeedsApproval``, the loop cannot simply abort and later re-run from
scratch: the model has no memory of having been gated, so a fresh run would
request the same tool call and hit the same gate forever. Instead ``run``
returns a ``PendingApproval`` carrying the *entire* conversation up to and
including the assistant message that requested the pending call.

Two invariants keep that conversation valid for an OpenAI-compatible server
such as LM Studio, which rejects a request whose assistant turn has a
``tool_call_id`` with no matching tool-role reply:

  1. Every ``tool_call_id`` in an assistant turn gets exactly one tool-role
     reply -- a real result, an ``ERROR: ...`` string, or a
     ``NOT_EXECUTED_MESSAGE`` placeholder. So the batch is always walked to
     completion, even once a pause has been decided.
  2. Tool failures (hallucinated names, malformed JSON arguments, missing
     files) are reported back to the model instead of raised out of the loop,
     letting it self-correct on the next turn.

Because of (1), the gated call already has a placeholder reply in
``PendingApproval.messages``. ``resume_task`` (Task 11) therefore *replaces*
that trailing placeholder with the real result of the approved call before
handing the list back via ``resume_messages`` -- it must not simply append,
which would leave two replies for one ``tool_call_id``. Only one approval is
captured per pause; a second gated call in the same batch is stubbed out and
must be re-requested by the model after resumption.
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

# Placeholder tool result for calls in a batch that were skipped because an
# earlier call in the same assistant turn paused the run awaiting approval.
NOT_EXECUTED_MESSAGE = (
    "not executed: run paused pending approval of an earlier operation in this batch"
)

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
        # Set at the start of each run(); see _build_result.
        self._base_commit: str | None = None
        self._base_dirty_files: set[str] = set()

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
        # Baseline for files_changed. Without it a pre-dirty repo gets its
        # existing changes billed to the agent, and any commit the agent makes
        # cleans the tree back to a count of zero. A repo with no commits yet
        # (or no git at all) leaves the baseline None and falls back to the raw
        # working-tree count.
        try:
            self._base_commit = self.git_inspector.snapshot().commit
            self._base_dirty_files = self.git_inspector.dirty_files()
        except Exception:
            self._base_commit = None
            self._base_dirty_files = set()

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
            # (exception, tool_call_id) of the first call in this batch that
            # needs approval; the PendingApproval is only built once the whole
            # batch has been walked, so it captures the complete message list.
            approval_pause = None
            for call in tool_calls:
                if approval_pause is not None:
                    # An earlier call in this same batch paused the run. Every
                    # tool_call_id in an assistant turn still needs a tool-role
                    # response or the conversation is malformed and cannot be
                    # resumed, so stub the remaining calls out.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": NOT_EXECUTED_MESSAGE,
                        }
                    )
                    continue

                try:
                    name = call["function"]["name"]
                    args = json.loads(call["function"]["arguments"] or "{}")
                    tool_result = TOOL_DISPATCH[name](self.tools, args)
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": str(tool_result)}
                    )
                except ToolNeedsApproval as exc:
                    # Pause, but keep walking the batch so the remaining
                    # tool_call_ids get placeholder responses. ``PendingApproval``
                    # tracks a single call, so only the first one that needs
                    # approval is captured; any later one is stubbed out and
                    # must be re-requested by the model after resumption.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": NOT_EXECUTED_MESSAGE,
                        }
                    )
                    approval_pause = (exc, call["id"])
                except ToolDenied as exc:
                    # Denial is not fatal: tell the model so it can adapt.
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": f"DENIED: {exc}"}
                    )
                except Exception as exc:
                    # Hallucinated tool names, malformed JSON arguments, missing
                    # argument keys and ordinary runtime failures (a missing file
                    # is normal while exploring) must not kill the run. Report
                    # them back so the model can self-correct next turn.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": f"ERROR: {type(exc).__name__}: {exc}",
                        }
                    )

            if approval_pause is not None:
                exc, pending_call_id = approval_pause
                result = self._build_result(
                    task_id, "pending_approval", model, start_time, iterations,
                    f"Paused: {exc.operation} requires approval ({exc.detail}).",
                )
                pending = PendingApproval(
                    operation=exc.operation,
                    detail=exc.detail,
                    tool_call_id=pending_call_id,
                    messages=messages,
                )
                return result, pending

        result = self._build_result(
            task_id, "escalation_recommended", model, start_time, iterations,
            f"Reached max_iterations ({self.agent_config.max_iterations}) without completion.",
        )
        return result, None

    def _build_result(self, task_id, status, model, start_time, iterations, summary) -> TaskResult:
        duration = time.monotonic() - start_time
        # Count against the pre-task baseline so a commit made during the run
        # still counts as changed files (and pre-existing dirt does not).
        if self._base_commit is not None:
            files_changed = self.git_inspector.files_changed_since(
                self._base_commit, self._base_dirty_files
            )
        else:
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
