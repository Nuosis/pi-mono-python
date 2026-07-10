"""Native goal tools inherited by every Tau agent.

Goals are session-scoped: the `session_id` passed to `create_goal_tools` is
stamped onto the persisted goal so other sessions in the same project do not
inherit it. A session terminator callback is invoked when the goal is
explicitly cleared or moved to a terminal status so the owning AgentSession
can stop its loop.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent

from pi_coding_agent.core.goal import clear_goal, load_goal, set_goal, update_goal


def _text(payload: dict[str, Any]) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, sort_keys=True))],
        details=payload,
    )


def create_goal_tools(
    project_root: str,
    session_id: str,
    on_goal_terminated: Callable[[], None] | None = None,
) -> list[AgentTool]:
    def _notify() -> None:
        if on_goal_terminated is None:
            return
        try:
            on_goal_terminated()
        except Exception:
            # Notification is best-effort; the tool result is still authoritative.
            pass

    async def get_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, params, cancel, on_update
        goal = load_goal(project_root, session_id)
        return _text({"goal": goal.to_dict() if goal else None})

    async def set_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, cancel, on_update
        objective = str((params or {}).get("objective") or "").strip()
        success_criteria = (params or {}).get("success_criteria") or []
        required_inputs = (params or {}).get("required_inputs") or []
        try:
            goal = set_goal(
                project_root,
                session_id=session_id,
                objective=objective,
                success_criteria=[str(item) for item in success_criteria],
                required_inputs=[str(item) for item in required_inputs],
            )
        except ValueError as exc:
            return _text({"error": str(exc)})
        return _text({"goal": goal.to_dict()})

    async def update_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, cancel, on_update
        status = str((params or {}).get("status") or "").strip()
        reason = (params or {}).get("reason")
        try:
            goal = update_goal(
                project_root,
                session_id=session_id,
                status=status,
                reason=reason,
            )
        except ValueError as exc:
            return _text({"error": str(exc)})
        if status in ("complete", "blocked") and goal is not None:
            _notify()
        return _text({"goal": goal.to_dict() if goal else None})

    async def clear_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, params, cancel, on_update
        clear_goal(project_root, session_id)
        _notify()
        return _text({"goal": None, "cleared": True})

    return [
        AgentTool(
            name="get_goal",
            label="get_goal",
            description="Get the current session-scoped Tau goal (or null).",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            execute=get_goal_execute,
        ),
        AgentTool(
            name="set_goal",
            label="set_goal",
            description=(
                "Set or replace the active Tau goal for this session. "
                "Other sessions in the same project are unaffected."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "objective": {"type": "string"},
                    "success_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "required_inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["objective"],
                "additionalProperties": False,
            },
            execute=set_goal_execute,
        ),
        AgentTool(
            name="update_goal",
            label="update_goal",
            description=(
                "Update the status of the active Tau goal for this session. "
                "Use status='complete' when the objective is achieved, "
                "'blocked' if a required input is missing, or 'active' to resume."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "complete", "blocked"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["status"],
                "additionalProperties": False,
            },
            execute=update_goal_execute,
        ),
        AgentTool(
            name="clear_goal",
            label="clear_goal",
            description=(
                "Clear the session-scoped Tau goal. Other sessions in the same "
                "project are unaffected. Signals the agent loop to stop after "
                "the current turn."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            execute=clear_goal_execute,
        ),
    ]