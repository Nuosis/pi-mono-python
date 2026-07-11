"""Session-scoped, in-memory goal tools.

Goal state lives on AgentSession as `_goal: SessionGoal | None`. The tools
operate on the session directly; nothing is written to disk.

Opt-in: the four tools are added to the session's tool registry only when
the active tool list explicitly names a goal tool, or names the "goal"
alias (which expands to all four). Without that opt-in, the tools are not
registered and the goal machinery cost is zero on the harness.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession


GOAL_TOOL_NAMES: tuple[str, ...] = (
    "get_goal",
    "set_goal",
    "update_goal",
    "clear_goal",
)
GOAL_ALIAS = "goal"


def _text(payload: dict[str, Any]) -> AgentToolResult:
    return AgentToolResult(
        content=[
            TextContent(type="text", text=json.dumps(payload, indent=2, sort_keys=True))
        ],
        details={},
    )


def _goal_payload(session: "AgentSession") -> dict[str, Any] | None:
    state = session.goal_state
    if state is None:
        return None
    return {"objective": state.objective, "status": state.status}


def create_goal_tools(session: "AgentSession") -> list:
    """Return the four goal tools bound to `session`.

    The tools read/mutate session.goal_state directly. No filesystem is
    touched. The terminator callback (session.terminate_goal) is fired
    by the session when a terminal status transition happens, so callers
    see goal completion through the same hook the CLI uses.
    """
    async def get_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, params, cancel, on_update
        return _text({"goal": _goal_payload(session)})

    async def set_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, cancel, on_update
        objective = str((params or {}).get("objective") or "").strip()
        if not objective:
            return _text({"error": "objective is required"})
        session.set_current_goal(objective)
        return _text({"goal": _goal_payload(session)})

    async def update_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, cancel, on_update
        status = str((params or {}).get("status") or "").strip()
        if status not in ("active", "complete", "blocked"):
            return _text({"error": f"unknown status: {status!r}"})
        try:
            session.update_goal_status(status)
        except ValueError as exc:
            return _text({"error": str(exc)})
        return _text({"goal": _goal_payload(session)})

    async def clear_goal_execute(tool_call_id, params, cancel=None, on_update=None):
        del tool_call_id, params, cancel, on_update
        session.clear_goal()
        return _text({"goal": None, "cleared": True})

    from pi_agent.types import AgentTool

    return [
        AgentTool(
            name="get_goal",
            label="get_goal",
            description=(
                "Get the current session-scoped Tau goal (or null). Goal state is in-memory only."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            execute=get_goal_execute,
        ),
        AgentTool(
            name="set_goal",
            label="set_goal",
            description=(
                "Set the session-scoped Tau goal. Replaces any existing goal. Goal state is in-memory only."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The goal objective for this session.",
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
                "Update the current session-scoped Tau goal's status. Status transitions to "
                "'complete' or 'blocked' are terminal and signal the agent loop to stop after "
                "the current turn."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "complete", "blocked"],
                        "description": "New status for the active goal.",
                    },
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
                "Clear the session-scoped Tau goal. Signals the agent loop to stop after "
                "the current turn. In-memory only."
            ),
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            execute=clear_goal_execute,
        ),
    ]


def goal_tool_names() -> tuple[str, ...]:
    """Return the four goal tool names. Used by agent_session for opt-in detection."""
    return GOAL_TOOL_NAMES


def goal_alias() -> str:
    """Return the alias that expands to all four goal tools."""
    return GOAL_ALIAS
