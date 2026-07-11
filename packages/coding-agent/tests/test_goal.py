"""In-memory goal loop tests.

Goal state lives on AgentSession as `_goal: SessionGoal | None` and is
exposed via `goal_state`, `set_current_goal`, `update_goal_status`,
`clear_goal`, `get_current_goal`. There is no filesystem persistence.

The four tools (get_goal / set_goal / update_goal / clear_goal) are also
tested via the tool factory in `core/tools/goal.py`, which delegates to
the same in-memory methods.
"""

from __future__ import annotations

import asyncio

import pytest

from pi_coding_agent.core.agent_session import SessionGoal
from pi_coding_agent.core.tools.goal import (
    GOAL_ALIAS,
    GOAL_TOOL_NAMES,
    create_goal_tools,
    goal_alias,
    goal_tool_names,
)


class _FakeSession:
    """Minimal stand-in for AgentSession — only the methods the tools touch."""

    def __init__(self) -> None:
        self._goal: SessionGoal | None = None
        self._terminated_count: int = 0

    @property
    def goal_state(self) -> SessionGoal | None:
        return self._goal

    def get_current_goal(self) -> str | None:
        if self._goal is not None and self._goal.status == "active":
            return self._goal.objective
        return None

    def set_current_goal(self, objective):
        # Mirrors AgentSession.set_current_goal: clearing via this entry
        # point does NOT fire the terminator. Only clear_goal / terminal
        # update_goal_status do.
        value = (objective or "").strip()
        if value:
            self._goal = SessionGoal(objective=value, status="active")
        else:
            self._goal = None

    def update_goal_status(self, status: str) -> None:
        if status not in ("active", "complete", "blocked"):
            raise ValueError(f"unknown goal status: {status!r}")
        if self._goal is None:
            return
        old_status = self._goal.status
        self._goal = SessionGoal(objective=self._goal.objective, status=status)
        if old_status == "active" and status in ("complete", "blocked"):
            self._terminated_count += 1

    def clear_goal(self) -> None:
        if self._goal is not None:
            self._goal = None
            self._terminated_count += 1


def _run(coro):
    return asyncio.run(coro)


def test_goal_tool_names_and_alias() -> None:
    assert goal_tool_names() == ("get_goal", "set_goal", "update_goal", "clear_goal")
    assert goal_alias() == "goal"
    assert GOAL_ALIAS == "goal"
    assert set(GOAL_TOOL_NAMES) == {"get_goal", "set_goal", "update_goal", "clear_goal"}


def test_session_goal_starts_empty() -> None:
    session = _FakeSession()
    assert session.goal_state is None
    assert session.get_current_goal() is None


def test_set_current_goal_seeds_in_memory_state() -> None:
    session = _FakeSession()
    session.set_current_goal("Ship the lookup")
    assert session.get_current_goal() == "Ship the lookup"
    assert session.goal_state is not None
    assert session.goal_state.objective == "Ship the lookup"
    assert session.goal_state.status == "active"


def test_set_current_goal_with_empty_string_clears() -> None:
    session = _FakeSession()
    session.set_current_goal("First")
    session.set_current_goal("")
    assert session.get_current_goal() is None
    assert session.goal_state is None


def test_set_current_goal_does_not_trigger_terminator() -> None:
    # CLI / direct set_current_goal should NOT fire the terminator — only
    # the goal tools (set_complete / clear) should.
    session = _FakeSession()
    session.set_current_goal("x")
    session.set_current_goal("")
    assert session._terminated_count == 0


def test_update_goal_status_active_to_complete_fires_terminator() -> None:
    session = _FakeSession()
    session.set_current_goal("x")
    session.update_goal_status("complete")
    assert session.goal_state.status == "complete"
    assert session.get_current_goal() is None  # complete is not visible
    assert session._terminated_count == 1


def test_update_goal_status_blocked_fires_terminator() -> None:
    session = _FakeSession()
    session.set_current_goal("x")
    session.update_goal_status("blocked")
    assert session.goal_state.status == "blocked"
    assert session._terminated_count == 1


def test_update_goal_status_rejects_unknown_value() -> None:
    session = _FakeSession()
    session.set_current_goal("x")
    with pytest.raises(ValueError):
        session.update_goal_status("done")


def test_update_goal_status_when_no_goal_is_noop() -> None:
    session = _FakeSession()
    # No goal yet — should not raise.
    session.update_goal_status("complete")
    assert session.goal_state is None
    assert session._terminated_count == 0


def test_clear_goal_from_set_state_fires_terminator() -> None:
    session = _FakeSession()
    session.set_current_goal("x")
    session.clear_goal()
    assert session.goal_state is None
    assert session._terminated_count == 1


def test_clear_goal_from_empty_is_noop() -> None:
    session = _FakeSession()
    session.clear_goal()
    assert session._terminated_count == 0


def test_session_goal_to_dict() -> None:
    g = SessionGoal(objective="ship", status="active")
    assert g.to_dict() == {"objective": "ship", "status": "active"}
    assert g.is_terminal is False
    assert SessionGoal(objective="x", status="complete").is_terminal is True
    assert SessionGoal(objective="x", status="blocked").is_terminal is True


async def _call(tool, params):
    return await tool.execute(
        tool_call_id="t1",
        params=params,
    )


def test_get_goal_tool_returns_current_or_null() -> None:
    session = _FakeSession()
    tools = create_goal_tools(session)
    get_t = next(t for t in tools if t.name == "get_goal")

    result = _run(_call(get_t, {}))
    import json
    payload = json.loads(result.content[0].text)
    assert payload == {"goal": None}

    session.set_current_goal("wire it up")
    result = _run(_call(get_t, {}))
    payload = json.loads(result.content[0].text)
    assert payload == {"goal": {"objective": "wire it up", "status": "active"}}


def test_set_goal_tool_requires_objective() -> None:
    session = _FakeSession()
    tools = create_goal_tools(session)
    set_t = next(t for t in tools if t.name == "set_goal")
    import json

    result = _run(_call(set_t, {}))
    payload = json.loads(result.content[0].text)
    assert "error" in payload

    result = _run(_call(set_t, {"objective": "  ship it  "}))
    payload = json.loads(result.content[0].text)
    assert payload["goal"]["objective"] == "ship it"
    assert payload["goal"]["status"] == "active"
    # No terminator firing on plain set.
    assert session._terminated_count == 0


def test_update_goal_tool_validates_and_transitions() -> None:
    session = _FakeSession()
    session.set_current_goal("x")
    tools = create_goal_tools(session)
    update_t = next(t for t in tools if t.name == "update_goal")
    import json

    # Bad status.
    result = _run(_call(update_t, {"status": "weird"}))
    payload = json.loads(result.content[0].text)
    assert "error" in payload

    # Good transition fires the terminator.
    result = _run(_call(update_t, {"status": "complete"}))
    payload = json.loads(result.content[0].text)
    assert payload["goal"]["status"] == "complete"
    assert session._terminated_count == 1


def test_clear_goal_tool_fires_terminator() -> None:
    session = _FakeSession()
    session.set_current_goal("x")
    tools = create_goal_tools(session)
    clear_t = next(t for t in tools if t.name == "clear_goal")
    import json

    result = _run(_call(clear_t, {}))
    payload = json.loads(result.content[0].text)
    assert payload == {"goal": None, "cleared": True}
    assert session.goal_state is None
    assert session._terminated_count == 1


def test_create_goal_tools_returns_four_tools() -> None:
    session = _FakeSession()
    tools = create_goal_tools(session)
    assert [t.name for t in tools] == ["get_goal", "set_goal", "update_goal", "clear_goal"]
