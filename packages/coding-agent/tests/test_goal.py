"""Tests for the session-scoped goal state and the create_goal_tools factory.

These exercise:
  - disk round-trip: set → load → clear
  - session isolation: two sessions in the same project root don't share a goal
  - tool factory: get_goal/set_goal/clear_goal produce AgentTool that wire up
  - terminator callback: invoked when a tool moves the goal to a terminal status
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_coding_agent.core.goal import (
    GoalState,
    clear_goal,
    goal_state_path,
    load_goal,
    save_goal,
    set_goal,
    update_goal,
)
from pi_coding_agent.core.tools.goal import create_goal_tools


SESSION_ID = "test-session"


def _run(coro):
    return asyncio.run(coro)


def test_goal_state_round_trip(tmp_path: Path) -> None:
    goal = set_goal(
        str(tmp_path),
        session_id=SESSION_ID,
        objective="Ship the Atlas user lookup",
        success_criteria=["GET /users/{id} returns 200", "P99 latency under 100ms"],
    )
    assert goal.status == "active"
    assert goal.objective == "Ship the Atlas user lookup"
    assert goal.completed_at is None
    assert goal.blocked_at is None
    assert goal.session_id == SESSION_ID

    loaded = load_goal(str(tmp_path), session_id=SESSION_ID)
    assert loaded is not None
    assert loaded.objective == goal.objective
    assert loaded.success_criteria == goal.success_criteria
    # Round-trip preserves the on-disk file shape
    assert Path(goal_state_path(str(tmp_path), SESSION_ID)).exists()


def test_goal_is_session_scoped(tmp_path: Path) -> None:
    set_goal(str(tmp_path), session_id="A", objective="Fix A")
    set_goal(str(tmp_path), session_id="B", objective="Fix B")

    a = load_goal(str(tmp_path), session_id="A")
    b = load_goal(str(tmp_path), session_id="B")
    assert a is not None and b is not None
    assert a.objective == "Fix A"
    assert b.objective == "Fix B"
    assert a.session_id == "A"
    assert b.session_id == "B"


def test_clear_goal_deletes_only_target_session(tmp_path: Path) -> None:
    set_goal(str(tmp_path), session_id="A", objective="A")
    set_goal(str(tmp_path), session_id="B", objective="B")

    clear_goal(str(tmp_path), session_id="A")

    assert load_goal(str(tmp_path), session_id="A") is None
    assert load_goal(str(tmp_path), session_id="B") is not None


def test_update_goal_marks_complete(tmp_path: Path) -> None:
    set_goal(str(tmp_path), session_id=SESSION_ID, objective="Finish the work")
    updated = update_goal(str(tmp_path), session_id=SESSION_ID, status="complete", reason="done")
    assert updated is not None
    assert updated.status == "complete"
    assert updated.completed_at is not None
    assert updated.blocked_at is None
    assert updated.reason == "done"


def test_update_goal_marks_blocked(tmp_path: Path) -> None:
    set_goal(str(tmp_path), session_id=SESSION_ID, objective="Resolve ambiguity")
    updated = update_goal(str(tmp_path), session_id=SESSION_ID, status="blocked", reason="need input")
    assert updated is not None
    assert updated.status == "blocked"
    assert updated.blocked_at is not None
    assert updated.completed_at is None
    assert updated.reason == "need input"


def test_update_goal_returns_none_when_absent(tmp_path: Path) -> None:
    # No prior set_goal for this session.
    assert update_goal(str(tmp_path), session_id=SESSION_ID, status="complete") is None


def test_create_goal_tools_exposes_four_tools(tmp_path: Path) -> None:
    tools = create_goal_tools(str(tmp_path), session_id=SESSION_ID)
    names = {t.name for t in tools}
    assert names == {"get_goal", "set_goal", "update_goal", "clear_goal"}


def test_set_goal_tool_persists_and_callback_fires_on_terminal(tmp_path: Path) -> None:
    calls: list[int] = {"count": 0}

    def notify() -> None:
        calls["count"] += 1

    tools = create_goal_tools(str(tmp_path), session_id=SESSION_ID, on_goal_terminated=notify)
    set_t = next(t for t in tools if t.name == "set_goal")
    update_t = next(t for t in tools if t.name == "update_goal")

    async def run_one() -> None:
        await set_t.execute(
            tool_call_id="tc1",
            params={
                "objective": "Wire the lookup",
            },
        )
        return await update_t.execute(
            tool_call_id="tc2",
            params={
                "status": "complete",
                "reason": "shipped",
            },
        )

    _run(run_one())
    # update_goal with status="complete" is a terminal transition → notify fires
    assert calls["count"] == 1
    loaded = load_goal(str(tmp_path), session_id=SESSION_ID)
    assert loaded is not None
    assert loaded.status == "complete"


def test_set_goal_tool_does_not_fire_callback_on_create(tmp_path: Path) -> None:
    # set_goal creates an active goal; it must NOT trigger the terminator
    # callback (that's reserved for terminal transitions).
    calls: list[int] = {"count": 0}

    def notify() -> None:
        calls["count"] += 1

    tools = create_goal_tools(str(tmp_path), session_id=SESSION_ID, on_goal_terminated=notify)
    set_t = next(t for t in tools if t.name == "set_goal")

    async def run_one() -> None:
        return await set_t.execute(
            tool_call_id="tc1",
            params={"objective": "Just starting"},
        )

    _run(run_one())
    assert calls["count"] == 0


def test_clear_goal_tool_fires_callback(tmp_path: Path) -> None:
    set_goal(str(tmp_path), session_id=SESSION_ID, objective="To be cleared")

    calls: list[int] = {"count": 0}

    def notify() -> None:
        calls["count"] += 1

    tools = create_goal_tools(str(tmp_path), session_id=SESSION_ID, on_goal_terminated=notify)
    clear_t = next(t for t in tools if t.name == "clear_goal")

    async def run_one() -> None:
        return await clear_t.execute(
            tool_call_id="tc2",
            params={},
        )

    _run(run_one())
    assert calls["count"] == 1
    assert load_goal(str(tmp_path), session_id=SESSION_ID) is None


def test_create_goal_tools_works_without_callback(tmp_path: Path) -> None:
    # The on_goal_terminated argument is optional. Verify it stays that way.
    tools = create_goal_tools(str(tmp_path), session_id=SESSION_ID)
    assert len(tools) == 4
