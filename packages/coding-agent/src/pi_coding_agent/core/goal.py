"""Session-scoped goal state for Tau agents.

Each session owns its own goal. A goal activated by session A does NOT
inherit into session B even when they share a project root.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

from pi_coding_agent.config import CONFIG_DIR_NAME

GoalStatus = Literal["active", "complete", "blocked"]


@dataclass
class GoalState:
    objective: str
    session_id: str = ""
    status: GoalStatus = "active"
    success_criteria: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None
    blocked_at: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def goal_state_path(project_root: str, session_id: str) -> str:
    """Session-scoped goal file path."""
    return os.path.join(
        os.path.abspath(project_root),
        CONFIG_DIR_NAME,
        "agent",
        "goals",
        f"{session_id}.json",
    )


def load_goal(project_root: str, session_id: str) -> GoalState | None:
    """Load the goal for the given session.

    Returns None when:
      - no goal file exists for this session
      - the file on disk belongs to a different session
      - the file is malformed
    """
    path = goal_state_path(project_root, session_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    # Defensive: refuse a goal stamped with a different session_id.
    owner = data.get("session_id")
    if owner and owner != session_id:
        return None
    objective = data.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        return None
    return GoalState(
        session_id=session_id,
        objective=objective.strip(),
        status=data.get("status", "active"),
        success_criteria=[
            str(item).strip() for item in data.get("success_criteria", [])
            if str(item).strip()
        ],
        required_inputs=[
            str(item).strip() for item in data.get("required_inputs", [])
            if str(item).strip()
        ],
        created_at=float(data.get("created_at") or 0.0),
        updated_at=float(data.get("updated_at") or 0.0),
        completed_at=data.get("completed_at"),
        blocked_at=data.get("blocked_at"),
        reason=data.get("reason"),
    )


def save_goal(project_root: str, session_id: str, goal: GoalState) -> GoalState:
    """Persist the goal, stamping it with the session_id."""
    now = time.time()
    if not goal.created_at:
        goal.created_at = now
    goal.updated_at = now
    goal.session_id = session_id
    path = goal_state_path(project_root, session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(goal.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp_path, path)
    return goal


def set_goal(
    project_root: str,
    *,
    session_id: str,
    objective: str,
    success_criteria: list[str] | None = None,
    required_inputs: list[str] | None = None,
) -> GoalState:
    """Create (or replace) the active goal for this session."""
    objective = objective.strip()
    goal = GoalState(
        session_id=session_id,
        objective=objective,
        status="active",
        success_criteria=[item.strip() for item in (success_criteria or []) if item.strip()],
        required_inputs=[item.strip() for item in (required_inputs or []) if item.strip()],
    )
    return save_goal(project_root, session_id, goal)


def update_goal(
    project_root: str,
    *,
    session_id: str,
    status: GoalStatus,
    reason: str | None = None,
) -> GoalState | None:
    """Update the status of this session's goal. Returns None if no goal exists."""
    goal = load_goal(project_root, session_id)
    if goal is None:
        return None
    goal.status = status
    goal.reason = reason.strip() if isinstance(reason, str) and reason.strip() else None
    now = time.time()
    if status == "complete":
        goal.completed_at = now
        goal.blocked_at = None
    elif status == "blocked":
        goal.blocked_at = now
        goal.completed_at = None
    else:
        goal.completed_at = None
        goal.blocked_at = None
    return save_goal(project_root, session_id, goal)


def clear_goal(project_root: str, session_id: str) -> None:
    """Delete this session's goal file (no-op if absent)."""
    try:
        os.remove(goal_state_path(project_root, session_id))
    except FileNotFoundError:
        return