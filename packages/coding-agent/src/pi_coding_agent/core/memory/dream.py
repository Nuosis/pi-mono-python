"""Memory dream/consolidation pass.

The dream pass is orchestration around the curator, not a separate durable
writer. It scans exact conversation memory since the last dream, asks the
curator for grounded durable candidates, and can apply those candidates through
the normal semantic memory write path.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as datetime_time, timedelta, timezone
from typing import Callable

from pi_coding_agent.config import CONFIG_DIR_NAME

from .curator import CommitDecision, Curator, Evidence
from .models import ConversationTurn
from .store import MemoryStore

EmptyLlm = Callable[[str, str], str]


@dataclass
class DreamProposal:
    title: str
    content: str
    memory_type: str
    key: str
    source_ids: list[str]
    confidence: float
    rationale: str
    verdict: str
    status: str = "proposed"


@dataclass
class DreamResult:
    project_root: str
    dry_run: bool
    since: float | None
    dreamed_at: float
    conversation_rows_scanned: int
    proposals: list[DreamProposal] = field(default_factory=list)
    written_ids: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    archive_plan_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScheduledDreamResult:
    project_root: str
    ran: bool
    reason: str
    now: float
    next_dream_at: float
    dream: DreamResult | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_llm(_system: str, _user: str) -> str:
    return '{"decisions":[]}'


def parse_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _state_path(project_root: str) -> str:
    return os.path.join(project_root, CONFIG_DIR_NAME, "memory", "dream_state.json")


def _load_state(project_root: str) -> dict:
    path = _state_path(project_root)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(project_root: str, **updates: float) -> None:
    path = _state_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = _load_state(project_root)
    data.update(updates)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def last_dream_at(project_root: str) -> float | None:
    value = _load_state(project_root).get("last_dream_at")
    return value if isinstance(value, (int, float)) else None


def next_dream_at(project_root: str) -> float | None:
    value = _load_state(project_root).get("next_dream_at")
    return value if isinstance(value, (int, float)) else None


def _next_local_246_after(now: float | None = None) -> float:
    now_dt = datetime.fromtimestamp(time.time() if now is None else now).astimezone()
    target = datetime.combine(now_dt.date(), datetime_time(hour=2, minute=46), tzinfo=now_dt.tzinfo)
    if now_dt >= target:
        target = target + timedelta(days=1)
    return target.timestamp()


def _turns_to_evidence(turns: list[ConversationTurn]) -> list[Evidence]:
    evidence: list[Evidence] = []
    for turn in turns:
        if turn.role == "user":
            kind = "user_turn"
        elif turn.role == "tool":
            kind = "tool_result"
        else:
            kind = "assistant_output"
        evidence.append(Evidence(id=turn.id, kind=kind, text=turn.content))
    return evidence


def _active_keys(store: MemoryStore) -> set[str]:
    rows = store._conn.execute(
        "SELECT key FROM semantic_memory WHERE project=? AND status='active'",
        (store.project_root,),
    ).fetchall()
    return {str(row["key"]) for row in rows}


def _to_proposal(decision: CommitDecision, existing_keys: set[str]) -> DreamProposal:
    status = "skipped_existing" if decision.key in existing_keys else "proposed"
    return DreamProposal(
        title=decision.title,
        content=decision.content,
        memory_type=decision.memory_type,
        key=decision.key,
        source_ids=decision.source_ids,
        confidence=decision.confidence,
        rationale=decision.rationale,
        verdict=decision.verdict,
        status=status,
    )


def dream_memory(
    project_root: str,
    *,
    dry_run: bool = True,
    since: float | None = None,
    llm_fn: EmptyLlm | None = None,
) -> DreamResult:
    project_root = os.path.abspath(project_root)
    store = MemoryStore(project_root)
    dreamed_at = time.time()
    scan_since = last_dream_at(project_root) if since is None else since
    turns = store.turns_since(scan_since)
    curator = Curator(
        llm_fn=llm_fn or _empty_llm,
        store=store,
        provenance=f"dream:{int(dreamed_at)}",
        verify=False,
    )
    decisions = [
        decision for decision in curator.curate(_turns_to_evidence(turns))
        if decision.verdict == "auto_commit"
    ]
    existing_keys = _active_keys(store)
    proposals = [_to_proposal(decision, existing_keys) for decision in decisions]
    writable = [
        decision for decision in decisions
        if decision.key not in existing_keys
    ]

    written_ids: list[str] = []
    if not dry_run and writable:
        written_ids = curator.commit(writable)
    if not dry_run:
        _write_state(project_root, last_dream_at=dreamed_at)

    return DreamResult(
        project_root=project_root,
        dry_run=dry_run,
        since=scan_since,
        dreamed_at=dreamed_at,
        conversation_rows_scanned=len(turns),
        proposals=proposals,
        written_ids=written_ids,
        skipped_existing=[p.key for p in proposals if p.status == "skipped_existing"],
        archive_plan_count=0,
    )


def run_scheduled_dream(
    project_root: str,
    *,
    now: float | None = None,
    llm_fn: EmptyLlm | None = None,
) -> ScheduledDreamResult:
    project_root = os.path.abspath(project_root)
    current = time.time() if now is None else now
    scheduled = next_dream_at(project_root)
    if scheduled is None:
        scheduled = _next_local_246_after(current)
        _write_state(project_root, next_dream_at=scheduled)
        return ScheduledDreamResult(project_root, False, "scheduled", current, scheduled)
    if current < scheduled:
        return ScheduledDreamResult(project_root, False, "not_due", current, scheduled)

    store = MemoryStore(project_root)
    since = last_dream_at(project_root)
    has_new_turns = bool(store.turns_since(since))
    store.close()
    next_scheduled = _next_local_246_after(current)
    if not has_new_turns:
        _write_state(project_root, next_dream_at=next_scheduled)
        return ScheduledDreamResult(project_root, False, "no_new_conversation", current, next_scheduled)

    result = dream_memory(project_root, dry_run=False, since=since, llm_fn=llm_fn)
    _write_state(project_root, next_dream_at=next_scheduled)
    return ScheduledDreamResult(project_root, True, "ran", current, next_scheduled, result)
