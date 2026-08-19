"""P5 gate — MemoryIntegration bundle: recall + record_turn + compress as a pipeline."""
from __future__ import annotations

import json
import os

from pi_coding_agent.core.memory import CtxBlock, WorkingContextConfig
from pi_coding_agent.core.memory.integration import MemoryIntegration, memory_enabled


class StubLlm:
    def __call__(self, system: str, user: str) -> str:
        if "Verify" in system:
            return json.dumps({"supported": True})
        return json.dumps({"decisions": [{
            "title": "reconnect", "content": "MAX_RECONNECT_ATTEMPTS is 7741",
            "memory_type": "file_api", "key": "fileapi:max_reconnect",
            "source_ids": ["e1"], "verdict": "auto_commit", "confidence": 0.9}]})


class EmptyStubLlm:
    def __call__(self, system: str, user: str) -> str:
        if "Verify" in system:
            return json.dumps({"supported": True})
        return json.dumps({"decisions": []})


def _integration(tmp_path, llm_fn=None):
    os.environ["PI_MEMORY_EMBED"] = "deterministic"   # hermetic embeddings
    cfg = WorkingContextConfig(floor_tokens=200, ceiling_tokens=2000,
                               head_tokens=120, tail_tokens=120, recall_k=3)
    return MemoryIntegration(str(tmp_path), llm_fn=llm_fn or StubLlm(), config=cfg)


def test_env_force_flag_defaults_off():
    os.environ.pop("PI_MEMORY_ENABLED", None)
    os.environ.pop("PI_CODING_AGENT_MEMORY_ENABLED", None)
    assert memory_enabled() is False
    os.environ["PI_MEMORY_ENABLED"] = "1"
    assert memory_enabled() is True
    os.environ.pop("PI_MEMORY_ENABLED", None)
    os.environ["PI_CODING_AGENT_MEMORY_ENABLED"] = "1"
    assert memory_enabled() is True
    os.environ.pop("PI_CODING_AGENT_MEMORY_ENABLED", None)


def test_record_then_recall(tmp_path):
    from pi_coding_agent.core.memory import Evidence
    mi = _integration(tmp_path)
    written = mi.record_turn([Evidence("e1", "tool_result",
                                        "config/net.py: MAX_RECONNECT_ATTEMPTS = 7741")])
    assert written                                   # curator committed
    block = mi.recall_block("what is MAX_RECONNECT_ATTEMPTS?")
    assert block and "7741" in block                 # recall surfaces it
    mi.close()


def test_explicit_preference_from_turn_is_recallable(tmp_path):
    from pi_coding_agent.core.memory import Evidence
    mi = _integration(tmp_path, llm_fn=EmptyStubLlm())
    written = mi.record_turn([Evidence(
        "u1",
        "user_turn",
        (
            "I want a new procedural rule: when listing ClickUp primitives, "
            "list the title or semantically rich tags, not the id."
        ),
    )])

    assert written
    block = mi.recall_block("how should ClickUp primitives be listed?")
    assert block
    assert "[preference]" in block
    assert "title or semantically rich tags" in block
    mi.close()


def test_dream_dry_run_proposes_without_mutating_memory(tmp_path):
    from pi_coding_agent.core.memory import (
        ConversationTurn,
        DeterministicEmbeddingProvider,
        MemoryStore,
        dream_memory,
        last_dream_at,
    )

    os.environ["PI_MEMORY_EMBED"] = "deterministic"
    store = MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())
    store.append_turn(ConversationTurn(
        id="u1",
        project="",
        role="user",
        content=(
            "I want a new procedural rule: when listing ClickUp primitives, "
            "list the title or semantically rich tags, not the id."
        ),
        created_at=100,
    ))

    result = dream_memory(str(tmp_path), dry_run=True, since=0)

    semantic_count = store._conn.execute(
        "SELECT COUNT(*) FROM semantic_memory WHERE project=?",
        (store.project_root,),
    ).fetchone()[0]
    conversation_count = store._conn.execute(
        "SELECT COUNT(*) FROM conversation_memory WHERE project=?",
        (store.project_root,),
    ).fetchone()[0]
    assert result.dry_run is True
    assert result.conversation_rows_scanned == 1
    assert len(result.proposals) == 1
    assert result.proposals[0].memory_type == "preference"
    assert "procedural rule" in result.proposals[0].content
    assert semantic_count == 0
    assert conversation_count == 1
    assert last_dream_at(str(tmp_path)) is None
    store.close()


def test_dream_apply_writes_recallable_semantic_memory_and_state(tmp_path):
    from pi_coding_agent.core.memory import (
        ConversationTurn,
        DeterministicEmbeddingProvider,
        MemoryStore,
        build_recall_block,
        dream_memory,
        last_dream_at,
    )

    os.environ["PI_MEMORY_EMBED"] = "deterministic"
    store = MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())
    store.append_turn(ConversationTurn(
        id="u1",
        project="",
        role="user",
        content=(
            "I want a new procedural rule: when listing ClickUp primitives, "
            "list the title or semantically rich tags, not the id."
        ),
        created_at=100,
    ))

    result = dream_memory(str(tmp_path), dry_run=False, since=0)

    assert result.dry_run is False
    assert len(result.written_ids) == 1
    assert last_dream_at(str(tmp_path)) is not None
    block = build_recall_block(store, "how should ClickUp primitives be listed?")
    assert block is not None
    assert "[preference]" in block
    assert "semantically rich tags" in block
    conversation_count = store._conn.execute(
        "SELECT COUNT(*) FROM conversation_memory WHERE project=?",
        (store.project_root,),
    ).fetchone()[0]
    assert conversation_count == 1
    store.close()


def test_parse_memory_dream_command_defaults_to_dry_run():
    from pi_coding_agent.main import _parse_memory_command

    parsed = _parse_memory_command(["memory", "dream", "--json", "--since", "2026-06-29T00:00:00Z"])

    assert parsed is not None
    assert parsed["command"] == "dream"
    assert parsed["dry_run"] is True
    assert parsed["json"] is True
    assert parsed["since"] == "2026-06-29T00:00:00Z"


def test_scheduled_dream_initializes_next_local_246_without_running(tmp_path):
    from datetime import datetime

    from pi_coding_agent.core.memory import next_dream_at, run_scheduled_dream

    now = datetime(2026, 6, 29, 1, 30).timestamp()

    result = run_scheduled_dream(str(tmp_path), now=now)

    assert result.ran is False
    assert result.reason == "scheduled"
    assert next_dream_at(str(tmp_path)) == result.next_dream_at
    scheduled = datetime.fromtimestamp(result.next_dream_at)
    assert scheduled.hour == 2
    assert scheduled.minute == 46


def test_scheduled_dream_not_due_is_idle(tmp_path):
    from datetime import datetime

    from pi_coding_agent.core.memory import run_scheduled_dream

    first = datetime(2026, 6, 29, 1, 30).timestamp()
    second = datetime(2026, 6, 29, 2, 0).timestamp()
    run_scheduled_dream(str(tmp_path), now=first)

    result = run_scheduled_dream(str(tmp_path), now=second)

    assert result.ran is False
    assert result.reason == "not_due"


def test_scheduled_dream_due_without_new_conversation_advances_schedule(tmp_path):
    from datetime import datetime

    from pi_coding_agent.core.memory import run_scheduled_dream

    first = datetime(2026, 6, 29, 1, 30).timestamp()
    due = datetime(2026, 6, 29, 3, 0).timestamp()
    run_scheduled_dream(str(tmp_path), now=first)

    result = run_scheduled_dream(str(tmp_path), now=due)

    assert result.ran is False
    assert result.reason == "no_new_conversation"
    next_run = datetime.fromtimestamp(result.next_dream_at)
    assert next_run.day == 30
    assert next_run.hour == 2
    assert next_run.minute == 46


def test_scheduled_dream_due_with_new_conversation_applies_memory(tmp_path):
    from datetime import datetime

    from pi_coding_agent.core.memory import (
        ConversationTurn,
        DeterministicEmbeddingProvider,
        MemoryStore,
        build_recall_block,
        run_scheduled_dream,
    )

    os.environ["PI_MEMORY_EMBED"] = "deterministic"
    first = datetime(2026, 6, 29, 1, 30).timestamp()
    due = datetime(2026, 6, 29, 3, 0).timestamp()
    run_scheduled_dream(str(tmp_path), now=first)
    store = MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())
    store.append_turn(ConversationTurn(
        id="u1",
        project="",
        role="user",
        content="When I give you a command, consider that approval; do not request approval again.",
        created_at=first + 60,
    ))

    result = run_scheduled_dream(str(tmp_path), now=due)

    assert result.ran is True
    assert result.reason == "ran"
    assert result.dream is not None
    assert len(result.dream.written_ids) == 1
    block = build_recall_block(store, "approval command")
    assert block is not None
    assert "consider that approval" in block
    store.close()


# test_compress_uses_store_recall removed: working-context positional compression
# was dropped (design §12); active compression (Headroom/CCR) replaces it.
