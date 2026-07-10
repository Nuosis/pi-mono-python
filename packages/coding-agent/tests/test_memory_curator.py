"""P2 gate — curator (writer-of-record): extraction, structural guards, grounding.

Stub LLM (no network) per the stub-vs-live-eval split: these unit-test the curator's
control logic; a live-model extraction quality eval lives in evals/."""
from __future__ import annotations

import json

from pi_coding_agent.core.memory import (
    Curator,
    DeterministicEmbeddingProvider,
    Evidence,
    MemoryStore,
)


class StubLlm:
    """Returns canned extract JSON; for the verify pass returns `supported`."""
    def __init__(self, decisions: list[dict], supported: bool = True):
        self.decisions = decisions
        self.supported = supported

    def __call__(self, system: str, user: str) -> str:
        if "Verify" in system:
            return json.dumps({"supported": self.supported})
        return json.dumps({"decisions": self.decisions})


def _store(tmp_path):
    return MemoryStore(str(tmp_path), embedder=DeterministicEmbeddingProvider())


def _ev():
    return [Evidence("e1", "tool_result", "config/net.py: MAX_RECONNECT_ATTEMPTS = 7741"),
            Evidence("e2", "assistant_output", "I think we should use 7741")]


def test_extract_and_commit(tmp_path):
    s = _store(tmp_path)
    c = Curator(StubLlm([{
        "title": "reconnect limit", "content": "MAX_RECONNECT_ATTEMPTS is 7741",
        "memory_type": "file_api", "key": "fileapi:max_reconnect",
        "source_ids": ["e1"], "verdict": "auto_commit", "confidence": 0.9}]), s)
    written = c.curate_and_commit(_ev())
    assert len(written) == 1
    hits = s.search("MAX_RECONNECT_ATTEMPTS")
    assert hits and "7741" in hits[0].memory.content
    s.close()


def test_guard_hallucinated_source_rejected(tmp_path):
    s = _store(tmp_path)
    c = Curator(StubLlm([{
        "title": "x", "content": "y", "memory_type": "decision", "key": "decision:x",
        "source_ids": ["e99"], "verdict": "auto_commit", "confidence": 0.9}]), s)  # e99 absent
    decs = c.curate(_ev())
    assert decs[0].verdict == "reject"
    assert c.commit(decs) == []
    s.close()


def test_assistant_output_ineligible(tmp_path):
    s = _store(tmp_path)
    c = Curator(StubLlm([{
        "title": "x", "content": "y", "memory_type": "decision", "key": "decision:x",
        "source_ids": ["e2"], "verdict": "auto_commit", "confidence": 0.9}]), s)  # e2 = assistant
    assert c.curate(_ev())[0].verdict == "reject"
    s.close()


def test_invalid_type_rejected(tmp_path):
    s = _store(tmp_path)
    c = Curator(StubLlm([{
        "title": "x", "content": "y", "memory_type": "gossip", "key": "k",
        "source_ids": ["e1"], "verdict": "auto_commit", "confidence": 0.9}]), s)
    assert c.curate(_ev())[0].verdict == "reject"
    s.close()


def test_verification_downgrade_to_review(tmp_path):
    s = _store(tmp_path)
    c = Curator(StubLlm([{
        "title": "reconnect", "content": "MAX_RECONNECT_ATTEMPTS is 7741",
        "memory_type": "file_api", "key": "fileapi:max_reconnect",
        "source_ids": ["e1"], "verdict": "auto_commit", "confidence": 0.9}],
        supported=False), s)
    decs = c.curate(_ev())
    assert decs[0].verdict == "needs_review"      # verifier said unsupported
    assert c.commit(decs) == []                   # not written as active
    assert s.search("MAX_RECONNECT_ATTEMPTS") == []   # not recallable
    s.close()


def test_no_direct_active_write_for_needs_review(tmp_path):
    s = _store(tmp_path)
    c = Curator(StubLlm([{
        "title": "t", "content": "c", "memory_type": "preference", "key": "pref:t",
        "source_ids": ["e1"], "verdict": "needs_review", "confidence": 0.5}]), s)
    c.curate_and_commit(_ev())
    assert s.search("c") == []                    # needs_review not active in recall
    s.close()


def test_explicit_user_preference_commits_when_llm_extracts_nothing(tmp_path):
    """Regression for live sessions where curator ran but committed_count stayed 0.

    This proves the curator itself promotes an explicit user preference into
    active semantic memory even when the model extractor returns no candidates.
    """
    s = _store(tmp_path)
    c = Curator(StubLlm([]), s)
    evidence = [
        Evidence(
            "u1",
            "user_turn",
            (
                "I want a new procedural rule: when listing ClickUp primitives, "
                "list the title or semantically rich tags, not the id."
            ),
        ),
        Evidence(
            "a1",
            "assistant_output",
            "Understood. I will use titles first.",
        ),
    ]

    written = c.curate_and_commit(evidence)

    assert len(written) == 1
    memory = s.get(written[0])
    assert memory is not None
    assert memory.memory_type == "preference"
    assert memory.status == "active"
    assert "ClickUp primitives" in memory.content
    assert "title or semantically rich tags" in memory.content
    assert memory.metadata["source_ids"] == ["u1"]
    assert s.search("ClickUp primitive title rich tags")[0].memory.id == written[0]
    s.close()


def test_explicit_procedural_instruction_commits_when_llm_extracts_nothing(tmp_path):
    s = _store(tmp_path)
    c = Curator(StubLlm([]), s)
    evidence = [
        Evidence(
            "u1",
            "user_turn",
            "When I give you a command, consider that approval; do not request approval again.",
        ),
    ]

    written = c.curate_and_commit(evidence)

    assert len(written) == 1
    memory = s.get(written[0])
    assert memory is not None
    assert memory.memory_type == "preference"
    assert "consider that approval" in memory.content
    assert memory.metadata["source_ids"] == ["u1"]
    assert s.search("approval command")[0].memory.id == written[0]
    s.close()


class AsyncStubLlm:
    def __init__(self, decisions, supported=True):
        self.decisions = decisions
        self.supported = supported

    async def __call__(self, system: str, user: str) -> str:
        if "Verify" in system:
            return json.dumps({"supported": self.supported})
        return json.dumps({"decisions": self.decisions})


async def test_acurate_and_commit(tmp_path):
    """Live write path: async curator extracts + commits (used by the running session)."""
    s = _store(tmp_path)
    c = Curator(store=s, allm_fn=AsyncStubLlm([{
        "title": "reconnect", "content": "MAX_RECONNECT_ATTEMPTS is 7741",
        "memory_type": "file_api", "key": "fileapi:max_reconnect",
        "source_ids": ["e1"], "verdict": "auto_commit", "confidence": 0.9}]))
    written = await c.acurate_and_commit(_ev())
    assert len(written) == 1
    assert "7741" in s.search("MAX_RECONNECT_ATTEMPTS")[0].memory.content
    s.close()


async def test_acurate_assistant_ineligible(tmp_path):
    s = _store(tmp_path)
    c = Curator(store=s, allm_fn=AsyncStubLlm([{
        "title": "x", "content": "y", "memory_type": "decision", "key": "decision:x",
        "source_ids": ["e2"], "verdict": "auto_commit", "confidence": 0.9}]))  # e2=assistant
    decs = await c.acurate(_ev())
    assert decs[0].verdict == "reject"
    s.close()
