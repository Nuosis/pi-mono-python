"""
Curator — the writer-of-record (P2).

The ONLY path to durable memory. Extracts atomic candidates (the DECIDED taxonomy) from
*grounded evidence*, verifies grounding, applies structural guards, and commits. The
agent never writes durable memory directly — it may only `propose` (audit), and the
curator decides. Mirrors Claire's `TierMemoryCuratorAgent` (design doc §8).

The LLM is injected (`llm_fn(system, user) -> str` returning JSON) so this is unit-
tested with a stub and eval'd with a real local/tier model.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

from .models import MemoryType, SemanticMemory
from .store import MemoryStore

LlmFn = Callable[[str, str], str]               # sync (system, user) -> json text
AsyncLlmFn = Callable[[str, str], Awaitable[str]]  # async variant for the live loop

# evidence the curator is ALLOWED to ground a durable memory in. Assistant output is
# deliberately excluded — the agent's own prose is not evidence for owner memory.
ELIGIBLE_KINDS = {"user_turn", "tool_result", "proposal"}
_VALID_TYPES = {t.value for t in MemoryType}
_EXPLICIT_PREFERENCE_MARKERS = (
    "preference",
    "procedural rule",
    "new rule",
    "going forward",
    "from now on",
    "remember that",
    "please always",
    "when i give you a command",
    "consider that approval",
    "do not request approval",
    "don't request approval",
)

Verdict = Literal["auto_commit", "needs_review", "reject"]


@dataclass
class Evidence:
    id: str
    kind: str          # user_turn | tool_result | proposal | assistant_output
    text: str


@dataclass
class CommitDecision:
    title: str
    content: str
    memory_type: str
    key: str
    source_ids: list[str]
    verdict: Verdict
    confidence: float
    rationale: str = ""


SYSTEM_EXTRACT = (
    "You curate durable memory for a coding agent. From the EVIDENCE, extract atomic, "
    "reusable memories — each one of these types only: decision, constraint, file_api, "
    "task_state, error_fix, preference. Ground every memory in specific evidence ids; "
    "never invent. Give each a canonical `key` (e.g. decision:db_choice, "
    "fileapi:config/net.py). Reply ONLY JSON: "
    '{"decisions":[{"title","content","memory_type","key","source_ids":[..],'
    '"verdict":"auto_commit|needs_review|reject","confidence":0..1,"rationale"}]}'
)
SYSTEM_VERIFY = (
    "Verify whether the CLAIM is directly supported by the cited EVIDENCE. "
    'Reply ONLY JSON: {"supported": true|false, "why": "<short>"}.'
)


def _extract_json(text: str) -> dict:
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        return {}
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return {}


class Curator:
    def __init__(self, llm_fn: LlmFn | None = None, store: MemoryStore = None, *,
                 allm_fn: AsyncLlmFn | None = None, provenance: str = "curator",
                 min_confidence: float = 0.05, verify: bool = True) -> None:
        self.llm_fn = llm_fn
        self.allm_fn = allm_fn
        self.store = store
        self.provenance = provenance
        self.min_confidence = min_confidence
        self.verify = verify

    # ── sync path (tests) ─────────────────────────────────────────────────────

    def curate(self, evidence: list[Evidence]) -> list[CommitDecision]:
        eligible, ev_by_id, packet = self._prep(evidence)
        decisions = self._parse_and_guard(
            self.llm_fn(SYSTEM_EXTRACT, f"EVIDENCE:\n{packet}"), eligible)
        for d in decisions:
            if d.verdict == "auto_commit" and self.verify:
                self._apply_verify(d, self.llm_fn(SYSTEM_VERIFY,
                                                  self._verify_prompt(d, ev_by_id)))
        decisions.extend(self._explicit_preference_fallbacks(evidence, decisions))
        return decisions

    # ── async path (live session, via stream_simple) ─────────────────────────

    async def acurate(self, evidence: list[Evidence]) -> list[CommitDecision]:
        eligible, ev_by_id, packet = self._prep(evidence)
        decisions = self._parse_and_guard(
            await self.allm_fn(SYSTEM_EXTRACT, f"EVIDENCE:\n{packet}"), eligible)
        for d in decisions:
            if d.verdict == "auto_commit" and self.verify:
                self._apply_verify(d, await self.allm_fn(
                    SYSTEM_VERIFY, self._verify_prompt(d, ev_by_id)))
        decisions.extend(self._explicit_preference_fallbacks(evidence, decisions))
        return decisions

    async def acurate_and_commit(self, evidence: list[Evidence]) -> list[str]:
        return self.commit(await self.acurate(evidence))

    # ── shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _prep(evidence: list[Evidence]):
        eligible = {e.id for e in evidence if e.kind in ELIGIBLE_KINDS}
        ev_by_id = {e.id: e for e in evidence}
        packet = "\n".join(f"[{e.id}] ({e.kind}) {e.text}" for e in evidence)
        return eligible, ev_by_id, packet

    def _parse_and_guard(self, raw: str, eligible: set[str]) -> list[CommitDecision]:
        out: list[CommitDecision] = []
        for d in _extract_json(raw).get("decisions", []):
            dec = self._coerce(d)
            if dec is None:
                continue
            out.append(self._apply_guards(dec, eligible))
        return out

    @staticmethod
    def _verify_prompt(d: CommitDecision, ev_by_id: dict[str, Evidence]) -> str:
        cited = "\n".join(f"[{sid}] {ev_by_id[sid].text}" for sid in d.source_ids
                          if sid in ev_by_id)
        return f"CLAIM: {d.title} — {d.content}\n\nEVIDENCE:\n{cited}"

    def _apply_verify(self, d: CommitDecision, raw: str) -> None:
        if not _extract_json(raw).get("supported", False):
            d.verdict = "needs_review"
            d.rationale = f"verify: unsupported; {d.rationale}"

    def commit(self, decisions: list[CommitDecision]) -> list[str]:
        """Write auto_commit → active; needs_review → stored inactive (audit); reject → drop."""
        written: list[str] = []
        for d in decisions:
            if d.verdict == "reject":
                continue
            status = "active" if d.verdict == "auto_commit" else "needs_review"
            mid = self.store.write_semantic(SemanticMemory(
                id="", project="", memory_type=d.memory_type, title=d.title,
                content=d.content, key=d.key, provenance=self.provenance, status=status,
                metadata={"source_ids": d.source_ids, "confidence": d.confidence,
                          "rationale": d.rationale}))
            if d.verdict == "auto_commit":
                written.append(mid)
        return written

    def curate_and_commit(self, evidence: list[Evidence]) -> list[str]:
        return self.commit(self.curate(evidence))

    # ── guards ────────────────────────────────────────────────────────────────

    @staticmethod
    def _coerce(d: dict) -> CommitDecision | None:
        try:
            return CommitDecision(
                title=str(d["title"]), content=str(d["content"]),
                memory_type=str(d.get("memory_type", "")),
                key=str(d.get("key", "")).strip(),
                source_ids=[str(x) for x in d.get("source_ids", [])],
                verdict=d.get("verdict", "needs_review"),
                confidence=float(d.get("confidence", 0.0)),
                rationale=str(d.get("rationale", "")))
        except (KeyError, TypeError, ValueError):
            return None

    def _apply_guards(self, d: CommitDecision, eligible: set[str]) -> CommitDecision:
        """Structural guards → downgrade to reject. (1) valid type, (2) non-empty key,
        (3) every source_id references ELIGIBLE evidence (real + not assistant-output),
        (4) confidence floor."""
        reason = None
        if d.memory_type not in _VALID_TYPES:
            reason = f"invalid memory_type {d.memory_type!r}"
        elif not d.key:
            reason = "missing canonical key"
        elif not d.source_ids or not all(sid in eligible for sid in d.source_ids):
            reason = "ungrounded: source_ids not all in eligible evidence"
        elif d.verdict == "auto_commit" and d.confidence < self.min_confidence:
            reason = "below confidence floor"
        if reason:
            d.verdict = "reject"
            d.rationale = f"guard: {reason}; {d.rationale}"
        return d

    def _explicit_preference_fallbacks(
        self,
        evidence: list[Evidence],
        existing: list[CommitDecision],
    ) -> list[CommitDecision]:
        """Promote clear user-declared preferences even if model extraction misses.

        The LLM curator remains the general writer. This deterministic path is
        deliberately narrow: it only uses direct user turns with explicit
        preference/rule language, cites only that user evidence id, and writes a
        project-scoped preference. Assistant acknowledgements are not evidence.
        """
        already_cited = {
            source_id
            for decision in existing
            if decision.memory_type == MemoryType.PREFERENCE.value
            and decision.verdict != "reject"
            for source_id in decision.source_ids
        }
        fallbacks: list[CommitDecision] = []
        for ev in evidence:
            if ev.kind != "user_turn" or ev.id in already_cited:
                continue
            text = _clean_space(ev.text)
            if not _is_explicit_preference(text):
                continue
            title = _preference_title(text)
            key = "preference:" + hashlib.sha1(text.lower().encode("utf-8")).hexdigest()[:16]
            fallbacks.append(CommitDecision(
                title=title,
                content=text,
                memory_type=MemoryType.PREFERENCE.value,
                key=key,
                source_ids=[ev.id],
                verdict="auto_commit",
                confidence=1.0,
                rationale="deterministic explicit user preference fallback",
            ))
        return fallbacks


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_explicit_preference(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _EXPLICIT_PREFERENCE_MARKERS)


def _preference_title(text: str) -> str:
    normalized = _clean_space(text)
    match = re.search(
        r"\b(?:when|always|prefer|use|list|lead|show|format)\b(.{0,90})",
        normalized,
        re.IGNORECASE,
    )
    if match:
        title = _clean_space(match.group(0))
    else:
        title = normalized
    title = title.rstrip(" .:")
    if len(title) > 96:
        title = title[:93].rstrip() + "..."
    return title or "user preference"
