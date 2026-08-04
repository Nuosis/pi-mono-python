"""Provider-independent recognition of structurally corrupt generations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any

from pi_ai.types import AssistantMessage, TextContent, ToolCall


_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ERROR_PREFIX = "provider_generation_degenerate:"


@dataclass(frozen=True)
class DegenerateGeneration:
    """Observed structural evidence that a provider completion is corrupt."""

    kind: str
    fingerprint: str
    repeat_unit_tokens: int
    repeat_count: int
    repeated_token_fraction: float

    def details(self) -> dict[str, Any]:
        return asdict(self)


def detect_degenerate_generation(
    message: AssistantMessage,
) -> DegenerateGeneration | None:
    """Recognize a length-bound response dominated by an exact repeating cycle.

    This is a semantic corruption check, not an output limit. It examines the
    provider's completed response and requires structural evidence: at least
    four consecutive copies of one exact token cycle covering at least half of
    the returned text. Non-periodic length truncation remains a valid response.
    """

    if message.stop_reason != "length":
        return None
    if any(isinstance(block, ToolCall) for block in message.content):
        return None
    text = "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )
    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    token_count = len(tokens)
    if not tokens:
        return None

    for period in range(1, token_count // 4 + 1):
        unit = tokens[-period:]
        repeat_count = 1
        cursor = token_count - period
        while cursor >= period and tokens[cursor - period : cursor] == unit:
            repeat_count += 1
            cursor -= period
        repeated_tokens = repeat_count * period
        if repeat_count >= 4 and cursor == 0:
            return None
        if repeat_count < 4 or repeated_tokens * 2 < token_count:
            continue
        fingerprint = hashlib.sha256(" ".join(unit).encode("utf-8")).hexdigest()
        return DegenerateGeneration(
            kind="exact_repetition_at_length_boundary",
            fingerprint=fingerprint,
            repeat_unit_tokens=period,
            repeat_count=repeat_count,
            repeated_token_fraction=repeated_tokens / token_count,
        )
    return None


def as_generation_error(
    message: AssistantMessage,
    failure: DegenerateGeneration,
) -> AssistantMessage:
    """Replace corrupt content with a typed terminal error safe to persist."""

    payload = json.dumps(failure.details(), sort_keys=True, separators=(",", ":"))
    return message.model_copy(
        update={
            "content": [],
            "stop_reason": "error",
            "error_message": f"{_ERROR_PREFIX}{payload}",
        }
    )


def generation_error_details(message: AssistantMessage) -> dict[str, Any] | None:
    """Read the typed failure details from a replacement error message."""

    value = message.error_message or ""
    if not value.startswith(_ERROR_PREFIX):
        return None
    try:
        parsed = json.loads(value[len(_ERROR_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
