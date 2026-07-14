"""Agentic-flow instrumentation → Clarity instrumentation endpoint (Langfuse).

The tau harness runs the real agent brain (system prompt build, provider
request/response, tool calls). None of that was observable when a live flow went
sideways. This module emits each of those as a structured event to the Clarity
instrumentation sink (``POST {base}/clarify/instrumentation/events``), which
clarity-backend forwards to Langfuse. Query them back per session/correlation.

Design:
- **Env-gated.** No-op unless a base URL + token are configured. Never a hard
  dependency, never changes agent behaviour.
- **Fire-and-forget.** Emits are scheduled on the running loop and awaited
  nowhere; a slow or failing sink can never block or crash the agent turn.
- **Safe serialization + size caps.** Arbitrary provider payloads are coerced to
  JSON-able structures and capped so a huge context can't blow the sink.

Config (first match wins):
- URL:   TAU_INSTRUMENTATION_URL | CLARITY_BACKEND_BASE_URL | CLARITY_BASE_URL
- Token: TAU_INSTRUMENTATION_TOKEN | CLARITY_API_KEY
- Correlation: CLAIRE_SESSION_ID | TAU_SESSION_ID  (tags every event)
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_MAX_FIELD_CHARS = 200_000  # keep full system prompts; cap only the absurd
_seq = itertools.count(1)


def _config() -> tuple[str, str] | None:
    base = (
        os.environ.get("TAU_INSTRUMENTATION_URL")
        or os.environ.get("CLARITY_BACKEND_BASE_URL")
        or os.environ.get("CLARITY_BASE_URL")
        or ""
    ).strip().rstrip("/")
    token = (
        os.environ.get("TAU_INSTRUMENTATION_TOKEN")
        or os.environ.get("CLARITY_API_KEY")
        or ""
    ).strip()
    if not base or not token:
        return None
    return base, token


def enabled() -> bool:
    return _config() is not None


def _session_id() -> str | None:
    return os.environ.get("CLAIRE_SESSION_ID") or os.environ.get("TAU_SESSION_ID")


def _coerce(value: Any, _depth: int = 0) -> Any:
    """Best-effort JSON-able coercion, size-capped, never raises."""
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= _MAX_FIELD_CHARS else value[:_MAX_FIELD_CHARS] + "…[truncated]"
        if _depth >= 6:
            return _coerce(str(value), _depth + 1)
        for attr in ("model_dump", "dict", "to_dict"):
            fn = getattr(value, attr, None)
            if callable(fn):
                try:
                    return _coerce(fn(), _depth + 1)
                except Exception:
                    pass
        if isinstance(value, dict):
            return {str(k): _coerce(v, _depth + 1) for k, v in list(value.items())[:200]}
        if isinstance(value, (list, tuple)):
            return [_coerce(v, _depth + 1) for v in list(value)[:200]]
        return _coerce(str(value), _depth + 1)
    except Exception:
        try:
            return str(value)[:_MAX_FIELD_CHARS]
        except Exception:
            return "<unserializable>"


async def _post(name: str, input: Any, output: Any, metadata: dict[str, Any]) -> None:
    cfg = _config()
    if cfg is None:
        return
    base, token = cfg
    body = {
        "name": name,
        "input": _coerce(input),
        "output": _coerce(output),
        "metadata": metadata,
        "level": "DEFAULT",
    }
    try:
        import httpx

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{base}/clarify/instrumentation/events",
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code >= 400:
            logger.warning(
                "tau.instrumentation emit rejected name=%s status=%s body=%s",
                name,
                resp.status_code,
                resp.text[:300],
            )
    except Exception as exc:  # noqa: BLE001 — telemetry must never break the agent
        logger.warning("tau.instrumentation emit failed name=%s err=%s", name, exc)


def emit(
    name: str,
    *,
    input: Any = None,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Schedule one instrumentation event. Fire-and-forget; never raises.

    ``name`` must be alphanumeric + ``_.:-`` per the Clarity route validator.
    """
    if _config() is None:
        return
    md: dict[str, Any] = {"source": "tau", "seq": next(_seq)}
    sid = _session_id()
    if sid:
        md["session_id"] = sid
        # Group every event from one session under a single queryable trace so
        # the whole agentic flow is retrievable via
        # GET /clarify/instrumentation/traces?correlation_id=<session_id>.
        md["correlation_id"] = sid
    if metadata:
        md.update(metadata)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_post(name, input, output, md))
    except RuntimeError:
        # No running loop (rare, e.g. sync init path) — best-effort synchronous.
        try:
            asyncio.run(_post(name, input, output, md))
        except Exception:
            pass


__all__ = ["emit", "enabled"]
