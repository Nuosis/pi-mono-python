"""Tests for streaming functions (with mocked providers)."""
from __future__ import annotations

import pytest
from unittest.mock import patch

from pi_ai import (
    Context,
    UserMessage,
    AssistantMessage,
    SimpleStreamOptions,
    EventStart,
    EventTextStart,
    EventTextDelta,
    EventTextEnd,
    EventDone,
    EventError,
    TextContent,
    ToolResultMessage,
    Usage,
    get_model,
    stream_simple,
    complete_simple,
)


def make_user_message(text: str) -> UserMessage:
    import time
    return UserMessage(role="user", content=text, timestamp=int(time.time() * 1000))


def make_context(text: str) -> Context:
    return Context(messages=[make_user_message(text)])


def make_assistant_message(model, text: str = "Hello!") -> AssistantMessage:
    import time
    return AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text=text)],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def make_tool_result(tool_name: str, text: str) -> ToolResultMessage:
    import time
    return ToolResultMessage(
        tool_call_id=f"{tool_name}-1",
        tool_name=tool_name,
        content=[TextContent(type="text", text=text)],
        timestamp=int(time.time() * 1000),
    )


async def mock_stream_simple_fn(model, context, options=None):
    """A mock stream function that yields a simple text response."""
    import time
    partial = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )
    yield EventStart(type="start", partial=partial)

    partial_with_text = partial.model_copy(update={"content": [TextContent(type="text", text="")]})
    yield EventTextStart(type="text_start", content_index=0, partial=partial_with_text)

    partial_with_delta = partial.model_copy(update={"content": [TextContent(type="text", text="Hello!")]})
    yield EventTextDelta(type="text_delta", content_index=0, delta="Hello!", partial=partial_with_delta)

    yield EventTextEnd(type="text_end", content_index=0, content="Hello!", partial=partial_with_delta)

    final = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Hello!")],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )
    yield EventDone(type="done", reason="stop", message=final)


def test_compress_context_preserves_prompts_and_read_results():
    import pi_ai
    from pi_ai.compression import compress_context

    pi_ai.register_compressor(lambda text: f"<<C>>{text[:5]}")
    try:
        ctx = Context(
            messages=[
                make_user_message("live prompt"),
                make_tool_result("read", "fresh file content"),
                make_tool_result("bash", "large command output"),
            ]
        )

        out = compress_context(ctx)

        assert out.messages[0].content == "live prompt"
        assert out.messages[1].content[0].text == "fresh file content"
        assert out.messages[2].content[0].text.startswith("<<C>>")
    finally:
        pi_ai.unregister_compressor()


@pytest.mark.asyncio
async def test_stream_simple_collects_events():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    context = make_context("Hello")

    provider = type("P", (), {"stream_simple": staticmethod(mock_stream_simple_fn), "stream": staticmethod(mock_stream_simple_fn)})()
    with patch("pi_ai.stream.get_api_provider", return_value=provider):
        events = []
        async for event in stream_simple(model, context):
            events.append(event)

    assert len(events) > 0
    assert isinstance(events[0], EventStart)
    assert any(isinstance(e, EventDone) for e in events)


@pytest.mark.asyncio
async def test_stream_simple_normalizes_legacy_dict_events():
    import time

    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    context = make_context("Hello")
    timestamp = int(time.time() * 1000)
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "api": model.api,
        "provider": model.provider,
        "model": model.id,
        "usage": {
            "input": 1,
            "output": 2,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 3,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        },
        "stop_reason": "stop",
        "timestamp": timestamp,
    }

    async def legacy_stream(_model, _context, _options=None):
        yield {"type": "start", "partial": {**message, "content": []}}
        yield {"type": "done", "reason": "stop", "message": message}

    provider = type("P", (), {"stream_simple": staticmethod(legacy_stream), "stream": staticmethod(legacy_stream)})()
    with patch("pi_ai.stream.get_api_provider", return_value=provider):
        events = []
        async for event in stream_simple(model, context):
            events.append(event)

    assert isinstance(events[0], EventStart)
    assert isinstance(events[1], EventDone)
    assert isinstance(events[1].message, AssistantMessage)
    assert isinstance(events[1].message.content[0], TextContent)
    assert events[1].message.content[0].text == "Hello!"


@pytest.mark.asyncio
async def test_complete_simple_returns_assistant_message():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    context = make_context("Hello")

    provider = type("P", (), {"stream_simple": staticmethod(mock_stream_simple_fn), "stream": staticmethod(mock_stream_simple_fn)})()
    with patch("pi_ai.stream.get_api_provider", return_value=provider):
        result = await complete_simple(model, context)

    assert isinstance(result, AssistantMessage)
    assert result.stop_reason == "stop"
    assert any(
        isinstance(b, TextContent) and "Hello" in b.text
        for b in result.content
    )


@pytest.mark.asyncio
async def test_stream_resolves_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    context = make_context("Hello")

    captured_opts = []

    async def capturing_stream(m, ctx, opts=None):
        captured_opts.append(opts)
        async for e in mock_stream_simple_fn(m, ctx, opts):
            yield e

    provider = type("P", (), {"stream_simple": staticmethod(capturing_stream), "stream": staticmethod(capturing_stream)})()
    with patch("pi_ai.stream.get_api_provider", return_value=provider):
        async for _ in stream_simple(model, context):
            pass

    assert captured_opts[0].api_key == "test-key-123"


@pytest.mark.asyncio
async def test_stream_unknown_api_raises():
    from pi_ai.types import Model, ModelCost
    model = Model(
        id="fake-model",
        name="Fake",
        api="unknown-api",
        provider="unknown",
        base_url="https://fake.api",
        cost=ModelCost(),
        context_window=4096,
        max_tokens=1024,
    )
    context = make_context("Hello")

    with pytest.raises(ValueError, match="No stream function"):
        async for _ in stream_simple(model, context):
            pass


# ── Regression: MiniMax-style message_delta with usage=None must not crash ──────
#
# MiniMax-M3 is served over the anthropic-messages API, so its streamed response
# flows through pi_ai.providers.anthropic. It emits a `message_delta` whose
# `usage` is null. The official Anthropic SDK accumulates usage *inside* the
# stream iterator (accumulate_event runs during __anext__, before the event is
# handed to us) and dereferences usage unconditionally:
#   current_snapshot.usage.output_tokens = event.usage.output_tokens
# raising `AttributeError: 'NoneType' object has no attribute 'output_tokens'`.
# Before the guard this killed the turn with stop_reason="error". The provider
# must tolerate it and finalize with the usage already accumulated in `partial`.
#
# This test drives the REAL SDK accumulator via a faithful fake stream, so it
# reproduces the exact crash path rather than merely simulating it.


def _minimax_none_usage_events():
    """Raw SDK events ending in a message_delta whose usage is None."""
    import anthropic.types as T
    from anthropic.types.raw_message_start_event import RawMessageStartEvent
    from anthropic.types.raw_content_block_start_event import RawContentBlockStartEvent
    from anthropic.types.raw_content_block_delta_event import RawContentBlockDeltaEvent
    from anthropic.types.raw_content_block_stop_event import RawContentBlockStopEvent
    from anthropic.types.raw_message_delta_event import RawMessageDeltaEvent, Delta
    from anthropic.types.text_delta import TextDelta

    msg = T.Message.model_construct(
        id="m1", type="message", role="assistant", model="MiniMax-M3",
        content=[], stop_reason=None, stop_sequence=None,
        usage=T.Usage(input_tokens=10, output_tokens=0),
    )
    text_block = T.TextBlock.model_construct(type="text", text="", citations=None)
    return [
        RawMessageStartEvent.model_construct(type="message_start", message=msg),
        RawContentBlockStartEvent.model_construct(
            type="content_block_start", index=0, content_block=text_block),
        RawContentBlockDeltaEvent.model_construct(
            type="content_block_delta", index=0,
            delta=TextDelta.model_construct(type="text_delta", text="hi")),
        RawContentBlockStopEvent.model_construct(type="content_block_stop", index=0),
        # The offending event: message_delta with usage=None.
        RawMessageDeltaEvent.model_construct(
            type="message_delta",
            delta=Delta.model_construct(stop_reason="end_turn", stop_sequence=None),
            usage=None),
    ]


class _FakeAntStream:
    """Mimics anthropic's MessageStream: accumulates via the real SDK function
    while yielding each event through __anext__, exactly as the SDK does."""

    def __init__(self, events):
        self._events = list(events)
        self._i = 0
        self._snapshot = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        from anthropic.lib.streaming._messages import accumulate_event
        if self._i >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._i]
        self._i += 1
        # Real SDK accumulation — raises AttributeError on the None-usage delta.
        self._snapshot = accumulate_event(event=event, current_snapshot=self._snapshot)
        return event

    async def get_final_message(self):
        return self._snapshot


class _FakeAntClient:
    def __init__(self, events):
        stream = _FakeAntStream

        class _Messages:
            def stream(self_inner, **params):
                return stream(events)

        self.messages = _Messages()


def test_accumulate_event_none_usage_reproduces_sdk_crash():
    """Guard the guard: the chosen event genuinely crashes the real SDK
    accumulator, so the provider test below exercises the real failure path."""
    from anthropic.lib.streaming._messages import accumulate_event

    snapshot = None
    crashed = False
    for event in _minimax_none_usage_events():
        try:
            snapshot = accumulate_event(event=event, current_snapshot=snapshot)
        except AttributeError as exc:
            crashed = True
            assert "output_tokens" in str(exc)
    assert crashed, "expected the None-usage message_delta to crash the SDK accumulator"


@pytest.mark.asyncio
async def test_anthropic_stream_survives_none_usage_message_delta():
    from pi_ai.types import Model, ModelCost
    from pi_ai.providers.anthropic import stream_simple as anthropic_stream_simple

    model = Model(
        id="MiniMax-M3",
        name="MiniMax M3",
        api="anthropic-messages",
        provider="minimax",
        base_url="https://api.example/anthropic",
        cost=ModelCost(),
        context_window=4096,
        max_tokens=1024,
    )
    context = make_context("hi")

    events = []
    with patch(
        "pi_ai.providers.anthropic._build_client",
        return_value=(_FakeAntClient(_minimax_none_usage_events()), False),
    ):
        async for ev in anthropic_stream_simple(model, context):
            events.append(ev)

    # The turn must NOT die: the terminal event is a normal completion, not an
    # error carrying the SDK's 'NoneType'...'output_tokens' message.
    terminal = events[-1]
    assert isinstance(terminal, EventDone), (
        f"expected EventDone, got {type(terminal).__name__}: "
        f"{getattr(getattr(terminal, 'error', None), 'error_message', None)}"
    )
    assert terminal.reason == "stop"

    final = terminal.message
    assert final.stop_reason == "stop"
    assert final.error_message is None
    # Usage falls back to what was accumulated before the null-usage delta.
    assert final.usage.input == 10
    assert final.usage.output == 0
    # Streamed text still made it through.
    assert any(isinstance(e, EventTextDelta) for e in events)
