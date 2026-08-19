"""
Tests for extended provider modules (all mocked).

Covers simple_options, google_shared, openai_responses_shared,
and stream function scaffolding for bedrock, vertex, azure, responses, codex.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_model(
    id_: str = "claude-3-5-sonnet-20241022",
    provider: str = "anthropic",
    api: str = "anthropic-messages",
    max_tokens: int = 8096,
    reasoning: bool = False,
    input_types: list[str] | None = None,
):
    m = MagicMock()
    m.id = id_
    m.provider = provider
    m.api = api
    m.max_tokens = max_tokens
    m.reasoning = reasoning
    m.input = input_types or ["text"]
    m.headers = {}
    m.base_url = None
    m.cost = MagicMock(cache_read=0, cache_write=0, input=0, output=0)
    return m


def _make_context(messages=None, system_prompt=None, tools=None):
    ctx = MagicMock()
    ctx.messages = messages or []
    ctx.system_prompt = system_prompt
    ctx.tools = tools or []
    return ctx


async def _collect_async(stream):
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# google_shared
# ---------------------------------------------------------------------------

class TestGoogleShared:
    def test_requires_tool_call_id_for_claude(self):
        from pi_ai.providers.google_shared import requires_tool_call_id
        assert requires_tool_call_id("claude-3-5-sonnet") is True
        assert requires_tool_call_id("gemini-1.5-pro") is False

    def test_is_thinking_part_true(self):
        from pi_ai.providers.google_shared import is_thinking_part
        assert is_thinking_part({"thought": True}) is True
        assert is_thinking_part({"thought": False}) is False
        assert is_thinking_part({}) is False

    def test_retain_thought_signature_incoming_wins(self):
        from pi_ai.providers.google_shared import retain_thought_signature
        assert retain_thought_signature("old", "new_sig") == "new_sig"

    def test_retain_thought_signature_keeps_existing_when_no_incoming(self):
        from pi_ai.providers.google_shared import retain_thought_signature
        assert retain_thought_signature("existing", None) == "existing"
        assert retain_thought_signature("existing", "") == "existing"

    def test_map_stop_reason_stop(self):
        from pi_ai.providers.google_shared import map_stop_reason
        assert map_stop_reason("STOP") == "stop"

    def test_map_stop_reason_max_tokens(self):
        from pi_ai.providers.google_shared import map_stop_reason
        assert map_stop_reason("MAX_TOKENS") == "length"

    def test_map_stop_reason_safety(self):
        from pi_ai.providers.google_shared import map_stop_reason
        assert map_stop_reason("SAFETY") == "error"

    def test_map_stop_reason_string(self):
        from pi_ai.providers.google_shared import map_stop_reason_string
        assert map_stop_reason_string("STOP") == "stop"
        assert map_stop_reason_string("MAX_TOKENS") == "length"
        assert map_stop_reason_string("OTHER") == "error"

    def test_map_tool_choice(self):
        from pi_ai.providers.google_shared import map_tool_choice
        assert map_tool_choice("auto") == "AUTO"
        assert map_tool_choice("none") == "NONE"
        assert map_tool_choice("any") == "ANY"
        assert map_tool_choice("unknown") == "AUTO"

    def test_convert_tools_empty(self):
        from pi_ai.providers.google_shared import convert_tools
        assert convert_tools([]) is None

    def test_convert_tools_basic(self):
        from pi_ai.providers.google_shared import convert_tools
        tool = MagicMock()
        tool.name = "my_tool"
        tool.description = "Does stuff"
        tool.parameters = {"type": "object", "properties": {}}
        result = convert_tools([tool])
        assert result is not None
        assert len(result) == 1
        assert result[0]["functionDeclarations"][0]["name"] == "my_tool"

    def test_convert_tools_use_parameters(self):
        from pi_ai.providers.google_shared import convert_tools
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.parameters = {}
        result = convert_tools([tool], use_parameters=True)
        assert "parameters" in result[0]["functionDeclarations"][0]

    def test_convert_tools_use_parameters_json_schema(self):
        from pi_ai.providers.google_shared import convert_tools
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.parameters = {}
        result = convert_tools([tool], use_parameters=False)
        assert "parametersJsonSchema" in result[0]["functionDeclarations"][0]


# ---------------------------------------------------------------------------
# openai_responses_shared
# ---------------------------------------------------------------------------

class TestOpenAIResponsesShared:
    def test_convert_responses_messages_normalizes_cross_provider_tool_ids(self):
        from pi_ai.providers.openai_responses_shared import convert_responses_messages
        from pi_ai.types import AssistantMessage, Context, Model, ModelCost, ToolCall

        model = Model(
            id="target",
            name="Target",
            api="openai-responses",
            provider="openai",
            base_url="https://api.openai.com/v1",
            cost=ModelCost(),
            context_window=128000,
            max_tokens=4096,
        )
        source = AssistantMessage(
            content=[ToolCall(id="call_123|item.456", name="read", arguments={"path": "x"})],
            api="anthropic-messages",
            provider="anthropic",
            model="source",
            timestamp=1,
        )

        result = convert_responses_messages(model, Context(messages=[source]))

        assert result[0]["type"] == "function_call"
        assert result[0]["call_id"] == "call_123"
        assert result[0]["id"] == "fc_item_456"

    def test_convert_responses_messages_uses_provider_safe_tool_name(self):
        from pi_ai.providers.openai_responses_shared import convert_responses_messages
        from pi_ai.types import AssistantMessage, Context, Model, ModelCost, ToolCall

        model = Model(
            id="target",
            name="Target",
            api="openai-responses",
            provider="openai",
            base_url="https://api.openai.com/v1",
            cost=ModelCost(),
            context_window=128000,
            max_tokens=4096,
        )
        source = AssistantMessage(
            content=[
                ToolCall(
                    id="call_123|fc_item_456",
                    name="sms.send_booking_link_to_caller",
                    arguments={"lead_qualified": True},
                )
            ],
            api="openai-responses",
            provider="openai",
            model="target",
            timestamp=1,
        )

        result = convert_responses_messages(
            model,
            Context(messages=[source]),
            tool_name_map={
                "sms.send_booking_link_to_caller": "sms_send_booking_link_abc123",
            },
        )

        assert result[0]["name"] == "sms_send_booking_link_abc123"

    def test_convert_responses_messages_strips_reasoning_response_status(self):
        from pi_ai.providers.openai_responses_shared import convert_responses_messages
        from pi_ai.types import (
            AssistantMessage,
            Context,
            Model,
            ModelCost,
            ThinkingContent,
        )

        model = Model(
            id="gpt-5.4-mini",
            name="GPT-5.4 mini",
            api="openai-responses",
            provider="openai",
            base_url="https://api.openai.com/v1",
            reasoning=True,
            cost=ModelCost(),
            context_window=128000,
            max_tokens=4096,
        )
        source = AssistantMessage(
            content=[
                ThinkingContent(
                    thinking="Checked the turn.",
                    thinking_signature=json.dumps(
                        {
                            "type": "reasoning",
                            "id": "rs_1",
                            "encrypted_content": "encrypted",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "Checked the turn.",
                                }
                            ],
                            "status": "completed",
                        }
                    ),
                )
            ],
            api="openai-responses",
            provider="openai",
            model="gpt-5.4-mini",
            timestamp=1,
        )

        result = convert_responses_messages(model, Context(messages=[source]))

        assert result == [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "encrypted",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Checked the turn.",
                    }
                ],
            }
        ]

    def test_convert_responses_tools(self):
        from pi_ai.providers.openai_responses_shared import convert_responses_tools
        tool = MagicMock()
        tool.name = "bash"
        tool.description = "Run bash"
        tool.parameters = {"type": "object", "properties": {"cmd": {"type": "string"}}}
        result = convert_responses_tools([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "bash"

    def test_convert_responses_tools_strict(self):
        from pi_ai.providers.openai_responses_shared import convert_responses_tools
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.parameters = {}
        result = convert_responses_tools([tool], strict=True)
        assert result[0]["strict"] is True

    def test_provider_safe_tool_names_do_not_collide_with_existing_name(self):
        from pi_ai.providers.openai_responses_shared import (
            build_responses_tool_name_map,
            provider_safe_tool_name,
        )

        dotted = MagicMock(name="dotted")
        dotted.name = "sms.send"
        already_safe = MagicMock(name="already_safe")
        already_safe.name = provider_safe_tool_name(dotted.name)

        mapping = build_responses_tool_name_map([dotted, already_safe])

        assert len(set(mapping.values())) == 2


# ---------------------------------------------------------------------------
# simple_options
# ---------------------------------------------------------------------------

class TestSimpleOptionsFull:
    def test_adjust_thinking_high(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t, budget = adjust_max_tokens_for_thinking(32000, 200000, "high")
        # high budget is 16384
        assert max_t == 32000 + 16384
        assert budget == 16384

    def test_adjust_thinking_xhigh_clamped_to_high(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t1, budget1 = adjust_max_tokens_for_thinking(32000, 200000, "xhigh")
        max_t2, budget2 = adjust_max_tokens_for_thinking(32000, 200000, "high")
        assert max_t1 == max_t2
        assert budget1 == budget2

    def test_custom_budgets_override_defaults(self):
        from pi_ai.providers.simple_options import adjust_max_tokens_for_thinking
        max_t, budget = adjust_max_tokens_for_thinking(
            32000, 200000, "low", custom_budgets={"low": 5000}
        )
        assert budget == 5000


class TestAnthropicProviderAuth:
    def test_custom_anthropic_base_url_sends_explicit_x_api_key_header(self):
        from pi_ai.providers import anthropic as anthropic_provider

        model = _make_model(provider="minimax", reasoning=True)
        model.base_url = "https://api.minimax.io/anthropic"

        with patch("pi_ai.providers.anthropic._anthropic.AsyncAnthropic") as client_cls:
            anthropic_provider._build_client(model, "secret-key")

        _, kwargs = client_cls.call_args
        assert kwargs["api_key"] == "secret-key"
        assert kwargs["base_url"] == "https://api.minimax.io/anthropic"
        assert kwargs["default_headers"]["X-Api-Key"] == "secret-key"

    def test_official_anthropic_base_url_does_not_force_x_api_key_header(self):
        from pi_ai.providers import anthropic as anthropic_provider

        model = _make_model(provider="anthropic", reasoning=True)
        model.base_url = "https://api.anthropic.com"

        with patch("pi_ai.providers.anthropic._anthropic.AsyncAnthropic") as client_cls:
            anthropic_provider._build_client(model, "secret-key")

        _, kwargs = client_cls.call_args
        assert "X-Api-Key" not in kwargs["default_headers"]


class TestAnthropicProviderResponseInstrumentation:
    class RawProviderEvent:
        pass

    class RawContentBlockStartEvent:
        index = 0
        content_block = SimpleNamespace(type="text")

    class FakeAnthropicStream:
        def __init__(self, raw_event, final_message, *, error=None):
            self.raw_event = raw_event
            self.final_message = final_message
            self.error = error
            self.iteration = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.iteration == 0:
                self.iteration += 1
                return self.raw_event
            if self.error is not None:
                raise self.error
            raise StopAsyncIteration

        async def get_final_message(self):
            return self.final_message

    class FakeStreamContext:
        def __init__(self, stream):
            self.stream = stream

        async def __aenter__(self):
            return self.stream

        async def __aexit__(self, exc_type, exc, tb):
            return False

    @staticmethod
    def _client_for(stream):
        messages = SimpleNamespace(stream=lambda **_kwargs: TestAnthropicProviderResponseInstrumentation.FakeStreamContext(stream))
        return SimpleNamespace(messages=messages)

    @staticmethod
    def _final_message():
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=7,
                output_tokens=3,
                cache_read_input_tokens=2,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
        )

    @pytest.mark.asyncio
    async def test_anthropic_emits_one_raw_response_envelope(self):
        from pi_ai.providers import anthropic as anthropic_provider
        from pi_ai.types import SimpleStreamOptions

        raw_event = self.RawProviderEvent()
        final_message = self._final_message()
        stream = self.FakeAnthropicStream(raw_event, final_message)
        responses = []

        async def on_response(response, model):
            responses.append((response, model))

        with patch(
            "pi_ai.providers.anthropic._build_client",
            return_value=(self._client_for(stream), False),
        ):
            events = [
                event
                async for event in anthropic_provider.stream_simple(
                    _make_model(provider="minimax"),
                    _make_context(),
                    SimpleStreamOptions(api_key="test-key", on_response=on_response),
                )
            ]

        assert events[-1].type == "done"
        assert len(responses) == 1
        envelope, callback_model = responses[0]
        assert envelope["events"] == [raw_event]
        assert envelope["final_message"] is final_message
        assert callback_model.provider == "minimax"

    @pytest.mark.asyncio
    async def test_claude_4_6_adaptive_level_reaches_provider_payload(self):
        from pi_ai.providers import anthropic as anthropic_provider
        from pi_ai.types import SimpleStreamOptions

        stream = self.FakeAnthropicStream(
            self.RawProviderEvent(),
            self._final_message(),
        )
        captured = {}

        def start_stream(**kwargs):
            captured.update(kwargs)
            return self.FakeStreamContext(stream)

        client = SimpleNamespace(
            messages=SimpleNamespace(stream=start_stream)
        )
        with patch(
            "pi_ai.providers.anthropic._build_client",
            return_value=(client, False),
        ):
            events = [
                event
                async for event in anthropic_provider.stream_simple(
                    _make_model(
                        id_="claude-sonnet-4-6",
                        provider="anthropic",
                        api="anthropic-messages",
                        reasoning=True,
                    ),
                    _make_context(),
                    SimpleStreamOptions(api_key="test-key", reasoning="adaptive"),
                )
            ]

        assert events[-1].type == "done"
        assert captured["thinking"] == {"type": "adaptive"}
        assert captured["output_config"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_anthropic_emits_partial_raw_response_when_stream_fails(self):
        from pi_ai.providers import anthropic as anthropic_provider
        from pi_ai.types import SimpleStreamOptions

        raw_event = self.RawProviderEvent()
        stream = self.FakeAnthropicStream(
            raw_event,
            self._final_message(),
            error=RuntimeError("stream broke after first event"),
        )
        responses = []

        async def on_response(response, _model):
            responses.append(response)

        with patch(
            "pi_ai.providers.anthropic._build_client",
            return_value=(self._client_for(stream), False),
        ):
            events = [
                event
                async for event in anthropic_provider.stream_simple(
                    _make_model(provider="minimax"),
                    _make_context(),
                    SimpleStreamOptions(api_key="test-key", on_response=on_response),
                )
            ]

        assert events[-1].type == "error"
        assert len(responses) == 1
        assert responses[0]["events"] == [raw_event]
        assert responses[0]["final_message"] is None

    @pytest.mark.asyncio
    async def test_anthropic_emits_partial_raw_response_when_consumer_closes_stream(self):
        from pi_ai.providers import anthropic as anthropic_provider
        from pi_ai.types import SimpleStreamOptions

        raw_event = self.RawContentBlockStartEvent()
        stream = self.FakeAnthropicStream(raw_event, self._final_message())
        responses = []

        async def on_response(response, _model):
            responses.append(response)

        with patch(
            "pi_ai.providers.anthropic._build_client",
            return_value=(self._client_for(stream), False),
        ):
            provider_stream = anthropic_provider.stream_simple(
                _make_model(provider="minimax"),
                _make_context(),
                SimpleStreamOptions(api_key="test-key", on_response=on_response),
            )
            assert (await anext(provider_stream)).type == "start"
            assert (await anext(provider_stream)).type == "text_start"
            await provider_stream.aclose()

        assert len(responses) == 1
        assert responses[0]["events"] == [raw_event]
        assert responses[0]["final_message"] is None

    @pytest.mark.asyncio
    async def test_minimax_retries_once_when_stream_is_idle_before_first_event(
        self, monkeypatch
    ):
        from pi_ai.providers import anthropic as anthropic_provider
        from pi_ai.types import SimpleStreamOptions

        class IdleStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()

            async def get_final_message(self):
                raise AssertionError("an idle stream has no final message")

        raw_event = self.RawProviderEvent()
        final_message = self._final_message()
        streams = [
            IdleStream(),
            self.FakeAnthropicStream(raw_event, final_message),
        ]
        calls = []

        def open_stream(**_kwargs):
            calls.append(1)
            return self.FakeStreamContext(streams.pop(0))

        client = SimpleNamespace(messages=SimpleNamespace(stream=open_stream))
        responses = []

        async def on_response(response, _model):
            responses.append(response)

        monkeypatch.setenv("TAU_PROVIDER_STREAM_IDLE_TIMEOUT_SECONDS", "0.01")
        with patch(
            "pi_ai.providers.anthropic._build_client",
            return_value=(client, False),
        ):
            events = await asyncio.wait_for(
                _collect_async(
                    anthropic_provider.stream_simple(
                        _make_model(provider="minimax"),
                        _make_context(),
                        SimpleStreamOptions(
                            api_key="test-key", on_response=on_response
                        ),
                    )
                ),
                timeout=0.5,
            )

        assert events[-1].type == "done"
        assert len(calls) == 2
        assert len(responses) == 1
        assert responses[0]["events"] == [raw_event]
        assert responses[0]["final_message"] is final_message
        assert responses[0]["idle_retries"] == 1

    @pytest.mark.asyncio
    async def test_minimax_does_not_retry_after_provider_emits_an_event(
        self, monkeypatch
    ):
        from pi_ai.providers import anthropic as anthropic_provider
        from pi_ai.types import SimpleStreamOptions

        raw_event = self.RawProviderEvent()

        class PartialIdleStream:
            def __init__(self):
                self.iteration = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.iteration == 0:
                    self.iteration += 1
                    return raw_event
                await asyncio.Event().wait()

            async def get_final_message(self):
                raise AssertionError("a partial idle stream has no final message")

        calls = []

        def open_stream(**_kwargs):
            calls.append(1)
            return self.FakeStreamContext(PartialIdleStream())

        client = SimpleNamespace(messages=SimpleNamespace(stream=open_stream))
        responses = []

        async def on_response(response, _model):
            responses.append(response)

        monkeypatch.setenv("TAU_PROVIDER_STREAM_IDLE_TIMEOUT_SECONDS", "0.01")
        with patch(
            "pi_ai.providers.anthropic._build_client",
            return_value=(client, False),
        ):
            events = await _collect_async(
                anthropic_provider.stream_simple(
                    _make_model(provider="minimax"),
                    _make_context(),
                    SimpleStreamOptions(
                        api_key="test-key", on_response=on_response
                    ),
                )
            )

        assert events[-1].type == "error"
        assert len(calls) == 1
        assert len(responses) == 1
        assert responses[0]["events"] == [raw_event]
        assert responses[0]["idle_retries"] == 0

    @pytest.mark.asyncio
    async def test_minimax_stops_after_one_retry_when_both_streams_are_idle(
        self, monkeypatch
    ):
        from pi_ai.providers import anthropic as anthropic_provider
        from pi_ai.types import SimpleStreamOptions

        class IdleStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()

            async def get_final_message(self):
                raise AssertionError("an idle stream has no final message")

        streams = [IdleStream(), IdleStream()]
        calls = []

        def open_stream(**_kwargs):
            calls.append(1)
            return self.FakeStreamContext(streams.pop(0))

        client = SimpleNamespace(messages=SimpleNamespace(stream=open_stream))
        responses = []

        async def on_response(response, _model):
            responses.append(response)

        monkeypatch.setenv("TAU_PROVIDER_STREAM_IDLE_TIMEOUT_SECONDS", "0.01")
        with patch(
            "pi_ai.providers.anthropic._build_client",
            return_value=(client, False),
        ):
            events = await asyncio.wait_for(
                _collect_async(
                    anthropic_provider.stream_simple(
                        _make_model(provider="minimax"),
                        _make_context(),
                        SimpleStreamOptions(
                            api_key="test-key", on_response=on_response
                        ),
                    )
                ),
                timeout=0.5,
            )

        assert events[-1].type == "error"
        assert "remained idle" in events[-1].error.error_message
        assert len(calls) == 2
        assert len(responses) == 1
        assert responses[0]["events"] == []
        assert responses[0]["final_message"] is None
        assert responses[0]["idle_retries"] == 1


# ---------------------------------------------------------------------------
# Provider stream functions return EventStream
# ---------------------------------------------------------------------------

class TestOpenAIResponsesParams:
    def test_openai_responses_params_leave_stream_to_sdk_call(self):
        from pi_ai.providers.openai_responses import _build_params

        params = _build_params(_make_model(api="openai-responses"), _make_context(), {})

        assert "stream" not in params

    def test_azure_openai_responses_params_leave_stream_to_sdk_call(self):
        from pi_ai.providers.azure_openai_responses import _build_params

        params = _build_params(_make_model(api="azure-openai-responses"), _make_context(), {}, "gpt-deployment")

        assert "stream" not in params

    def test_openai_codex_responses_body_omits_standard_max_output_tokens(self):
        from pi_ai.providers.openai_codex_responses import _build_request_body

        body = _build_request_body(
            _make_model(id_="gpt-5.5", provider="openai", api="openai-codex-responses"),
            _make_context(),
            {"max_tokens": 8192},
            [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        )

        assert "max_output_tokens" not in body

    @pytest.mark.asyncio
    async def test_openai_completions_on_payload_mutation_reaches_sdk_request(self):
        from pi_ai.providers.openai_completions import stream_simple
        from pi_ai.types import Context, SimpleStreamOptions, Tool

        class EmptyStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        create = AsyncMock(return_value=EmptyStream())
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        async def restrict_tools(payload, model):
            assert model.id == "MiniMax-M3"
            return {
                **payload,
                "tools": [payload["tools"][1]],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "sms.notify_staff_owner"},
                },
                "parallel_tool_calls": False,
            }

        tools = [
            Tool(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
            )
            for name in ("people_ops.calendar_check", "sms.notify_staff_owner")
        ]

        with patch(
            "pi_ai.providers.openai_completions._openai.AsyncOpenAI",
            return_value=client,
        ):
            events = [
                event
                async for event in stream_simple(
                    _make_model(
                        id_="MiniMax-M3",
                        provider="minimax",
                        api="openai-completions",
                    ),
                    Context(tools=tools),
                    SimpleStreamOptions(
                        api_key="test-key",
                        on_payload=restrict_tools,
                    ),
                )
            ]

        assert events[0].type == "start"
        request = create.await_args.kwargs
        assert [item["function"]["name"] for item in request["tools"]] == [
            "sms.notify_staff_owner"
        ]
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "sms.notify_staff_owner"},
        }
        assert request["parallel_tool_calls"] is False

    @pytest.mark.asyncio
    async def test_openrouter_gpt_5_6_sends_xhigh_reasoning_effort(self):
        from pi_ai.providers.openai_completions import stream_simple
        from pi_ai.types import Context, SimpleStreamOptions

        class EmptyStream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        create = AsyncMock(return_value=EmptyStream())
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch(
            "pi_ai.providers.openai_completions._openai.AsyncOpenAI",
            return_value=client,
        ):
            events = [
                event
                async for event in stream_simple(
                    _make_model(
                        id_="gpt-5.6-luna",
                        provider="openrouter",
                        api="openai-completions",
                        reasoning=True,
                    ),
                    Context(),
                    SimpleStreamOptions(api_key="test-key", reasoning="xhigh"),
                )
            ]

        assert events[0].type == "start"
        assert create.await_args.kwargs["reasoning_effort"] == "xhigh"

    def test_openai_codex_responses_body_always_sends_instructions(self):
        from pi_ai import Context
        from pi_ai.providers.openai_codex_responses import _build_request_body

        body = _build_request_body(
            _make_model(id_="gpt-5.5", provider="openai", api="openai-codex-responses"),
            Context(messages=[]),
            {},
            [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        )

        assert body["instructions"]

    def test_openai_codex_responses_body_omits_unsupported_temperature(self):
        from pi_ai.providers.openai_codex_responses import _build_request_body

        body = _build_request_body(
            _make_model(id_="gpt-5.5", provider="openai", api="openai-codex-responses"),
            _make_context(),
            {"temperature": 0},
            [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        )

        assert "temperature" not in body

    async def _fake_response_events(self):
        yield {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}}
        yield {"type": "response.output_text.delta", "delta": "Hello"}
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Hello"}],
            },
        }
        yield {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 2},
                },
            },
        }

    @pytest.mark.asyncio
    async def test_openai_responses_awaits_async_create_and_streams_typed_output(self):
        from pi_ai.providers.openai_responses import stream_openai_responses

        responses = SimpleNamespace()
        responses.create = AsyncMock(side_effect=lambda **_: self._fake_response_events())
        client = SimpleNamespace(responses=responses)

        with patch("openai.AsyncOpenAI", return_value=client):
            stream = stream_openai_responses(
                _make_model(id_="gpt-5.5", provider="openai", api="openai-responses"),
                _make_context(),
                {"api_key": "test-key"},
            )
            events = [event async for event in stream]

        assert responses.create.await_count == 1
        assert events[-1]["type"] == "done"
        result = await stream.result()
        assert result.stop_reason == "stop"
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello"
        assert result.usage.input == 5
        assert result.usage.cache_read == 2

    @pytest.mark.asyncio
    async def test_openai_responses_serializes_typed_reasoning_summary(self):
        from pi_ai.providers.openai_responses import stream_openai_responses

        class TypedSummary:
            text = "Checked every direct question."

            def model_dump(self):
                return {"type": "summary_text", "text": self.text}

        class TypedReasoning:
            type = "reasoning"
            id = "rs_1"
            encrypted_content = "encrypted"
            summary = [TypedSummary()]

            def model_dump(self):
                return {
                    "type": self.type,
                    "id": self.id,
                    "encrypted_content": self.encrypted_content,
                    "summary": [item.model_dump() for item in self.summary],
                }

        reasoning = TypedReasoning()

        async def response_events():
            yield {"type": "response.output_item.added", "item": reasoning}
            yield {"type": "response.output_item.done", "item": reasoning}
            yield {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            }

        responses = SimpleNamespace(
            create=AsyncMock(side_effect=lambda **_: response_events())
        )
        client = SimpleNamespace(responses=responses)

        with patch("openai.AsyncOpenAI", return_value=client):
            stream = stream_openai_responses(
                _make_model(
                    id_="gpt-5.4-mini",
                    provider="openai",
                    api="openai-responses",
                    reasoning=True,
                ),
                _make_context(),
                {"api_key": "test-key", "reasoning_effort": "low"},
            )
            events = [event async for event in stream]

        assert events[-1]["type"] == "done"
        result = await stream.result()
        assert result.content[0].thinking == "Checked every direct question."
        assert json.loads(result.content[0].thinking_signature)["summary"] == [
            {
                "type": "summary_text",
                "text": "Checked every direct question.",
            }
        ]

    @pytest.mark.asyncio
    async def test_openai_responses_emits_one_raw_response_envelope(self):
        from pi_ai.providers.openai_responses import stream_openai_responses

        raw_events = [event async for event in self._fake_response_events()]

        async def event_stream():
            for event in raw_events:
                yield event

        responses = SimpleNamespace()
        responses.create = AsyncMock(side_effect=lambda **_: event_stream())
        client = SimpleNamespace(responses=responses)
        observed = []

        async def on_response(response, model):
            observed.append((response, model))

        with patch("openai.AsyncOpenAI", return_value=client):
            stream = stream_openai_responses(
                _make_model(id_="gpt-5.5", provider="openai", api="openai-responses"),
                _make_context(),
                {"api_key": "test-key", "on_response": on_response},
            )
            events = [event async for event in stream]

        assert events[-1]["type"] == "done"
        assert len(observed) == 1
        envelope, callback_model = observed[0]
        assert envelope["events"] == raw_events
        assert envelope["final_response"] == raw_events[-1]["response"]
        assert callback_model.id == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_openai_responses_round_trips_dotted_tool_name(self):
        from pi_ai.providers.openai_responses import stream_openai_responses
        from pi_ai.types import Context, Tool

        async def tool_events(provider_tool_name):
            item = {
                "type": "function_call",
                "id": "fc_item_1",
                "call_id": "call_1",
                "name": provider_tool_name,
                "arguments": '{"lead_qualified":true,"lead_accepted_sms":true}',
            }
            yield {"type": "response.output_item.added", "item": item}
            yield {
                "type": "response.function_call_arguments.done",
                "arguments": item["arguments"],
            }
            yield {"type": "response.output_item.done", "item": item}
            yield {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "total_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 0},
                    },
                },
            }

        async def create(**request):
            provider_tool_name = request["tools"][0]["name"]
            assert "." not in provider_tool_name
            return tool_events(provider_tool_name)

        responses = SimpleNamespace(create=AsyncMock(side_effect=create))
        client = SimpleNamespace(responses=responses)
        tool = Tool(
            name="sms.send_booking_link_to_caller",
            description="Send a booking link",
            parameters={"type": "object", "properties": {}},
        )

        with patch("openai.AsyncOpenAI", return_value=client):
            stream = stream_openai_responses(
                _make_model(id_="gpt-5.4-mini", provider="openai", api="openai-responses"),
                Context(tools=[tool]),
                {"api_key": "test-key"},
            )
            events = [event async for event in stream]

        assert events[-1]["type"] == "done"
        result = await stream.result()
        assert result.content[0].name == "sms.send_booking_link_to_caller"

    @pytest.mark.asyncio
    async def test_openai_responses_emits_partial_raw_response_when_stream_fails(self):
        from pi_ai.providers.openai_responses import stream_openai_responses

        raw_event = {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_partial"},
        }

        async def failing_event_stream():
            yield raw_event
            raise RuntimeError("stream broke after first event")

        responses = SimpleNamespace()
        responses.create = AsyncMock(side_effect=lambda **_: failing_event_stream())
        client = SimpleNamespace(responses=responses)
        observed = []

        async def on_response(response, _model):
            observed.append(response)

        with patch("openai.AsyncOpenAI", return_value=client):
            stream = stream_openai_responses(
                _make_model(id_="gpt-5.5", provider="openai", api="openai-responses"),
                _make_context(),
                {"api_key": "test-key", "on_response": on_response},
            )
            events = [event async for event in stream]

        assert events[-1]["type"] == "error"
        assert len(observed) == 1
        assert observed[0]["events"] == [raw_event]
        assert observed[0]["final_response"] is None

    @pytest.mark.asyncio
    async def test_azure_openai_responses_handles_awaitable_create_and_typed_output(self):
        from pi_ai.providers.azure_openai_responses import stream_azure_openai_responses

        responses = SimpleNamespace()
        responses.create = AsyncMock(side_effect=lambda **_: self._fake_response_events())
        client = SimpleNamespace(responses=responses)

        with patch("pi_ai.providers.azure_openai_responses._create_client", return_value=client):
            stream = stream_azure_openai_responses(
                _make_model(id_="gpt-5.5", provider="azure-openai-responses", api="azure-openai-responses"),
                _make_context(),
                {"api_key": "test-key"},
            )
            events = [event async for event in stream]

        assert responses.create.await_count == 1
        assert events[-1]["type"] == "done"
        result = await stream.result()
        assert result.stop_reason == "stop"
        assert result.content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_openai_codex_responses_streams_typed_output(self):
        from pi_ai.providers.openai_codex_responses import stream_openai_codex_responses

        lines = [
            'data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}',
            'data: {"type":"response.output_text.delta","delta":"Hello"}',
            'data: {"type":"response.output_item.done","item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"Hello"}]}}',
            'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10,"input_tokens_details":{"cached_tokens":2}}}}',
            "data: [DONE]",
        ]

        class FakeResponse:
            status_code = 200

            async def aread(self):
                return b""

            async def aiter_lines(self):
                for line in lines:
                    yield line

        class FakeStreamContext:
            async def __aenter__(self):
                return FakeResponse()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, *_args, **_kwargs):
                return FakeStreamContext()

        with patch("httpx.AsyncClient", return_value=FakeClient()):
            stream = stream_openai_codex_responses(
                _make_model(id_="gpt-5.5", provider="openai", api="openai-codex-responses"),
                _make_context(),
                {"api_key": "oauth-token"},
            )
            events = [event async for event in stream]

        assert events[-1]["type"] == "done"
        result = await stream.result()
        assert result.stop_reason == "stop"
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello"
        assert result.usage.input == 5
        assert result.usage.cache_read == 2


class TestProviderStreamReturn:
    """Verify all provider stream functions return EventStreams immediately without blocking."""

    def _close_scheduled_coroutine(self, coro):
        coro.close()
        return MagicMock()

    def test_amazon_bedrock_returns_event_stream(self):
        from pi_ai.providers.amazon_bedrock import stream_bedrock
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future", side_effect=self._close_scheduled_coroutine):
            stream = stream_bedrock(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_google_vertex_returns_event_stream(self):
        from pi_ai.providers.google_vertex import stream_google_vertex
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future", side_effect=self._close_scheduled_coroutine):
            stream = stream_google_vertex(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_openai_responses_returns_event_stream(self):
        from pi_ai.providers.openai_responses import stream_openai_responses
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future", side_effect=self._close_scheduled_coroutine):
            stream = stream_openai_responses(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_azure_openai_responses_returns_event_stream(self):
        from pi_ai.providers.azure_openai_responses import stream_azure_openai_responses
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future", side_effect=self._close_scheduled_coroutine):
            stream = stream_azure_openai_responses(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_openai_codex_responses_returns_event_stream(self):
        from pi_ai.providers.openai_codex_responses import stream_openai_codex_responses
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future", side_effect=self._close_scheduled_coroutine):
            stream = stream_openai_codex_responses(_make_model(), _make_context())
            assert isinstance(stream, EventStream)

    def test_google_gemini_cli_returns_event_stream(self):
        from pi_ai.providers.google_gemini_cli import stream_google_gemini_cli
        from pi_ai.utils.event_stream import EventStream

        with patch("asyncio.ensure_future", side_effect=self._close_scheduled_coroutine):
            stream = stream_google_gemini_cli(_make_model(), _make_context())
            assert isinstance(stream, EventStream)


# ---------------------------------------------------------------------------
# register_builtins
# ---------------------------------------------------------------------------

class TestRegisterBuiltins:
    def test_register_builtins_idempotent(self):
        from pi_ai.providers.register_builtins import register_builtins, reset_api_providers
        reset_api_providers()
        register_builtins()
        register_builtins()  # Should not raise or duplicate

    def test_all_core_providers_registered(self):
        from pi_ai.api_registry import get_api_provider
        from pi_ai.providers.register_builtins import register_builtins, reset_api_providers
        reset_api_providers()
        register_builtins()

        for api in ("anthropic-messages", "openai-completions", "google-generative-ai"):
            p = get_api_provider(api)
            assert p is not None, f"Provider {api!r} not registered"
