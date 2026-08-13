"""Tests for model registry — mirrors packages/ai/test/ model tests."""
import pytest

from pi_ai import (
    Usage,
    calculate_cost,
    get_model,
    get_models,
    get_providers,
    supports_adaptive,
    supports_xhigh,
)


def test_get_model_anthropic():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    assert model.id == "claude-3-5-sonnet-20241022"
    assert model.provider == "anthropic"
    assert model.api == "anthropic-messages"
    assert model.context_window == 200000


def test_get_model_openai():
    model = get_model("openai", "gpt-5.4-nano")
    assert model.id == "gpt-5.4-nano"
    assert model.provider == "openai"
    assert model.api == "openai-responses"
    assert model.context_window == 400_000


def test_get_model_google():
    model = get_model("google", "gemini-2.0-flash")
    assert model.id == "gemini-2.0-flash"
    assert model.provider == "google"
    assert model.api == "google-generative-ai"


def test_get_model_not_found():
    result = get_model("nonexistent", "fake-model")
    assert result is None


def test_get_providers():
    providers = get_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "google" in providers


def test_get_models_all():
    models = get_models()
    assert len(models) >= 650


def test_current_openai_registry_models():
    expected = {"gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}
    assert {m.id for m in get_models("openai")} == expected
    assert {m.id for m in get_models("azure-openai-responses")} == expected
    assert {m.id for m in get_models("openai-codex")} == expected


def test_old_direct_openai_model_names_are_removed():
    old_ids = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4.1",
        "gpt-4o",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3-codex",
        "o1",
        "o3",
        "o4-mini",
    ]
    for provider in ("openai", "azure-openai-responses", "openai-codex"):
        for model_id in old_ids:
            assert get_model(provider, model_id) is None


def test_minimax_m3_registered():
    model = get_model("minimax", "MiniMax-M3")
    assert model.id == "MiniMax-M3"
    assert model.provider == "minimax"
    # Aligned to the Node reference: MiniMax-M3 uses the Anthropic-compatible
    # endpoint (matches pi-mono-node-reference models.generated.ts).
    assert model.api == "anthropic-messages"
    assert model.base_url == "https://api.minimax.io/anthropic"
    assert model.context_window == 1_048_576
    assert model.input == ["text", "image"]

    assert get_model("minimax-cn", "MiniMax-M3") is None


def test_get_models_by_provider():
    anthropic_models = get_models("anthropic")
    assert all(m.provider == "anthropic" for m in anthropic_models)
    assert len(anthropic_models) > 0


def test_calculate_cost():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    usage = Usage(input=1_000_000, output=500_000, cache_read=0, cache_write=0)
    cost = calculate_cost(model, usage)
    # 1M input * $3/M + 0.5M output * $15/M = $3 + $7.5 = $10.5
    assert abs(cost - 10.5) < 0.01


def test_supports_xhigh_false():
    model = get_model("anthropic", "claude-3-5-sonnet-20241022")
    assert supports_xhigh(model) is False


def test_supports_xhigh_true():
    from pi_ai.types import Model, ModelCost
    model = Model(
        id="gpt-5.4-nano",
        name="GPT-5.5",
        api="openai-responses",
        provider="openai",
        base_url="https://api.openai.com/v1",
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32768,
    )
    assert supports_xhigh(model) is True


def test_supports_xhigh_for_gpt_5_6_compatible_model():
    from pi_ai.types import Model, ModelCost

    model = Model(
        id="gpt-5.6-luna",
        name="GPT-5.6 Luna",
        api="openai-completions",
        provider="openrouter",
        reasoning=True,
        base_url="https://openrouter.ai/api/v1",
        cost=ModelCost(),
        context_window=1_050_000,
        max_tokens=128_000,
    )

    assert supports_xhigh(model) is True


def test_supports_adaptive_for_claude_4_6_models_only():
    assert supports_adaptive(get_model("anthropic", "claude-opus-4-6")) is True
    assert supports_adaptive(get_model("anthropic", "claude-sonnet-4-6")) is True
    assert supports_adaptive(get_model("anthropic", "claude-3-5-sonnet-20241022")) is False
