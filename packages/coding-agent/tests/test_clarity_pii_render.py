"""The owner must never be shown `[PII:PHONE:1]` where a real value belongs.

Masking is a provider-facing measure. Once the response comes back, everything
rendered to the owner — streamed deltas, the final assistant message — has to be
restored from the *session* vault. Before this fix pi_ai's universal filter minted
a throwaway vault per call, so it could only restore tokens it had created itself;
tokens the session extension minted upstream survived all the way to the screen.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pi_ai import pii as pi_ai_pii
from pi_coding_agent import clarity_pii
from pi_coding_agent.clarity_pii import Vault
from pi_coding_agent.clarity_pii.extension import extension_factory


class _FakePi:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.commands: dict[str, Any] = {}

    def register_flag(self, _name: str, _definition: dict[str, Any]) -> None:
        return None

    def get_flag(self, _name: str) -> None:
        return None

    def on(self, name: str, handler: Any) -> None:
        self.handlers[name] = handler

    def register_command(self, name: str, definition: dict[str, Any]) -> None:
        self.commands[name] = definition["handler"]


@pytest.fixture(autouse=True)
def _isolate_active_vault():
    """Never leak an active vault between tests — it is process-global state."""
    yield
    clarity_pii.clear_active_vault()


def _start_extension(tmp_path, monkeypatch, session_id: str) -> _FakePi:
    monkeypatch.setenv("PI_AGENT_DIR", str(tmp_path))
    monkeypatch.delenv(clarity_pii.DISABLE_ENV, raising=False)
    pi = _FakePi()
    extension_factory(pi)
    pi.handlers["session_start"]({}, SimpleNamespace(session_id=session_id))
    return pi


# --------------------------------------------------------------------------- #
# The regression itself
# --------------------------------------------------------------------------- #


def test_session_vault_becomes_the_filter_vault(tmp_path, monkeypatch):
    _start_extension(tmp_path, monkeypatch, "session-render-1")
    vault = clarity_pii.get_active_vault()
    assert vault is not None

    token = vault.tokenize("Call me at +1 415-555-0132")
    assert "[PII:PHONE:1]" in token

    tokenize, detokenize = pi_ai_pii._factory()
    assert detokenize(token) == "Call me at +1 415-555-0132"


def test_streamed_text_delta_is_restored_before_render(tmp_path, monkeypatch):
    _start_extension(tmp_path, monkeypatch, "session-render-2")
    vault = clarity_pii.get_active_vault()
    masked = vault.tokenize("reach lance at +1 415-555-0132")

    _tokenize, detokenize = pi_ai_pii._factory()
    event = pi_ai_pii.detok_event(
        SimpleNamespace(
            type="text_delta",
            delta=masked,
            model_copy=lambda update: SimpleNamespace(type="text_delta", **update),
        ),
        detokenize,
    )
    assert event.delta == "reach lance at +1 415-555-0132"
    assert "[PII:" not in event.delta


def test_text_end_content_is_restored(tmp_path, monkeypatch):
    _start_extension(tmp_path, monkeypatch, "session-render-3")
    vault = clarity_pii.get_active_vault()
    masked = vault.tokenize("email marcus@claritybusinesssolutions.ca")

    _tokenize, detokenize = pi_ai_pii._factory()
    event = pi_ai_pii.detok_event(
        SimpleNamespace(
            type="text_end",
            content=masked,
            model_copy=lambda update: SimpleNamespace(type="text_end", **update),
        ),
        detokenize,
    )
    assert event.content == "email marcus@claritybusinesssolutions.ca"


def test_without_an_active_vault_session_tokens_survive(tmp_path, monkeypatch):
    """Pins the old behaviour as the *cause*: a per-call vault cannot restore what
    the session minted. If this ever passes with tokens restored, the fix has moved
    somewhere else and this test should move with it."""
    monkeypatch.delenv(clarity_pii.DISABLE_ENV, raising=False)
    clarity_pii.clear_active_vault()

    session_vault = Vault()
    masked = session_vault.tokenize("Call me at +1 415-555-0132")

    _tokenize, detokenize = pi_ai_pii._factory()
    assert detokenize(masked) == masked  # unrestorable — this was the bug on screen


# --------------------------------------------------------------------------- #
# Vault identity and persistence at the chokepoint
# --------------------------------------------------------------------------- #


def test_tokens_minted_at_the_chokepoint_are_persisted(tmp_path, monkeypatch):
    _start_extension(tmp_path, monkeypatch, "session-render-4")
    tokenize, detokenize = pi_ai_pii._factory()

    # A value the extension never saw — e.g. one that only appears in the system
    # prompt, which protect_context tokenizes.
    masked = tokenize("escalate to +1 415-555-0177")
    assert "[PII:PHONE:" in masked

    artifact = tmp_path / "pii_vault" / "session-render-4.json"
    assert artifact.exists(), "chokepoint-minted mappings must survive into the artifact"
    assert detokenize(masked) == "escalate to +1 415-555-0177"


@pytest.mark.asyncio
async def test_clear_repoints_the_filter_at_the_new_vault(tmp_path, monkeypatch):
    pi = _start_extension(tmp_path, monkeypatch, "session-render-5")
    old = clarity_pii.get_active_vault()
    old.tokenize("+1 415-555-0132")
    assert len(old) == 1

    assert await pi.commands["pii"]("clear") == "PII vault cleared."

    new = clarity_pii.get_active_vault()
    assert new is not old, "a stale vault would keep detokenizing cleared PII"
    assert len(new) == 0

    _tokenize, detokenize = pi_ai_pii._factory()
    assert detokenize("[PII:PHONE:1]") == "[PII:PHONE:1]"


@pytest.mark.asyncio
async def test_tool_call_restores_incomplete_tokens_before_execution(
    tmp_path, monkeypatch
):
    pi = _start_extension(tmp_path, monkeypatch, "session-tool-call")
    vault = clarity_pii.get_active_vault()
    email_token = vault.tokenize("jas@fvwireless.com")
    phone_token = vault.tokenize("+1 604-576-6635")

    result = await pi.handlers["tool_call"](
        {
            "input": {
                "contacts": [
                    {
                        "email": email_token.removesuffix("]"),
                        "phone": phone_token.removesuffix("]"),
                    }
                ]
            }
        },
        SimpleNamespace(session_id="session-tool-call"),
    )

    assert result == {
        "arguments": {
            "contacts": [
                {
                    "email": "jas@fvwireless.com",
                    "phone": "+1 604-576-6635",
                }
            ]
        }
    }
    assert "[PII:" not in str(result)
    assert vault.detokenize("[PII:EMAIL:999") == "[PII:EMAIL:999"


# --------------------------------------------------------------------------- #
# Kill-switch
# --------------------------------------------------------------------------- #


def test_kill_switch_yields_identity_functions(tmp_path, monkeypatch):
    _start_extension(tmp_path, monkeypatch, "session-render-6")
    monkeypatch.setenv(clarity_pii.DISABLE_ENV, "1")

    tokenize, detokenize = pi_ai_pii._factory()
    assert tokenize("+1 415-555-0132") == "+1 415-555-0132"
    assert detokenize("[PII:PHONE:1]") == "[PII:PHONE:1]"
