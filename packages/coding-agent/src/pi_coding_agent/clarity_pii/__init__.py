"""Tau by Clarity — first-class, on-by-default PII tokenization for tau.

Ships inside pi_coding_agent and is auto-loaded for every agent via
`resource_loader` (kill-switch: env `PI_CLARITY_PII_DISABLED=1`). Real PII never
leaves the machine; the per-session vault is persisted as a lazy,
session-referenced artifact under `pii_vault/`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .detect import detect
from .vault import ARTIFACT_SCHEMA, Vault, load_artifact, save_artifact

DISABLE_ENV = "PI_CLARITY_PII_DISABLED"

__all__ = [
    "detect",
    "Vault",
    "load_artifact",
    "save_artifact",
    "ARTIFACT_SCHEMA",
    "DISABLE_ENV",
    "is_enabled",
    "builtin_extension_path",
    "register_with_pi_ai",
    "set_active_vault",
    "clear_active_vault",
    "get_active_vault",
]

# The session vault currently in play, published by the extension. pi_ai's
# universal filter uses it so the detokenizer it hands back can restore tokens
# the *session* minted — not just the ones minted inside a single call.
_active: dict[str, Any] = {"vault": None, "on_change": None}


def set_active_vault(vault: Vault, on_change: Callable[[], None] | None = None) -> None:
    """Publish the session vault as the one pi_ai's universal filter should use.

    `on_change` (the extension's persist hook) is invoked whenever tokenizing
    through the filter mints a new mapping, so vault entries created at the
    provider chokepoint survive into the artifact.
    """
    _active["vault"] = vault
    _active["on_change"] = on_change


def clear_active_vault() -> None:
    _active["vault"] = None
    _active["on_change"] = None


def get_active_vault() -> Vault | None:
    return _active["vault"]


def is_enabled() -> bool:
    """Always-on unless explicitly disabled via the kill-switch env var."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() not in ("1", "true", "yes")


def builtin_extension_path() -> str:
    """Absolute path to the bundled extension module, for the loader to pick up."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "extension.py")


def register_with_pi_ai() -> None:
    """Install Tau by Clarity as pi_ai's universal outbound filter, so EVERY LLM call
    — agent sessions, the outer loop, evals, any direct pi_ai use — tokenizes PII
    before it reaches the provider, regardless of source.

    Registered unconditionally; the per-call factory honors the kill-switch at call
    time (identity functions when disabled), so toggling `PI_CLARITY_PII_DISABLED`
    takes effect without re-registering.

    When a session vault is active, the filter uses IT rather than a throwaway
    per-call vault. That is what lets pi_ai's response detokenizer restore tokens
    the session minted upstream — masking exists to protect the provider, not to
    show the owner `[PII:PHONE:1]` where a phone number (or a date) belongs.
    """
    try:
        from pi_ai import register_pii_filter
    except Exception:
        return
    from .vault import Vault

    def _factory():
        if not is_enabled():
            return (lambda s: s, lambda s: s)
        vault = _active["vault"]
        if vault is None:
            vault = Vault()
            return (vault.tokenize, vault.detokenize)

        on_change = _active["on_change"]

        def _tokenize(text: str) -> str:
            before = len(vault)
            out = vault.tokenize(text)
            if on_change is not None and len(vault) != before:
                try:
                    on_change()
                except Exception:
                    pass
            return out

        return (_tokenize, vault.detokenize)

    register_pii_filter(_factory)


# Self-register on import so any importer of clarity_pii installs the filter.
register_with_pi_ai()
