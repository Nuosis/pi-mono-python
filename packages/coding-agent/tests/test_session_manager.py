"""
Tests for SessionManager — mirrors packages/coding-agent/test/ session tests.
Updated to use the new per-session SessionManager API.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace

import pytest

from pi_coding_agent.core.session_manager import SessionEntry, SessionManager
from pi_coding_agent.core.settings_manager import Settings, SettingsManager
from pi_coding_agent.core.auth_storage import AuthStorage


@pytest.fixture
def session_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def session_manager(session_dir):
    """Create a fresh per-session SessionManager using the new factory."""
    return SessionManager.create(cwd=session_dir, session_dir=session_dir)


# ── SessionManager tests ──────────────────────────────────────────────────────

def test_create_session(session_manager):
    """Session ID is a UUID (matching TypeScript randomUUID() behavior)."""
    sid = session_manager.get_session_id()
    assert sid
    assert len(sid) == 36  # full UUID (TypeScript randomUUID() parity)


def test_create_session_with_label(session_dir):
    sm = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    sm.append_session_info(name="My Session")
    entries = sm.load_entries()
    assert any(e.type == "session_info" for e in entries)


def test_append_and_load_message(session_manager):
    msg = {"role": "user", "content": "Hello", "timestamp": 1234567890}
    entry_id = session_manager.append_message(msg)
    assert entry_id

    messages = session_manager.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"


def test_multiple_messages(session_manager):
    messages_to_add = [
        {"role": "user", "content": f"Message {i}", "timestamp": i}
        for i in range(5)
    ]
    for msg in messages_to_add:
        session_manager.append_message(msg)

    loaded = session_manager.get_messages()
    assert len(loaded) == 5
    for i, msg in enumerate(loaded):
        assert msg["content"] == f"Message {i}"


def test_open_restores_current_leaf_and_message_context(session_dir):
    original = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    first_id = original.append_message(
        {"role": "user", "content": "first", "timestamp": 1}
    )
    last_id = original.append_message(
        {"role": "assistant", "content": [{"type": "text", "text": "second"}], "timestamp": 2}
    )

    reopened = SessionManager.open(original.get_session_file())

    assert first_id != last_id
    assert reopened.get_leaf_id() == last_id
    assert [message["role"] for message in reopened.get_messages()] == [
        "user",
        "assistant",
    ]


def test_model_change(session_manager):
    session_manager.append_model_change("anthropic", "claude-3-5-sonnet-20241022")

    entries = session_manager.load_entries()
    model_entries = [e for e in entries if e.type == "model_change"]
    assert len(model_entries) == 1
    # Data uses camelCase (TypeScript parity)
    assert model_entries[0].data.get("modelId") == "claude-3-5-sonnet-20241022"


def test_thinking_level_change(session_manager):
    session_manager.append_thinking_level_change("high")

    entries = session_manager.load_entries()
    level_entries = [e for e in entries if e.type == "thinking_level_change"]
    assert len(level_entries) == 1
    # Data uses camelCase (TypeScript parity)
    assert level_entries[0].data.get("thinkingLevel") == "high"


def test_compaction(session_manager):
    # append_compaction(summary, first_kept_entry_id, tokens_before)
    session_manager.append_compaction("Summary text", "id1")

    entries = session_manager.load_entries()
    compact_entries = [e for e in entries if e.type == "compaction"]
    assert len(compact_entries) == 1
    assert compact_entries[0].data["summary"] == "Summary text"


def test_list_sessions(session_dir):
    """list_sessions() should return all sessions in the sessions directory."""
    managers = [
        SessionManager.create(cwd=session_dir, session_dir=session_dir)
        for _ in range(3)
    ]
    # Any of the managers shares the same sessions_dir
    sessions = managers[0].list_sessions()
    assert len(sessions) == 3


def test_delete_session(session_dir):
    sm = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    sid = sm.get_session_id()
    sm.delete_session()
    # Create a new manager to list sessions
    sm2 = SessionManager.create(cwd=session_dir, session_dir=session_dir)
    sessions = sm2.list_sessions()
    assert all(s.session_id != sid for s in sessions)


def test_set_label(session_manager):
    session_manager.set_label(session_manager.get_session_id(), "My Label")

    entries = session_manager.load_entries()
    # The entry type is "label" (TypeScript parity)
    label_entries = [e for e in entries if e.type == "label"]
    assert len(label_entries) == 1
    assert label_entries[0].data["label"] == "My Label"


# ── SettingsManager tests ─────────────────────────────────────────────────────

def test_settings_defaults():
    settings = Settings()
    assert settings.thinking_level == "off"
    assert settings.auto_compact is True


def test_settings_from_dict():
    data = {"thinking_level": "high", "auto_compact": False, "theme": "light"}
    settings = Settings.from_dict(data)
    assert settings.thinking_level == "high"
    assert settings.auto_compact is False
    assert settings.theme == "light"


def test_settings_merge():
    base = Settings(thinking_level="off", theme="dark")
    override = Settings(thinking_level="high")
    merged = base.merge(override)
    assert merged.thinking_level == "high"
    assert merged.theme == "dark"  # Not overridden


def test_settings_manager_load_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = tmpdir
        manager = SettingsManager(project_root=project_root)

        # Use the new API: save individual keys
        manager.save_project("thinking_level", "medium")
        manager.save_project("theme", "light")

        manager2 = SettingsManager(project_root=project_root)
        loaded = manager2.get()
        assert loaded.thinking_level == "medium"
        assert loaded.theme == "light"


def test_settings_manager_accepts_trailing_commas():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = tmpdir
        settings_dir = os.path.join(project_root, ".tau")
        os.makedirs(settings_dir)
        with open(os.path.join(settings_dir, "settings.json"), "w", encoding="utf-8") as f:
            f.write(
                '{\n'
                '  "name": "Devin",\n'
                '  "tools": [],\n'
                '  "extensions": ["/tmp/ext.py",],\n'
                '  "memory_enabled": true,\n'
                '}\n'
            )

        manager = SettingsManager(project_root=project_root, inherit_global=False)

        assert manager.get_agent_name() == "Devin"
        assert manager.get_tools() == []
        assert manager.get_extensions() == ["/tmp/ext.py"]
        assert manager.get().memory_enabled is True
        assert manager.drain_errors() == []


def test_settings_manager_storage_backend_project_trust_and_node_aliases():
    from pi_coding_agent import InMemorySettingsStorage
    from pi_coding_agent.core.settings_manager import SettingsManager

    storage = InMemorySettingsStorage(
        global_value=json.dumps({"theme": "dark", "transport": "sse", "compaction": {"enabled": True}}),
        project_value=json.dumps({"theme": "project", "compaction": {"keepRecentTokens": 123}}),
    )
    manager = SettingsManager.fromStorage(storage)

    assert manager.isProjectTrusted() is True
    assert manager.getTheme() == "project"
    assert manager.getCompactionSettings() == {
        "enabled": True,
        "reserveTokens": 16384,
        "keepRecentTokens": 123,
    }

    manager.setProjectTrusted(False)
    assert manager.getTheme() == "dark"
    with pytest.raises(RuntimeError, match="Project is not trusted"):
        manager.setProjectPackages(["pkg"])

    manager.setProjectTrusted(True)
    manager.setProjectPackages(["pkg"])
    assert json.loads(storage.project_value or "{}")["packages"] == ["pkg"]


def test_settings_manager_reload_keeps_previous_settings_when_storage_json_invalid():
    from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage(
        global_value=json.dumps({"theme": "dark", "extensions": ["/before.py"]}),
        project_value="{}",
    )
    manager = SettingsManager.fromStorage(storage)

    storage.global_value = "{ invalid json"
    manager.reload()

    assert manager.getTheme() == "dark"
    assert manager.getExtensionPaths() == ["/before.py"]
    errors = manager.drainErrors()
    assert len(errors) == 1
    assert errors[0]["scope"] == "global"
    assert manager.drainErrors() == []


def test_settings_manager_node_defaults_and_exports():
    import pi_coding_agent
    from pi_coding_agent.core.settings_manager import FileSettingsStorage, InMemorySettingsStorage, SettingsManager

    manager = SettingsManager.inMemory()

    assert manager.getTransport() == "auto"
    assert manager.getBlockImages() is False
    manager.setBlockImages(True)
    assert manager.getBlockImages() is True
    assert pi_coding_agent.FileSettingsStorage is FileSettingsStorage
    assert pi_coding_agent.InMemorySettingsStorage is InMemorySettingsStorage


def test_settings_manager_node_typed_setters_persist_to_storage():
    from pi_coding_agent.core.settings_manager import InMemorySettingsStorage, SettingsManager

    storage = InMemorySettingsStorage(global_value="{}", project_value="{}")
    manager = SettingsManager.fromStorage(storage)

    manager.setTheme("solarized")
    manager.setDefaultModelAndProvider("openai", "gpt-5.4-nano")
    manager.setSteeringMode("all")
    manager.setFollowUpMode("all")
    manager.setCompactionEnabled(False)
    manager.setRetryEnabled(False)
    manager.setShowImages(False)
    manager.setImageWidthCells(42)
    manager.setClearOnShrink(True)
    manager.setImageAutoResize(False)
    manager.setExtensionPaths(["extensions/*.py"])
    manager.setSkillPaths(["skills/*/SKILL.md"])
    manager.setPromptTemplatePaths(["prompts/*.md"])
    manager.setThemePaths(["themes/*.json"])
    manager.setEnabledModels(["openai/gpt-5.4-nano"])
    manager.setDoubleEscapeAction("tree")
    manager.setTreeFilterMode("labeled-only")
    manager.setEditorPaddingX(2)
    manager.setAutocompleteMaxVisible(8)
    manager.setNpmCommand(["pnpm", "exec"])
    manager.setWarnings({"permissions": False})

    assert manager.getTheme() == "solarized"
    assert manager.getDefaultProvider() == "openai"
    assert manager.getDefaultModel() == "gpt-5.4-nano"
    assert manager.getSteeringMode() == "all"
    assert manager.getFollowUpMode() == "all"
    assert manager.getCompactionEnabled() is False
    assert manager.getRetryEnabled() is False
    assert manager.getShowImages() is False
    assert manager.getImageWidthCells() == 42
    assert manager.getClearOnShrink() is True
    assert manager.getImageAutoResize() is False
    assert manager.getExtensionPaths() == ["extensions/*.py"]
    assert manager.getSkillPaths() == ["skills/*/SKILL.md"]
    assert manager.getPromptTemplatePaths() == ["prompts/*.md"]
    assert manager.getThemePaths() == ["themes/*.json"]
    assert manager.getEnabledModels() == ["openai/gpt-5.4-nano"]
    assert manager.getDoubleEscapeAction() == "tree"
    assert manager.getTreeFilterMode() == "labeled-only"
    assert manager.getEditorPaddingX() == 2
    assert manager.getAutocompleteMaxVisible() == 8
    assert manager.getNpmCommand() == ["pnpm", "exec"]
    assert manager.getProviderRetrySettings() == {
        "timeoutMs": None,
        "maxRetries": None,
        "maxRetryDelayMs": 60000,
    }
    assert manager.getWarnings() == {"permissions": False}

    raw = json.loads(storage.global_value or "{}")
    assert raw["theme"] == "solarized"
    assert raw["defaultProvider"] == "openai"
    assert raw["terminal"]["showImages"] is False
    assert raw["images"]["autoResize"] is False
    assert raw["npmCommand"] == ["pnpm", "exec"]
    assert raw["warnings"] == {"permissions": False}


# ── AuthStorage tests ──────────────────────────────────────────────────────────

def test_auth_storage_api_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = False
        auth._data = {}

        auth.set_api_key("anthropic", "sk-test-key-123")
        key = auth.get_api_key("anthropic")
        assert key == "sk-test-key-123"


def test_auth_storage_oauth_token():
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = False
        auth._data = {}

        token = {"access_token": "tok_123", "expires_at": 9999999}
        auth.set_oauth_token("github-copilot", token)

        loaded = auth.get_oauth_token("github-copilot")
        assert loaded == token


def test_auth_storage_oauth_token_wins_over_later_api_key():
    auth = AuthStorage.in_memory({})
    token = {"access_token": "oauth-token", "expires_at": 9999999999}

    auth.set_oauth_token("openai", token)
    auth.set_api_key("openai", "api-key")

    assert auth.get_oauth_token("openai") == token
    assert auth.get_api_key("openai") == "api-key"
    assert auth.is_using_oauth("openai") is True
    assert auth.resolve_api_key("openai") == "oauth-token"


def test_auth_storage_oauth_token_wins_over_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
    auth = AuthStorage.in_memory({})
    auth.set_oauth_token("openai", {"access_token": "oauth-token", "expires_at": 9999999999})

    assert auth.resolve_api_key("openai") == "oauth-token"


def test_auth_storage_delete_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = False
        auth._data = {}

        auth.set_api_key("openai", "sk-test")
        auth.delete_api_key("openai")
        assert auth.get_api_key("openai") is None


def test_auth_storage_targeted_delete_preserves_other_credential():
    auth = AuthStorage.in_memory({})
    token = {"access_token": "oauth-token", "expires_at": 9999999999}

    auth.set_oauth_token("openai", token)
    auth.set_api_key("openai", "api-key")
    auth.delete_api_key("openai")

    assert auth.get_api_key("openai") is None
    assert auth.get_oauth_token("openai") == token

    auth.set_api_key("openai", "api-key")
    auth.delete_oauth_token("openai")

    assert auth.get_oauth_token("openai") is None
    assert auth.get_api_key("openai") == "api-key"


def test_auth_storage_env_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-xyz")
    with tempfile.TemporaryDirectory() as tmpdir:
        auth = AuthStorage()
        auth.AUTH_FILE = os.path.join(tmpdir, "auth.json")
        auth.AUTH_DIR = tmpdir
        auth._loaded = True
        auth._data = {}

        resolved = auth.resolve_api_key("anthropic")
        assert resolved == "env-key-xyz"


def test_auth_storage_in_memory_backend_and_node_shape(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    auth = AuthStorage.in_memory({"openai": {"type": "api_key", "key": "stored-key"}})

    assert auth.get("openai") == {"type": "api_key", "key": "stored-key"}
    assert auth.get_api_key("openai") == "stored-key"
    assert auth.list() == ["openai"]
    assert auth.has("openai") is True
    assert auth.has_auth("openai") is True
    assert auth.get_auth_status("openai") == {"configured": True, "source": "stored"}

    auth.set_runtime_api_key("openai", "runtime-key")
    assert auth.resolve_api_key("openai") == "runtime-key"
    assert auth.get_auth_status("missing") == {"configured": False}

    auth.remove_runtime_api_key("openai")
    auth.set_fallback_resolver(lambda provider: "fallback-key" if provider == "custom" else None)
    assert auth.resolve_api_key("custom") == "fallback-key"
    assert auth.get_auth_status("custom") == {
        "configured": False,
        "source": "fallback",
        "label": "custom provider config",
    }


def test_auth_storage_file_backend_exports_and_legacy_shape(tmp_path):
    from pi_coding_agent import FileAuthStorageBackend, InMemoryAuthStorageBackend

    auth_path = tmp_path / "auth.json"
    auth_path.write_text('{"api_keys":{"anthropic":"legacy-key"}}', encoding="utf-8")
    auth = AuthStorage.create(str(auth_path))

    assert isinstance(AuthStorage.fromStorage(InMemoryAuthStorageBackend()), AuthStorage)
    assert isinstance(FileAuthStorageBackend(str(auth_path)), FileAuthStorageBackend)
    assert auth.get("anthropic") == {"type": "api_key", "key": "legacy-key"}

    auth.set("openai", {"type": "api_key", "key": "node-key"})
    encrypted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert encrypted["encrypted"] is True
    assert "node-key" not in auth_path.read_text(encoding="utf-8")
    reloaded = AuthStorage.create(str(auth_path))
    assert reloaded.get("openai") == {"type": "api_key", "key": "node-key"}

    auth.remove("anthropic")
    assert auth.get("anthropic") is None


def test_auth_storage_keeps_openai_and_anthropic_subscription_tokens_separate():
    auth = AuthStorage.in_memory()

    auth.set_oauth_token(
        "openai",
        {
            "access_token": "openai-access",
            "refresh_token": "openai-refresh",
            "access": "openai-access",
            "refresh": "openai-refresh",
            "expires": 4_102_444_800_000,
            "oauth_provider": "openai-codex",
        },
    )
    auth.set_oauth_token(
        "anthropic",
        {
            "access_token": "anthropic-access",
            "refresh_token": "anthropic-refresh",
            "access": "anthropic-access",
            "refresh": "anthropic-refresh",
            "expires": 4_102_444_800_000,
            "oauth_provider": "anthropic",
        },
    )

    assert auth.list_stored_providers() == ["anthropic", "openai"]
    assert auth.resolve_api_key("openai") == "openai-access"
    assert auth.resolve_api_key("anthropic") == "anthropic-access"
    assert auth.get_oauth_token("openai")["oauth_provider"] == "openai-codex"
    assert auth.get_oauth_token("anthropic")["oauth_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_auth_storage_refreshes_the_subscription_credential_written_by_login(tmp_path, monkeypatch):
    from pi_ai.utils.oauth.types import OAuthCredentials

    auth_path = tmp_path / "auth.json"
    auth = AuthStorage.create(str(auth_path))
    auth.set_oauth_token(
        "anthropic",
        {
            "access_token": "expired-access",
            "refresh_token": "stored-refresh",
            "expires_at": 1,
            "oauth_provider": "anthropic",
        },
    )
    calls = []

    async def refresh(provider, credentials):
        calls.append((provider, credentials))
        return OAuthCredentials(
            refresh="rotated-refresh",
            access="fresh-access",
            expires=4_102_444_800_000,
        )

    monkeypatch.setattr("pi_ai.utils.oauth.refresh_oauth_token", refresh)

    assert await auth.resolve_api_key_async("anthropic") == "fresh-access"
    assert calls[0][0] == "anthropic"
    assert calls[0][1].refresh == "stored-refresh"
    stored = AuthStorage.create(str(auth_path)).get_oauth_token("anthropic")
    assert stored["access_token"] == "fresh-access"
    assert stored["refresh_token"] == "rotated-refresh"
    assert stored["expires_at"] == 4_102_444_800
    assert stored["access"] == "fresh-access"
    assert stored["refresh"] == "rotated-refresh"
    assert stored["expires"] == 4_102_444_800_000
    assert stored["oauth_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_auth_storage_does_not_refresh_a_valid_subscription_credential(monkeypatch):
    auth = AuthStorage.in_memory()
    auth.set_oauth_token(
        "anthropic",
        {
            "access_token": "valid-access",
            "refresh_token": "stored-refresh",
            "expires_at": 4_102_444_800,
            "oauth_provider": "anthropic",
        },
    )

    async def unexpected_refresh(*_args):
        raise AssertionError("valid credentials must not be refreshed")

    monkeypatch.setattr("pi_ai.utils.oauth.refresh_oauth_token", unexpected_refresh)

    assert await auth.resolve_api_key_async("anthropic") == "valid-access"


@pytest.mark.asyncio
async def test_async_auth_resolution_preserves_runtime_override_precedence(monkeypatch):
    auth = AuthStorage.in_memory()
    auth.set_oauth_token(
        "anthropic",
        {
            "access_token": "expired-access",
            "refresh_token": "stored-refresh",
            "expires_at": 1,
            "oauth_provider": "anthropic",
        },
    )
    auth.set_runtime_api_key("anthropic", "runtime-key")

    async def unexpected_refresh(*_args):
        raise AssertionError("runtime override must bypass stored OAuth refresh")

    monkeypatch.setattr("pi_ai.utils.oauth.refresh_oauth_token", unexpected_refresh)

    assert await auth.resolve_api_key_async("anthropic") == "runtime-key"


@pytest.mark.asyncio
async def test_file_auth_serializes_rotating_refresh_across_instances(tmp_path, monkeypatch):
    from pi_ai.utils.oauth.types import OAuthCredentials

    auth_path = tmp_path / "auth.json"
    writer = AuthStorage.create(str(auth_path))
    writer.set_oauth_token(
        "anthropic",
        {
            "access_token": "expired-access",
            "refresh_token": "stored-refresh",
            "expires_at": 1,
            "oauth_provider": "anthropic",
        },
    )
    first = AuthStorage.create(str(auth_path))
    second = AuthStorage.create(str(auth_path))
    refresh_calls = 0

    async def refresh(_provider, _credentials):
        nonlocal refresh_calls
        refresh_calls += 1
        await asyncio.sleep(0.05)
        return OAuthCredentials(
            refresh="rotated-refresh",
            access="fresh-access",
            expires=4_102_444_800_000,
        )

    monkeypatch.setattr("pi_ai.utils.oauth.refresh_oauth_token", refresh)

    assert await asyncio.gather(
        first.resolve_api_key_async("anthropic"),
        second.resolve_api_key_async("anthropic"),
    ) == ["fresh-access", "fresh-access"]
    assert refresh_calls == 1


@pytest.mark.asyncio
async def test_auth_storage_never_returns_an_expired_token_when_refresh_fails(monkeypatch):
    auth = AuthStorage.in_memory()
    auth.set_oauth_token(
        "anthropic",
        {
            "access_token": "expired-access",
            "refresh_token": "stored-refresh",
            "expires_at": 1,
            "oauth_provider": "anthropic",
        },
    )

    async def failed_refresh(*_args):
        raise RuntimeError("provider rejected refresh")

    monkeypatch.setattr("pi_ai.utils.oauth.refresh_oauth_token", failed_refresh)

    with pytest.raises(RuntimeError, match="provider rejected refresh"):
        await auth.resolve_api_key_async("anthropic")
    assert auth.resolve_api_key("anthropic") is None


@pytest.mark.asyncio
async def test_agent_session_uses_the_async_subscription_resolver():
    from pi_coding_agent.core.agent_session import AgentSession

    calls = []

    class AsyncAuthStorage:
        async def resolve_api_key_async(self, provider):
            calls.append(provider)
            return "fresh-access"

        def resolve_api_key(self, _provider):
            raise AssertionError("agent session must not use the stale synchronous path")

        def get_oauth_token(self, _provider):
            return {"access_token": "fresh-access"}

        def get_api_key(self, _provider):
            return None

    session = SimpleNamespace(_auth_storage=AsyncAuthStorage())

    assert await AgentSession._resolve_api_key(session, "anthropic") == "fresh-access"
    assert calls == ["anthropic"]


def test_auth_storage_file_backend_encrypts_on_write(tmp_path):
    auth_path = tmp_path / "auth.json"
    auth = AuthStorage.create(str(auth_path))

    auth.set_api_key("openai", "secret-key")

    raw_text = auth_path.read_text(encoding="utf-8")
    raw = json.loads(raw_text)
    assert raw["encrypted"] is True
    assert "secret-key" not in raw_text
    reloaded = AuthStorage.create(str(auth_path))
    assert reloaded.get_api_key("openai") == "secret-key"
