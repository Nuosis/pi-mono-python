"""Tests for the agent-to-agent extension.

Covers:
  - Pydantic models: A2ASelf / A2APeer / A2ADirectory
  - directory CRUD: write to .tau/settings.json, read back, peer lookup
  - JSON-schema subset validator: required fields, type coercion
  - extension_factory: registers the expected set of tools
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pi_coding_agent.a2a.extension import (
    A2ADirectory,
    A2APeer,
    A2ASelf,
    _load_directory,
    _schema_error,
    _settings_payload,
    _write_directory,
    extension_factory,
)


def test_a2a_self_requires_slug() -> None:
    with pytest.raises(ValueError):
        A2ASelf(slug="")


def test_a2a_directory_peer_lookup_by_slug_or_name() -> None:
    directory = A2ADirectory(
        self=A2ASelf(slug="us"),
        peers=[
            A2APeer(slug="alpha", name="Alpha", clarity_agent_id="ca_1"),
            A2APeer(slug="beta", name="Beta", clarity_agent_id="ca_2"),
        ],
    )
    assert directory.peer("alpha") is not None
    assert directory.peer("ALPHA") is not None  # case-insensitive
    assert directory.peer("beta") is not None
    assert directory.peer("unknown") is None


def test_a2a_directory_persists_to_settings_json(tmp_path: Path) -> None:
    directory = A2ADirectory(
        self=A2ASelf(slug="us"),
        peers=[A2APeer(slug="alpha", name="Alpha", clarity_agent_id="ca_1", status="approved")],
        unknown_sender_policy="block",
    )
    _write_directory(tmp_path, directory)

    settings_path = tmp_path / ".tau" / "settings.json"
    assert settings_path.exists()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert "a2a" in payload
    assert payload["a2a"]["self"]["slug"] == "us"
    assert payload["a2a"]["agents"][0]["slug"] == "alpha"
    assert payload["a2a"]["unknown_sender_policy"] == "block"


def test_load_directory_round_trip(tmp_path: Path) -> None:
    original = A2ADirectory(
        self=A2ASelf(slug="us"),
        peers=[A2APeer(slug="alpha", name="Alpha", clarity_agent_id="ca_1")],
    )
    _write_directory(tmp_path, original)
    loaded = _load_directory(tmp_path)
    assert loaded.self.slug == "us"
    assert loaded.peers[0].slug == "alpha"
    assert loaded.peers[0].name == "Alpha"


def test_load_directory_falls_back_to_legacy_agents_json(tmp_path: Path) -> None:
    legacy = {
        "self": {"slug": "us", "name": "Us", "clarity_agent_id": None},
        "peers": [{"slug": "alpha", "name": "Alpha", "clarity_agent_id": "ca_1"}],
        "unknown_sender_policy": "block",
    }
    legacy_path = tmp_path / ".tau" / "agents.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = _load_directory(tmp_path)
    assert loaded.self.slug == "us"
    assert loaded.peers[0].slug == "alpha"


def test_settings_payload_omits_legacy_fields_for_new_format(tmp_path: Path) -> None:
    directory = A2ADirectory(
        self=A2ASelf(slug="us"),
        agents=[],
    )
    payload = _settings_payload(directory)
    assert "self" in payload
    assert "agents" in payload
    assert "unknown_sender_policy" in payload


def test_schema_error_required_fields() -> None:
    schema = {
        "type": "object",
        "required": ["foo", "bar"],
        "properties": {"foo": {"type": "string"}, "bar": {"type": "string"}},
    }
    assert _schema_error(schema, {"foo": "x"}) == "payload missing required field(s): bar"
    assert _schema_error(schema, {"foo": "x", "bar": "y"}) is None
    assert _schema_error(schema, {"foo": "x", "bar": "y", "baz": 1}) is None


def test_schema_error_type_validation() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
        },
    }
    assert _schema_error(schema, {}) is None  # no required fields
    assert _schema_error(schema, {"name": 123}) == "payload field name must be string"
    assert _schema_error(schema, {"count": "ten"}) == "payload field count must be integer"
    assert _schema_error(schema, {"ratio": 1}) is None  # int is a number
    assert _schema_error(schema, {"flag": "yes"}) == "payload field flag must be boolean"


def test_schema_error_rejects_non_object_schemas() -> None:
    assert _schema_error({"type": "array"}, {}) == "only object payload schemas are supported"
    assert _schema_error({"type": "string"}, {}) == "only object payload schemas are supported"


def test_schema_error_accepts_empty_schema() -> None:
    assert _schema_error({}, {"anything": "goes"}) is None


def test_extension_factory_registers_expected_tools() -> None:
    """Smoke test: extension_factory(pi) registers the documented tool set."""

    class _FakeTool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeExtension:
        def __init__(self):
            self.tools: dict[str, _FakeTool] = {}

    class _FakeAPI:
        def __init__(self):
            self._extension = _FakeExtension()

        def register_tool(self, name, description, parameters, execute, **kwargs):
            self._extension.tools[name] = _FakeTool(
                name=name,
                description=description,
                parameters=parameters,
                execute=execute,
                **kwargs,
            )

    api = _FakeAPI()
    extension_factory(api)
    expected = {
        "a2a_list_known_agents",
        "a2a_send_message",
        "a2a_poll_inbox",
        "a2a_reply",
        "a2a_approve_sender",
        "a2a_block_sender",
        "a2a_mark_message",
    }
    assert expected.issubset(set(api._extension.tools.keys()))


def test_extension_factory_schemas_reference_pydantic_models() -> None:
    """Every registered tool's parameters must be a JSON-serializable dict."""

    class _FakeExtension:
        def __init__(self):
            self.tools: dict = {}

    class _FakeAPI:
        def __init__(self):
            self._extension = _FakeExtension()

        def register_tool(self, name, description, parameters, execute, **kwargs):
            self._extension.tools[name] = {
                "name": name,
                "description": description,
                "parameters": parameters,
            }

    api = _FakeAPI()
    extension_factory(api)
    for name, tool in api._extension.tools.items():
        assert isinstance(tool["parameters"], dict), f"{name} parameters not a dict"
        assert "properties" in tool["parameters"] or tool["parameters"] == {}, (
            f"{name} parameters missing 'properties'"
        )
