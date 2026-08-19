"""Agent-to-agent peer communication extension.

Clarity owns global agent identity and message brokering. The local Tau agent
owns its peer directory and trust decisions in <project>/.tau/settings.json
under the "a2a" key. Legacy <project>/.tau/agents.json is read only as a
backward-compatibility fallback.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent


KNOWN_STATUSES = {"approved", "pending", "blocked"}


class A2ASelf(BaseModel):
    slug: str = Field(min_length=1)
    name: str | None = None
    clarity_agent_id: str | None = None


class A2APeer(BaseModel):
    slug: str = Field(min_length=1)
    name: str | None = None
    path: str | None = None
    clarity_agent_id: str | None = None
    status: Literal["approved", "pending", "blocked"] = "pending"
    functionality: str | None = None
    description: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    delivery_mode: Literal["broker", "poll", "webhook"] = "broker"
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ADirectory(BaseModel):
    self: A2ASelf
    peers: list[A2APeer] = Field(default_factory=list)
    unknown_sender_policy: Literal["surface_for_approval", "block"] = "surface_for_approval"

    @model_validator(mode="before")
    @classmethod
    def _accept_settings_agents_key(cls, value: Any) -> Any:
        if isinstance(value, dict) and "peers" not in value and "agents" in value:
            normalized = dict(value)
            normalized["peers"] = normalized.pop("agents")
            return normalized
        return value

    def peer(self, slug_or_name: str) -> A2APeer | None:
        needle = slug_or_name.strip().lower()
        for peer in self.peers:
            if peer.slug.lower() == needle or (peer.name or "").lower() == needle:
                return peer
        return None


class SendParams(BaseModel):
    to_agent: str = Field(min_length=1)
    message: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None
    parent_message_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    payload_schema_version: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None


class PollParams(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)


class ReplyParams(BaseModel):
    message_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    payload_schema_version: str | None = None
    correlation_id: str | None = None
    idempotency_key: str | None = None


class SenderDecisionParams(BaseModel):
    slug: str = Field(min_length=1)
    name: str | None = None
    path: str | None = None
    clarity_agent_id: str | None = None
    functionality: str | None = None
    description: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    delivery_mode: Literal["broker", "poll", "webhook"] = "broker"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_slug(self) -> "SenderDecisionParams":
        self.slug = self.slug.strip()
        return self


class MarkParams(BaseModel):
    message_id: str = Field(min_length=1)
    status: Literal["pending", "processing", "completed", "failed"]
    delivery_status: Literal[
        "queued",
        "delivering",
        "delivered",
        "processing",
        "needs_approval",
        "completed",
        "failed",
        "dead_letter",
    ] | None = None
    last_error: str | None = None


def _result(payload: dict[str, Any]) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, sort_keys=True))],
        details=payload,
    )


def _project_root(ctx: Any) -> Path:
    cwd = getattr(ctx, "cwd", None) if ctx is not None else None
    return Path(cwd or os.getcwd()).resolve()


def _settings_path(root: Path) -> Path:
    return root / ".tau" / "settings.json"


def _legacy_directory_path(root: Path) -> Path:
    return root / ".tau" / "agents.json"


def _load_directory(root: Path) -> A2ADirectory:
    settings_path = _settings_path(root)
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(settings, dict) and settings.get("a2a"):
            return A2ADirectory.model_validate(settings["a2a"])
    legacy_path = _legacy_directory_path(root)
    if legacy_path.exists():
        return A2ADirectory.model_validate_json(legacy_path.read_text(encoding="utf-8"))
    raise ValueError(f"A2A directory not found in {settings_path} under key 'a2a'")


def _settings_payload(directory: A2ADirectory) -> dict[str, Any]:
    payload = directory.model_dump(mode="json")
    payload["agents"] = payload.pop("peers")
    return payload


def _write_directory(root: Path, directory: A2ADirectory) -> None:
    settings_path = _settings_path(root)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings: dict[str, Any] = {}
    if settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                settings = loaded
        except json.JSONDecodeError:
            settings = {}
    settings["a2a"] = _settings_payload(directory)
    tmp = settings_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(settings_path)


def _clarity_base() -> str:
    return os.environ.get("CLARITY_API_BASE", "https://api.claritybusinesssolutions.ca").rstrip("/")


def _clarity_secret() -> str:
    secret = os.environ.get("CLARITY_SECRET") or os.environ.get("SECRET_KEY")
    if not secret:
        raise ValueError("CLARITY_SECRET or SECRET_KEY is required for A2A Clarity calls")
    return secret


def _auth_header(body: str) -> str:
    ts = str(int(time.time()))
    sig = hmac.new(_clarity_secret().encode(), f"{ts}.{body}".encode(), hashlib.sha256).hexdigest()
    return f"Bearer {sig}.{ts}"


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True) if body is not None else ""
    data = encoded.encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"{_clarity_base()}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": _auth_header(encoded if body is not None else ""),
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8")
    return json.loads(text) if text else {}


def _schema_error(schema: dict[str, Any], payload: dict[str, Any]) -> str | None:
    """Small JSON-schema subset validator for local peer contracts."""
    if not schema:
        return None
    if schema.get("type") and schema.get("type") != "object":
        return "only object payload schemas are supported"
    required = schema.get("required") or []
    if isinstance(required, list):
        missing = [str(key) for key in required if str(key) not in payload]
        if missing:
            return f"payload missing required field(s): {', '.join(missing)}"
    properties = schema.get("properties") or {}
    if isinstance(properties, dict):
        for key, spec in properties.items():
            if key not in payload or not isinstance(spec, dict):
                continue
            expected = spec.get("type")
            value = payload[key]
            if expected == "string" and not isinstance(value, str):
                return f"payload field {key} must be string"
            if expected == "object" and not isinstance(value, dict):
                return f"payload field {key} must be object"
            if expected == "array" and not isinstance(value, list):
                return f"payload field {key} must be array"
            if expected == "boolean" and not isinstance(value, bool):
                return f"payload field {key} must be boolean"
            if expected == "number" and not isinstance(value, (int, float)):
                return f"payload field {key} must be number"
            if expected == "integer" and not isinstance(value, int):
                return f"payload field {key} must be integer"
    return None


def _thread_id(params: SendParams, directory: A2ADirectory) -> str:
    if params.thread_id:
        return params.thread_id
    seed = f"{directory.self.slug}:{params.to_agent}:{params.message}:{time.time_ns()}"
    return "a2a-" + hashlib.sha256(seed.encode()).hexdigest()[:24]


def _params_error(exc: ValidationError | ValueError) -> AgentToolResult:
    return _result({"ok": False, "error": str(exc)})


def extension_factory(pi):
    async def list_known_agents(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        del tool_call_id, params, cancel, on_update
        try:
            directory = _load_directory(_project_root(ctx))
        except ValueError as exc:
            return _params_error(exc)
        return _result(
            {
                "ok": True,
                "self": directory.self.model_dump(mode="json"),
                "peers": [peer.model_dump(mode="json") for peer in directory.peers],
                "unknown_sender_policy": directory.unknown_sender_policy,
            }
        )

    async def send_message(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        del tool_call_id, cancel, on_update
        try:
            parsed = SendParams.model_validate(params or {})
            directory = _load_directory(_project_root(ctx))
        except (ValidationError, ValueError) as exc:
            return _params_error(exc)
        peer = directory.peer(parsed.to_agent)
        if peer is None:
            return _result({"ok": False, "error": f"unknown peer: {parsed.to_agent}"})
        if peer.status != "approved":
            return _result({"ok": False, "error": f"peer {peer.slug} is {peer.status}, not approved"})
        schema_error = _schema_error(peer.input_schema, parsed.payload)
        if schema_error:
            return _result({"ok": False, "error": schema_error})

        thread_id = _thread_id(parsed, directory)
        body = {
            "session_id": thread_id,
            "thread_id": thread_id,
            "parent_message_id": parsed.parent_message_id,
            "from_agent": directory.self.slug,
            "to_agent": peer.slug,
            "message": parsed.message,
            "payload": parsed.payload,
            "payload_schema_version": parsed.payload_schema_version,
            "metadata": {
                **parsed.metadata,
                "a2a": True,
                "from_clarity_agent_id": directory.self.clarity_agent_id,
                "to_clarity_agent_id": peer.clarity_agent_id,
            },
            "message_type": "a2a",
            "kind": "request",
            "correlation_id": parsed.correlation_id,
            "idempotency_key": parsed.idempotency_key,
            "delivery_mode": peer.delivery_mode,
        }
        try:
            created = _request("POST", "/agent/agent_messages", body)
        except Exception as exc:
            return _params_error(ValueError(f"Clarity A2A send failed: {exc}"))
        return _result({"ok": True, "message": created})

    async def poll_inbox(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        del tool_call_id, cancel, on_update
        try:
            parsed = PollParams.model_validate(params or {})
            directory = _load_directory(_project_root(ctx))
        except (ValidationError, ValueError) as exc:
            return _params_error(exc)
        query = urllib.parse.urlencode({"to_agent": directory.self.slug, "limit": parsed.limit})
        try:
            inbox = _request("GET", f"/agent/agent_messages/pending?{query}")
        except Exception as exc:
            return _params_error(ValueError(f"Clarity A2A poll failed: {exc}"))
        messages = inbox.get("messages", []) if isinstance(inbox, dict) else []
        known = {peer.slug.lower(): peer for peer in directory.peers if peer.status == "approved"}
        for message in messages:
            sender = str(message.get("from_agent") or "").lower()
            message["sender_known"] = sender in known
            if sender not in known:
                message["local_decision_required"] = directory.unknown_sender_policy
        return _result({"ok": True, "messages": messages, "total": len(messages)})

    async def reply(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        del tool_call_id, cancel, on_update
        try:
            parsed = ReplyParams.model_validate(params or {})
            directory = _load_directory(_project_root(ctx))
        except (ValidationError, ValueError) as exc:
            return _params_error(exc)
        body = {
            "id": parsed.message_id,
            "from_agent": directory.self.slug,
            "message": parsed.message,
            "payload": parsed.payload,
            "metadata": parsed.metadata,
            "payload_schema_version": parsed.payload_schema_version,
            "correlation_id": parsed.correlation_id,
            "idempotency_key": parsed.idempotency_key,
        }
        try:
            created = _request("POST", "/agent/agent_messages/reply", body)
        except Exception as exc:
            return _params_error(ValueError(f"Clarity A2A reply failed: {exc}"))
        return _result({"ok": True, "message": created})

    async def approve_sender(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        del tool_call_id, cancel, on_update
        root = _project_root(ctx)
        try:
            parsed = SenderDecisionParams.model_validate(params or {})
            directory = _load_directory(root)
        except (ValidationError, ValueError) as exc:
            return _params_error(exc)
        peer = directory.peer(parsed.slug)
        if peer is None:
            peer = A2APeer(
                slug=parsed.slug,
                name=parsed.name,
                path=parsed.path,
                clarity_agent_id=parsed.clarity_agent_id,
                status="approved",
                functionality=parsed.functionality,
                description=parsed.description,
                capabilities=parsed.capabilities,
                input_schema=parsed.input_schema,
                output_schema=parsed.output_schema,
                delivery_mode=parsed.delivery_mode,
                metadata=parsed.metadata,
            )
            directory.peers.append(peer)
        else:
            peer.status = "approved"
            peer.name = parsed.name or peer.name
            peer.path = parsed.path or peer.path
            peer.clarity_agent_id = parsed.clarity_agent_id or peer.clarity_agent_id
            peer.functionality = parsed.functionality or peer.functionality
            peer.description = parsed.description or peer.description
            peer.capabilities = parsed.capabilities or peer.capabilities
            peer.input_schema = parsed.input_schema or peer.input_schema
            peer.output_schema = parsed.output_schema or peer.output_schema
            peer.delivery_mode = parsed.delivery_mode or peer.delivery_mode
            peer.metadata.update(parsed.metadata or {})
        _write_directory(root, directory)
        return _result({"ok": True, "peer": peer.model_dump(mode="json")})

    async def block_sender(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        del tool_call_id, cancel, on_update
        root = _project_root(ctx)
        try:
            parsed = SenderDecisionParams.model_validate(params or {})
            directory = _load_directory(root)
        except (ValidationError, ValueError) as exc:
            return _params_error(exc)
        peer = directory.peer(parsed.slug)
        if peer is None:
            peer = A2APeer(slug=parsed.slug, name=parsed.name, status="blocked")
            directory.peers.append(peer)
        else:
            peer.status = "blocked"
        _write_directory(root, directory)
        return _result({"ok": True, "peer": peer.model_dump(mode="json")})

    async def mark_message(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        del tool_call_id, cancel, on_update, ctx
        try:
            parsed = MarkParams.model_validate(params or {})
        except ValidationError as exc:
            return _params_error(exc)
        body = {
            "id": parsed.message_id,
            "status": parsed.status,
            "delivery_status": parsed.delivery_status,
            "last_error": parsed.last_error,
        }
        try:
            updated = _request("POST", "/agent/agent_messages/id/status", body)
        except Exception as exc:
            return _params_error(ValueError(f"Clarity A2A mark failed: {exc}"))
        return _result({"ok": True, "message": updated})

    pi.register_tool(
        "a2a_list_known_agents",
        description="List configured A2A peers from .tau/settings.json key 'a2a' (legacy .tau/agents.json is read only as fallback).",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        execute=list_known_agents,
    )
    pi.register_tool(
        "a2a_send_message",
        description="Send a typed message to an approved peer agent through Clarity agent_agent_messages.",
        parameters=SendParams.model_json_schema(),
        execute=send_message,
    )
    pi.register_tool(
        "a2a_poll_inbox",
        description="Poll Clarity for pending messages addressed to this agent and flag unknown senders.",
        parameters=PollParams.model_json_schema(),
        execute=poll_inbox,
    )
    pi.register_tool(
        "a2a_reply",
        description="Reply to an A2A message in the same Clarity thread.",
        parameters=ReplyParams.model_json_schema(),
        execute=reply,
    )
    pi.register_tool(
        "a2a_approve_sender",
        description="Approve a sender locally by adding/updating .tau/settings.json key 'a2a'.",
        parameters=SenderDecisionParams.model_json_schema(),
        execute=approve_sender,
    )
    pi.register_tool(
        "a2a_block_sender",
        description="Block a sender locally by adding/updating .tau/settings.json key 'a2a'.",
        parameters=SenderDecisionParams.model_json_schema(),
        execute=block_sender,
    )
    pi.register_tool(
        "a2a_mark_message",
        description="Update Clarity status/delivery_status for an A2A message.",
        parameters=MarkParams.model_json_schema(),
        execute=mark_message,
    )
