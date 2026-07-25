import asyncio

import httpx
import pytest

from pi_coding_agent.core import clarity_instrumentation


@pytest.mark.asyncio
async def test_flush_waits_for_scheduled_native_trace(monkeypatch):
    release = asyncio.Event()
    posted: list[str] = []

    async def fake_post(name, input, output, metadata):
        await release.wait()
        posted.append(name)

    monkeypatch.setenv("TAU_INSTRUMENTATION_URL", "https://clarity.example.test")
    monkeypatch.setenv("TAU_INSTRUMENTATION_TOKEN", "test-token")
    monkeypatch.setattr(clarity_instrumentation, "_post", fake_post)

    task = clarity_instrumentation.emit("tau.provider_response", output={"ok": True})

    assert task is not None
    assert task.done() is False
    release.set()
    await clarity_instrumentation.flush(timeout_seconds=1.0)

    assert posted == ["tau.provider_response"]
    assert task.done() is True


@pytest.mark.asyncio
async def test_post_retries_transient_transport_failure(monkeypatch):
    attempts = 0

    class FakeResponse:
        status_code = 200
        text = ""

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, *_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectTimeout("transient")
            return FakeResponse()

    async def no_delay(_delay):
        return None

    monkeypatch.setenv("TAU_INSTRUMENTATION_URL", "https://clarity.example.test")
    monkeypatch.setenv("TAU_INSTRUMENTATION_TOKEN", "test-token")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: FakeClient())
    monkeypatch.setattr(asyncio, "sleep", no_delay)

    await clarity_instrumentation._post(
        "tau.provider_response",
        None,
        {"ok": True},
        {},
    )

    assert attempts == 2
