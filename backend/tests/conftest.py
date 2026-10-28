"""Shared fixtures: ContextGC app with a mock sink injected as http_client.

The mock sink lives behind httpx.MockTransport, so acceptance tests don't
require a running Ollama server.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app.config import Settings


@pytest.fixture
def settings(monkeypatch) -> Settings:
    s = Settings(gc_enabled=False, sink="http://mock-sink", proxy_port=0)
    monkeypatch.setattr("app.config.get_settings", lambda: s)
    return s


@pytest.fixture
def settings_factory(monkeypatch) -> Callable[..., Settings]:
    """Builds fresh Settings for core-only tests (gc pipeline etc.)."""
    def _set(**kw: object) -> Settings:
        base = {"sink": "http://mock-sink", "proxy_port": 0}
        base.update(kw)  # type: ignore[assignment]
        s = Settings(**base)  # type: ignore[arg-type]
        monkeypatch.setattr("app.config.get_settings", lambda: s)
        return s
    return _set


@pytest.fixture(autouse=True)
def fresh_store_for_tests(tmp_path, monkeypatch):
    """Keep all fixture-created forwards out of the repo's real DB.

    main.py accesses the store via module attribute (`store_mod.get_store()`)
    precisely so this monkeypatch reaches it regardless of import style.
    """
    from app.core import store as store_mod

    monkeypatch.setattr(
        store_mod, "get_store", lambda: store_mod.Store(str(tmp_path / "proxy-test.db"))
    )
    from app.core import anchored_state, num_ctx, run_state
    num_ctx.reset_state()
    run_state.reset()
    anchored_state.reset_state()  # per-run anchor state must not leak across tests
    yield
    num_ctx.reset_state()
    run_state.reset()
    anchored_state.reset_state()


@pytest.fixture
def app_client(monkeypatch, settings):
    """Returns (TestClient, calls). `calls` records requests that reached the sink."""
    from app.main import app

    calls: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(
            {
                "path": req.url.path,
                "method": req.method,
                "headers": dict(req.headers),
                "content": req.read(),
            }
        )
        if (
            req.headers.get("content-type", "").startswith("application/json")
            and json.loads(req.content).get("stream")
        ):
            return sse_stream_response(req)
        return json_response(req)

    app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-sink"
    )
    tc = TestClient(app)
    try:
        yield tc, calls
    finally:
        tc.close()
        # Do not close app.state.http_client: the app lifespan owns it in prod,
        # and TestClient with lifespan disabled means we close it here.
        try:
            import asyncio

            asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
                close_client(app.state.http_client)
            )
        except Exception:
            pass


async def close_client(c: httpx.AsyncClient) -> None:
    await c.aclose()


def json_response(req: httpx.Request) -> httpx.Response:
    """Echo the parsed JSON body back, like an inference sink would relplay a minimal completion."""
    payload = json.loads(req.content)
    return httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"echo": payload},
        request=req,
    )


def sse_stream_response(req: httpx.Request, n_chunks: int = 5) -> httpx.Response:
    """Return server-sent-event chunks like Ollama/openai streams do."""
    chunks = []
    for i in range(n_chunks):
        chunks.append(('data: {"chunk": "%d"}' % i).encode())
    chunks.append(b"data: [DONE]")
    body = b"".join(c + b"\n\n" for c in chunks)
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream; charset=utf-8"},
        content=body,
        request=req,
    )
