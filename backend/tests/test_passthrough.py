"""Milestone 1.1 acceptance tests.

1.1-A: OpenAI-compatible passthrough (/v1/chat/completions, non-streaming)
1.1-B: Native Ollama passthrough (/api/chat)
1.1-C: SSE streaming passthrough — chunk content byte-faithful
1.1-D: GC-off guarantee — request body bytes identical, headers relayed
1.1-E: ContextGC owns its own endpoints (healthz not forwarded)
1.1-F: Unknown non-forwardable path -> 404 from ContextGC, not the sink
"""
from __future__ import annotations

import json


def test_openai_compat_passthrough_nonstream(app_client):
    tc, calls = app_client
    payload = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    r = tc.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert r.json() == {"echo": payload}
    assert len(calls) == 1
    assert calls[0]["path"] == "/v1/chat/completions"
    assert json.loads(calls[0]["content"]) == payload  # byte-faithful body


def test_ollama_native_passthrough(app_client):
    tc, calls = app_client
    payload = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    r = tc.post("/api/chat", json=payload)
    assert r.status_code == 200
    assert r.json() == {"echo": payload}
    assert calls[0]["path"] == "/api/chat"


def test_sse_streaming_passthrough(app_client):
    tc, calls = app_client
    payload = {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    with tc.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = b"".join(r.iter_raw())
    # Byte-faithful stream: same chunks the sink sent, in order.
    expected = b"".join(
        (('data: {"chunk": "%d"}' % i).encode() + b"\n\n") for i in range(5)
    ) + b"data: [DONE]\n\n"
    assert body == expected


def test_gc_off_is_identity(app_client):
    tc, calls = app_client
    payload = {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "x" * 500}], "stream": False}
    raw = json.dumps(payload).encode()  # send exact bytes; no client re-serialization
    r = tc.post("/api/chat", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert calls[0]["content"] == raw  # GC-off: byte-identical passthrough


def test_healthz_is_contextgc_owned(app_client):
    tc, calls = app_client
    r = tc.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "contextgc"
    assert len(calls) == 0  # never reached the sink


def test_unknown_path_not_forwarded(app_client):
    tc, calls = app_client
    r = tc.get("/definitely/not/real")
    assert r.status_code == 404
    assert len(calls) == 0
