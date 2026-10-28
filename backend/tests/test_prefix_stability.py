from __future__ import annotations

import json

from app.core.gc import process_payload


def _message_bytes(body: bytes) -> bytes:
    messages = json.loads(body)["messages"]
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()


def test_append_only_cleaned_messages_keep_prefix(settings_factory):
    settings_factory(gc_enabled=True, numctx_default=32768)
    first = {"model": "m", "messages": [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "remember the goal: keep this prefix"},
    ]}
    second = {"model": "m", "messages": first["messages"] + [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
    ]}
    out1, r1 = process_payload("/api/chat", json.dumps(first).encode(), run_id="prefix")
    out2, r2 = process_payload("/api/chat", json.dumps(second).encode(), run_id="prefix")
    assert r2["incremental_view"] is True
    prefix = _message_bytes(out1)
    current = _message_bytes(out2)
    assert current.startswith(prefix[:-1])
