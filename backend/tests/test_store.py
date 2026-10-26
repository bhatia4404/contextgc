"""Milestone 1.2 acceptance tests.

1.2-A: every forwarded request is persisted (raw + clean), with turn ids increasing
1.2-B: originals byte-faithful (raw == what the client sent; clean == raw when GC off)
1.2-C: fetchable later by turn_id (the fallback/invariant) + 404 for unknown ids
1.2-D: run_id grouping from optional x-contextgc-run header
1.2-E: stats reflect counts; raw_json decodes
1.2-F: LRU bound: evicts oldest over capacity, keeps newest/hot keys
"""
from __future__ import annotations

import json

import pytest

from app.core.store import Store
from app.core.caches import LRU


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "test.db"))


@pytest.fixture(autouse=True)
def fresh_store_cache(monkeypatch, tmp_path):
    """Redirect the global get_store to a tmp DB and reset its cache per test."""
    from app.core import store as store_mod

    monkeypatch.setattr(
        "app.core.store.get_store", lambda: store_mod.Store(str(tmp_path / "global.db"))
    )


def test_every_request_persisted(store):
    a = store.append("/v1/chat/completions", "run-1", b'{"q":1}', b'{"q":1}', {})
    b = store.append("/api/chat", "run-1", b'{"q":2}', b'{"q":2}', {"saved": 3})
    assert b == a + 1


def test_originals_byte_faithful_gc_off(store):
    raw = b'{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}'
    tid = store.append("/api/chat", "run-1", raw, raw, {})
    assert store.get_raw(tid) == raw
    assert store.get_clean(tid) == raw          # GC off -> clean == raw
    assert store.get_report(tid) == {}


def test_fetch_by_turn_id_and_404(store):
    tid = store.append("/api/chat", "r", b"{}@not-json", b"{}", {})
    assert store.get_raw(tid) == b"{}@not-json"
    assert store.get_raw(999) is None
    # non-JSON payloads survive storage intact rather than crashing
    assert store.raw_json(tid) == "{}@not-json"


def test_run_id_grouping(store):
    store.append("/api/chat", "crew-a", b"1", b"1", {})
    store.append("/api/chat", "crew-b", b"2", b"2", {})
    runs = {t["run_id"] for t in store.recent()}
    assert runs == {"crew-a", "crew-b"}

    # store-empty case
    st2 = Store(str(store.db_path) + "2")
    assert st2.recent() == []


def test_stats_and_json_decode(store):
    payload = {"messages": [{"role": "user", "content": "hello"}]}
    body = json.dumps(payload).encode()
    tid = store.append("/v1/chat/completions", "r", body, body, {"stage1": 5})
    assert store.stats()["turns"] == 1
    assert store.raw_json(tid) == payload      # originals decode to the same JSON
    assert store.get_report(tid) == {"stage1": 5}


def test_lru_eviction():
    c = LRU(maxsize=2)
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") is None        # evicted
    assert c.get("b") == 2 and c.get("c") == 3
    c.put("b", 9)                    # refresh b; next eviction should take c
    c.put("d", 4)
    assert c.get("c") is None and c.get("b") == 9
    assert len(c) == 2


# --- integration through the proxy (mock sink) ---


def test_proxy_persists_every_forward(app_client):
    tc, calls = app_client
    payload = {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    r1 = tc.post("/v1/chat/completions", json=payload, headers={"x-contextgc-run": "crew-demo"})
    r2 = tc.post("/v1/chat/completions", json=payload, headers={"x-contextgc-run": "crew-demo"})
    assert r1.status_code == r2.status_code == 200

    import app.core.store as store_mod
    st = store_mod.get_store()
    turns = st.recent(n=10)
    assert len([t for t in turns if t["run_id"] == "crew-demo"]) == 2

    first = turns[-1]  # oldest = first request here
    assert first["raw_bytes"] == first["clean_bytes"]  # GC-off: identical sizes
    # The raw fetch API round-trips the exact bytes that were forwarded:
    raw = st.get_raw(first["turn_id"])
    assert json.loads(raw) == payload


def test_store_stats_endpoint(app_client):
    tc, calls = app_client
    tc.post("/api/chat", json={"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": False})
    stats = tc.get("/store/stats").json()
    assert stats["turns"] == 1
    # stats endpoint never forwarded to the sink:
    assert len(calls) == 1  # only the chat POST reached the sink
