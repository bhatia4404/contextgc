"""Milestone 1.8 acceptance tests — telemetry bus.

1.8-A Unified per-turn event: every proxied turn emits `turn_telemetry`
      (tokens, TTFT, latency, stage savings, negotiation) — GC off case
1.8-B Per-stage savings visible in the event with GC on (dedup + num_ctx +
      CPU KV gauge)
1.8-C ref_reinject shows up as NEGATIVE savings (accuracy contract visible)
1.8-D WebSocket replays the ring buffer
1.8-E VRAM sampler degrades gracefully (never raises, honest about source)
1.8-F KV-token gauge helper (CPU-only fallback per Iteration-1/L3)
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core import hash_filter, offload, telemetry, vram

from tests.conftest import json_response, sse_stream_response


def _chat_body(msgs: list[dict]) -> str:
    return json.dumps({"model": "llama3.2:3b", "messages": msgs})


def _turn_events(client_events: list[dict]) -> list[dict]:
    return [e for e in client_events if e.get("type") == "turn_telemetry"]


def _collect_events(tc: TestClient) -> list[dict]:
    return _filter_turn(json.loads(tc.get("/telemetry/recent").text)["events"])


def _filter_turn(events: list[dict]) -> list[dict]:
    return _turn_events(events)


# --- 1.8-A: unified event on the default (GC-off) path ---


def test_turn_telemetry_gc_off(app_client):
    tc, _ = app_client
    tc.post("/api/chat", content=_chat_body(
        [{"role": "user", "content": "hello"}]).encode(),
        headers={"content-type": "application/json"})
    events = _turn_events(json.loads(tc.get("/telemetry/recent").text)["events"])
    ev = events[-1]
    assert ev["turn_id"] >= 1
    assert ev["raw_tokens"] == ev["clean_tokens"]       # GC off -> nothing saved
    assert ev["tokens_saved"] == 0
    assert ev["ttft_s"] is not None and ev["ttft_s"] >= 0
    assert ev["total_latency_s"] >= ev["ttft_s" ]        # never before first byte
    assert ev["upstream_status"] == 200
    assert "num_ctx" not in ev            # numctx injected only when GC on
    assert ev["stage_tokens_saved"] == {}


# --- 1.8-B: per-stage savings with GC on ---


@pytest.fixture
def gc_app_client(monkeypatch, settings_factory, tmp_path):
    """Same shape as conftest app_client, but GC fully enabled."""
    s = settings_factory(
        gc_enabled=True, scratchpad_dir=str(tmp_path / "scratch"))
    hash_filter.reset_state()
    from app.main import app

    def handler(req: httpx.Request) -> httpx.Response:
        if (req.headers.get("content-type", "").startswith("application/json")
                and json.loads(req.content).get("stream")):
            return sse_stream_response(req)
        return json_response(req)

    app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-sink")
    tc = TestClient(app)
    yield tc
    tc.close()
    hash_filter.reset_state()


def test_turn_telemetry_stage_savings(gc_app_client):
    tc = gc_app_client
    dup = {"role": "user",
           "content": "please fix the tests in backend/util.py. " + ("context line for duplicate weight. " * 80)}
    body = _chat_body([dup, dup,
                       {"role": "user", "content": "fix the tests please"},
                       {"role": "assistant", "content": "on it"}])
    tc.post("/api/chat", content=body.encode(),
            headers={"content-type": "application/json"})
    ev = _turn_events(
        json.loads(tc.get("/telemetry/recent").text)["events"])[-1]

    assert ev["stage_tokens_saved"]["stage1_dedup"] > 0
    assert ev["tokens_saved"] > 0 and ev["tokens_saved_pct"] > 0
    assert ev["clean_tokens"] < ev["raw_tokens"]
    # num_ctx negotiation surfaced for the dashboard's window gauge
    assert ev["num_ctx"]["chosen_num_ctx"] >= 2048
    # CPU-only fallback gauge present (host has a KV bytes/token configured)
    assert ev["cpu_kv_gauge_bytes"] == ev["clean_tokens"] * 114688
    assert ev["meta"]["state_pinned"] is True


# --- 1.8-C: reinjection = visible negative savings ---


def _unit_report_with_reinject(tokens: int) -> dict:
    return {
        "stage1_dedup": {"tokens_saved_estimate": 500},
        "ref_reinject": {"restored": [
            {"ref": 3, "tokens_restored_estimate": tokens}]},
    }


def test_reinject_counts_as_negative_saving():
    out = telemetry.summarize_gc_report(_unit_report_with_reinject(300))
    assert out["stage_tokens_saved"]["stage1_dedup"] == 500
    assert out["stage_tokens_saved"]["ref_reinject"] == -300
    assert out["tokens_saved_net"] == 200


# --- 1.8-D: WebSocket replays the ring buffer ---


def test_ws_replays_ring_buffer():
    from app.main import app

    tag = json.dumps({"r": "1.8-D"}, separators=(",", ":"))
    telemetry.broadcast(telemetry.emit("ws_probe", {"tag": hash(tag)}))
    with TestClient(app) as tc:   # context manager -> lifespan runs
        with tc.websocket_connect("/ws/telemetry") as ws:
            found = False
            for _ in range(min(len(telemetry.recent()), 600)):
                event = json.loads(ws.receive_text())
                if event.get("type") == "ws_probe" \
                        and event.get("tag") == hash(tag):
                    found = True
                    break
    assert found


# --- 1.8-E: sampler is graceful on every machine ---


def test_vram_sampler_never_raises():
    snap = vram.sample()
    assert set(snap) >= {"ts", "source", "vram_used_mb", "vram_total_mb",
                         "gpu_name", "error"}
    assert snap["source"] in {"pynvml", "nvidia_smi", "none"}
    if snap["source"] == "none":
        assert snap["vram_used_mb"] is None and snap["error"]
    else:
        assert snap["vram_used_mb"] > 0 and snap["vram_total_mb"] > 0


# --- 1.8-F: KV gauge helper ---


def test_kv_gauge_bytes():
    assert telemetry.kv_gauge_bytes(100, 114688) == 11_468_800
    assert telemetry.kv_gauge_bytes(0, 114688) == 0
    assert telemetry.kv_gauge_bytes(100, 0) is None   # gauge disabled
