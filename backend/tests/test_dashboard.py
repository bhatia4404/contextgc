"""Phase 4 dashboard: the page is served, and telemetry carries Ollama's own
per-request numbers (cached prompt tokens, load time) for the cache/reload gauges."""
from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from app.main import _ollama_stats, app


OLLAMA_FINAL = {"model": "m", "done": True, "message": {"role": "assistant", "content": "ok"},
                "prompt_eval_count": 1200, "prompt_eval_cached_count": 1100, "eval_count": 7,
                "load_duration": 25_000_000, "prompt_eval_duration": 90_000_000, "eval_duration": 300_000_000}


def test_stats_from_the_last_ndjson_line():
    body = (json.dumps({"message": {"content": "o"}, "done": False}) + "\n" + json.dumps(OLLAMA_FINAL) + "\n").encode()
    stats = _ollama_stats(body)
    assert stats["prompt_eval_count"] == 1200 and stats["prompt_eval_cached_count"] == 1100
    assert stats["load_s"] == 0.025 and stats["eval_s"] == 0.3


def test_no_stats_for_non_ollama_responses():
    assert _ollama_stats(b'data: {"chunk": "1"}\n\ndata: [DONE]\n\n') is None
    assert _ollama_stats(b"") is None


def test_dashboard_page_is_served(app_client):
    tc, calls = app_client
    r = tc.get("/dashboard")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<title>ContextGC Dashboard</title>" in r.text
    assert calls == []                      # served by the proxy, never forwarded


def test_turn_telemetry_carries_ollama_stats(settings):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"},
                              json=OLLAMA_FINAL, request=req)

    with TestClient(app) as tc:            # lifespan creates the real client; swap in the mock
        app.state.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://mock-sink")
        tc.post("/api/chat", content=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
                headers={"content-type": "application/json"})
        events = [e for e in tc.get("/telemetry/recent").json()["events"] if e.get("type") == "turn_telemetry"]
    assert events[-1]["ollama"]["prompt_eval_cached_count"] == 1100
    assert "compaction" in events[-1]


def test_results_listing_and_fetch_stay_inside_result_folders(app_client, tmp_path, monkeypatch):
    import app.main as main_mod
    root = tmp_path / "results"
    (root / "modal" / "run1").mkdir(parents=True)
    (root / "modal" / "run1" / "r.json").write_text('{"runs": []}', encoding="utf-8")
    (tmp_path / "secret.json").write_text('{"secret": 1}', encoding="utf-8")
    monkeypatch.setattr(main_mod, "RESULT_ROOTS", {"benchmark": root})
    tc, _ = app_client
    assert tc.get("/dashboard/results").json()["files"] == ["benchmark/modal/run1/r.json"]
    assert tc.get("/dashboard/results/benchmark/modal/run1/r.json").json() == {"runs": []}
    assert tc.get("/dashboard/results/benchmark/../secret.json").status_code == 404
    assert tc.get("/dashboard/results/benchmark/%2E%2E/secret.json").status_code == 404
    assert tc.get("/dashboard/results/other/r.json").status_code == 404
