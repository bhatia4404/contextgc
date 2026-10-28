"""v3 acceptance tests — fixed per-run num_ctx pinning.

1.7-A native /api/chat: options.num_ctx injected per estimate, 1024-snapped
1.7-B estimated tokens scale with payload size (floor respected)
1.7-C client-set options.num_ctx is RESPECTED (no override)
1.7-D /v1 + Ollama sink (auto mode): top-level num_ctx injected
1.7-E /v1 + cloud sink: report-only, no extra keys leaked
1.7-F num_ctx negotiation runs AFTER dedup — sized to the CLEANED payload
1.7-H gc off: no negotiation at all (identity passthrough unaffected)
"""
from __future__ import annotations

import json

import pytest

from app.core import hash_filter, num_ctx
from app.core.gc import process_payload

ROUND = 1024


@pytest.fixture(autouse=True)
def stage_env(settings_factory):
    settings_factory(gc_enabled=True)
    hash_filter.reset_state()
    yield
    hash_filter.reset_state()


def test_native_injection_and_floor(settings_factory):
    s = settings_factory(gc_enabled=True)
    payload = {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "hi"}] * 10}
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    neg = report["numctx_negotiation"]
    chosen = neg["chosen_num_ctx"]
    fwd = json.loads(out)
    assert chosen >= s.numctx_min and chosen % ROUND == 0  # 1024-snapped
    # negotiation is sized to the CLEANED payload (dedup ran first — 1.7-F):
    assert neg["estimated_tokens"] == num_ctx.estimate_tokens(fwd)
    assert len(fwd["messages"]) == 1  # the freshest copy (newest wins; empty anchor skipped)
    assert fwd["options"]["num_ctx"] == chosen
    assert report["pipeline"]["modified"] is True


def test_estimate_scales_and_floor(settings_factory):
    s = settings_factory(gc_enabled=True)
    small = {"messages": [{"role": "user", "content": "tiny"}]}
    big = {"messages": [{"role": "user", "content": "x" * 40000}]}
    e_small, e_big = num_ctx.estimate_tokens(small), num_ctx.estimate_tokens(big)
    assert e_small < 100 and e_big > 9000
    for e in (e_small, e_big):
        chosen = min(max(num_ctx._round_up_to(int(e) or 1), s.numctx_min), s.numctx_max)
        assert chosen % ROUND == 0 and chosen >= s.numctx_min


def test_client_num_ctx_respected(settings_factory):
    settings_factory(gc_enabled=True)
    payload = {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "hello"}],
        "options": {"num_ctx": 6144},
    }
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    neg = report["numctx_negotiation"]
    assert neg["respected_client"] is True and neg["client_num_ctx"] == 6144
    assert json.loads(out)["options"]["num_ctx"] == 6144  # untouched


def test_v1_ollama_sink_injection(settings_factory):
    s = settings_factory(gc_enabled=True, sink="ollama")
    payload = {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "hi"}]}
    out, report = process_payload("/v1/chat/completions", json.dumps(payload).encode(), run_id="r")
    neg = report["numctx_negotiation"]
    assert neg["injected_where"] == "top_level"
    fwd = json.loads(out)
    assert fwd["num_ctx"] == neg["chosen_num_ctx"]
    assert "options" not in fwd  # no native-endpoint pollution on /v1


def test_v1_cloud_sink_report_only(settings_factory):
    settings_factory(gc_enabled=True, sink="https://baseten.example/v1")
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    out, report = process_payload("/v1/chat/completions", json.dumps(payload).encode(), run_id="r")
    neg = report["numctx_negotiation"]
    assert neg["injected_where"] == "report_only"
    fwd = json.loads(out)
    assert "num_ctx" not in fwd and "options" not in fwd  # clean toward cloud sinks


def test_negotiation_sizes_after_dedup(settings_factory):
    settings_factory(gc_enabled=True)
    shared = "Understood. Proceeding with the identical boilerplate sentence for the record now."  * 2
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "assistant", "content": shared},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": shared},
            {"role": "user", "content": "go"},
        ],
    }
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    dedup_after = report["stage1_dedup"]["messages_after"]
    neg = report["numctx_negotiation"]
    # estimate must reflect the dedup'd set: count messages that actually went downstream
    fwd = json.loads(out)
    assert len(fwd["messages"]) == dedup_after          # no state fields -> no anchor pin
    assert len(fwd["messages"]) < len(payload["messages"]) + 2  # (+2 = anchor pin allowance)
    assert neg["estimated_tokens"] == num_ctx.estimate_tokens(fwd)  # sized to clean payload


def test_window_is_identical_across_turns(settings_factory):
    settings_factory(gc_enabled=True, numctx_default=16384)
    first = {"messages": [{"role": "user", "content": "short"}]}
    second = {"messages": [{"role": "user", "content": "short"},
                            {"role": "assistant", "content": "x" * 30000}]}
    out1, r1 = process_payload("/api/chat", json.dumps(first).encode(), run_id="fixed")
    out2, r2 = process_payload("/api/chat", json.dumps(second).encode(), run_id="fixed")
    assert r1["numctx_negotiation"]["chosen_num_ctx"] == 16384
    assert r2["numctx_negotiation"]["chosen_num_ctx"] == 16384
    assert json.loads(out1)["options"]["num_ctx"] == json.loads(out2)["options"]["num_ctx"]


def test_gc_off_no_negotiation():
    import app.config as config_mod
    real = config_mod.get_settings
    config_mod.get_settings = lambda: SettingsLike()
    try:
        out, report = process_payload("/api/chat", json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "x"}]}).encode(), run_id="r")
        assert report == {} and b"num_ctx" not in out
    finally:
        config_mod.get_settings = real


class SettingsLike:
    gc_enabled = False
