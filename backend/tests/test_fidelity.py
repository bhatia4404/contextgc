"""Tier 2 / Milestone 2.3 tests — deterministic fidelity fallback.

2.3-A Repetition loop (same assistant turn x3) -> raw passthrough, no GC
2.3-B Identical tool-call signature x3 -> recovery turn
2.3-C Confusion markers (2 distinct phrases) -> recovery turn
2.3-D Below threshold (x2) -> GC runs normally
2.3-E Healthy payload -> fallback silent, GC applies
2.3-F Trigger fires on RAW evidence even though GC would have deduped it
      (detection must see the history as the framework sent it)
2.3-G Telemetry: the trigger surfaces in summarize_gc_report meta
"""
from __future__ import annotations

import json

import pytest

from app.core import fidelity, hash_filter
from app.core.gc import process_payload


def _run(payload, run_id="default", **kw):
    from app.config import Settings
    base = {
        "sink": "http://mock-sink", "proxy_port": 0,
        "gc_enabled": True, "fidelity_enabled": True,
        "scratchpad_dir": "data/scratchpad", **kw,
    }
    import app.config as config_mod
    s = Settings(**base)
    original = config_mod.get_settings
    config_mod.get_settings = lambda: s
    try:
        return process_payload("/api/chat", json.dumps(payload).encode(), run_id=run_id)
    finally:
        config_mod.get_settings = original


def _payload(*msgs):
    return {"model": "llama3.2:3b", "messages": list(msgs)}


# --- 2.3-A: repetition loop ---


def test_repetition_loop_triggers_raw_passthrough():
    dup = {"role": "assistant", "content": "I will edit the file now."}
    payload = _payload(*([dup] * 3), {"role": "user", "content": "go on please"})
    raw = json.dumps(payload).encode()
    out, report = _run(payload)
    assert report["fidelity_fallback"]["triggered"] is True
    assert any("repetition_loop" in s for s in report["fidelity_fallback"]["signals"])
    # recovery turn == byte-identical RAW passthrough (no GC applied)
    assert out == raw
    assert report["pipeline"]["stages"] == ["fidelity_fallback"]
    assert report["pipeline"]["modified"] is False


# --- 2.3-B: tool-call loop ---


def test_tool_call_loop_triggers():
    tc_msg = {"role": "assistant", "content": " ", "tool_calls": [
        {"id": "1", "function": {"name": "bash", "arguments": {"cmd": "pytest -q"}}}]}
    payload = _payload(*([tc_msg] * 3), {"role": "user", "content": "run again please"})
    out, report = _run(payload)
    assert report["fidelity_fallback"]["triggered"] is True
    assert any("tool_call_loop" in s for s in report["fidelity_fallback"]["signals"])


# --- 2.3-C: confusion markers ---


def test_confusion_markers_trigger():
    payload = _payload(
        {"role": "assistant", "content": "I apologize for the confusion, let me fix that output"},
        {"role": "assistant", "content": "I'm sorry, there seems there was an error in the run"},
        {"role": "user", "content": "continue the task"},
    )
    out, report = _run(payload)
    assert report["fidelity_fallback"]["triggered"] is True
    assert any("confusion_markers" in s for s in report["fidelity_fallback"]["signals"])


# --- 2.3-D: below threshold -> normal GC ---


@pytest.fixture(autouse=True)
def stage_state():
    hash_filter.reset_state()
    yield
    hash_filter.reset_state()


def test_below_threshold_gc_runs_normally():
    dup = {"role": "assistant", "content": "repeated only twice for the dedup layer today okay"}
    payload = _payload(*([dup] * 2), {"role": "user", "content": "continue the task now"})
    out, report = _run(payload)
    assert "fidelity_fallback" not in report            # signal below threshold
    assert report["pipeline"]["stages"] and report["pipeline"]["modified"] in (True, False)
    # GC DID get the chance to run (stage0 at least); dedup should merge x2
    assert "stage1_dedup" in report["pipeline"]["stages"]
    assert report["pipeline"]["modified"] is True       # dedup removed the repeat


# --- 2.3-E: healthy payload ---


def test_healthy_payload_no_fallback():
    payload = _payload(
        {"role": "user", "content": "hello there friendly agent"},
        {"role": "assistant", "content": "working on the task as requested"},
        {"role": "user", "content": "great continue"},
    )
    out, report = _run(payload)
    assert "fidelity_fallback" not in report


# --- 2.3-F: raw evidence wins over what GC would have done ---


def test_fallback_sees_raw_history():
    # three identical assistant turns WOULD be dedup-expanded to 1 by S1,
    # but the fallback must trigger on the raw history BEFORE any GC runs
    dup = {"role": "assistant", "content": "identical turn repeated as loop evidence"}
    payload = _payload(*([dup] * 3), {"role": "user", "content": "please continue"})
    out, report = _run(payload)
    assert out == json.dumps(payload).encode()          # raw, not deduped
    assert "stage1_dedup" not in report["pipeline"]["stages"]


# --- 2.3-G: telemetry ---


def test_telemetry_surfaces_fidelity_trigger():
    dup = {"role": "assistant", "content": "loop again loop again loop again ok"}
    payload = _payload(*([{"role": "assistant", "content": "same repeated assistant turn here"}] * 3),
                       {"role": "user", "content": "go on"})
    out, report = _run(payload)
    from app.core.telemetry import summarize_gc_report
    summary = summarize_gc_report(report)
    assert summary["meta"]["fidelity_fallback"]["triggered"] is True
    assert summary["tokens_saved_net"] == 0             # recovery turn saves nothing
