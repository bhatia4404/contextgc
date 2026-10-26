"""Milestone 1.3 acceptance tests — Stage 0 rule-based classifier."""
from __future__ import annotations

import json

from app.core.classifier import MemoryType, classify_message, classify_payload


def test_system_is_procedural():
    ann = classify_message("system", "You are a helpful agent. Never leak secrets.")
    assert ann["type"] == MemoryType.PROCEDURAL
    assert ann["confidence"] == 1.0


def test_tools_lock_payload():
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "run_cmd", "parameters": {}}}],
    }
    report = classify_payload(payload)
    assert report["procedural_locked"] is True


def test_tool_result_is_episodic_tool_tag():
    payload = {
        "messages": [
            {"role": "user", "content": "run tests"},
            {"role": "tool", "tool_call_id": "9", "content": "5,000 lines of output..."},
        ]
    }
    report = classify_payload(payload)
    ann = report["annotations"][1]
    assert ann["type"] == MemoryType.EPISODIC
    assert ann["tool_result"] is True
    assert report["counts"]["tool_results"] == 1


def test_semantic_detection():
    ann = classify_message("user", "remember the goal: make the build pass")
    assert ann["type"] == MemoryType.SEMANTIC
    ann2 = classify_message("assistant", "the port 8855 is important: config lives there")
    assert ann2["type"] == MemoryType.SEMANTIC


def test_default_turns_are_episodic():
    for role, content in (
        ("user", "so what happened next?"),
        ("assistant", "I searched the repository and found three issues."),
    ):
        assert classify_message(role, content)["type"] == MemoryType.EPISODIC


def test_classification_cache_hits():
    a = classify_message("assistant", "unique message content for cache test")
    assert a["cache"] == "miss"
    b = classify_message("assistant", "unique message content for cache test")
    assert b["cache"] == "hit"
    assert b["type"] == a["type"] and b["reasons"] == a["reasons"]


def test_non_chat_payload_skipped():
    payload = {"model": "m", "prompt": "raw completion", "stream": False}
    report = classify_payload(payload)
    assert "skipped" in report
    assert report["counts"]["episodic"] == 0


# --- integration: pipeline stays byte-faithful with GC ON (no modifiers yet) ---


def test_gc_on_no_modifier_yet_byte_faithful(settings_factory):
    settings_factory(gc_enabled=True)  # anchor on: pin is the one modification GC-on always makes
    from app.core.gc import process_payload

    body = (
        b'{"model":"llama3.2:3b","messages":['
        b'{"role":"system","content":"You are a build agent."},'
        b'{"role":"user","content":"remember the goal: fix tests"}],'
        b'"tools":[{"type":"function","function":{"name":"run_cmd"}}]}'
    )
    out, report = process_payload("/v1/chat/completions", body)
    # with 1.6 anchor ON, the pin is the one modification GC always makes;
    # content-wise the rest must be byte-equal (no stage removed anything):
    from tests.helpers import without_anchor
    assert without_anchor(out) == json.loads(body)["messages"]
    counts = report["counts"]
    assert counts["procedural"] == 1 and counts["semantic"] == 1
    assert report["procedural_locked"] is True
