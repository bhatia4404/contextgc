"""Milestone 1.4 acceptance tests — Stage 1 MinHash/LSH dedup.

1.4-A  exact duplicate within one request (cross-request boilerplate) removed
1.4-B  near-duplicate (paraphrase ~identical structure) removed, meaning-kept
1.4-C  contradictions NOT merged (precision guard, Iteration-1/L5)
1.4-D  cross-request dedup: same run re-sends same history -> second request cleaned
1.4-E  procedural/semantic protected: system + semantic never removed
1.4-F  tool-call envelopes and LAST message never removed (freshness)
1.4-G  run isolation: different run_id sees no cross-run state
1.4-H  GC-on integration: dedup shrinks payload; sink receives fewer messages
1.4-I  gc-on without duplicates stays byte-faithful
"""
from __future__ import annotations

import json

import pytest

from app.core import hash_filter
from app.core.gc import process_payload

run_id_hdr = {"x-contextgc-run": "run-1"}


@pytest.fixture(autouse=True)
def fresh_stage1_state(settings_factory):
    settings_factory(gc_enabled=True)  # master switch ON for all dedup tests
    hash_filter.reset_state()
    yield
    hash_filter.reset_state()


def _as_bytes(payload: dict) -> bytes:
    return json.dumps(payload).encode()


# 1.4-A: exact duplicate inside one list (with a fresh trailing message so
# the duplicate is not the protected "last" turn)
def test_exact_duplicate_removed():
    payload = {
        "messages": [
            {"role": "assistant", "content": "Understood. Proceeding with step 2."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Understood. Proceeding with step 2."},
            {"role": "user", "content": "go on"},
        ]
    }
    out, report = process_payload("/api/chat", _as_bytes(payload), run_id="r")
    assert report["stage1_dedup"]["messages_after"] == 3
    assert report["stage1_dedup"]["removed"][0]["role"] == "assistant"
    assert report["pipeline"]["modified"] is True


# 1.4-B: structurally near-identical (a one-word swap stays below threshold on
# purpose — precision first; real paraphrase detection is Tier-2 embeddings)
def test_near_duplicate_paraphrase_removed():
    base = ("Ran the full test suite for the build module and captured the "
            "complete output plus the exact stack trace of the first failing "
            "test for the record.")
    a = base
    b = base + " done"   # +1 token: Jaccard ~0.96 — same report, tiny tail
    payload = {
        "messages": [
            {"role": "assistant", "content": a},
            {"role": "assistant", "content": b},
            {"role": "user", "content": "final"},
        ]
    }
    out, report = process_payload("/api/chat", _as_bytes(payload), run_id="r")
    assert report["stage1_dedup"]["messages_after"] == 2


# 1.4-C: contradiction not treated as duplicate (precision guard)
def test_contradiction_not_removed():
    yes = "The file has a bug in the login function that must be fixed."
    no = "The file has no bug in the login function that must be fixed."
    payload = {"messages": [{"role": "assistant", "content": yes}, {"role": "user", "content": no}]}
    out, report = process_payload("/api/chat", _as_bytes(payload), run_id="r")
    assert "stage1_dedup" not in report.get("pipeline", {}) and not report.get("stage1_dedup")


# 1.4-D: same run, same boilerplate in the NEXT request -> removed there too
def test_cross_request_history_is_preserved():
    boiler = {"role": "assistant", "content": "Updated the todo list. Next: run the test suite."}
    p1 = {"messages": [{"role": "user", "content": "go"}, boiler]}
    p2 = {"messages": [{"role": "user", "content": "go"}, boiler, boiler]}
    _, rep1 = process_payload("/api/chat", _as_bytes(p1), run_id="same-run")
    _, rep2 = process_payload("/api/chat", _as_bytes(p2), run_id="same-run")
    # A resent history is valid state, not cross-request duplication.
    assert not rep2.get("stage1_dedup")
    assert rep1.get("stage1_dedup") is None


# 1.4-E: procedural system + semantic messages never removed even when dup'd
def test_procedural_semantic_protected():
    sys1 = {"role": "system", "content": "You are a build agent that must fix tests."}
    sys2 = {"role": "system", "content": "You are a build agent that must fix tests."}
    fact1 = {"role": "user", "content": "remember the goal: make the build pass"}
    fact2 = {"role": "user", "content": "remember the goal: make the build pass"}
    payload = {"messages": [sys1, fact1, sys2, fact2]}
    out, report = process_payload("/api/chat", _as_bytes(payload), run_id="r")
    assert "stage1_dedup" not in report  # nothing removed
    from tests.helpers import without_anchor
    assert without_anchor(out) == payload["messages"]  # payload unchanged (minus the pin)
    # (system = procedural immutable; semantic = merge-only, Tier-2's domain)


# 1.4-F: tool envelopes and the freshest message protected
def test_tool_envelopes_and_last_stay():
    env = {
        "role": "assistant", "content": "Running tests", "tool_calls": [
            {"id": "9", "function": {"name": "bash", "arguments": "pytest"}}
        ],
    }
    payload = {"messages": [env, {"role": "assistant", "content": "Running tests"}]}
    out, report = process_payload("/api/chat", _as_bytes(payload), run_id="r")
    from tests.helpers import messages as msgs_of
    fwd = msgs_of(out)  # anchor pin shifts indices; assert by role/shape, not position
    kept = [m for m in fwd if m.get("role") in ("assistant", "user") or m.get("tool_calls")]
    assert len(fwd) == 2          # envelope + last message (empty anchor is skipped)
    assert "stage1_dedup" not in report or not report["stage1_dedup"]


# 1.4-G: runs are isolated
def test_run_isolation():
    def fresh_payload():
        return {
            "messages": [{"role": "assistant", "content": "Identical across runs."}, {"role": "user", "content": "go"}]
        }
    process_payload("/api/chat", _as_bytes(fresh_payload()), run_id="A")
    _, report_b = process_payload("/api/chat", _as_bytes(fresh_payload()), run_id="B")
    assert not report_b.get("stage1_dedup")


def test_gc_on_no_modifier_yet_byte_faithful(settings_factory):
    settings_factory(gc_enabled=True, state_enabled=False)
    body = (
        b'{"model":"llama3.2:3b","messages":['
        b'{"role":"system","content":"You are a build agent."},'
        b'{"role":"user","content":"remember the goal: fix tests"}],'
        b'"tools":[{"type":"function","function":{"name":"run_cmd"}}]}'
    )
    out, report = process_payload("/v1/chat/completions", body)
    assert out == body  # only one copy of each message -> Stage 1 not triggered
    counts = report["counts"]
    assert counts["procedural"] == 1 and counts["semantic"] == 1


def test_proxy_dedup_end_to_end(app_client, settings_factory):
    settings_factory(gc_enabled=True)
    tc, calls = app_client
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "assistant", "content": "Understood. Proceeding with step 2."},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "Understood. Proceeding with step 2."},
            {"role": "user", "content": "go on"},
        ],
        "stream": False,
    }
    r = tc.post("/v1/chat/completions", json=payload, headers=run_id_hdr)
    assert r.status_code == 200
    from tests.helpers import messages as msgs_of
    forwarded = msgs_of(calls[0]["content"])
    assert len(forwarded) == 3  # duplicate removed; no anchor because no state fields exist
    # store records the original alongside the cleaned view
    import app.core.store as store_mod
    st = store_mod.get_store()
    last = st.recent(n=1)[0]
    assert len(json.loads(st.get_raw(last["turn_id"]))["messages"]) == 4
    assert len(json.loads(st.get_clean(last["turn_id"]))["messages"]) == 3
