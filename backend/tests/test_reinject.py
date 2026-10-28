"""Milestone 1.5b acceptance tests — on-demand re-injection (accuracy contract).

1.5b-A current turn mentions "Ref #1" -> the pointed dump is FULLY restored for that turn
1.5b-B no mention -> nothing restored (prompt stays lean)
1.5b-C restored content drops the scratchpad header (clean readback)
1.5b-D unknown ref mentioned -> reported, no crash, no modification
1.5b-E reinject runs before num_ctx: negotiated estimate INCLUDES the restored dump
1.5b-F only explicitly-mentioned refs restored (blast-radius control)
"""
from __future__ import annotations

import json

import pytest

from app.core.offload import _is_pointer
from app.core import hash_filter, num_ctx, offload  # noqa: F401
from app.core.gc import process_payload

DUMP_LINES = [f"line {i} of the build log content" for i in range(400)]
DUMP = ("\n".join(DUMP_LINES) + "\nFAILED: 2 tests failed.")


@pytest.fixture(autouse=True)
def stage_env(settings_factory, tmp_path, monkeypatch):
    s = settings_factory(gc_enabled=True, scratchpad_dir=str(tmp_path / "scratch"))
    hash_filter.reset_state()
    yield s
    hash_filter.reset_state()
    offload._REF_MAP.clear()
    offload._CONTENT_TO_REF.clear()


def _seed_pointer(tmp_path, ref_id_content=DUMP) -> dict:
    """Turn with a dump + a short newer result -> creates pointered msg, returns it."""
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "assistant", "content": "running build", "tool_calls": [
                {"id": "7", "function": {"name": "bash", "arguments": {"cmd": "pytest"}}}]},
            {"role": "tool", "name": "bash", "tool_call_id": "7", "content": ref_id_content},
            {"role": "assistant", "content": "running current", "tool_calls": [
                {"id": "9", "function": {"name": "bash", "arguments": {"cmd": "pytest"}}}]},
            {"role": "tool", "name": "bash", "tool_call_id": "9", "content": "ok all green"},
            {"role": "assistant", "content": "running stale check", "tool_calls": [
                {"id": "3", "function": {"name": "bash", "arguments": {"cmd": "echo stale"}}}]},
            {"role": "tool", "name": "bash", "tool_call_id": "3", "content": "stale run done"},
            {"role": "user", "content": "what now?"},
        ],
    }
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="seedrun")
    assert "tool_offload" in report
    dumped_ref = [o for o in report["tool_offload"]["offloads"] if o["tool_call_id"] == "7"][0]["ref"]
    # locate by content marker — the 1.6 anchor shifted positions
    from tests.helpers import find as find_msg
    pointer = find_msg(out, role="tool", contains="Key result:")["content"]
    return {"model": "llama3.2:3b", "messages": [
        {"role": "tool", "name": "bash", "tool_call_id": "7", "content": pointer},
        {"role": "user", "content": "placeholder turn"},
    ]}, dumped_ref


def test_mention_restores_full_dump(stage_env, tmp_path):
    p, ref = _seed_pointer(tmp_path)
    p["messages"][1]["content"] = f"show me the exact stack trace from Ref #{ref}"
    out, report = process_payload("/api/chat", json.dumps(p).encode(), run_id="seedrun")
    ri = report["ref_reinject"]
    from tests.helpers import find as find_msg
    tool_c = find_msg(out, role="tool", contains="FULL OUTPUT RESTORED")["content"]
    assert ri["restored"][0]["ref"] == ref
    assert "line 200 of the build log content" in tool_c          # full log back
    assert "line 399 of the build log content" in tool_c          # all 400 lines
    assert "read_ref" not in tool_c                               # pointer text gone
    assert report["pipeline"]["modified"] is True


def test_no_mention_stays_lean(stage_env, tmp_path):
    p, _ = _seed_pointer(tmp_path)
    p["messages"][1]["content"] = "what happened overall?"
    out, report = process_payload("/api/chat", json.dumps(p).encode(), run_id="seedrun")
    assert "ref_reinject" not in report
    from tests.helpers import find as find_msg
    tool = find_msg(out, role="tool", contains="Key result:")
    assert tool is not None and _is_pointer(tool["content"])       # still pointered


def test_header_dropped_in_readback(stage_env, tmp_path):
    p, ref = _seed_pointer(tmp_path)
    p["messages"][1]["content"] = f"read_ref {ref} please"
    out, report = process_payload("/api/chat", json.dumps(p).encode(), run_id="seedrun")
    from tests.helpers import find as find_msg
    tool_c = find_msg(out, role="tool", contains="FULL OUTPUT RESTORED")["content"]
    assert "# ContextGC tool-offload original" not in tool_c
    body_line = tool_c.splitlines()[1]
    assert body_line.startswith("line 0 of")                       # dump starts immediately


def test_unknown_ref_reported_not_crash(stage_env, tmp_path):
    p, ref = _seed_pointer(tmp_path)
    p["messages"][1]["content"] = "show Ref #912 but nothing else"   # only the unknown
    out, report = process_payload("/api/chat", json.dumps(p).encode(), run_id="seedrun")
    assert report["ref_reinject"]["restored"] == []                  # nothing restorable
    assert report["ref_reinject"]["unknown_or_missing"] == [912]     # reported, not crash
    # the pointer for `ref` remains untouched (its own content skipped from mentions)


def test_reinject_before_numctx_scaling(stage_env, tmp_path):
    p, ref = _seed_pointer(tmp_path)
    p["messages"][1]["content"] = f"paste Ref #{ref} details"
    out, report = process_payload("/api/chat", json.dumps(p).encode(), run_id="seedrun")
    neg = report["numctx_negotiation"]
    from tests.helpers import messages as msgs_of
    fwd = json.loads(out)
    assert neg["estimated_tokens"] == num_ctx.estimate_tokens(fwd)   # includes restore
    assert neg["estimated_tokens"] > 800                              # grew with the dump


def test_only_declared_pointer_refs_restored(stage_env, tmp_path):
    """Two pointers exist; only the ref that both (a) is mentioned and
    (b) still points at a real scratchpad file gets restored."""
    p, ref = _seed_pointer(tmp_path)          # pointer msg at messages[0]
    decoy = {"role": "tool", "name": "bash", "tool_call_id": "99",
             "content": "[Tool Output Ref #99 — decoy]\nKey result: none\n]"}
    p["messages"].append(decoy)               # placeholder shift: pointer msg + decoy
    # Only the active/latest assistant message is a reinjection request.
    p["messages"][1]["content"] = f"compare Ref #{ref} with Ref #99"
    out, report = process_payload("/api/chat", json.dumps(p).encode(), run_id="seedrun")
    ri = report["ref_reinject"]
    assert set(ri["mentioned"]) == {ref, 99}
    assert [r["ref"] for r in ri["restored"]] == [ref]   # only the REAL pointer restored
    assert 99 in ri["unknown_or_missing"]                # decoy reported
