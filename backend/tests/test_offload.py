"""Milestone 1.5 acceptance tests — pointer-based tool offload.

1.5-A dump-sized tool result -> replaced by pointer; original written verbatim to scratchpad
1.5-B pointer contains: ref id, tool label, size + a status-bearing summary line
1.5-C freshness: freshest N (=keep_recent) stay raw; last message always intact
1.5-D sub-threshold tool result stays raw (precision-first)
1.5-E envelope integrity: tool_call_id / name preserved on the pointer message
1.5-F idempotence: pointer content is never re-offloaded on the next turn
1.5-G non-tool roles are never offloaded (user/assistant = dedup's domain)
1.5-H integration: offload runs BEFORE num_ctx -> negotiated window sizes the post-offload payload
"""
from __future__ import annotations

import json
from typing import Callable

import pytest

from app.config import Settings
from app.core import hash_filter, num_ctx
from app.core.gc import process_payload
from app.core.offload import _is_pointer, _extract_summary

DUMP = "\n".join(f"[test line {i}] output row detail ..." for i in range(400)) \
       + "\nFAILED: 3 tests failed."     # dump-sized (>800 tokens) with a status line


@pytest.fixture(autouse=True)
def stage_env(settings_factory, tmp_path, monkeypatch) -> Settings:
    """GC on + a per-test scratchpad dir; resets Stage-1 state."""
    s = settings_factory(gc_enabled=True, scratchpad_dir=str(tmp_path / "scratch"))
    hash_filter.reset_state()
    yield s
    hash_filter.reset_state()


def _payload(*tool_msgs, tools=True):
    msgs = []
    if tools:
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": "9", "function": {"name": "bash", "arguments": "pytest"}}
        ]})
    msgs.extend(tool_msgs)
    msgs.append({"role": "user", "content": "what now?"})   # fresh last message
    return {"model": "llama3.2:3b", "messages": msgs}


def _tool_msg(name="bash", content=DUMP, id="9"):
    return {"role": "tool", "name": name, "tool_call_id": id, "content": content}


def test_dump_offloaded_and_file_written(stage_env, tmp_path):
    # 3 dumps: keep_recent=2 protects the two FRESHEST; the oldest gets the pointer
    payload = _payload(
        _tool_msg(id="1", content=DUMP),
        _tool_msg(id="2", content=DUMP),
        _tool_msg(id="3", content=DUMP),
    )
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    t = report["tool_offload"]
    assert len(t["offloads"]) == 1
    offl = t["offloads"][0]
    offl = t["offloads"][0]
    assert offl["index"] == 1 and offl["tool_call_id"] == "1"
    assert offl["tokens_before"] > offl["tokens_after"] * 30
    from tests.helpers import find as find_msg
    pointer_msg = find_msg(out, role="tool", contains="Key result:")  # pointer-only marker
    assert pointer_msg is not None and _is_pointer(pointer_msg["content"])
    pointer = pointer_msg["content"]
    assert "Key result: FAILED: 3 tests failed." in pointer
    assert "Key result: FAILED: 3 tests failed." in pointer
    assert offl["tool"] == "bash"

    files = list((tmp_path / "scratch").glob(f"ref_{offl['ref']}_*.txt"))
    assert files
    content = files[0].read_text(encoding="utf-8")
    assert DUMP in content and "FAILED: 3 tests failed." in content   # verbatim original
    assert t["tokens_saved_estimate"] == offl["tokens_before"] - offl["tokens_after"]


def test_summary_extraction_prioritizes_status():
    assert "FAILED: 3 tests failed." in _extract_summary(DUMP)
    plain = "started\nrunning\ndone"           # no status words
    assert _extract_summary(plain).startswith("started")


def test_freshness_keep_recent_and_last(stage_env):
    payload = _payload(
        _tool_msg(id="1", content=DUMP),
        _tool_msg(id="2", content=DUMP),
        _tool_msg(id="3", content=DUMP),
        _tool_msg(id="4", content=DUMP),
    )
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    fwd = json.loads(out)["messages"]
    pointers = [i for i, m in enumerate(fwd)
                if m["role"] == "tool" and _is_pointer(m["content"])]
    raw_tools = [i for i, m in enumerate(fwd)
                 if m["role"] == "tool" and not _is_pointer(m["content"])]
    assert len(pointers) == 2                      # keep_recent=2 -> oldest two offloaded
    assert len(raw_tools) == 2                     # freshest two stay raw
    assert fwd[-1]["content"] == "what now?"       # last message untouched


def test_subthreshold_stays_raw(stage_env):
    small = "short log line"
    payload = _payload(_tool_msg(content=small))
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    assert "tool_offload" not in report
    from tests.helpers import find as find_msg
    small_msg = find_msg(out, role="tool", contains=small)
    assert small_msg is not None and small_msg["content"] == small  # index-independent


def test_envelope_integrity(stage_env):
    payload = _payload(_tool_msg())
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    fwd = json.loads(out)["messages"]
    tool_msgs = [m for m in fwd if m["role"] == "tool"]
    assert all(m.get("tool_call_id") == "9" for m in tool_msgs)    # pairing intact
    assert {m.get("name") for m in tool_msgs} == {"bash"}


def test_idempotence_pointer_not_reoffloaded(stage_env):
    payload = {
        "model": "m",
        "messages": [
            {"role": "assistant", "content": "x", "tool_calls": [{"id": "9", "function": {"name": "bash"}}]},
            {"role": "tool", "tool_call_id": "9",
             "content": "[Tool Output Ref #1 — bash | saved to scratchpad: data/ref_1.txt\n"
                        "Key result: ok\n Retrieve via \"read_ref 1\"]"},
            {"role": "user", "content": "what now?"},
        ],
    }
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    assert "tool_offload" not in report            # pointer content never re-offloaded


def test_non_tool_roles_never_offloaded(stage_env):
    big = "user context dump " * 400
    payload = {"model": "m", "messages": [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
    ]}
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    assert "tool_offload" not in report


def test_integration_offload_sized_before_numctx(stage_env):
    """1.5-H: pre-offload estimate (original) > post-pointer estimate; num_ctx uses POST."""
    def est(p):
        return num_ctx.estimate_tokens(p)

    payload = _payload(
        _tool_msg(id="1", content=DUMP),
        _tool_msg(id="2", content=DUMP),
        _tool_msg(id="3", content=DUMP),
    )
    pre = est(payload)
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="r")
    fwd = json.loads(out)
    from tests.helpers import without_anchor
    post_no_pin = est({"messages": without_anchor(out)})     # post-offload, pre-pin size
    assert report["tool_offload"]["tokens_saved_estimate"] == pre - post_no_pin
    assert report["numctx_negotiation"]["estimated_tokens"] == est(fwd)  # sized AFTER offload+pin
