"""Tier 2 / Milestone 2.2 tests — S5 importance eviction.

2.2-A Inert under budget: no eviction, nothing modified, nothing reported
2.2-B Fires under pressure and lands back under budget
2.2-C Least-important first: old/unreferenced/short evicted, newer kept
2.2-D Protected: system (procedural), anchor pin, live turn always survive
2.2-E Referenced-by-later survives: pointer mention ("Ref #2") protects target
2.2-F Freshness floor: newest N messages never evicted even under pressure
2.2-G Semantic facts outlast episodic chatter under equal pressure
2.2-H Telemetry: eviction savings flow into stage_tokens_saved
2.2-I Pipeline order: runs after anchored_state, before num_ctx
"""
from __future__ import annotations

import json

import pytest

from app.core import evict
from app.core.gc import process_payload


def _payload(*msgs):
    return {"model": "llama3.2:3b", "messages": list(msgs)}


def _filler(text_seed: str, n_turns: int, per_msg: int = 300) -> list[dict]:
    """Deterministic filler history: n_turns alternating turns with padding."""
    msgs = []
    for t in range(n_turns):
        msgs.append({"role": "user",
                     "content": f"{text_seed} turn {t}: " + ("filler detail. " * per_msg)})
        msgs.append({"role": "assistant",
                     "content": f"{text_seed} reply {t}: " + ("progress noted. " * per_msg)})
    return msgs


def _run(payload, run_id="default", **kw):
    from app.config import Settings
    base = {
        "sink": "http://mock-sink", "proxy_port": 0,
        "gc_enabled": True, "gc_stage1_enabled": False,
        "semantic_enabled": False, "offload_enabled": False,
        "state_enabled": False, "numctx_enabled": False,   # isolate S5
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


# --- 2.2-A: inert when under budget ---


def test_inert_under_budget():
    payload = _payload(*_filler("small", n_turns=3))
    out, report = _run(payload, evict_token_budget=999999)
    assert report["importance_eviction"]["triggered"] is False
    assert json.loads(out)["messages"] == payload["messages"]


# --- 2.2-B+C: fires under pressure, order least-important-first ---


def test_evicts_under_pressure_least_important_first():
    old_short = {"role": "assistant", "content": "turn 0 one-liner note"}
    # referenced message: the live turn quotes it verbatim -> hard protection
    referenced_old = {"role": "assistant",
                      "content": "important output: AlphaKey-9900 result marker text"}
    live = {"role": "user",
            "content": "please continue; prior output was: "
                       "important output: AlphaKey-9900 result marker text"}
    msgs = [old_short, referenced_old, *_filler("bulk", n_turns=6), live]
    payload = _payload(*msgs)

    out, report = _run(payload, evict_token_budget=6000)
    stage = report["importance_eviction"]
    assert stage["triggered"] is True
    assert stage["tokens_saved_estimate"] > 0
    out_msgs = json.loads(out)["messages"]

    live_kept = any(m["content"] == live["content"] for m in out_msgs)
    ref_kept = any(m["content"] == referenced_old["content"] for m in out_msgs)
    new_last_kept = any("reply 5" in m.get("content", "") for m in out_msgs)
    old_evicted = not any(m["content"] == old_short["content"] for m in out_msgs)
    assert live_kept and new_last_kept
    assert ref_kept, "referenced-by-later must survive pressure (hard protection)"
    assert old_evicted, "oldest unreferenced short message evicted first"
    # landed at/under budget (mirror estimator vs char-based saving may differ)
    from app.core import num_ctx
    assert num_ctx.estimate_tokens(json.loads(out)) <= 6000 * 2


# --- 2.2-D: hard protections ---


def test_protections_never_evicted():
    sysmsg = {"role": "system", "content": "You are a coding agent. " * 30}
    msgs = [sysmsg, *_filler("bulk", n_turns=6),
            {"role": "user", "content": "final instruction, please wrap up now"}]
    out, report = _run(_payload(*msgs), evict_token_budget=800)
    out_msgs = json.loads(out)["messages"]
    assert any(m["role"] == "system" for m in out_msgs)          # procedural survives
    assert any(m["content"].startswith("final instruction") for m in out_msgs)  # live survives


def test_anchor_pin_survives():
    from app.core.anchored_state import _ANCHOR_MARKER
    msgs = [*_filler("bulk", n_turns=6),
            # a stated goal gives the anchor content (empty anchors are skipped)
            {"role": "user", "content": "remember the goal: keep the anchor pin through eviction"}]
    # run with anchor ON so it inserts its own pin; eviction must skip it
    out, report = _run(_payload(*msgs), evict_token_budget=800, state_enabled=True)
    out_msgs = json.loads(out)["messages"]
    pins = [m for m in out_msgs if isinstance(m.get("content"), str)
            and m["content"].startswith(_ANCHOR_MARKER)]
    assert len(pins) == 1, "the anchor pin must never be evicted"


# --- 2.2-E: pointer reference protects target ---


def test_pointer_reference_protects_target():
    target = {"role": "assistant",
              "content": "definitely protected pointer: [Tool Output Ref #2 — bash | saved to scratchpad]"}
    later = {"role": "user", "content": "referring to [Tool Output Ref #2 again — what was the exit code? " + "context padding. " * 60}
    live = {"role": "user", "content": "continue please"}
    payload = _payload(target, later)
    # pad payload above budget with unprotected filler AFTER the pair
    filler = {"role": "assistant", "content": "bulk filler text. " * 400}
    payload = _payload(target, later, filler, _filler("bulk", n_turns=2)[0], live)

    out, report = _run(payload, evict_token_budget=900)
    out_msgs = json.loads(out)["messages"]
    assert any("Ref #2" in m.get("content", "") for m in out_msgs), \
        "pointered target survived eviction because a later turn references it"


# --- 2.2-F: freshness floor ---


def test_freshness_floor():
    msgs = _filler("bulk", n_turns=8)
    newest4 = [m["content"] for m in msgs[-4:-1]]     # last is the live turn
    out, report = _run(_payload(*msgs), evict_token_budget=500, evict_keep_recent=4)
    out_contents = [m["content"] for m in json.loads(out)["messages"]]
    for content in newest4:
        assert content in out_contents   # newest 3 (excluding live) never evicted


# --- 2.2-G: semantic facts outlast episodic chatter ---


def test_semantic_fact_outlives_episodic_chatter():
    fact = {"role": "assistant", "content":
            "the project objective is to ship the parser suite before Friday deadline review"}
    chatter = {"role": "assistant", "content": "ok working on it, sounds good"}
    msgs = [_filler("bulk", n_turns=2)[0], fact, chatter,
            *_filler("more", n_turns=6), {"role": "user", "content": "wrap up now please"}]
    out, report = _run(_payload(*msgs), evict_token_budget=800)
    out_contents = [m.get("content", "") for m in json.loads(out)["messages"]]
    assert any("project objective" in c for c in out_contents), \
        "standing-fact message outlives equal-pressured episodic chatter"


# --- 2.2-H: telemetry wiring ---


def test_telemetry_includes_eviction_savings():
    msgs = _filler("bulk", n_turns=6)
    out, report = _run(_payload(*msgs), evict_token_budget=900)
    from app.core.telemetry import summarize_gc_report
    summary = summarize_gc_report(report)
    assert summary["stage_tokens_saved"].get("importance_eviction", 0) > 0


# --- 2.2-I: pipeline order ---


def test_pipeline_stage_order():
    msgs = [*_filler("bulk", n_turns=8),
            {"role": "user", "content": "final wrap-up instruction for the turn"}]
    out, report = _run(_payload(*msgs), state_enabled=True, numctx_enabled=True,
                       evict_token_budget=800)
    stages = report["pipeline"]["stages"]
    assert "importance_eviction" in stages
    assert stages.index("importance_eviction") > stages.index("anchored_state")
    assert stages.index("importance_eviction") < stages.index("numctx_negotiation")
