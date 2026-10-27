"""Tier 2 / Milestone 2.1 tests — S4 semantic contraction.

2.1-A Near-duplicate rephrasing merges (cosine > threshold), most-recent kept
2.1-B Below threshold: distinct facts never merge
2.1-C Negation guard: contradictory phrasings NEVER merge (Iteration-1/L5)
2.1-D Number guard: differing numerics never merge
2.1-E Cross-namespace (role) statements never merge
2.1-F Procedural (system) messages locked; tool envelopes skipped
2.1-G Encoder unavailable -> graceful skip, payload untouched
2.1-H Telemetry: semantic savings flow into stage_tokens_saved
2.1-I Pipeline integration: stage runs in gc.py order (after reinject, before anchor)
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import pytest

from app.core import semantic
from app.core.gc import process_payload


# --- deterministic fake encoder: word-hash bag-of-words, unit-normalized ---

_DIM = 64
_WORD_VECTORS: dict[str, tuple[float, ...]] = {}


def _word_vec(word: str) -> tuple[float, ...]:
    if word not in _WORD_VECTORS:
        h = hashlib.blake2b(word.encode(), digest_size=16).digest()
        v = [((h[i % 16] + h[(i * 3 + 1) % 16]) / 255.0) - 0.5 for i in range(_DIM)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        _WORD_VECTORS[word] = tuple(x / norm for x in v)
    return _WORD_VECTORS[word]


def fake_encoder(text: str) -> list[float]:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return [0.0] * _DIM
    v = [sum(wv[k] for wv in map(_word_vec, words)) for k in range(_DIM)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


@pytest.fixture(autouse=True)
def fake_embed():
    semantic.set_encoder(fake_encoder)
    yield
    semantic.reset_state()


def _payload(*msgs):
    return {"model": "llama3.2:3b", "messages": list(msgs)}


def _run(payload, run_id="default", **kw):
    from app.config import Settings
    base = {
        "sink": "http://mock-sink", "proxy_port": 0,
        "gc_enabled": True, "gc_stage1_enabled": False,  # isolate S4
        "state_enabled": False, "numctx_enabled": False,  # no anchor/window side effects
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


# --- 2.1-A: merge + most-recent kept ---

def test_near_duplicate_rephrasing_merges():
    # high word overlap, not byte-identical (S1 dedup's blind spot)
    old = "Remember that the parser fails on comments whenever comments appear inside the parser input file"
    new = "remember that the parser fails on comments whenever comments appear inside the parser input file too"
    payload = _payload(
        {"role": "assistant", "content": old},
        {"role": "assistant", "content": new},
        {"role": "user", "content": "ok continue working on the parser task now please"},
    )
    fake_vec_a, fake_vec_b = fake_encoder(old), fake_encoder(new)
    from app.core.semantic import _cosine
    assert _cosine(fake_vec_a, fake_vec_b) > 0.85   # pre-condition: fake says merge

    out, report = _run(json.loads(json.dumps(payload)))
    assert report["semantic_contraction"]["merged"] == 1
    assert report["semantic_contraction"]["tokens_saved_estimate"] > 0
    msgs = json.loads(out)["messages"]
    contents = [m["content"] for m in msgs]
    assert old not in contents               # older phrasing dropped
    assert new in contents                   # most-recent phrasing kept


# --- 2.1-B: below threshold -> no merge ---

def test_distinct_facts_never_merge():
    payload = _payload(
        {"role": "assistant", "content": "Note that the rate limit hits 50 requests per hour outbound"},
        {"role": "assistant", "content": "The database schema needs three migration steps tomorrow"},
        {"role": "user", "content": "continue the work please and thank you agent"},
    )
    out, report = _run(payload)
    assert report["semantic_contraction"]["merged"] == 0
    assert len(json.loads(out)["messages"]) == 3


# --- 2.1-C: negation guard ---

def test_negation_never_merges():
    old = "the parser fails on comments inside the parser input file every single run"
    new = "the parser does not fail on comments inside the parser input file this run"
    from app.core.semantic import _cosine
    assert _cosine(fake_encoder(old), fake_encoder(new)) > 0.85   # vectors say merge
    payload = _payload(
        {"role": "assistant", "content": old},
        {"role": "assistant", "content": new},
        {"role": "user", "content": "carry on with the parser debugging task please"},
    )
    out, report = _run(payload)
    assert report["semantic_contraction"]["merged"] == 0   # guard blocked it
    assert len(json.loads(out)["messages"]) == 3


# --- 2.1-D: number guard ---

def test_numbers_never_merge():
    old = "About 3 tests failed in the parser suite after the latest big refactor"
    new = "Roughly 2 tests failed in the parser suite after the latest big refactor"
    from app.core.semantic import _cosine
    assert _cosine(fake_encoder(old), fake_encoder(new)) > 0.85   # vectors say merge
    payload = _payload(
        {"role": "assistant", "content": old},
        {"role": "assistant", "content": new},
        {"role": "user", "content": "please continue fixing the parser suite now"},
    )
    out, report = _run(payload)
    assert report["semantic_contraction"]["merged"] == 0
    assert len(json.loads(out)["messages"]) == 3


# --- 2.1-E: cross-namespace (role) never merges ---

def test_cross_role_never_merges():
    text = "Remember that the parser fails on comments whenever comments appear inside the parser input file"
    payload = _payload(
        {"role": "user", "content": text},
        {"role": "assistant", "content": text},
        {"role": "user", "content": "continue with the parser work please and report status"},
    )
    out, report = _run(payload)
    assert report["semantic_contraction"]["merged"] == 0
    assert len(json.loads(out)["messages"]) == 3


# --- 2.1-F: procedural locked; tool envelope skipped ---

def test_system_and_tool_envelope_locked():
    sys_text = "Remember that the parser fails on comments whenever comments appear inside the parser input file"
    payload = _payload(
        {"role": "system", "content": sys_text},
        {"role": "assistant", "content": sys_text, "tool_calls": [
            {"id": "1", "function": {"name": "bash", "arguments": "x"}}]},
        {"role": "user", "content": "keep going with the parser work now please"},
    )
    out, report = _run(payload)
    assert report["semantic_contraction"]["merged"] == 0


# --- 2.1-G: encoder unavailable -> graceful skip ---

def test_encoder_unavailable_skips_cleanly():
    semantic.set_encoder(lambda text: None)   # simulate missing sentence-transformers
    payload = _payload(
        {"role": "assistant", "content": "Remember that the parser fails on comments whenever comments appear inside the parser input file"},
        {"role": "assistant", "content": "Remember that the parser fails on comments whenever comments appear inside the parser input file"},
        {"role": "user", "content": "continue parsing work pretty please"},
    )
    out, report = _run(payload)
    sc = report["semantic_contraction"]
    assert sc["merged"] == 0 and sc.get("skipped_reason")
    assert len(json.loads(out)["messages"]) == 3   # untouched


# --- 2.1-H: telemetry wiring ---

def test_telemetry_includes_semantic_savings():
    text = "Remember that the parser fails on comments whenever comments appear inside the parser input file"
    payload = _payload(
        {"role": "assistant", "content": text},
        {"role": "assistant", "content": text + " today as well"},
        {"role": "user", "content": "continue the parser task now please thank you"},
    )
    out, report = _run(payload)
    from app.core.telemetry import summarize_gc_report
    summary = summarize_gc_report(report)
    assert summary["stage_tokens_saved"].get("semantic_contraction", 0) > 0
    assert summary["tokens_saved_net"] > 0


# --- 2.1-I: pipeline order (stage listed, after reinject, before anchor) ---

def test_pipeline_stage_order():
    text = "Remember that the parser fails on comments whenever comments appear inside the parser input file"
    payload = _payload(
        {"role": "assistant", "content": text},
        {"role": "assistant", "content": text + " today as well"},
        {"role": "user", "content": "continue the parser task now please thank you"},
    )
    out, report = _run(payload, state_enabled=True, numctx_enabled=True)
    stages = report["pipeline"]["stages"]
    assert "semantic_contraction" in stages
    assert stages.index("semantic_contraction") < stages.index("anchored_state")
    assert stages.index("semantic_contraction") < stages.index("numctx_negotiation")
