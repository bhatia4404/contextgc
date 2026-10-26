"""Stage 0 — Rule-based Memory-Type Classifier (Milestone 1.3).

Cognitive framing (CoALA, per final_plan_v2.md): every message gets exactly one
memory type, which decides how every later GC stage may treat it:

  PROCEDURAL  - system prompts, tool definitions, guardrails  -> NEVER touched
  SEMANTIC    - standing facts / assertions                   -> merge-only (1.5/Tier2)
  EPISODIC    - turn history, thoughts, tool outputs          -> aggressive GC

Design constraints:
- NO LLM in the hot path (Iteration 1, L8): pure heuristics, microseconds.
- Classifications are cached by content prefix (LRU from 1.2) — agent loops
  re-send nearly the same messages every turn, so repeats hit the cache.
- The proxy NEVER adds fields to the payload sent downstream; classification
  output travels out-of-band (in gc_report -> telemetry/store/dashboard).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.caches import LRU

# Cache: one entry per (role, content) pair. Agent loops re-send history —
# the second occurrence of a message costs a hash, not a re-scan.
_CLASS_CACHE = LRU(maxsize=8192)


class MemoryType:
    PROCEDURAL = "procedural"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


# Patterns that mark a user/assistant message as likely *standing fact*.
# Deliberately cheap word-level heuristics — precision of the merge stage
# (Tier 2) is guarded separately; here we only decide "treat with care".
_SEMANTIC_HINTS = (
    "remember ", "note that ", "the fact ", "important:", "important ",
    "my name ", "the project ", "the objective ", "the goal ",
    "constraint", "requirement", "convention", "specification",
)
_SEMANTIC_SUFFIXES = (" is ", " are ", " must ", " will bearing", " exists at ")

_TOOL_RESULT_ROLES = {"tool", "function", "tool_result"}
_TOOL_PAYLOAD_KEYS = ("tool_calls", "tool_call_id")


def _content_hash(role: str, content: str) -> str:
    h = hashlib.blake2b(digest_size=10)
    h.update(role.encode())
    h.update(b"\x00")
    h.update(content.encode("utf-8", errors="replace")[:8192])
    return h.hexdigest()


def classify_message(role: str, content: str, **extra: Any) -> dict:
    """Classify one chat message. Returns an out-of-band annotation dict."""
    cache_key = _content_hash(role, content)
    cached = _CLASS_CACHE.get(cache_key)
    if cached is not None:
        hit = dict(cached)
        hit["cache"] = "hit"
        return hit

    reasons: list[str] = []

    if role == "system":
        mtype, conf = MemoryType.PROCEDURAL, 1.0
        reasons.append("role=system (immutable by definition)")
    else:
        mtype, conf = MemoryType.EPISODIC, 0.7
        reasons.append(f"role={role} default (history default)")

    lowered = content.lower()
    if mtype is MemoryType.EPISODIC:
        if any(mark in lowered for mark in _SEMANTIC_HINTS) or any(
            marker in lowered for marker in _SEMANTIC_SUFFIXES
        ):
            mtype, conf = MemoryType.SEMANTIC, 0.6
            reasons.append("standing-fact pattern (heuristic)")

    ann = {
        "type": mtype,
        "confidence": conf,
        "reasons": reasons,
        "cache": "miss",
    }
    if extra.get("name"):
        ann["name"] = extra["name"]
    _CLASS_CACHE.put(cache_key, {k: v for k, v in ann.items() if k != "cache"})
    return ann


def classify_payload(payload: dict) -> dict:
    """Stage 0 over a whole request payload. Returns the stage-0 report.

    Contract: does NOT modify the payload. Tool definitions live at the
    payload level (`tools`) -> PROCEDURAL lock for the whole payload.
    """
    messages = payload.get("messages")
    report: dict = {
        "counts": {"procedural": 0, "semantic": 0, "episodic": 0, "tool_results": 0},
        "procedural_locked": False,
        "annotations": [],
    }
    if not isinstance(messages, list):
        report["skipped"] = "no messages array (not a chat payload)"
        return report

    if payload.get("tools"):
        report["procedural_locked"] = True  # tool schemas: never GC'd (immutable)

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "unknown"))
        content = msg.get("content")
        text = content if isinstance(content, str) else json.dumps(content, default=str)

        ann = classify_message(role, text, name=msg.get("name"))

        if role in _TOOL_RESULT_ROLES:
            ann["type"] = MemoryType.EPISODIC          # tool OUTPUTS age aggressively
            ann["tool_result"] = True
            report["counts"]["tool_results"] += 1
        if role == "system" or msg.get("tool_calls") or any(k in msg for k in _TOOL_PAYLOAD_KEYS[:1]):
            if role != "system" and (msg.get("tool_calls") or k_tool_payload_tool_calls(msg)):
                ann["procedural_note"] = (
                    "tool-call envelope present; message body eligible for GC, "
                    "tool definition list remains locked"
                )
        report["counts"][ann["type"]] += 1
        report["annotations"].append({"index": i, "role": role, **ann})

    return report


def k_tool_payload_tool_calls(msg: dict) -> bool:  # tiny helper kept for clarity
    return "tool_call_id" in msg


def annotate(path: str, body: bytes) -> tuple[dict, dict]:
    """Main-line stage-0 entry for gc.pipeline: (payload, report).

    Passthrough rules: non-chat payloads return (payload, {'skipped': ...}).
    Raises ValueError on undecodable JSON — gc.pipeline falls back to identity.
    """
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("not a JSON object")
    return payload, classify_payload(payload)
