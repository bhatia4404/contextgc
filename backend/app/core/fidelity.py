"""Deterministic Fidelity Fallback (Tier 2, Milestone 2.3; plan v2; Iteration-1/L7).

The plan v1 idea — "detect hallucination, restore context" — was unreproducible
(there is no ground truth in the proxy). Replacement: DETERMINISTIC quality
signals, observable in the raw payload the framework sends:

  1. Repetition loop — the same assistant turn (content hash) appears
     >= `fidelity_loop_threshold` times in the history (the model going in
     circles is exactly when cleaning hurt most), or
  2. Identical tool-call signatures (function + args) repeated >= threshold
     (the agent hammering the same failing call), or
  3. Confusion markers — >= 2 DISTINCT known confusion phrases ("i apologize",
     "let me try again", ...) or the same phrase >= threshold times.

Action on trigger: THIS request is served byte-identically from the raw view
(the recovery turn) — no GC applies. No cross-request state needed: the loop
evidence is visible in the history of the arriving request. Restore is safe
because of the non-destructive invariant: the raw body was never modified.

Why deterministic: a rule engine is auditable ("explain the fallback" is one
grep), demo-safe, and the plan's own L7 requirement.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app import config

# Known confusion markers (lowercase). Two DISTINCT ones or one repeated
# `fidelity_loop_threshold` times triggers the recovery turn.
_CONFUSION_PHRASES = (
    "i apologize", "i'm sorry", "i am sorry", "let me try again",
    "it seems there was an error", "something went wrong",
    "let me retry", "attempting again",
)

_HASH_CACHE_MAX = 4096  # one hash per (content) — small, request-local anyway


def _content_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8", errors="replace")[:2048],
                           digest_size=8).hexdigest()


def _tool_call_key(msg: dict) -> str | None:
    calls = msg.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    sigs = []
    for c in calls:
        try:
            fn = c.get("function") or {}
            sigs.append(f"{fn.get('name')}:{json.dumps(fn.get('arguments'), sort_keys=True, default=str)}")
        except Exception:
            return None
    return _content_key("|".join(sigs))


def detect(payload: dict) -> list[str]:
    """Returns the list of triggered signal descriptions ([] = healthy)."""
    settings = config.get_settings()
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    signals: list[str] = []
    threshold = settings.fidelity_loop_threshold

    # 1) identical assistant content repeated
    counts: dict[str, int] = {}
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
            counts[_content_key(m["content"])] = counts.get(_content_key(m["content"]), 0) + 1
    if counts and max(counts.values()) >= threshold:
        signals.append(f"repetition_loop: identical assistant turn x{max(counts.values())} "
                       f"(threshold {threshold})")

    # 2) identical tool-call signature repeated
    tc_counts: dict[str, int] = {}
    for m in messages:
        if isinstance(m, dict):
            k = _tool_call_key(m)
            if k:
                tc_counts[k] = tc_counts.get(k, 0) + 1
    if tc_counts and max(tc_counts.values()) >= threshold:
        signals.append(f"tool_call_loop: identical tool_call x{max(tc_counts.values())}")

    # 3) confusion markers
    confusion_counts: dict[str, int] = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c, default=str)
        lowered = (text or "").lower()
        for phrase in _CONFUSION_PHRASES:
            if phrase in lowered:
                confusion_counts[phrase] = confusion_counts.get(phrase, 0) + 1
    if confusion_counts:
        if len(confusion_counts) >= 2:
            signals.append(f"confusion_markers: {len(confusion_counts)} distinct phrases")
        elif max(confusion_counts.values()) >= threshold:
            signals.append(f"confusion_markers: '{max(confusion_counts, key=confusion_counts.get)}' "
                           f"x{max(confusion_counts.values())}")

    return signals
