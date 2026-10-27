"""S5 — Importance Eviction (Tier 2, Milestone 2.2; plan v2; Iteration-1/L6 fix).

The plan's original TF-IDF x exp(-lambda dt) formula was statistically weak
(a 15-turn conversation is not a corpus). Replacement pragmatic score, all
stdlib, all deterministic:

    importance(msg) = log1p(length) x recency x protected-by-later-turn

Protected (NEVER evicted, in precedence order):
  - the live turn (last message)
  - procedural memory (system role) — locked at S0
  - the anchored-state pin (S3's own message)
  - messages explicitly referenced by a later turn (pointer mention "Ref #n",
    or content quoted verbatim in a later message) — someone still needs them
  - the freshness floor (newest `evict_keep_recent` messages)

Pressure model: eviction ONLY fires when the cleaned payload's estimated
tokens exceed `evict_token_budget` — otherwise the stage is inert, so normal
runs prove nothing was needlessly removed. Eviction removes the LOWEST-score
candidates (old, unreferenced, short: length grows importance via 1/log
scaling, so a huge stale dump out-ranks a one-liner) until back under budget.

Same contract as every stage: True if payload modified; originals stay in the
append-only store, so eviction is a view change, never a deletion.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from app import config
from app.core import num_ctx as num_ctx_mod

_POINTER_REF_RE = re.compile(r"\[(Tool Output Ref #\d+)")
_ANCHOR_MARKER = "[ContextGC State Anchor]"


def _ref_id(text: str) -> str | None:
    """Pointer id ('[Tool Output Ref #7') of a pointered message, or None."""
    m = _POINTER_REF_RE.search(text)
    return m.group(1) if m else None


def _text_of(msg: dict) -> str:
    content = msg.get("content")
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _referenced_by_later(messages: list, i: int, ref_i: str | None) -> bool:
    """True when any later message depends on message i."""
    for j in range(i + 1, len(messages)):
        later = _text_of(messages[j]).lower()
        if not later:
            continue
        if ref_i and ref_i.lower() in later:            # explicit pointer mention
            return True
        core = _text_of(messages[i]).strip()
        if len(core) >= 40 and core.lower() in later:   # verbatim quote (same turn logic)
            return True
    return False


def importance(messages: list, i: int, annotations: dict[int, dict]) -> float:
    """Defensible L6 score: bigger = more important, eviction order = ascending.

    Hard protections (never evicted) live in `evictables`; this score only
    orders the REMAINING evictable messages: newest and longer survive longer.
    """
    text = _text_of(messages[i])
    age = len(messages) - 1 - i
    # recency at the evictable depth: monotonic decay with age
    recency = 1.0 / (1.0 + age * 0.5)
    # length term (L6): long content outvalues tiny fragments (1/log scaling)
    length_term = math.log1p(len(text))
    return recency * length_term


def evictables(messages: list, annotations: dict[int, dict]) -> list[int]:
    """Indexes eligible for eviction — everything NOT hard-protected.

    Hard protections (plan v2 L6, precedence in docstring):
      live turn * freshness floor * procedural (system) * anchor pin
      * referenced-by-later (pointer mention or >=40-char verbatim quote)
      * semantic memory (S0 marked "treat with care"; S4 already contracted it)
    """
    settings = config.get_settings()
    protected: set[int] = {len(messages) - 1}          # live turn
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            protected.add(i)
            continue
        if (len(messages) - 1 - i) < settings.evict_keep_recent:
            protected.add(i)                            # freshness floor
        if str(msg.get("role")) == "system":
            protected.add(i)                            # procedural lock
        t = _text_of(msg)
        if _ANCHOR_MARKER in t:
            protected.add(i)                            # S3's own pin
        if (annotations.get(i) or {}).get("type") == "semantic":
            protected.add(i)                            # standing fact
        elif _referenced_by_later(messages, i, _ref_id(t)):
            protected.add(i)                            # referenced later
    return [i for i in range(len(messages)) if i not in protected]


def _align_annotations(report: dict, messages: list) -> dict[int, dict]:
    """Re-key classifier annotations onto the CURRENT message list.

    S3 prepends the anchor pin after S0 ran, shifting original indexes by 1.
    Detected self-descriptively (anchor marker at index 0), no coupling to
    pipeline internals.
    """
    annotations = {int(a.get("index", -1)): a for a in report.get("annotations", [])}
    if not annotations:
        return annotations
    first = messages[0] if messages else None
    if isinstance(first, dict) and _ANCHOR_MARKER in _text_of(first):
        return {k + 1: v for k, v in annotations.items()}
    return annotations


def run_evict(payload: dict, report: dict, run_id: str = "default") -> bool:
    """Evict under window pressure. Contract: True if payload was modified."""
    settings = config.get_settings()
    messages = payload.get("messages")
    if not isinstance(messages, list) or report.get("skipped"):
        return False

    annotations = _align_annotations(report, messages)
    est = num_ctx_mod.estimate_tokens(payload)
    stage = report.setdefault("importance_eviction", {"budget": settings.evict_token_budget})
    if est <= settings.evict_token_budget:
        stage["triggered"] = False
        return False                                    # inert unless under pressure

    cand = evictables(messages, annotations)
    # least important first: (importance, index) ascending
    scored = sorted(cand, key=lambda i: (importance(messages, i, annotations), i))
    target_save = est - settings.evict_token_budget
    evicted: list[dict] = []          # frozen BEFORE mutation (indexes stay valid)
    saved = 0
    for i in scored:
        if saved >= target_save:
            break
        t = _text_of(messages[i])
        imp = importance(messages, i, annotations)
        evicted.append({"index": i, "chars": len(t), "importance": round(imp, 3)})
        saved += len(t) // 4 + 8                        # mirror of the estimator

    if not evicted:
        stage["triggered"] = False
        stage["note"] = "under pressure but nothing evictable (all protected)"
        return False

    evict_set = {d["index"] for d in evicted}
    payload["messages"] = [
        m for i, m in enumerate(messages) if i not in evict_set
    ]
    stage.update({
        "triggered": True,
        "est_before": est,
        "est_after": num_ctx_mod.estimate_tokens(payload),
        "evicted": evicted,
        "tokens_saved_estimate": saved,
    })
    counts = report.setdefault("counts", {})
    counts["evicted"] = len(evicted)
    return True
