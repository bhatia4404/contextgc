"""Stage 1 — MinHash/LSH exact structural dedup (Milestone 1.4).

Dedupes EPISODIC messages that are structurally near-identical, two ways:
  1. Within the request list (agent loops emit the same boilerplate several
     times inside one payload).
  2. Across requests of the same run (CrewAI/AutoGen re-send the whole
     conversation each turn — the cross-agent duplication from plan v2).

NEVER removes: procedural (system), semantic, tool-call envelopes, tool
results (1.5's offload territory), or the LAST message (the freshest
instruction must always stay — the agent's current turn).

Similarity: word 3-shingles → MinHash(128 perms) → MinHashLSH(threshold).
Zero neural cost (Iteration 1/L5: precision-first; threshold 0.92 keeps
contradictions like "has a bug"/"has no bug" far above but ~0.2 Jaccard apart).

Run state lives in an in-process LRU of MinHashLSH indexes keyed by run_id —
this is the cross-request cache the 1.2 caches module was built for.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

from datasketch import MinHash, MinHashLSH

from app import config
from app.core.classifier import MemoryType

_NUM_PERM = 128
_K_SHINGLE = 3


def _shingles(text: str, k: int = _K_SHINGLE) -> set[bytes]:
    words = text.split()
    if len(words) < k:
        if not text:
            return set()
        # short texts: single-token shingles (exact-ish match only)
        return {f"⟬w⟭ {w}".encode("utf-8") for w in words}
    return {" ".join(words[i : i + k]).encode("utf-8") for i in range(len(words) - k + 1)}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=_NUM_PERM)
    for sh in _shingles(text):
        m.update(sh)
    return m


class RunLSHIndex:
    """Per-run LSH index of seen episodic message signatures."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._lsh = MinHashLSH(threshold=threshold, num_perm=_NUM_PERM)
        self._lock = threading.Lock()

    def find_similar(self, key: str, mh: MinHash) -> list[str]:
        with self._lock:
            try:
                return list(self._lsh.query(mh))
            except ValueError:  # empty vector edge
                return []

    def insert(self, key: str, mh: MinHash) -> None:
        with self._lock:
            self._lsh.insert(key, mh, check_duplication=False)


class Stage1State:
    """LRU of run indexes: bounded memory, old runs auto-evicted."""

    def __init__(self, max_runs: int = 128, threshold: float = 0.92):
        self.max_runs = max_runs
        self.threshold = threshold
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, RunLSHIndex] = OrderedDict()
        self._seq: dict[str, int] = {}

    def next_key(self, run_id: str) -> str:
        """Monotonically increasing LSH keys per run — NEVER reused across
        requests, or the self-key filter would hide earlier turns (bug: keys
        used to reset each request and identical re-sent history 'became' self)."""
        with self._lock:
            n = self._seq.get(run_id, 0) + 1
            self._seq[run_id] = n
            return f"s1-{run_id}-{n}"

    def for_run(self, run_id: str) -> RunLSHIndex:
        with self._lock:
            if run_id in self._runs:
                self._runs.move_to_end(run_id)
                return self._runs[run_id]
            idx = RunLSHIndex(self.threshold)
            self._runs[run_id] = idx
            while len(self._runs) > self.max_runs:
                self._runs.popitem(last=False)
            return idx

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()
            self._seq.clear()


_state = Stage1State(threshold=0.92)


def revert_default_state() -> None:
    global _state
    _state = Stage1State(threshold=config.get_settings().gc_stage1_threshold)


def reset_state(threshold: float | None = None) -> None:
    revert_default_state() if threshold is None else _reset(threshold)


def _reset(threshold: float) -> None:
    global _state
    _state = Stage1State(threshold=threshold)


def _removal_allowed(msg: dict, index: int, total: int) -> bool:
    if index == total - 1:
        return False                      # never drop the freshest turn
    role = msg.get("role")
    if role == "system":
        return False                      # procedural: immutable
    if msg.get("tool_calls") or msg.get("tool_call_id"):
        return False                      # tool-call envelope integrity
    if msg.get("is_tool_result") or role in {"tool", "function"}:
        return False                      # tool results: Stage 1.5's domain
    return True


def run_stage1(payload: dict, run_id: str, report: dict) -> bool:
    """Dedup payload['messages'] in place. Returns True if payload modified."""
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False

    settings = config.get_settings()
    lsh = _state.for_run(run_id)
    kept: list[dict] = []
    removed: list[dict] = []
    saved_chars = 0

    for i, msg in enumerate(list(messages)):
        if not isinstance(msg, dict):
            kept.append(msg)
            continue
        role = str(msg.get("role", "unknown"))
        content = msg.get("content")
        text = content if isinstance(content, str) else json.dumps(content, default=str)

        # Semantic memory is merge-only territory (Tier 2) — Stage 1 never removes it.
        ann = next((a for a in report.get("annotations", []) if a.get("index") == i), None)
        if ann and ann.get("type") == "semantic":
            kept.append(msg)
            continue

        mh = _minhash(text)
        key = _state.next_key(run_id)
        similar = lsh.find_similar(key, mh)
        similar = [s for s in similar if s != key]

        if similar and _removal_allowed(msg, i, len(messages)):
            removed.append({
                "index": i,
                "similar_to": similar[0],
                "role": role,
                "chars_saved": len(text),
            })
            saved_chars += len(text)
            continue  # drop the duplicate from the forwarded payload

        lsh.insert(key, mh)
        kept.append(msg)

    if removed:
        payload["messages"] = kept
        token_est = saved_chars // 4
        report["stage1_dedup"] = {
            "removed": removed,
            "messages_before": len(messages),
            "messages_after": len(kept),
            "chars_saved": saved_chars,
            "tokens_saved_estimate": token_est,
        }
        counts = report.setdefault("counts", {})
        counts["dedup_removed"] = len(removed)
        return True
    return False
