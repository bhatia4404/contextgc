"""Stage 1.6 — Anchored State (Milestone 1.6: the task-drift pin).

A pinned 4-key state block is prepended to the TOP of every outgoing prompt —
the only payload content that is *guaranteed stable* across turns, updated by a
heuristic state manager (no LLM in the hot path, consistent with Stage 0):

  {"user_intent":       the mission (stated once, persists — never overwritten by
                        lesser info)
   "completed_actions": rolling summaries of what's done (dedup'd, cap 5)
   "active_blocker":    the current blocking failure (key-result of the last dump)
   "next_step":         latest "next..." guidance or carried forward}

Why: recency bias on small local models (goal stated at turn 1 falls out of
attention by turn 30) AND our own aggressive GC (1.4/1.5) prunes the oldest —
i.e., the mission. GC must "forget history, never the mission". The pin also
acts as the drift *detector* for the Tier-2 fidelity fallback (2.3): a
noise-turn counter (consecutive informative-turns-without progress while a
blocker is unresolved) is carried in the report.

Insertion contract:
  - runs AFTER all cleaning stages (dedup -> offload -> reinject) so the pin is
    never touched by GC; num_ctx negotiation runs after it and includes its size.
  - injected as a SYNTHETIC system message (existing system content is never
    modified — destructive-free, env-invariant). Marker allows idempotence.
  - per-run state lives in a bounded run registry (old runs evicted).
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict

from app import config
from app.core.offload import _extract_summary

_ANCHOR_MARKER = "[ContextGC State Anchor]"
_GOAL_RE = re.compile(
    r"(?:remember the goal|the (?:goal|objective))\s*[:\-]?\s*(.{5,120})", re.IGNORECASE)
_NEXT_RE = re.compile(r"next(?: step| onto)?\s*[:\-]?\s*(.{5,120})", re.IGNORECASE)
_FAILURE_RE = re.compile(r"failed|error|mismatch|exception|❌|✗", re.IGNORECASE)
_NOISE_HINTS = ("git status", "git state", "what now?", "ok")


def _run_key(run_id: str, hashable: str) -> tuple[str, str]:
    return (run_id, hashable)


class RunStates:
    """Bounded per-run state registry (old runs evicted)."""

    def __init__(self, max_runs: int = 128):
        self.max_runs = max_runs
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, dict] = OrderedDict()

    def for_run(self, run_id: str) -> dict:
        with self._lock:
            if run_id not in self._runs:
                self._runs[run_id] = {
                    "user_intent": None,
                    "completed_actions": [],
                    "active_blocker": None,
                    "next_step": None,
                    "noise_turns": 0,
                }
            self._runs.move_to_end(run_id)
            while len(self._runs) > self.max_runs:
                self._runs.popitem(last=False)
            return self._runs[run_id]

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


_STATE = RunStates()


def reset_state() -> None:
    _STATE.clear()


def _is_anchor(msg: dict) -> bool:
    c = msg.get("content") or ""
    return isinstance(c, str) and c.lstrip().startswith(_ANCHOR_MARKER)


def update_state(state: dict, raw_payload: dict, report: dict) -> None:
    """Per-turn heuristic updates: goal once, blocker from the lastSuccessful
    dump summary, next-step from latest user guidance, completed roll."""
    updates: list[str] = []
    messages = [m for m in raw_payload.get("messages", []) if isinstance(m, dict)]
    users = [m for m in messages if m.get("role") == "user" and not _is_anchor(m)]
    latest_user = users[-1] if users else None
    latest_user_text = (latest_user.get("content") or "") if latest_user else ""

    # --- goal: stated once; persist (never overwritten by lesser turns) ---
    # scan ALL user messages chronologically (frameworks re-send history; the
    # mission statement may be a first-turn copy, not the latest turn)
    if state["user_intent"] is None:
        for m in users:
            g = _GOAL_RE.search(m.get("content") or "")
            if g:
                state["user_intent"] = g.group(1).strip()
                updates.append("user_intent")
                break

    # --- blocker: status line from the last dump (pointered or raw tool msg) ---
    tool_msgs = [m for m in messages if m.get("role") in ("tool", "function")]
    if tool_msgs:
        last_tool = tool_msgs[-1]
        c = last_tool.get("content") or ""
        text = c if isinstance(c, str) else json.dumps(c, default=str)
        if "Key result:" in text:
            summary = text.split("Key result:", 1)[1].splitlines()[0].strip()
        else:
            summary = _extract_summary(text)
        if _FAILURE_RE.search(summary) and summary != state["active_blocker"]:
            state["active_blocker"] = summary
            updates.append("active_blocker")
            state["noise_turns"] = 0
        elif "Key result:" in text and not _FAILURE_RE.search(summary) and state["active_blocker"]:
            # a newer SUCCESSIVE dump that resolved the failure keeps the blocker truthful
            state["active_blocker"] = None
            updates.append("active_blocker=NULL (resolved)")
            state["noise_turns"] = 0

    # --- next step: latest "next..." guidance; else carry forward ---
    n = _NEXT_RE.search(latest_user_text)
    if n:
        state["next_step"] = n.group(1).strip()
        updates.append("next_step")

    # --- completed actions: summaries of recent tool results (dedup'd) ---
    if tool_msgs:
        done = state["completed_actions"]
        before = list(done)
        for m in tool_msgs[-2:]:
            c = m.get("content") or ""
            text = c if isinstance(c, str) else json.dumps(c, default=str)
            s = _extract_summary(text)
            if s and s not in done:
                done.append(s)
        state["completed_actions"] = done[-config.get_settings().state_max_completed:]
        if state["completed_actions"] != before:   # only signal ACTUAL change
            updates.append("completed_actions")

    # --- drift signal: informative-but-inert turns (e.g. 'ok what now?') ---
    inert = (not updates) and latest_user and any(
        hint in latest_user_text.lower() for hint in _NOISE_HINTS)
    if inert and state["active_blocker"]:
        state["noise_turns"] += 1

    report["anchored_state_updates"] = updates
    if state["noise_turns"] >= 3:
        report["drift_signal_note"] = (
            "noise-turns >=3 while active_blocker unresolved; "
            "Tier-2 fidelity fallback would trigger here"
        )


def run_anchor(payload: dict, report: dict, run_id: str) -> bool:
    """Update per-run state + place the anchor immediately before latest user.

    Keeping the synthetic state near the active request avoids rewriting the
    stable system/history prefix. The compaction coordinator will eventually
    control when this suffix is refreshed.
    """
    if not config.get_settings().state_enabled:
        return False
    state = _STATE.for_run(run_id)
    update_state(state, payload, report)

    block = {
        "user_intent": state["user_intent"],
        "completed_actions": state["completed_actions"],
        "active_blocker": state["active_blocker"],
        "next_step": state["next_step"],
    }
    if not any(block.values()):
        return False
    content = f"{_ANCHOR_MARKER}\n{json.dumps(block, ensure_ascii=False)}"

    messages = payload.setdefault("messages", [])
    # Idempotence for a payload that already contains our synthetic message.
    for m in messages:
        if _is_anchor(m):
            m["content"] = content
            return False
    user_indexes = [i for i, m in enumerate(messages)
                    if isinstance(m, dict) and m.get("role") == "user"]
    insert_at = user_indexes[-1] if user_indexes else len(messages)
    messages.insert(insert_at, {"role": "system", "content": content})
    report["state_pinned"] = True
    return True


def explain_last_turn() -> str:  # pragma: no cover - CLI helper
    run = sorted(_STATE._runs, key=_STATE._runs.get)[-1]
    return json.dumps(_STATE.for_run(run), indent=1)
