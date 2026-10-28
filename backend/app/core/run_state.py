"""Per-run append-only baseline and compaction state."""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RunState:
    raw_messages: list[dict]
    cleaned_payload: dict[str, Any]
    epoch: int = 0
    turns: int = 1
    raw_payload: dict[str, Any] | None = None
    compaction_floor: int = 0
    last_seen: float = 0.0


_states: dict[str, RunState] = {}
_lock = threading.RLock()


def reset() -> None:
    with _lock:
        _states.clear()


def _same_prefix(old: list, new: list) -> bool:
    return len(new) >= len(old) and new[:len(old)] == old


def prepare(run_id: str, raw_payload: dict, threshold: float,
            estimate_fn) -> tuple[dict, RunState | None, bool, bool, bool]:
    """Return (working payload, prior state, append_only, compact, retry)."""
    messages = raw_payload.get("messages")
    if not isinstance(messages, list):
        return copy.deepcopy(raw_payload), None, False, False, False
    with _lock:
        _prune_locked(time.time())
        state = _states.get(run_id)
        if state is None:
            return copy.deepcopy(raw_payload), None, False, False, False
        if (state.raw_payload and raw_payload.get("model") == state.raw_payload.get("model")
                and raw_payload.get("messages") == state.raw_payload.get("messages")):
            # A retry of the same request reuses the already-cleaned view.
            return copy.deepcopy(state.cleaned_payload), state, False, False, True
        if not _same_prefix(state.raw_messages, messages) or len(messages) == len(state.raw_messages):
            return copy.deepcopy(raw_payload), state, False, False, False
        delta = copy.deepcopy(raw_payload)
        delta["messages"] = copy.deepcopy(messages[len(state.raw_messages):])
        combined = copy.deepcopy(state.cleaned_payload)
        combined["messages"] = combined.get("messages", []) + delta["messages"]
        limit = int(raw_payload.get("_contextgc_pinned_window") or 0)
        estimate = estimate_fn(combined)
        compaction = bool(limit and estimate >= int(limit * threshold)
                          and (not state.compaction_floor
                               or estimate > int(state.compaction_floor * 1.10)))
        return delta, state, True, compaction, False


def combine(state: RunState | None, cleaned_delta: dict, append_only: bool) -> dict:
    if not append_only or state is None:
        return cleaned_delta
    result = copy.deepcopy(state.cleaned_payload)
    result.update({k: copy.deepcopy(v) for k, v in cleaned_delta.items() if k != "messages"})
    result["messages"] = result.get("messages", []) + copy.deepcopy(cleaned_delta.get("messages", []))
    return result


def commit(run_id: str, raw_payload: dict, cleaned_payload: dict,
           epoch: int = 0, append_only: bool = False) -> None:
    messages = raw_payload.get("messages")
    if not isinstance(messages, list):
        return
    with _lock:
        previous = _states.get(run_id)
        try:
            from app.core.num_ctx import estimate_tokens
            estimate = estimate_tokens(cleaned_payload)
        except Exception:
            estimate = 0
        _states[run_id] = RunState(
            raw_messages=copy.deepcopy(messages),
            cleaned_payload=copy.deepcopy(cleaned_payload),
            epoch=(previous.epoch if previous else 0) + (1 if epoch else 0),
            turns=(previous.turns if previous else 0) + 1,
            raw_payload=copy.deepcopy(raw_payload),
            compaction_floor=(estimate if epoch else (previous.compaction_floor if previous else 0)),
            last_seen=time.time(),
        )


def _prune_locked(now: float) -> None:
    from app import config
    settings = config.get_settings()
    expired = [key for key, value in _states.items()
               if now - value.last_seen > settings.run_state_ttl_s]
    for key in expired:
        _states.pop(key, None)
        try:
            from app.core import num_ctx
            num_ctx.forget_run(key)
        except Exception:
            pass
    while len(_states) > settings.run_state_max:
        oldest = min(_states, key=lambda key: _states[key].last_seen)
        _states.pop(oldest, None)
