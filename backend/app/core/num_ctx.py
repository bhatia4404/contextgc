"""Context-window estimation and v3 fixed-per-run pinning.

Ollama sizes the KV cache from ``num_ctx`` at model-load time.  Changing it on
every request reloads the model and destroys the prompt-prefix cache, so the
proxy now pins one value for the lifetime of a run.  A future machine-capacity
probe can supply the tier; this module remains the deterministic policy layer.
"""
from __future__ import annotations

import json
import math
import threading
from typing import Any

from app import config

_CHARS_PER_TOKEN = 4
_RUN_WINDOWS: dict[str, int] = {}
_LOWERED: set[str] = set()   # runs that already took their one residency step-down
_LOCK = threading.Lock()


def estimate_tokens(payload: dict) -> int:
    total = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            text = content if isinstance(content, str) else json_dumps(content)
            total += len(text) // _CHARS_PER_TOKEN + 8
    tools = payload.get("tools")
    if isinstance(tools, list):
        total += len(json_dumps(tools)) // 3
    for k in ("system", "prompt"):
        v = payload.get(k)
        if isinstance(v, str):
            total += len(v) // _CHARS_PER_TOKEN + 8
    return total


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _round_down_tier(x: int) -> int:
    tiers = (4096, 8192, 16384, 32768)
    return max((tier for tier in tiers if tier <= x), default=2048)


def _round_up_to(x: int, base: int = 1024) -> int:
    """Compatibility helper for estimator/benchmark callers."""
    return int(math.ceil(x / base)) * base


def reset_state() -> None:
    with _LOCK:
        _RUN_WINDOWS.clear()
        _LOWERED.clear()


def pinned_window(run_id: str) -> int | None:
    with _LOCK:
        return _RUN_WINDOWS.get(run_id)


def forget_run(run_id: str) -> None:
    with _LOCK:
        _RUN_WINDOWS.pop(run_id, None)
        _LOWERED.discard(run_id)


def lower_window(run_id: str) -> int | None:
    """Drop one tier after a post-load residency check detects CPU spill.

    At most once per run: every change is a model reload, so a run never
    walks down the tiers turn after turn.
    """
    with _LOCK:
        current = _RUN_WINDOWS.get(run_id)
        if current is None:
            return None
        if run_id in _LOWERED:
            return current
        tiers = (2048, 4096, 8192, 16384, 32768)
        lower = [tier for tier in tiers if tier < current]
        if lower:
            _RUN_WINDOWS[run_id] = max(lower)
        _LOWERED.add(run_id)
        return _RUN_WINDOWS[run_id]


def _choose_initial_window(settings: config.Settings, estimated: int) -> int:
    # ``numctx_default`` is the safe fallback until the machine probe selects
    # a tier. It is deliberately independent of the current prompt size.
    default = max(settings.numctx_min, min(settings.numctx_default, settings.numctx_max))
    return _round_down_tier(default) if default >= 4096 else settings.numctx_min


def negotiate(payload: dict, report: dict, endpoint_family: str, run_id: str = "default") -> bool:
    """Inject one stable context window for a run; never resize per turn."""
    settings = config.get_settings()
    est = estimate_tokens(payload)
    existing_client = None
    if endpoint_family == "api":
        options = payload.get("options")
        if isinstance(options, dict) and options.get("num_ctx"):
            existing_client = int(options["num_ctx"])
    elif endpoint_family == "v1":
        if payload.get("num_ctx"):
            existing_client = int(payload["num_ctx"])

    with _LOCK:
        chosen = _RUN_WINDOWS.get(run_id)
    capacity_details = None
    if chosen is None:
        if existing_client:
            candidate = existing_client
        else:
            from app.core.capacity import choose_window
            candidate, capacity_details = choose_window(
                payload.get("model"), settings.numctx_default,
                settings.numctx_min, settings.numctx_max,
            )
        candidate = max(settings.numctx_min, min(candidate, settings.numctx_max))
        with _LOCK:
            chosen = _RUN_WINDOWS.setdefault(run_id, candidate)

    neg = report.setdefault("numctx_negotiation", {})
    neg.update({
        "estimated_tokens": est,
        "chosen_num_ctx": chosen,
        "pinned": True,
        "run_id": run_id,
        "client_num_ctx": existing_client,
    })
    if capacity_details is not None:
        neg["capacity_probe"] = capacity_details
    if existing_client is not None:
        neg["respected_client"] = existing_client == chosen
        if existing_client != chosen:
            neg["client_num_ctx_overridden"] = True

    if endpoint_family == "api":
        options = payload.setdefault("options", {})
        changed = options.get("num_ctx") != chosen
        options["num_ctx"] = chosen
        neg["injected_where"] = "options.num_ctx"
        return changed
    if settings.sink_is_ollama and settings.numctx_v1_mode == "auto":
        changed = payload.get("num_ctx") != chosen
        payload["num_ctx"] = chosen
        neg["injected_where"] = "top_level"
        return changed
    neg["injected_where"] = "report_only"
    neg["report_only_reason"] = "OpenAI-compatible non-Ollama sink"
    return False
