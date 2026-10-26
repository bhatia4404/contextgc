"""Dynamic num_ctx negotiation (Milestone 1.7 — plan v2, Iteration-1/L2 fix).

Ollama sizes the KV cache from the `num_ctx` SETTING, not from the actual
prompt length — so message pruning alone saves tokens/latency/accuracy but
NO VRAM. This stage closes the loop: it estimates the cleaned payload's
token count and right-sizes the request's context window:

    chosen = clamp(ceil(est_tokens * headroom), floor, cap)
             rounded UP to a 1024 multiple (deterministic gauges)

Injection matrix (verified against Ollama behavior, Iteration-11 search):
  /api/chat  (native)  -> options.num_ctx            [always honored]
  /v1/chat/completions -> top-level num_ctx          [honored on merged-PR builds;
      only injected when sink is Ollama (`auto` mode); cloud sinks
      (Baseten/Fireworks/Modal) get report-only so no extra keys leak]
  client explicitly set num_ctx/options.num_ctx -> RESPECTED (not overridden)
"""
from __future__ import annotations

import math
from typing import Any

from app import config

_CHARS_PER_TOKEN = 4        # calibrated against llama family tokenizers


def estimate_tokens(payload: dict) -> int:
    """Cheap deterministic estimator; no model call in the hot path."""
    total = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            text = content if isinstance(content, str) else json_dumps(content)
            total += len(text) // _CHARS_PER_TOKEN + 8   # role/frame overhead

    tools = payload.get("tools")
    if isinstance(tools, list):
        # tool schemas are JSON-heavy; ~2.5 chars/token when minified
        total += len(json_dumps(tools)) // 3

    for k in ("system", "prompt"):  # /api/generate compat
        v = payload.get(k)
        if isinstance(v, str):
            total += len(v) // _CHARS_PER_TOKEN + 8
    return total


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)


def _round_up_to(x: int, base: int = 1024) -> int:
    return int(math.ceil(x / base)) * base


def negotiate(payload: dict, report: dict, endpoint_family: str) -> bool:
    """Set num_ctx on the payload per the matrix. Returns True if modified."""
    settings = config.get_settings()
    est = estimate_tokens(payload)
    chosen = min(
        max(
            _round_up_to(int(math.ceil(est * settings.numctx_headroom) / 1) or 1,
                         base=1024),
            settings.numctx_min,
        ),
        settings.numctx_max,
    )
    neg = report.setdefault("numctx_negotiation", {})
    neg.update({
        "estimated_tokens": est,
        "chosen_num_ctx": chosen,
        "headroom": settings.numctx_headroom,
        "floor": settings.numctx_min,
        "cap": settings.numctx_max,
    })

    if endpoint_family == "api":  # native /api/chat
        options = payload.get("options")
        if isinstance(options, dict) and options.get("num_ctx"):
            neg["respected_client"] = True; neg["client_num_ctx"] = options["num_ctx"]
            return False
        payload.setdefault("options", {})["num_ctx"] = chosen
        neg["injected_where"] = "options.num_ctx"
        return True

    # OpenAI-compat /v1: only for Ollama sinks; never leak extras to cloud sinks
    if settings.sink_is_ollama and settings.numctx_v1_mode == "auto":
        existing = payload.get("num_ctx")
        if existing:
            neg["respected_client"] = True; neg["client_num_ctx"] = existing
            return False
        payload["num_ctx"] = chosen
        neg["injected_where"] = "top_level"
        return True

    neg["injected_where"] = "report_only"
    neg["report_only_reason"] = (
        "OpenAI-compat endpoint used with a non-sync sink "
        "(or numctx_v1_mode=report); benchmark should rely on native lane"
    )
    return False
