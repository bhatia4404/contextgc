"""Telemetry bus (Milestone 1.8).

In-memory ring buffer + WebSocket fan-out, plus the unified per-turn event
schema the Phase-3 dashboard consumes. VRAM sampling itself lives in
`core.vram.py` (sync) and main.py's lifespan loop (Ollama `/api/ps`).
"""
from __future__ import annotations

import asyncio
import time
import uuid
import weakref

from fastapi import WebSocket

_RING_SIZE = 500
_events: list[dict] = []
_subscribers: weakref.WeakSet = weakref.WeakSet()


def emit(event_type: str, data: dict) -> dict:
    event = {
        "ts": time.time(),
        "id": uuid.uuid4().hex[:12],
        "type": event_type,
        **data,
    }
    _events.append(event)
    del _events[:-_RING_SIZE]
    return event


async def ws_handler(ws: WebSocket) -> None:
    """Replays the ring buffer, then streams new events.

    Safe under concurrency: each subscriber consumes its own asyncio.Queue.
    """
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.add(queue)  # type: ignore[arg-type]
    try:
        for event in list(_events):
            await ws.send_json(event)
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except Exception:
        pass
    finally:
        _subscribers.discard(queue)  # type: ignore[arg-type]


def broadcast(event: dict) -> None:
    for queue in list(_subscribers):
        try:
            queue.put_nowait(event)  # type: ignore[union-attr]
        except Exception:
            _subscribers.discard(queue)


def recent(n: int = 100) -> list[dict]:
    """Ring-buffer tail without touching a private module member."""
    return list(_events[-n:])


# --- Unified per-turn event schema (dashboard contract) ---

_STAGES_WITH_SAVINGS = {                       # gc_report key -> savings field
    "stage1_dedup": "tokens_saved_estimate",
    "tool_offload": "tokens_saved_estimate",
}


def summarize_gc_report(report: dict) -> dict:
    """Extract per-stage savings + negotiation info from a gc_report.

    Stages that only annotate (classifier) or pin (anchored_state) have no
    token delta; they appear as metadata, not savings. `ref_reinject` is a
    *negative* saving (pointer restored on purpose) so the dashboard can show
    the accuracy contract working instead of silently undoing savings.
    """
    out: dict = {"stage_tokens_saved": {}, "meta": {}}
    saved_total = 0
    for stage, field in _STAGES_WITH_SAVINGS.items():
        if stage in report:
            n = int(report[stage].get(field) or 0)
            out["stage_tokens_saved"][stage] = n
            saved_total += n
    reinject = report.get("ref_reinject")
    if reinject:
        restored = sum(r.get("tokens_restored_estimate", 0)
                       for r in reinject.get("restored", []))
        out["stage_tokens_saved"]["ref_reinject"] = -restored
        saved_total -= restored
    out["tokens_saved_net"] = max(saved_total, 0)

    neg = report.get("numctx_negotiation")
    if neg:
        out["num_ctx"] = {k: neg[k] for k in (
            "estimated_tokens", "chosen_num_ctx", "injected_where",
            "respected_client", "client_num_ctx") if k in neg}
    for extra, key in (("state_pinned", "state_pinned"),
                       ("anchored_state_updates", "anchored_state_updates"),
                       ("procedural_locked", "procedural_locked")):
        if extra in report:
            out["meta"][key] = report[extra]
    return out


def build_turn_event(*, turn_id: int, run_id: str, path: str,
                     raw_bytes: int, clean_bytes: int,
                     raw_tokens: int, clean_tokens: int,
                     ttft_s: float | None, total_latency_s: float,
                     upstream_status: int, gc_report: dict,
                     last_vram: dict | None = None) -> dict:
    """One dashboard-ready event per proxied turn."""
    saved = max(raw_tokens - clean_tokens, 0)
    event = {
        "type": "turn_telemetry",
        "turn_id": turn_id,
        "run_id": run_id,
        "path": path.split("?")[0],
        "raw_tokens": raw_tokens,
        "clean_tokens": clean_tokens,
        "tokens_saved": saved,
        "tokens_saved_pct": round(saved / raw_tokens * 100, 1) if raw_tokens else 0.0,
        "raw_bytes": raw_bytes,
        "clean_bytes": clean_bytes,
        "ttft_s": round(ttft_s, 4) if ttft_s is not None else None,
        "total_latency_s": round(total_latency_s, 4),
        "upstream_status": upstream_status,
        **summarize_gc_report(gc_report),
    }
    if last_vram:                       # host VRAM at response completion
        event["vram"] = {k: last_vram.get(k) for k in (
            "source", "vram_used_mb", "vram_total_mb", "error")}
    return event


def kv_gauge_bytes(est_tokens: int, kv_bytes_per_token: int) -> int | None:
    """CPU-only fallback gauge: KV-cache footprint implied by token count."""
    return est_tokens * kv_bytes_per_token if kv_bytes_per_token else None  # type: ignore[arg-type]
