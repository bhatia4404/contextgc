"""ContextGC proxy shell — Milestone 1.1.

Transparent FastAPI reverse proxy in front of the local inference sink
(Ollama by default). Milestone scope:

- Dual endpoint passthrough: OpenAI-compatible /v1/chat/completions
  (what CrewAI / AutoGen / LangChain call) and native Ollama /api/chat.
- Full SSE streaming passthrough (chunk-by-chunk, no buffering).
- GC layers exist as a hook (`core.gc.process_payload`) and are toggled
  OFF by default -> byte-faithful passthrough.

Acceptance bar (plan v2 §Tier1 1.1):
  an OpenAI client with stream=True pointed at ContextGC behaves
  identically to pointing at Ollama directly.

Later endpoints (1.2+): dashboard routes, WS telemetry, benchmark hooks.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse

from app.core import store as store_mod
from app.core import gc, telemetry, vram
from app.core import num_ctx as num_ctx_mod
from app import config

# Paths we forward to the sink. Both endpoint families are supported from
# day one so agent frameworks (OpenAI clients) and Ollama SDKs both work.
FORWARD_PREFIXES = ("/api/", "/v1/")
# Health/identity endpoints that belong to ContextGC itself.
OWN_PATHS = ("/healthz", "/ws/telemetry", "/telemetry/recent",
             "/store/stats", "/store/recent", "/store")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = config.get_settings()
    timeout = httpx.Timeout(settings.request_timeout_s, connect=10.0)
    app.state.http_client = httpx.AsyncClient(timeout=timeout)
    app.state.emit = lambda t, d: telemetry.broadcast(telemetry.emit(t, d))
    sampler = asyncio.create_task(_vram_sampler_loop(app))
    yield
    sampler.cancel()
    try:
        await sampler
    except asyncio.CancelledError:
        pass
    await app.state.http_client.aclose()


async def _vram_sampler_loop(app: FastAPI) -> None:
    """Milestone 1.8: periodic host-VRAM sample (+ Ollama /api/ps residency)."""
    settings = config.get_settings()
    client: httpx.AsyncClient = app.state.http_client
    while True:
        await asyncio.sleep(settings.telemetry_vram_interval_s)
        event = vram.sample()
        if settings.sink_is_ollama:
            # Ollama itself reports model VRAM residency — most truthful gauge
            try:
                r = await client.get(f"{settings.ollama_base_url}/api/ps")
                models = r.json().get("models", []) if r.status_code == 200 else []
                if models:
                    event["ollama_vram_mb"] = int(
                        sum(m.get("size_vram", 0) for m in models) / 1024**2)
                    event["ollama_models"] = [m.get("name") for m in models]
            except Exception:
                pass  # sink down — host sample still goes out
        emit = getattr(app.state, "emit", None)
        if emit:
            emit("vram_sample", event)
        global _last_vram
        _last_vram = event


app = FastAPI(title="ContextGC", version="1.8.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    settings = config.get_settings()
    return JSONResponse(
        {
            "service": "contextgc",
            "version": app.version,
            "sink": settings.sink,
            "sink_base_url": settings.sink_base_url,
            "gc_enabled": settings.gc_enabled,
        }
    )


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    await telemetry.ws_handler(ws)


@app.get("/telemetry/recent")
async def telemetry_recent(n: int = 100) -> JSONResponse:
    return JSONResponse({"events": telemetry.recent(n)})


# --- Append-only store debug endpoints (Milestone 1.2) ---


@app.get("/store/stats")
async def store_stats() -> JSONResponse:
    return JSONResponse(store_mod.get_store().stats())


@app.get("/store/recent")
async def store_recent(n: int = 50) -> JSONResponse:
    return JSONResponse({"turns": store_mod.get_store().recent(n)})


@app.get("/store/{turn_id}/raw")
async def store_raw(turn_id: int) -> Response:
    """Returns the ORIGINAL (pre-GC) payload of a turn — the fallback source."""
    raw = store_mod.get_store().get_raw(turn_id)
    if raw is None:
        return JSONResponse({"error": "turn not found"}, status_code=404)
    return Response(content=raw, media_type="application/json")


@app.get("/store/{turn_id}/clean")
async def store_clean(turn_id: int) -> Response:
    clean = store_mod.get_store().get_clean(turn_id)
    if clean is None:
        return JSONResponse({"error": "turn not found"}, status_code=404)
    return Response(content=clean, media_type="application/json")


def _hop_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    """Copy downstream response headers, minus hop-by-hop ones."""
    skip = {"content-length", "transfer-encoding", "connection", "keep-alive"}
    return [(k, v) for k, v in headers.multi_items() if k.lower() not in skip]


def _payload_tokens(body: bytes) -> int | None:
    """Estimated tokens of an already-serialized body (None if not chat JSON)."""
    try:
        return num_ctx_mod.estimate_tokens(json.loads(body))
    except Exception:
        return None  # /api/tags, Ollama errors, anything non-chat


_last_vram: dict | None = None  # freshest sampler snapshot (sampler sets it)


def emit_turn_telemetry(turn_id: int, path: str, run_id: str,
                        raw_body: bytes, clean_body: bytes, report: dict, *,
                        ttft_s: float | None, total_s: float,
                        upstream_status: int) -> None:
    """Milestone 1.8: the unified per-turn event the Phase-3 dashboard shows."""
    raw_tokens = _payload_tokens(raw_body)
    clean_tokens = _payload_tokens(clean_body)
    event = telemetry.build_turn_event(
        turn_id=turn_id, run_id=run_id, path=path,
        raw_bytes=len(raw_body), clean_bytes=len(clean_body),
        raw_tokens=raw_tokens if raw_tokens is not None else 0,
        clean_tokens=clean_tokens if clean_tokens is not None else 0,
        ttft_s=ttft_s, total_latency_s=total_s,
        upstream_status=upstream_status, gc_report=report,
        last_vram=_last_vram,
    )
    if report.get("numctx_negotiation") and clean_tokens:
        # CPU-only fallback gauge (Iteration-1/L3): KV footprint implied by
        # the negotiated window, shown when the VRAM sampler has no GPU.
        kv_bytes = telemetry.kv_gauge_bytes(
            clean_tokens, config.get_settings().kv_bytes_per_token)
        if kv_bytes:
            event["cpu_kv_gauge_bytes"] = kv_bytes
    # Both lifespan and the test fallback use this exact bus call.
    telemetry.broadcast(telemetry.emit("turn_telemetry", event))


async def _forward(request: Request, path: str) -> Response:
    """Byte-faithful forward of any request to the sink, streamed back.

    Streaming-or-not is decided entirely by the downstream response
    (Ollama sends SSE chunks when stream=true, single JSON otherwise) —
    the proxy never needs to parse the payload to relay transport.
    """
    client: httpx.AsyncClient = request.app.state.http_client
    if hasattr(request.app.state, "emit"):
        emit = request.app.state.emit
    else:  # no lifespan (tests): fall back to default bus
        emit = lambda t, d: telemetry.broadcast(telemetry.emit(t, d))  # noqa: E731

    body = await request.body()
    raw_body = body  # keep the original; GC must never destroy it
    run_id = request.headers.get("x-contextgc-run", "default")
    stages_preset = request.headers.get("x-contextgc-stages") or None
    t_start = time.perf_counter()
    try:
        body, report = gc.process_payload(path, body, run_id=run_id,
                                          stages_preset=stages_preset)
    except ValueError as exc:  # e.g. unknown stages preset
        return JSONResponse({"error": str(exc)}, status_code=400)

    turn_id = store_mod.get_store().append(path, run_id, raw_body, body, report)

    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in {
        "host", "content-length", "connection", "keep-alive", "transfer-encoding",
        "x-contextgc-run", "x-contextgc-stages",
    }}

    settings = config.get_settings()
    url = settings.sink_base_url + path  # full URL: sink base + forwarded path
    req = client.build_request(
        method=request.method,
        url=url,
        content=body,
        headers=fwd_headers,
    )
    resp = await client.send(req, stream=True)

    emit(
        "store_append",
        {"turn_id": turn_id, "path": path, "run_id": run_id,
         "raw_bytes": len(raw_body), "clean_bytes": len(body), "gc_report": report},
    )
    emit(
        "proxy_forward",
        {"path": path, "upstream_status": resp.status_code, "gc_report": report},
    )

    async def byte_iter() -> AsyncIterator[bytes]:
        ttft: float | None = None
        try:
            async for chunk in resp.aiter_bytes():
                if ttft is None:  # first BYTE from the sink = TTFT milestone
                    ttft = time.perf_counter() - t_start
                yield chunk
        finally:
            await resp.aclose()
            emit_turn_telemetry(
                turn_id, path, run_id, raw_body, body, report,
                ttft_s=ttft,
                total_s=time.perf_counter() - t_start,
                upstream_status=resp.status_code,
            )

    return StreamingResponse(
        byte_iter(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
        headers=dict(_hop_headers(resp.headers)),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    full_path = "/" + path.lstrip("/")
    if full_path in OWN_PATHS or not any(full_path.startswith(p) for p in FORWARD_PREFIXES):
        return JSONResponse({"error": f"not forwardable: {full_path}"}, status_code=404)
    return await _forward(request, full_path)


def main() -> None:  # pragma: no cover - manual entrypoint
    import uvicorn

    settings = config.get_settings()
    uvicorn.run("app.main:app", host=settings.proxy_host, port=settings.proxy_port)


if __name__ == "__main__":  # pragma: no cover
    main()
