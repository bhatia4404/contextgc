# ContextGC — Python Backend (developer guide)

## What this is
A transparent FastAPI reverse proxy in front of Ollama. Agent frameworks
(CrewAI / AutoGen / LangChain) point their `base_url` here instead of at
Ollama; ContextGC garbage-collects the outgoing context payload before
forwarding it. All GC layers are OFF by default (GC_ENABLED=false)
→ byte-faithful passthrough.

## Quick start

```powershell
# 1) install deps
cd backend
python -m pip install -r requirements.txt

# 2) run the proxy (assumes Ollama on localhost:11434)
python -m uvicorn app.main:app --host 127.0.0.1 --port 11400
```

## Configuration (env vars, prefix CONTEXTGC_)

| Var | Default | Meaning |
|---|---|---|
| `CONTEXTGC_SINK` | `ollama` | `ollama` for native, or any full OpenAI-compat base URL (Baseten/Fireworks/Modal Endpoints) |
| `CONTEXTGC_OLLAMA_BASE_URL` | `http://localhost:11434` | used when sink=ollama |
| `CONTEXTGC_GC_ENABLED` | `false` | master GC toggle (all layers off = byte-faithful passthrough) |
| `CONTEXTGC_PROXY_PORT` | `11400` | proxy listen port |

## Endpoints

- Everything under `/api/*` and `/v1/*`: forwarded byte-faithfully (SSE included).
- `/healthz` — proxy identity + current config.
- `/ws/telemetry` — WebSocket: replays ring buffer, then streams live proxy events.
- `/telemetry/recent` — last 100 events as JSON.

## Point an agent framework at it

```python
# Any OpenAI client (CrewAI, AutoGen, LangChain):
client = OpenAI(base_url="http://localhost:11400/v1", api_key="contextgc")
```

## Tests

```powershell
python -m pytest tests -q
```

`tests/test_passthrough.py` encodes the proxy acceptance bar:
- A: OpenAI-compat `/v1/chat/completions` passthrough (body byte-faithful)
- B: Native `/api/chat` passthrough
- C: SSE streaming passthrough (chunks byte-faithful, `text/event-stream`)
- D: GC-off = identity (no payload modification at all)
- E: ContextGC-owned endpoints never leak to the sink
- F: Unknown paths 404 from the proxy, not the sink

## Repo layout

```
backend/
  app/main.py          # proxy shell + routing + streaming forward
  app/config.py        # env-driven settings (sink swap = env var, no code)
  app/core/gc.py       # GC pipeline orchestrator
  app/core/telemetry.py# ring buffer + WS fan-out + per-turn event builder
  app/core/vram.py     # VRAM sampler: pynvml -> nvidia-smi -> graceful none
  tests/               # pytest acceptance tests (mock sink, no Ollama needed)
```

<!-- Milestone-internal notes (dev log) — hidden for the public repo
## Milestone 1.8 recap — telemetry bus

- **`turn_telemetry` event** (one per proxied turn, on the WS + ring buffer):
  `raw_tokens`/`clean_tokens`/`tokens_saved{,_pct}`, `ttft_s` (time to first
  byte from the sink — measured inside the streaming loop), `total_latency_s`,
  `stage_tokens_saved` (dedup/offload positive; `ref_reinject` **negative**,
  making the accuracy contract visible), `num_ctx` negotiation summary,
  `meta` (state_pinned, procedural_locked adds), and a `vram` snapshot.
- **VRAM samplings** (`core/vram.py`, zero hard deps): source priority
  `pynvml` → `nvidia-smi` CLI → graceful `none`; the lifespan background loop
  (`telemetry_vram_interval_s`, default 3s) also polls **Ollama `/api/ps`**
  and emits `vram_sample` with `ollama_vram_mb` (model residency, most
  truthful gauge). CPU-only rigs degrade to the
  `cpu_kv_gauge_bytes` = clean_tokens × `kv_bytes_per_token` (default
  114,688 = llama3.2:3b KV/token) on each chat turn — per Iteration-1/L3.
- Dashboard contract: connect to `/ws/telemetry`, receive `turn_telemetry`
  + `vram_sample` events; `/telemetry/recent?n=` gives a REST tail.

### Seeing 1.8 live

```powershell
cd backend
python -m uvicorn app.main:app --port 11400
# then in another terminal, generate traffic through the proxy
python scripts\smoke_test.py
# every turn now emits turn_telemetry on the wire + /telemetry/recent
```
-->

<!-- Milestone-internal notes (dev log) — hidden for the public repo
## Milestone 1.5 recap (live-verified: 14,069 → 1,136 chars = 92% cut)

- A 301-line, ~2,541-token tool dump → `POINTERED` `[Tool Output Ref #1 — bash | 301 lines (~2,541 tokens) saved to scratchpad…  Key result: 3 tests failed]`; freshest `keep_recent=2` tool results stay raw; original written verbatim to `data/scratchpad/ref_1_*.txt` (retrieval: `read_ref` tool if defined, else proxy re-injection).
- **Ollama quirks discovered:** (a) native `/api/chat` requires `tool_calls.function.arguments` as a JSON object (or JSON-encoded object) — CLI-string args → `400 "Value looks like object, but can't find closing '}'"`; (b) assistant `content:""` also rejected; demo tools must send `content` non-empty. Real frameworks always send proper object args — production unaffected.
- Stage order: classify → dedup → offload → num_ctx (window sized to post-offload payload).
-->


## Design invariants

1. **GC is a view, not a deletion** — originals are never lost (append-only store).
2. **Visually provable** — every stage's effect is inspectable via telemetry & diffing.
3. **No LLM in the hot path** — GC stages must be cheap heuristics first.

<!-- Milestone-internal notes (dev log) — hidden for the public repo
## num_ctx behavior recap (live-verified, Milestone 1.7)

- Live run: `estimated_tokens: 450 → chosen_num_ctx: 2048` (floor applied), `injected_where: "options.num_ctx"`.
- Injection matrix (search-supported): native `/api/chat` = `options.num_ctx` always honored; `/v1` = officially ignored (top-level supported on PR-merged Ollama builds; `numctx_v1_mode=auto` injects on Ollama sinks only). Cloud sinks = report-only (no key pollution).
- Client-provided `num_ctx`/`options.num_ctx` is ALWAYS respected, never overridden.
-->
