# ContextGC

**by ZeroSpike**

> A drop-in, local-first middleware that keeps long agent runs inside a fixed,
> machine-sized context window while preserving the model's prompt-prefix cache.
> Point your CrewAI / AutoGen / LangChain `base_url` at it and your local agents use fewer
> tokens, stay accurate, and stop breaking on long runs — a live telemetry feed shows the
> cleaning happening message by message.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-proxy-009688)
![Tests](https://img.shields.io/badge/tests-97%20passing-brightgreen)
![License](https://img.shields.io/badge/license-—-lightgrey)

---

## The problem

Local multi-agent loops (researcher → coder → reviewer) often resend growing histories
and tool outputs. Four real, reproducible failures are:

1. **Silent truncation or context errors** — when history exceeds usable `num_ctx`,
   older task state may be lost or the runtime may reject the request.
2. **Context rot / accuracy collapse** — duplicated and stale turns degrade small-model
   (3B-class) reasoning even *inside* the context window.
3. **Long-run latency** — prompt rewrites, reloads, or cache misses make tool loops costly.
4. **KV-cache pressure** — an unnecessarily large `num_ctx` can push a model off the GPU.

Conventional fixes are cloud APIs, framework rewrites, or memory stores.
**ContextGC is none of those** — it's a transparent HTTP proxy between the framework you
already use and your local runtime.

---

## How it works

```
CrewAI / AutoGen / LangChain app
        │  base_url = http://localhost:11400     ← the only change required
        ▼
┌─────────────────────────────────────────────────────────────┐
│ ContextGC Proxy (FastAPI + httpx)                           │
│  • Dual passthrough: /v1/chat/completions + /api/chat       │
│  • Full SSE streaming, byte-faithful when GC is off         │
│  • Append-only store: originals are NEVER lost              │
│  • Per-run state (prefix baseline, pinned window, caches)   │
├─────────────────────────────────────────────────────────────┤
│ Request-time GC pipeline                                    │
│  S0  Memory classifier (procedural=lock, semantic, episodic) │
│  S1  Request-local MinHash/LSH structural dedup             │
│  S2  Pointer offload of dump-sized tool results → disk      │
│  S2b On-demand re-injection when an agent asks for a ref    │
│  S2c Semantic contraction: embedding merge of rephrased     │
│      facts (same-namespace + negation guard)                │
│  S3  Anchored 4-key state suffix at compaction boundaries   │
│  S2d Importance eviction under window pressure (recency ×   │
│      length; live turn, system prompts, state pin,          │
│      referenced-later and standing facts hard-protected)    │
│  S5  Fidelity fallback: loop/confusion rule engine serves   │
│      a recovery turn from the raw view                      │
│  S4  Machine-aware fixed num_ctx pinned per run              │
├─────────────────────────────────────────────────────────────┤
│ Telemetry bus: per-turn tokens, TTFT, per-stage savings,    │
│ VRAM (pynvml → nvidia-smi → Ollama /api/ps → CPU KV gauge)  │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
                  Ollama (llama3.2:3b)
```

The causal chain for memory savings:

**machine capacity → fixed safe `num_ctx` → stable KV cache.** Prompt cleaning keeps
the task inside that window; it does not by itself reduce VRAM. Compaction is an
occasional epoch boundary, not a per-turn rewrite.

---

## Design invariants

1. **GC is a view, never a deletion.** Every original payload lives in the append-only
   store; offloaded dumps sit in a disk scratchpad and are re-injected on demand. Nothing
   is ever deleted — GC only changes *what gets sent*.
2. **Visually provable.** Every stage's effect is inspectable via the telemetry bus and
   the per-turn raw/clean endpoints.
3. **No LLM in the hot path.** All GC stages are cheap deterministic heuristics.

---

## Quick start

```powershell
# 1) install deps
cd backend
python -m pip install -r requirements.txt

# 2) run the proxy (assumes Ollama on localhost:11434)
python -m uvicorn app.main:app --host 127.0.0.1 --port 11400

# 3) satellite check
curl http://localhost:11400/healthz
```

Point any OpenAI-compatible client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11400/v1", api_key="contextgc")
# CrewAI / AutoGen / LangChain: same idea — just set base_url.
```

That's it. Framework code stays untouched.

---

## The GC pipeline

| Stage | What it does | Mechanism |
|---|---|---|
| **S0 — Memory classifier** | Categorizes every message so we only GC what's safe | Rule-based (no LLM): role/shape/markers/length → `procedural` (immovable), `semantic`, `episodic` |
| **S1 — Structural dedup** | Removes safe duplicates within the current payload | Request-local MinHash/LSH; prior requests are never deletion evidence |
| **S2 — Pointer offload** | Frees the window from dump-sized tool results | Result → disk scratchpad, prompt keeps `[Tool Output Ref #n …  Key result: …]`; freshest N stay raw |
| **S2b — On-demand re-injection** | Restores accuracy when the agent asks for a ref | "Ref #12"-mention detection → full original restored for exactly that turn |
| **S3 — Anchored state** | Prevents task drift on long runs | Four-key state suffix refreshed at compaction boundaries |
| **S4 — fixed num_ctx** | Keeps KV allocation stable | One machine-sized tier is pinned for the run; no per-turn reloads |

Pipeline order matters: classify → dedup → offload → re-inject check → anchor → num_ctx
(the window is sized to the *post-cleaning* payload).

---

## Telemetry — see the GC working

Every proxied turn emits a unified event on `/ws/telemetry` (WebSocket) and `/telemetry/recent`:

```json
{
  "type": "turn_telemetry",
  "turn_id": 9,
  "raw_tokens": 5821,
  "clean_tokens": 1142,
  "tokens_saved": 4679,
  "tokens_saved_pct": 80.4,
  "ttft_s": 0.2841,
  "total_latency_s": 3.1026,
  "upstream_status": 200,
  "stage_tokens_saved": { "tool_offload": 4459, "stage1_dedup": 220 },
  "num_ctx": { "estimated_tokens": 1142, "chosen_num_ctx": 2048, "injected_where": "options.num_ctx" },
  "vram": { "source": "pynvml", "vram_used_mb": 3021, "vram_total_mb": 8192 }
}
```

VRAM sampling needs no hard dependencies: it tries `pynvml` → `nvidia-smi` CLI →
Ollama's own `/api/ps` (model residency) → and on CPU-only rigs gracefully falls back to
a **KV-token gauge** (`clean_tokens × kv_bytes_per_token`).

Watch it live:

```powershell
python scripts\smoke_test.py        # traffic through the proxy
# then connect to ws://localhost:11400/ws/telemetry
```

---

## Configuration

All knobs are environment variables (prefix `CONTEXTGC_`), so the sink/backend can be
swapped with zero code changes:

| Var | Default | Meaning |
|---|---|---|
| `CONTEXTGC_SINK` | `ollama` | `ollama`, or any full OpenAI-compatible base URL |
| `CONTEXTGC_OLLAMA_BASE_URL` | `http://localhost:11434` | used when sink is ollama |
| `CONTEXTGC_GC_ENABLED` | `false` | master GC switch (off = byte-faithful passthrough) |
| `CONTEXTGC_GC_STAGE1_ENABLED` | `true` | Stage-1 dedup |
| `CONTEXTGC_OFFLOAD_ENABLED` | `true` | pointer offload |
| `CONTEXTGC_OFFLOAD_THRESHOLD_TOKENS` | `800` | dump-size threshold |
| `CONTEXTGC_STATE_ENABLED` | `true` | anchored state block |
| `CONTEXTGC_NUMCTX_ENABLED` | `true` | fixed per-run window injection |
| `CONTEXTGC_NUMCTX_DEFAULT` | `8192` | fallback fixed context tier per run |
| `CONTEXTGC_KEEP_ALIVE` | `30m` | keep the Ollama model resident between turns |
| `CONTEXTGC_VRAM_OVERRIDE_MB` | unset | simulate a GPU capacity for benchmarks |
| `CONTEXTGC_TELEMETRY_VRAM_ENABLED` | `true` | background VRAM sampler |
| `CONTEXTGC_TELEMETRY_VRAM_INTERVAL_S` | `3.0` | sampler poll period |
| `CONTEXTGC_PROXY_PORT` | `11400` | proxy listen port |

---

## Repository layout

```
crewai_agent/                 # standalone CrewAI customer application
  code_maintenance_agent.py  # Agent 1: fixes an existing repository
  incident_response_agent.py # Agent 2: investigates a production incident
demo_repository/              # separate existing customer repository
  src/opsdesk/                # production-style incident service
  tests/                      # deterministic repository checks
  data/incidents/             # incident logs used by Agent 2
backend/
  app/main.py            # proxy shell, routing, streaming forward, telemetry wiring
  app/config.py          # env-driven settings
  app/core/
    classifier.py        # S0 memory-type classifier
    hash_filter.py       # S1 MinHash/LSH dedup
    offload.py           # S2 pointer offload + re-injection
    anchored_state.py    # S3 anchored state block
    num_ctx.py           # S4 fixed per-run window pinning
    store.py             # append-only SQLite store (originals never lost)
    caches.py            # bounded process caches
    gc.py                # pipeline orchestrator
    telemetry.py         # ring buffer + WS bus + per-turn event builder
    vram.py              # VRAM sampler (pynvml → nvidia-smi → graceful none)
  scripts/               # smoke test, savings report, replay/explain tools
  tests/                 # pytest acceptance suite (mock sink — no Ollama needed)
```

---

## Tests

```powershell
cd backend
python -m pytest tests -q
```

97 passing. The acceptance suite runs against a **mock sink**, so no Ollama or GPU is
required. It covers: byte-faithful passthrough (OpenAI + native + SSE streaming), the
store, the classifier, dedup, offload & re-injection, anchored state, num_ctx
pinning, and the telemetry bus.

---

## Status & roadmap

Implemented foundation — **proxy, correctness safeguards, and v3 cache-safety work**:

- **Tier 1 ✅** proxy + SSE passthrough · append-only store · memory classifier ·
   request-local dedup · pointer offload + re-injection · tail state anchor · fixed per-run num_ctx ·
  telemetry bus
- **Tier 2 ✅** semantic contraction (embedding merge, same-namespace + negation
  guard, never merges contradictions) · importance eviction (pressure-triggered,
  recency × length with hard protections for the live turn, system prompts, the
  state pin, referenced-later turns and standing facts) · deterministic fidelity
  fallback (loop/confusion rule engine → recovery turn replays the raw view)

**97 tests passing** (mock sink, no Ollama required). Runtime capacity probing,
incremental prefix views, and compaction epochs are implemented; hardware benchmark
execution remains environment-dependent.

### Pipeline order

`fidelity check (raw evidence) → S0 classify → S1 dedup → S2 offload →
S2b re-inject → S4 semantic contraction → S3 anchored state → S5 importance
eviction (pressure-triggered) → incremental combine/compaction → fixed num_ctx`

**Build in progress:**

- 🔲 ~~**Semantic contraction** — merge near-duplicate *facts* (embedding similarity,
  same-agent namespace, negation guard) beyond structural dedup~~ ✅ **shipped (Tier 2)**
- 🔲 ~~**Importance eviction** — policy-driven eviction (recency × referenced-later ×
  length × state-entailment) for stale-but-unique turns~~ ✅ **shipped (Tier 2)**
- 🔲 ~~**Fidelity fallback** — deterministic quality-signal detection → replay the
  uncompressed view for a recovery turn~~ ✅ **shipped (Tier 2)**
- ✅ **Benchmark harness** — records Ollama final-chunk timing/count fields and supports
   fixed-reply replay for Lane A/B latency comparisons
- 🔲 **Dashboard** — split-screen race mode, prompt diff viewer, live gauges

Benchmark numbers are reported only when measured — this README deliberately contains
no performance claims.

### Context-window configuration

ContextGC pins one context window per run. If a client supplies `num_ctx` on the
first request, that value becomes the run's pinned value. A later client change is
intentionally overridden and reported as `client_num_ctx_overridden`; restart the
run with a new `x-contextgc-run` value to select a different window.

The capacity probe uses VRAM telemetry, `/api/show` architecture metadata, and
`/api/tags` model size. If any input is unavailable—especially on CPU-only systems—
it falls back to `CONTEXTGC_NUMCTX_DEFAULT`.

---

## How it compares

| Existing work | What it is | Why ContextGC is different |
|---|---|---|
| Anthropic context editing + memory tool | Auto-clears stale tool results, keeps pointer refs | Cloud API only (Claude); ContextGC does it **locally, proxy-level, model-agnostic** |
| Letta / MemGPT | Virtual context, paging to disk | Rebuilds your app on their framework; ContextGC is **zero-code drop-in** |
| LLMLingua / LLMLingua-2 | Neural prompt compression | Token-level compression; no memory governance, no multi-agent namespace, no telemetry |
| AgentFold (Alibaba, '25) | Agent decides to fold subtasks | Requires custom agents/prompts; ContextGC is **transparent middleware** |
| Mem0 / Zep-Graphiti / Cognee | Long-term memory layers | Store-and-retrieve across sessions; ContextGC is **real-time working-memory GC** in the request path |
| LangChain `trim_messages`, AutoGen limits | Framework-internal trimming | Bound to one framework; no offload, no state anchoring, no telemetry |

**One-liner:** *everything above is a cloud API, a framework you must adopt, or a memory
store. ContextGC is a transparent garbage collector between the framework you already use
and Ollama — nothing is ever deleted, only what gets sent.*

---

## Documentation

- [`backend/README.md`](backend/README.md) — developer guide: configuration details,
  endpoints, test coverage, live-verified stage behavior.

---

**Built by ZeroSpike.**
