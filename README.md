# ContextGC

**by ZeroSpike**

> A drop-in, local-first middleware that keeps long agent runs inside a fixed,
> machine-sized context window while preserving the model's prompt-prefix cache.
> Point your CrewAI / AutoGen / LangChain `base_url` at it and your local agents use fewer
> tokens, stay accurate, and stop breaking on long runs — a live telemetry feed shows the
> cleaning happening message by message.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-proxy-009688)
![Tests](https://img.shields.io/badge/tests-113%20passing-brightgreen)
![License](https://img.shields.io/badge/license-—-lightgrey)

---

## Start here

| You are… | Read this | Time |
|---|---|---|
| **A judge** checking that it works | [Quick start: verify it works](#quick-start-verify-it-works-cpu-only-no-gpu-needed), steps 1–6 | 10 s (tests) to ~20 min (full walkthrough) |
| **New to the project** and want to use it with your own agent | [User guide](#user-guide-run-your-own-agent-through-contextgc) | 15 min |
| **Running it on a GPU** (local, cloud VM or Modal) | [Running anywhere](#running-anywhere-cpu-a-remote-gpu-box-or-modal) | 15 min |
| **Curious how it works** | [How it works](#how-it-works) and [The GC pipeline](#the-gc-pipeline) | 5 min |
| **A developer** changing the code | [Repository layout](#repository-layout), [Tests](#tests), [`backend/README.md`](backend/README.md) | — |

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

## Quick start: verify it works (CPU only, no GPU needed)

Every step below was run on a CPU-only Windows laptop (16 GB RAM, no GPU).
Commands are PowerShell; the bash form is shown where it differs.

| You need | Notes |
|---|---|
| Python 3.12 | `python --version` |
| [Ollama](https://ollama.com/download) | only for steps 3–6; step 2 needs nothing else |
| ~3 GB free disk | for the `qwen2.5:3b` model (1.9 GB) |

### 1. Install

```powershell
git clone <this repo> ; cd MetaAI/backend
python -m pip install -r requirements.txt
```

### 2. Run the test suite (no Ollama, no GPU, about 10 s)

```powershell
python -m pytest tests -q
```

Expected: `113 passed`. The tests talk to a mock model server, so this works on any machine.

### 3. Get Ollama and a model

```powershell
ollama pull qwen2.5:3b
ollama list                    # qwen2.5:3b should be listed
```

The Ollama desktop app starts the server automatically; otherwise run `ollama serve` in
its own terminal. `qwen2.5:3b` is the recommended model: it is small enough for a CPU and
good enough at tool calls to solve the benchmark tasks (`llama3.2:3b` is not).

### 4. Start the proxy with GC switched on

GC is **off by default** (the proxy is then a byte-for-byte passthrough), so turn it on:

```powershell
$env:CONTEXTGC_GC_ENABLED = "true"
python -m uvicorn app.main:app --host 127.0.0.1 --port 11400
```

```bash
CONTEXTGC_GC_ENABLED=true python -m uvicorn app.main:app --host 127.0.0.1 --port 11400
```

In a second terminal (inside `backend`):

```powershell
curl http://localhost:11400/healthz
# {"service":"contextgc", ..., "gc_enabled":true}
```

### 5. Send a request through it

```powershell
$env:CONTEXTGC_MODEL = "qwen2.5:3b"
python scripts/smoke_test.py
# [health] service=contextgc gc_enabled=True sink=ollama
# reply: 'hello world'   (total_tokens=42)
# [PASS] non-streaming passthrough works
```

See what the GC did to each request (raw vs. cleaned tokens, savings per stage, the
pinned window):

```powershell
curl http://localhost:11400/telemetry/recent
```

To watch GC trim a growing agent conversation turn by turn (big tool dumps arrive
every few turns; `sent_est=` is what the agent sent, `clean=` what reached the model):

```powershell
python benchmark/run_simulation.py --lane b --turns 8 --model qwen2.5:3b
#   turn  8 [chat]      sent_est= 2707 clean= 2707 saved=    0 (0.0%)
#   turn  9 [tool_dump] sent_est= 5028 clean= 2508 saved= 2520 (50.1%) stages={'tool_offload': 2202, ...}
#   turn 10 [chat]      sent_est= 5087 clean= 2567 saved= 2520 (49.5%)
```

Nothing is removed while the conversation fits. Once it crosses ~70% of the pinned
window (turn 9 here) ContextGC compacts it once, mainly by moving old tool output to
disk behind a pointer, and later turns build on the compacted prompt so Ollama's
prefix cache keeps working (note the ~1 s first-token time on turn 10). About 5 minutes on a CPU.

### 6. Run the task benchmark with one command

`scripts/run_stack.py` uses the Ollama and proxy that are already running (or starts
them if they are not), then runs the benchmark. This quick version runs one coding task
in all three lanes:

```powershell
python scripts/run_stack.py --model qwen2.5:3b -- benchmark/run_tasks.py --tasks fix_path_join --repeats 1 --small-num-ctx 4096 --pad-tokens 2500
```

| Lane | What it is |
|---|---|
| **A1** | raw Ollama with a big 32k window (the quality ceiling) |
| **A2** | raw Ollama with the same small window as B (plain truncation) |
| **B** | through ContextGC with the small window |

Each line reports: `PASS`/`fail` from the task's own pytest checker, `steps` taken by the
agent, `wall` time, `reloads` (model reloads; should be 0), `cache` (share of prompt
tokens served from Ollama's prefix cache) and `gpu` (share of the model in GPU memory;
0.0 on a CPU). Results are saved as JSON in `benchmark/results/`.

On a laptop CPU this takes about 4 minutes per lane (≈11 minutes in total). A single
run of a 3B model is noisy: any lane can pass or fail one task by chance, so this quick
run checks that everything works end to end, not which lane is better. Judge quality by
success rates over many runs. The full
benchmark (8 tasks × 3 lanes × 3 repeats) is `... benchmark/run_tasks.py --repeats 3`;
that is several hours on a CPU, so run it on a GPU.

### 7. On a GPU

- **Local NVIDIA GPU:** run the same commands. The window is sized from free VRAM
  automatically, and `gpu=1.0` in the output means the model sits fully on the GPU.
  `pip install pynvml` gives the most accurate VRAM readings.
- **Modal or any remote machine:** see [Running anywhere](#running-anywhere-cpu-a-remote-gpu-box-or-modal).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `could not connect to a running Ollama instance` / `[stack] Ollama is not running` | start the Ollama app or `ollama serve` |
| `model "…" not found` | `ollama pull qwen2.5:3b`, or pass the model you have with `--model` |
| `healthz` shows `"gc_enabled": false` | restart the proxy with `CONTEXTGC_GC_ENABLED=true` |
| `[stack] proxy ... is running with GC off` | stop that proxy, or start one with GC on |
| port 11400 already in use | another proxy is running; reuse it or pick `--port` / `--proxy-url` |
| very slow first request | normal on CPU: the first prompt of a run is processed cold; later turns hit the cache |

---

## User guide: run your own agent through ContextGC

This section teaches everything a new user needs: the few ideas behind ContextGC, how
to put it in front of your agent, how to see what it did, and how to tune or switch it off.

### The ideas in one minute

| Term | Meaning |
|---|---|
| **Proxy** | ContextGC listens on port `11400` and forwards every request to Ollama (`11434`). Your agent talks to the proxy instead of Ollama; nothing else changes. |
| **Raw vs. clean** | *Raw* is what your agent sent. *Clean* is what ContextGC forwarded after GC. The raw request is always stored, so nothing is ever lost. |
| **Run** | One agent session (one task). ContextGC keeps per-run state: the pinned window, offloaded tool outputs, the anchored task state. |
| **Pinned window** | The `num_ctx` (context size) ContextGC fixes for the whole run, sized to your machine. A fixed window means Ollama never reloads the model mid-run. |
| **Compaction** | When the conversation reaches ~70% of the window, ContextGC shrinks it once to ~40% (moves big tool outputs to disk, merges duplicates, evicts stale turns). Between compactions it only appends, so Ollama's prompt cache keeps working. |
| **Tool Output Ref** | A big tool result moved to disk is replaced by `[Tool Output Ref #n … Key result: …]`. If the agent later mentions `Ref #n`, the full output is put back for that turn. |

### 1. Start Ollama and the proxy

One command starts whatever is not already running (Ollama, the model download, the
proxy with GC on) and keeps it up until you press Ctrl+C:

```powershell
cd backend
python scripts/run_stack.py --model qwen2.5:7b
# [stack] ... stack is up; press Ctrl+C to stop
```

Or start the proxy by hand as in [Quick start step 4](#4-start-the-proxy-with-gc-switched-on).
Either way, `curl http://localhost:11400/healthz` must show `"gc_enabled": true`.

### 2. Point your agent at the proxy

**Use Ollama's native API (`/api/chat`) when you can.** Ollama ignores the context
window on its OpenAI-style `/v1` endpoint, so on `/v1` ContextGC still cleans the prompt
but cannot pin the window ([details](#context-window-configuration)).

| Client | Base URL | Window pinned? |
|---|---|---|
| Native Ollama API (`ollama` Python package, LangChain `ChatOllama`, LiteLLM `ollama_chat/…`) | `http://localhost:11400` | yes |
| OpenAI-compatible clients (`openai` package, most frameworks' default) | `http://localhost:11400/v1` | no: start Ollama with `OLLAMA_CONTEXT_LENGTH=8192` (or your window) instead |

**Plain Python, native API** (the most direct way, and what the benchmark uses):

```python
import httpx

r = httpx.post("http://localhost:11400/api/chat", timeout=600, headers={"x-contextgc-run": "my-task-1"}, json={
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "Summarise this log: ..."}],
    "options": {"num_ctx": 8192},   # optional: omit it and ContextGC sizes the window for your machine
    "stream": False,
})
print(r.json()["message"]["content"])
```

**LangChain** (`pip install langchain-ollama`):

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", base_url="http://localhost:11400", num_ctx=8192,
                 client_kwargs={"headers": {"x-contextgc-run": "my-task-1"}})
```

**OpenAI client** (works with anything OpenAI-compatible; window not pinned, see above):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11400/v1", api_key="contextgc",
                default_headers={"x-contextgc-run": "my-task-1"})
```

**CrewAI** uses LiteLLM underneath. Use the native provider so the window is pinned:
`LLM(model="ollama_chat/qwen2.5:7b", base_url="http://localhost:11400")`. The CrewAI
agents in `crewai_agent/` still use `/v1` today; moving them to the native API and
measuring them with and without ContextGC is the next step in
[`NEXT_STEPS.md`](NEXT_STEPS.md).

### 3. Tell ContextGC where a run starts

Send a header `x-contextgc-run: <any id>` with every request of one task, and a new id
for the next task. Without it, ContextGC derives the run from the model, the system
prompt and the first user message, which works for most agents. Use a new id when you
want a different window: the window is pinned on a run's first request, and later
changes to `num_ctx` within that run are ignored on purpose (reported as
`client_num_ctx_overridden`).

### 4. See what it did

**Open the dashboard: <http://localhost:11400/dashboard>** (served by the proxy, nothing to install).

- **Live tab:** requests, tokens saved, share of the prompt served from Ollama's cache,
  compactions, loop fallbacks and mid-run model reloads; which model is loaded, its
  window and how much of it is on the GPU; a chart of raw vs. forwarded prompt size per
  request with compaction markers; and a table of requests. Click a request (or open
  `/dashboard#turn=<id>`) for a side-by-side **prompt diff** of what the agent sent and
  what the model received.
- **Benchmark results tab:** pick any results file on this machine (`run_tasks.py` or the
  CrewAI `run_ab.py`) or drop one in: summary per lane, prompt tokens per lane, pass/fail
  per task, and a step-by-step "race" chart per task.

The same data is available as JSON:

| Endpoint | What you get |
|---|---|
| `GET /healthz` | service status, `gc_enabled`, where requests are forwarded |
| `GET /telemetry/recent?n=20` | one event per request: raw vs. clean tokens, % saved, tokens saved per stage, pinned window, first-token time, VRAM ([example](#telemetry--see-the-gc-working)) |
| `WS /ws/telemetry` | the same events live, as they happen |
| `GET /store/recent?n=20` | the last stored turns (ids, run ids, sizes) |
| `GET /store/<turn_id>/raw` | exactly what your agent sent for that turn |
| `GET /store/<turn_id>/clean` | exactly what ContextGC forwarded to the model |

Helper scripts (run from `backend/`, they read the local store in `data/contextgc.db`):

| Script | What it does |
|---|---|
| `python scripts/savings_report.py` | token and KV-memory savings across everything stored |
| `python scripts/explain_offload_turns.py [run_id]` | per-turn story: what came in, what went out, which stages acted |
| `python scripts/explain_last_offload.py` | explains the most recent tool-offload turn |
| `python scripts/replay_turn.py <turn_id>` | sends a stored *clean* prompt to Ollama again and prints the reply, to check the cleaned prompt still makes sense |

### 5. Tune it or switch it off

- **Switch GC off:** start the proxy without `CONTEXTGC_GC_ENABLED=true`. It then forwards
  every request byte for byte, which is a safe way to compare with and without.
- **Switch single stages off** with environment variables, e.g.
  `CONTEXTGC_SEMANTIC_ENABLED=false`, `CONTEXTGC_EVICT_ENABLED=false`,
  `CONTEXTGC_OFFLOAD_ENABLED=false`, `CONTEXTGC_FIDELITY_ENABLED=false`
  (all variables: [Configuration](#configuration)).
- **Per request**, send `x-contextgc-stages: <preset>`: `full` (default), `none` (proxy
  overhead only), `anchor_only`, `no_semantic` or `no_evict`.
- **When to compact:** `CONTEXTGC_COMPACTION_TRIGGER` (default `0.70` of the window) and
  `CONTEXTGC_COMPACTION_TARGET` (default `0.40`).
- **Window size:** chosen automatically from free GPU memory; on a CPU it is
  `CONTEXTGC_NUMCTX_DEFAULT` (8192). Send `num_ctx` on the first request to choose it yourself.
- **Another backend:** set `CONTEXTGC_SINK` to any OpenAI-compatible base URL to put
  ContextGC in front of something other than Ollama.

### FAQ

**Can ContextGC lose information?** It never deletes anything: the raw request is stored,
big tool outputs are kept on disk and come back when the agent refers to them, and if
the history shows the model looping, that turn is sent raw (fidelity fallback).

**Which model should I use?** A model that is good at tool calls: `qwen2.5:7b` on a GPU,
`qwen2.5:3b` on a CPU. In our tests `llama3.2:3b` failed coding tasks even without ContextGC.

**Why is the first request slow?** The first prompt of a run is processed from scratch;
later turns reuse Ollama's prompt cache. On a CPU the first request can take a minute.

**Does it work with streaming?** Yes, for both `/api/chat` and `/v1/chat/completions`.

**Does it need a GPU?** No. Everything works on a CPU, just more slowly. A GPU adds
automatic window sizing from VRAM.

---

## Running anywhere: CPU, a remote GPU box, or Modal

The same launcher runs the whole stack on any machine: `backend/scripts/run_stack.py`
reuses Ollama and the proxy if they are already up, otherwise starts them, pulls the
model, runs a benchmark, and stops what it started. The benchmark sizes its windows
for the machine it lands on (`--small-num-ctx auto`: VRAM-sized on a GPU, the
`CONTEXTGC_NUMCTX_DEFAULT` window on CPU) and records the hardware plus Ollama's
GPU residency (`size_vram / size`) in every result file.

**Local CPU or local GPU**

```powershell
cd backend
python scripts/run_stack.py --model qwen2.5:3b -- benchmark/run_tasks.py --repeats 3
```

**Any remote machine** (cloud VM, lab box): install Ollama and
`pip install -r backend/requirements.txt pynvml`, then run the same command there.
To keep the proxy on your machine and use a remote Ollama instead, pass
`--ollama-url http://<host>:11434` (the remote side needs `OLLAMA_HOST=0.0.0.0`).

**Modal GPUs** (`deploy/modal_app.py`): the container runs the same launcher, with
weights in a Modal Volume and results copied back to `backend/benchmark/results/modal/`.

```powershell
pip install modal
modal token new                                                  # once
modal run deploy/modal_app.py::pull --models qwen2.5:7b          # CPU container fills the weights Volume (cheap)
modal run deploy/modal_app.py::main --gpu t4 --models qwen2.5:7b --tasks fix_path_join --repeats 1 --small-num-ctx 8192   # smoke, ~3 min
modal run deploy/modal_app.py::main --gpu t4 --models qwen2.5:7b --small-num-ctx 8192          # full benchmark
modal run deploy/modal_app.py::sweep --gpus t4,l4 --models qwen2.5:7b,llama3.1:8b --overrides 0,8192,6144
```

What each command does: `pull` downloads model weights into a Modal Volume on a CPU
container, so GPU time is not spent downloading. `main` starts one GPU container that
runs Ollama, the proxy and the benchmark, copies the results to your machine and shuts
down; billing stops with it. `sweep` starts one container per GPU × model × simulated
VRAM combination, all in parallel (12 in the example), so it costs the most. Modal needs
a payment method on the account before it runs GPU functions; the Starter plan's monthly
credit covers the smoke test and several full runs.

**Pass `--small-num-ctx 8192` on big cards.** On a 16 GB T4 the probe picks a 32k window
for the small-window lanes too, so A2 and B would get the same window as A1 and nothing
would be under pressure. The run prints a warning when this happens.

**Stage keep/kill ablation** (plan 3.6): a lane `b-<preset>` is ContextGC with one stage switched off,
e.g. `--lanes b,b-no_semantic,b-no_state,b-no_fidelity,b-no_evict,b-no_offload`. Every ContextGC run also
records how often the proxy compacted and how often the loop fallback fired.

GPUs: `t4`, `l4`, `a10g`, plus `cpu`. `--vram-override-mb` simulates a smaller card for the
window choice (`CONTEXTGC_VRAM_OVERRIDE_MB`). It does not shrink real GPU memory, so
spill-to-CPU of the raw lane only shows up on a card that is actually small.
Each container stops after `CONTEXTGC_MODAL_TIMEOUT_H` hours (default 4).

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
  benchmark/             # run_tasks.py (A1/A2/B task success), run_simulation.py (latency)
  scripts/               # run_stack.py launcher, smoke test, savings report, replay tools
deploy/
  modal_app.py           # Modal GPU containers running scripts/run_stack.py
  tests/                 # pytest acceptance suite (mock sink — no Ollama needed)
```

---

## Tests

```powershell
cd backend
python -m pytest tests -q
```

113 passing. The acceptance suite runs against a **mock sink**, so no Ollama or GPU is
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

**113 tests passing** (mock sink, no Ollama required). Runtime capacity probing,
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
- ✅ **Dashboard** (`/dashboard`) — live gauges (cache hits, reloads, GPU residency),
  per-run prompt-size chart with compaction markers, prompt diff viewer, benchmark
  results viewer with per-task race chart

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

**`/v1` does not honor `num_ctx` (checked on Ollama 0.40.13).** Ollama's
OpenAI-compatible endpoint ignores `num_ctx` whether it is sent top-level or under
`options`; the model loads at the server default (`OLLAMA_CONTEXT_LENGTH`, 4096 unless
set). The pinned window is only enforced on the native `/api/chat` path. For
frameworks, either point them at the native API (e.g. LiteLLM's `ollama_chat/<model>`
provider) or start Ollama with `OLLAMA_CONTEXT_LENGTH` set to the window you want.

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
