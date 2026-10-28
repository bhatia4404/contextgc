# ContextGC — Plan v3 (cache-aware, machine-aware)

_Written 2026-10-28. Supersedes the execution order in `contextgc_final_plan_v2.md`;
v2 stays as history. Based on the Phase 2.1 benchmark findings and the code audit._

---

## 1. Goal (re-scoped)

> **"Your model fits on your machine — long agent tasks don't. ContextGC makes the task fit."**

ContextGC is a drop-in proxy that lets **long, tool-heavy local agent runs** work on
**small machines**:

- **Detect** the machine's capacity once per run (free VRAM, model footprint).
- **Pin** the largest context window that keeps the model fully on the GPU.
- **Keep** the agent's context inside that window — no silent truncation — while
  **preserving Ollama's prefix cache**, so per-turn latency stays at raw-Ollama parity.
- **Prove** results stay comparable (task success parity, measured).

Allowed claims (only once measured): lower steady VRAM for the same task length,
model stays 100% on GPU where raw would spill, comparable task success, faster total
task time on long runs. **Not** claimed: per-turn TTFT beating a cached baseline,
per-turn VRAM changes, running models whose weights don't fit.

---

## 2. What changed from v2 and why

| v2 assumption | Reality (measured / audited) | v3 response |
|---|---|---|
| Prefill cost grows every turn | Ollama/llama.cpp reuses the cached prompt prefix; append-only history only prefills the new suffix (Lane A ≈ 2s TTFT at 9k tokens) | GC must keep the prompt append-only between compactions |
| Dynamic num_ctx per turn saves VRAM | num_ctx is a load-time parameter; every change = model reload + empty KV cache | Pin one window per run, sized from machine capacity |
| Anchor pinned at the top | Rebuilt at index 1 every turn → everything after the system prompt re-prefilled | Move to tail; skip when empty |
| Cross-request dedup removes duplicates | **Bug:** deletes the normal re-sent history (all earlier user instructions) | Dedup within the current payload only |
| Mutations every turn are fine | Offload/evict/merge rewrite mid-history → cache bust every turn | Compaction epochs with hysteresis; sticky transforms |
| Savings numbers are real | Partly produced by the dedup bug (lost context) | Re-measure everything after fixes |

---

## 3. Phases

### Phase 3.0 — Correctness & cache-safety fixes (P0) ▶️ NEXT

| # | Item | Where | Acceptance |
|---|---|---|---|
| 3.0.1 | Dedup only within the current payload; never remove a message because an earlier request contained it | `core/hash_filter.py` | Probe: all prior user turns survive on turn N |
| 3.0.2 | Classifier annotations keyed by message identity (content hash), not list index | `core/classifier.py`, `semantic.py`, `evict.py` | Test: annotations correct after dedup/anchor shift |
| 3.0.3 | Eviction removes an assistant `tool_calls` message and its tool results together, or neither | `core/evict.py` | Test: no orphaned tool call/result after eviction |
| 3.0.4 | Re-injection scans only the latest user + latest assistant message | `core/offload.py` | Test: ref restored once, pointer again next turn |
| 3.0.5 | Fidelity fallback looks only at the last few turns; fallback requests still carry the pinned num_ctx | `core/fidelity.py`, `core/gc.py` | Test: no permanent fallback; no num_ctx-less forward |
| 3.0.6 | Anchor moved to just before the latest user message; skipped when all fields are empty | `core/anchored_state.py` | Prefix test (3.1.3) passes for anchor-only preset |
| 3.0.7 | `keep_alive` sent on every forwarded request (configurable, default `30m`) | `core/gc.py` / `main.py` | Model stays resident across idle gaps |
| 3.0.8 | GC off the event loop: `await asyncio.to_thread(gc.process_payload, ...)` | `main.py` | Concurrent requests don't serialize |
| 3.0.9 | Encoder loaded once (module-level / startup), batched encodes; still skips cleanly when not installed | `core/semantic.py` | Encoder constructed once per process |
| 3.0.10 | Remove per-turn num_ctx negotiation (replaced by 3.2) | `core/num_ctx.py` | num_ctx identical on every request of a run |

### Phase 3.1 — Measurement upgrade (do alongside 3.0)

| # | Item | Acceptance |
|---|---|---|
| 3.1.1 | Harness records Ollama's final-chunk fields: `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` | Per-turn JSON shows reloads (`load_duration` > 0) and tokens actually prefilled |
| 3.1.2 | Derived metrics: reload count, cache-hit ratio (1 − prefilled ÷ prompt tokens), total task time, sum of prefilled tokens | Printed in summary table |
| 3.1.3 | **Prefix-stability invariant test** (mock sink): for append-only input, turn N's cleaned output is a byte prefix of turn N+1's, except on compaction turns | `tests/test_prefix_stability.py` green |
| 3.1.4 | Teacher-forced mode: both lanes replay the same fixed assistant replies (latency comparisons); free-generation mode kept for quality runs | `--replay` flag |
| 3.1.5 | Lane A window set explicitly; record `ollama ps` CONTEXT and `size_vram` per lane | No unknown-default comparisons |
| 3.1.6 | Warm-up request after every lane switch | No reload counted against a lane |

### Phase 3.2 — Machine-aware fixed window

| # | Item | Acceptance |
|---|---|---|
| 3.2.1 | Capacity probe at run start: free VRAM (`vram.py`), model architecture from Ollama `/api/show` → exact KV bytes/token (replaces hard-coded 114,688), weight size | Correct KV bytes/token for llama3.2:3b and llama3.1:8b |
| 3.2.2 | Window = (free VRAM − weights − buffers − 10–15% margin) ÷ (KV bytes/token × parallel slots), rounded down to tier 4k/8k/16k/32k; config for KV type (`q8_0` halves) and `NUM_PARALLEL` | Unit tests over a table of (VRAM, model) cases |
| 3.2.3 | Post-load check via `/api/ps`: if `size_vram < size`, drop one tier and reload once | Model 100% on GPU after probe |
| 3.2.4 | Window pinned per run (keyed by `x-contextgc-run` or system-prompt hash); never shrinks mid-run | Zero reloads within a run |
| 3.2.5 | Fallbacks: no readable VRAM (AMD/Apple/CPU) → config default; CPU-only → pick for latency (smaller tier) | Graceful on this laptop (CPU) |
| 3.2.6 | `CONTEXTGC_VRAM_OVERRIDE_MB` to simulate small cards (4/6/8 GB) on big GPUs | Used by Phase 3.5 |
| 3.2.7 | Verify `/v1` honors injected num_ctx on the current Ollama build; if not, document native `/api/chat` as the supported path for frameworks | Recorded in README |

### Phase 3.3 — Cache-aware GC redesign

| # | Item | Acceptance |
|---|---|---|
| 3.3.1 | **Incremental view:** per run, keep last raw history + last cleaned prompt; if the new raw request = old raw + appended messages, send old cleaned prompt + cleaned new messages | Prefix test passes for the full pipeline |
| 3.3.2 | **Compaction epochs:** GC rewrites only when the prompt crosses ~70% of the pinned window; compacts to ~40% in one pass (offload, dedup, evict) | ≤ 1 cache bust per compaction; telemetry marks compaction turns |
| 3.3.3 | **Sticky transforms:** a message keeps its representation (raw/pointer) until the next compaction | No mid-history change between epochs |
| 3.3.4 | Eviction becomes part of compaction only (no per-turn trickle eviction); budget derived from the pinned window, not a fixed 6000 | — |
| 3.3.5 | Divergent requests (framework rebuilt or edited history) → treated as a new baseline, full clean once | No crash, logged |

### Phase 3.4 — Honest local benchmark

| # | Item | Acceptance |
|---|---|---|
| 3.4.1 | Real task suite (8–12 tasks) with deterministic checks: small coding/debugging tasks with real tool output (pytest, grep, file reads) | Each task has a pass/fail checker |
| 3.4.2 | Three lanes: **A1** raw Ollama, big window (quality ceiling) · **A2** raw Ollama, same small window as B (truncation) · **B** ContextGC, small window | Same tasks, same model, `--rotate ≥ 3` |
| 3.4.3 | Report per lane: task success, total time, steady VRAM, reload count, cache-hit ratio, prefilled tokens | Results JSON + table |
| 3.4.4 | Real framework run (CrewAI first) through the proxy; check whether the framework itself is append-only (if not, measure whether ContextGC stabilizes it) | Documented |

**Success bar:** B's task success ≈ A1's, at A2's window size, with reloads = 0 within a run
and total time ≤ A1's on long tasks.

### Phase 3.5 — Modal GPU-tier sweep (November credits)

- Budget: Starter plan, $30/month (hackathon $100 not received). Estimated sweep ≈ 8 GPU-hours ≈ $5–8.
- Optional now (before Oct 31 reset, ~$1): one T4 smoke test — Ollama in a Modal container, weights in a Modal Volume, harness connects.
- Real sweep after 3.0–3.4: T4 + L4 only; simulated 4/6/8 GB cards via 3.2.6; models llama3.2:3b and llama3.1:8b; lanes A1/A2/B × 3 repeats.
- Cost hygiene: weights in a Volume, one container session per tier, container timeout, usage limit stays at $30.
- Output: the pitch slide — "8B agent on an 8 GB GPU: raw spills to CPU, ContextGC stays 100% on GPU, same task success."

### Phase 3.6 — Quality gate for heuristic stages

Decide keep/kill from 3.4 data, not intuition:

| Stage | Keep if |
|---|---|
| Semantic contraction (needs `sentence-transformers`) | Improves success or saves tokens with no success loss |
| Anchored state | Measurable drift reduction on long tasks |
| Fidelity fallback | Fires on real loops, not on normal runs |
| Classifier "semantic" heuristic (`" is "` etc.) | Tightened; otherwise reduce to role/size rules |

### Phase 3.7 — Cross-backend validation (optional)

- Baseten/Fireworks only if credits are available. Validates: token savings and quality on bigger models,
  and that the cache-aware design also preserves vLLM-style prefix caching. **Cannot** validate window sizing
  or VRAM (server manages KV memory; num_ctx is report-only there).

### Phase 4 — Dashboard (after 3.4)

Split-screen race (A1/A2/B), live `ollama ps` VRAM + GPU-residency gauge, cache-hit ratio,
compaction markers, prompt diff viewer.

### Tier 3 — Parked (after Phase 4)

Delta offload for re-read files, multi-model switching, `OLLAMA_NUM_PARALLEL` per-agent slots study,
content-addressed message store (replaces full raw/clean payload per turn), tokenizer-based estimator.

---

## 4. Dropped / changed

- **Dropped:** per-turn num_ctx negotiation; cross-request "seen before" dedup; per-turn trickle eviction.
- **Changed:** anchor at tail; offload at compaction; claims reframed around small machines and long runs.

## 5. Docs to correct (with 3.0)

- README problem #3 ("prefill grows every turn") and #4 ("VRAM spikes") — rewrite per §1.
- README claim that Ollama truncation drops the system prompt — verify on the current build, fix wording.
- README pipeline diagram: S4 "dynamic num_ctx" → "machine-aware pinned window".

## 6. Risks

| Risk | Mitigation |
|---|---|
| Frameworks (CrewAI via `/v1`) ignore num_ctx | 3.2.7; document native path; test early |
| Compaction hurts quality more than truncation | 3.4 lanes A1/A2/B make it visible; tune thresholds |
| Heuristic stages net-negative | 3.6 keep/kill gate |
| Small-card results only simulated | Simulation via VRAM override + one real check on this laptop's hardware |
| Ollama already sizes context from VRAM on some builds | Differentiate: ContextGC also keeps the task inside the window, cache-safe |

## 7. Order of work

1. 3.0 + 3.1 (fixes and measurement, together)
2. 3.2 (machine-aware window)
3. 3.3 (cache-aware redesign)
4. 3.4 (local benchmark) → 3.6 (keep/kill)
5. 3.5 (Modal sweep, November credits)
6. Phase 4 dashboard → 3.7 → Tier 3
