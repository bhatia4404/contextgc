# ContextGC: next steps

Last updated: 2026-11-01. All work so far is committed and pushed to `main` (b82fb25).
Work only on `main`; commit after each step that works.

## Where we are

- The Modal GPU setup works end to end (workspace `shivambhatia4404`, T4). Weights for
  `qwen2.5:3b` and `qwen2.5:7b` are already in the `contextgc-models` Volume.
- On a 16 GB T4 the window probe gives every lane 32768, so benchmark runs must pass
  `--small-num-ctx 8192` to create window pressure.
- `qwen2.5:3b` is too weak (fails in every lane). `qwen2.5:7b` passes with the big window (A1).
- The padding is ~13.8k real tokens at `pad=8192`, so in an 8k window Ollama silently drops the
  oldest messages, including the task prompt. That is why A2 (raw, small window) fails.
- Fixed in b82fb25: the fidelity fallback mistook tool-call-only turns (empty text) for a loop
  and sent B's raw history, so B was truncated like A2. Not yet re-verified on GPU.

## Progress 2026-11-03

- Step 1 DONE. fix_path_join x3: A1 3/3, A2 0/3, B 1/3. Found a second bug: pointer text had a
  process-wide ref counter and a timestamped path, so every run's prompt differed (B's runs
  diverged at T=0). Fixed: per-run ref ids, no path in the prompt (106 tests pass).
- All 8 tasks x1, qwen2.5:7b, T4, small window 8192 (results/modal/step1b_all_tasks_7b_deterministic):
  A1 3/8 (1,187k prompt tokens), A2 2/8 (474k), **B 5/8 (399k = 34% of A1)**, 0 reloads, gpu=1.0.
  Single run per task: noisy, needs repeats/more models before claiming.
- Step 2 code written: crewai_agent/llm_config.py (LiteLLM ollama_chat -> native /api/chat with
  num_ctx; CrewAI 1.15's own ollama client uses /v1), run_ab.py (raw vs ContextGC, fresh repo copy,
  generated 2000-line outage log, pytest + test-tamper check), Modal `crewai` entrypoint.
  Local smoke: crew runs through the proxy with tools (run was cut off by session end).
  GPU run started detached: `crewai_ab_7b_t4` -> fetch with
  `modal volume get contextgc-results crewai_ab_7b_t4 backend/benchmark/results/modal/`.
- CrewAI A/B GPU run `crewai_ab_7b_t4` (qwen2.5:7b, 2 repeats): **0/4 raw, 0/4 ContextGC**, not a
  ContextGC verdict. The 7B model wrote to wrong paths (`repo/routing.py`), sent write calls with
  empty args, and CrewAI crashed on empty replies (EmptyLLMResponseError) in both lanes. Context
  stayed ~1k tokens/request because absolute paths (`/data/...`) were rejected by the tools, so the
  big log was never read. FIXED: tools now resolve any path form inside the repo
  (`llm_config.resolve_in_repo`). Rerun with qwen2.5:14b on L4 started detached (tag
  `crewai_ab_14b_l4`); fetch with
  `modal volume get contextgc-results crewai_ab_14b_l4 backend/benchmark/results/modal/`.
- Step 4 (3.6 keep/kill) tooling DONE: `run_tasks.py --lanes b,b-no_<stage>,...`; presets derived
  from one stage list (old no_semantic/no_evict also silently disabled fidelity); per-run proxy
  telemetry (compactions, fidelity fallbacks, stage savings). Ablation run `ablation_7b_t4` started
  detached; fetch with `modal volume get contextgc-results ablation_7b_t4 backend/benchmark/results/modal/`.
  Observation: the anchored state block is mostly nulls + log lines (weak; see ablation result).
- 2026-11-03 evening: both detached runs were STOPPED at 17:51:30 when the Claude session closed
  (`--detach` did not survive the local process being killed on Windows). Ablation 46/48 runs done;
  CrewAI 14B 1/8. For long runs use a deployed app + `.spawn()` instead of `modal run --detach`.
- Ablation `ablation_7b_t4` (1 run/task, noisy): B 6/8 (358k tok), no_semantic 4/8, no_state 5/8,
  no_fidelity 4/8, **no_evict 1/7**, no_offload 5/7 at **170k tok** (eviction replaced offload).
  Semantic contraction saved 0 tokens everywhere, yet B vs no_semantic still differ on 2 tasks:
  that is the noise floor (GPU/prompt-cache numerics), so only large gaps count. Readings:
  eviction = keep (clear); semantic = kill candidate (does nothing on this suite); state = no
  measurable effect; fidelity fired 10x in 8 B runs (check if real loops); offload vs eviction
  needs repeats. Need >=3 repeats per lane before deciding.
- CrewAI 14B (1 run): fail; ContextGC saved 0%. **CrewAI 1.15 defers large tool results itself**
  (`read_a_deferred_tool_result` tool), so prompts stay ~1k tokens and there is nothing for
  ContextGC to compact. Also the model kept reading the decoy `routing_code.txt`. Decide: disable
  CrewAI's deferral to compare, or position ContextGC as framework-agnostic (frameworks without it).
- 2026-11-03 night: long runs now go through `modal deploy deploy/modal_app.py` +
  `python deploy/launch.py main|ablation|crewai ...` (one spawned container per task; survives a
  closed session; `python deploy/launch.py status`). Campaigns launched: final_qwen7b_t4,
  ablation3_qwen7b_t4, crew_14b_l4 (4 lanes: raw/contextgc x CrewAI's own handling on/off via
  `-nocrew`), final_qwen14b_l4, final_llama8b_l4 (all 3 repeats). Fetch with
  `modal volume get contextgc-results <campaign> backend/benchmark/results/modal/`, merge with
  `backend/benchmark/merge_results.py`, then `deploy/build_showcase.py` and
  `modal deploy deploy/api_app.py` (data) and `cd frontend && npm run build && modal deploy ../deploy/frontend_app.py` (UI).
- Judge site: https://shivambhatia4404--contextgc-showcase-web.us-east.modal.run (React app in
  frontend/, served by deploy/frontend_app.py; backend API deploy/api_app.py at
  https://shivambhatia4404--contextgc-api-api.us-east.modal.run; results page +
  /live GPU run page, 1 T4 at a time, 25 runs/day cap). Workspace GPU limit is 10: live runs
  queue while campaigns run.
- 2026-11-04 results (3 repeats, all on the site and in README "Results"): qwen2.5:7b T4 A1 10/24,
  A2 10/24, B 16/24; qwen2.5:14b A10G 15/19/15; llama3.1:8b T4 3/13/2. B uses ~1/3 of A1's tokens on
  every model. Ablation x3: B 14/24, no_evict 6/24, others within 2. CrewAI 14B A10G (4 lanes): ~all
  fail, window never fills (tools cap reads at 16k chars), ContextGC saves 0-3%. L4 GPUs were never
  scheduled; use T4/A10G. Modal preempts containers: merge_results keeps the newest file per task.
- NEXT: on llama/14B, B behaves like full history (A1). Improve relevance filtering of old
  history (earlier-session noise survives compaction); then rerun B lanes on those models.
- Phase 4 dashboard DONE: `/dashboard` (live gauges, prompt-size chart with compaction markers,
  prompt diff, results viewer + race chart). Telemetry now carries Ollama stats (cached tokens,
  load time) and a compaction flag. 113 tests pass.
- Fixed: `demo_repository/data/` was gitignored (incident log missing on GitHub).
- Uncommitted: all of the above + README user guide + this file.
- Gotcha: on Windows, piping `modal run` output needs `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.
  Use `modal run --detach` for long runs so they survive a closed session.

## Step 1: confirm the fidelity fix on the GPU (~15 min, ~$0.15)

```
modal run deploy/modal_app.py::main --gpu t4 --models qwen2.5:7b --tasks fix_path_join --repeats 3 --small-num-ctx 8192
```

- Hoped for: A1 passes, A2 fails, B passes with far fewer prefilled tokens than A1.
- If B fails: read the trace in `backend/benchmark/results/modal/<tag>/*.json` (per-step tools,
  args, prompt_eval_count). Decide whether GC dropped needed context or the model erred; fix first.

## Step 2: make CrewAI a real with/without-ContextGC test (plan 3.4.4)

Files: `crewai_agent/code_maintenance_agent.py`, `crewai_agent/incident_response_agent.py`.

- [ ] Switch from the `/v1` endpoint (Ollama ignores `num_ctx` there) to the native API
      (`ollama_chat/<model>` with base URL `http://localhost:11400`, no `/v1`), and pass `num_ctx`.
- [ ] Default model `qwen2.5:7b` instead of `llama3.2:3b`.
- [ ] Run two lanes with the same small window: raw Ollama (11434) vs. ContextGC proxy (11400).
      Record pass/fail, tokens, wall time, reloads, cache hit ratio. Repeats + reset of the demo repo per run.
- [ ] Add `crewai` to the Modal image and a `crewai` entrypoint in `deploy/modal_app.py`.
- [ ] Smoke-test on T4, then several repeats.

## Step 3: full benchmark (1–3 GPU hours, ~$3–10)

```
modal run deploy/modal_app.py::main --gpu t4 --models qwen2.5:7b --small-num-ctx 8192
modal run deploy/modal_app.py::pull --models llama3.1:8b
modal run deploy/modal_app.py::main --gpu l4 --models llama3.1:8b --small-num-ctx 8192
```

- Optional small sweep over simulated VRAM (e.g. `--overrides 0,8192,6144` with qwen2.5:7b/llama3.1:8b).
  Note: the VRAM override only changes the window choice, not real GPU memory.
- Output: pass rate, token savings and speed numbers for the README.

## Step 4: if time allows

- [ ] Plan 3.6 keep/kill per GC stage: ablate stages one at a time with the stage filter; drop
      stages that do not help.
- [ ] Write the GPU and CrewAI results into the README with honest caveats (small samples, noise).

## Later

- [ ] Phase 4 dashboard.
- [ ] Plan 3.7 cross-backend validation (a second model server).

## Known gotchas

- `modal run` must stay attached unless `--detach` is used; results then stay in the
  `contextgc-results` Volume (`modal volume get contextgc-results <tag> .`).
- Containers stop by themselves when the script ends; 4-hour timeout cap.
- In bash heredocs, avoid backslashes when writing to the README (they were mangled once).
