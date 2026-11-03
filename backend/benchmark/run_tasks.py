r"""ContextGC — Phase 3.4 three-lane task-success benchmark.

Runs the deterministic task suite (benchmark/task_suite.py) with a real
tool-using agent loop over Ollama's native /api/chat, in three lanes:

    A1  raw Ollama, big window      (quality ceiling)
    A2  raw Ollama, small window    (what truncation does)
    B   ContextGC proxy, small window

Plan 3.6 keep/kill ablation: a lane `b-<preset>` is lane B with the proxy's
stage preset (x-contextgc-stages), e.g. `b-no_semantic`, `b-no_state`,
`b-no_fidelity`, `b-none` (proxy only, no GC). Every proxied run also records
the proxy's telemetry for that run: requests, compactions, fidelity
fallbacks and tokens saved per stage.

Each task starts from a history that already holds earlier exploration output
(`--pad-tokens`, deterministic CI-log noise), the way a long agent session
does. Without it the tiny fixtures never fill a 4k window and the lanes
cannot differ. The task prompt sits before that noise, so a lane that loses
early context loses the task.

Per run it records checker pass/fail, steps, wall time, reloads
(load_duration), and the cache-hit ratio (same estimator as run_simulation).

Usage (Ollama on :11434, proxy with CONTEXTGC_GC_ENABLED=true on :11400):
    cd backend
    python benchmark/run_tasks.py --repeats 3
    python benchmark/run_tasks.py --tasks fix_failing_sum --lanes b --repeats 1  # smoke
    python benchmark/run_tasks.py --lanes b,b-no_semantic,b-no_state,b-no_fidelity  # 3.6 ablation
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from task_suite import TASKS  # noqa: E402
from run_simulation import cache_stats  # noqa: E402
from app import config  # noqa: E402
from app.core import capacity  # noqa: E402

RESULTS_DIR = Path(os.getenv("CONTEXTGC_RESULTS_DIR") or ROOT / "results")

SYSTEM = (
    "You are a coding agent working in a small Python repository. Use the tools to "
    "list and read files, run pytest, and write fixed source files. Never edit test "
    "files. Always write the complete file content. When pytest passes, reply with "
    "the single word DONE."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List the files in the repository.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative file path"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Replace a source file with new full content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_pytest", "description": "Run the test suite and return its output.",
        "parameters": {"type": "object", "properties": {}}}},
]


# --- tools ---------------------------------------------------------------

def _safe(root: Path, rel: str) -> Path | None:
    path = (root / str(rel)).resolve()
    return path if root.resolve() in path.parents else None


def run_tool(root: Path, task, name: str, args: dict) -> str:
    if name == "list_files":
        return "\n".join(sorted(str(p.relative_to(root)).replace("\\", "/")
                                for p in root.rglob("*") if p.is_file()
                                and "__pycache__" not in p.parts
                                and ".pytest_cache" not in p.parts))
    if name == "read_file":
        path = _safe(root, args.get("path", ""))
        if not path or not path.is_file():
            return f"error: no such file {args.get('path')!r}"
        return path.read_text(encoding="utf-8")[:6000]
    if name == "write_file":
        rel = str(args.get("path", ""))
        path = _safe(root, rel)
        if not path:
            return "error: path outside repository"
        if Path(rel).name.startswith("test_"):
            return "error: test files must not be edited"
        content = args.get("content")
        if not isinstance(content, str) or not content.strip():
            return "error: content must be the full file text"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {rel} ({len(content)} chars)"
    if name == "run_pytest":
        _, output = task.run_checker(root)
        return output[-3000:]
    return f"error: unknown tool {name!r}"


_JSON_CALL = re.compile(r"\{.*\}", re.S)


def parse_calls(message: dict) -> list[tuple[str, dict]]:
    """Native tool_calls, or the JSON-in-content form small models often emit."""
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        calls.append((fn.get("name", ""), args if isinstance(args, dict) else {}))
    if calls:
        return calls
    match = _JSON_CALL.search(message.get("content") or "")
    if match:
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            return []
        if isinstance(obj, dict) and obj.get("name") in {t["function"]["name"] for t in TOOLS}:
            args = obj.get("parameters") or obj.get("arguments") or {}
            return [(obj["name"], args if isinstance(args, dict) else {})]
    return []


# --- earlier-session padding ----------------------------------------------

def _noise_dump(seed: int, tokens: int) -> str:
    lines, i = [], 0
    while len("\n".join(lines)) < tokens * 4:  # ~4 chars per token
        mod = (seed * 31 + i * 7) % 97
        lines.append(
            f"[ci-{seed}] {i:04d} services/billing/module_{mod}.py::test_case_{i % 13} "
            f"{'PASSED' if (i + seed) % 5 else 'SKIPPED (flaky upstream)'} "
            f"duration={(i * 37 + seed) % 900}ms worker=gw{i % 4}")
        i += 1
    return "\n".join(lines)


def padding_messages(pad_tokens: int, chunks: int = 3) -> list[dict]:
    """Earlier exploration: the agent read unrelated CI logs."""
    msgs = []
    per = max(1, pad_tokens // chunks)
    for c in range(chunks if pad_tokens else 0):
        call_id = f"pad_{c}"
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": call_id, "function": {
                         "name": "read_file", "arguments": {"path": f"logs/ci_run_{c}.log"}}}]})
        msgs.append({"role": "tool", "tool_call_id": call_id, "tool_name": "read_file",
                     "content": _noise_dump(c + 1, per)})
    if msgs:
        msgs.append({"role": "assistant", "content":
                     "Those CI logs are unrelated to the task. Back to the repository."})
    return msgs


# --- one agent run ----------------------------------------------------------

def run_task(client: httpx.Client, base: str, model: str, task, window: int,
             run_id: str, proxy: bool, max_steps: int, pad_tokens: int,
             stages: str | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"cgc-{task.name}-") as directory:
        root = Path(directory)
        task.materialize(root)
        for c in range(3 if pad_tokens else 0):
            (root / "logs").mkdir(exist_ok=True)
            (root / "logs" / f"ci_run_{c}.log").write_text(_noise_dump(c + 1, 200), encoding="utf-8")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task.prompt}] + padding_messages(pad_tokens)
        steps, ollama_rows, errors, done, streak = [], [], 0, False, 0
        headers = {"x-contextgc-run": run_id} if proxy else {}
        if proxy and stages:
            headers["x-contextgc-stages"] = stages
        t0 = time.perf_counter()
        for step in range(max_steps):
            payload = {"model": model, "messages": messages, "tools": TOOLS, "stream": False,
                       "options": {"num_ctx": window, "temperature": 0, "seed": 7,
                                   "num_predict": 512}}
            try:
                r = client.post(f"{base}/api/chat", json=payload, headers=headers)
                if r.status_code == 500:
                    # Ollama answers 500 when the model emits an unparseable tool
                    # call: a model error the agent should recover from.
                    raise ValueError(f"500: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                errors += 1
                streak += 1
                steps.append({"step": step + 1, "error": f"{type(exc).__name__}: {exc}"[:300]})
                if streak >= 3 or not isinstance(exc, ValueError):
                    break
                # Greedy decoding would repeat the same output; change the input.
                messages.append({"role": "user", "content":
                                 "Your last tool call was malformed. Make one valid tool call."})
                continue
            streak = 0
            row = {k: data.get(k) for k in ("load_duration", "prompt_eval_count",
                                            "prompt_eval_cached_count", "prompt_eval_duration", "eval_count", "eval_duration")}
            ollama_rows.append(row)
            message = data.get("message") or {}
            # One action per step: small models batch blind writes before reading.
            calls = parse_calls(message)[:1]
            assistant = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                assistant["tool_calls"] = [{"id": f"s{step}_{i}", "function": {
                    "name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]
            messages.append(assistant)
            steps.append({"step": step + 1, "tools": [n for n, _ in calls],
                          "args": [json.dumps(a)[:300] for _, a in calls],
                          "reload": (row["load_duration"] or 0) > 500_000_000,
                          "prompt_eval_count": row["prompt_eval_count"],
                          "prompt_eval_cached_count": row["prompt_eval_cached_count"],
                          "prompt_eval_s": round((row["prompt_eval_duration"] or 0) / 1e9, 2)})
            if not calls:
                if "DONE" in assistant["content"].upper():
                    ok, output = task.run_checker(root)
                    if ok:
                        done = True
                        break
                    messages.append({"role": "user", "content":
                                     "pytest still fails:\n" + output[-1500:] + "\nKeep fixing."})
                    continue
                messages.append({"role": "user", "content":
                                 "Use a tool call to continue, or reply DONE if pytest passes."})
                continue
            for i, (name, args) in enumerate(calls):
                messages.append({"role": "tool", "tool_call_id": f"s{step}_{i}", "tool_name": name,
                                 "content": run_tool(root, task, name, args)})
        passed, output = task.run_checker(root)
    cache = cache_stats(ollama_rows)
    return {
        "task": task.name, "passed": passed, "said_done": done, "steps": len(steps),
        "errors": errors, "wall_s": round(time.perf_counter() - t0, 1),
        "reloads": sum(1 for s in steps if s.get("reload")),
        "cache_hit_ratio": cache["cache_hit_ratio"],
        "prefilled_tokens_est": cache["prefilled_tokens_est"],
        "prompt_eval_tokens": sum(r["prompt_eval_count"] or 0 for r in ollama_rows),
        "checker_tail": output[-400:], "trace": steps,
    }


def warmup(client: httpx.Client, base: str, model: str, window: int) -> None:
    try:
        client.post(f"{base}/api/chat", json={
            "model": model, "messages": [{"role": "user", "content": "warm up"}],
            "stream": False, "options": {"num_ctx": window}})
    except Exception:
        pass


def residency(client: httpx.Client, ollama_url: str, model: str) -> dict:
    """Ollama's own view of where the model lives: 1.0 = fully on GPU."""
    try:
        for item in client.get(f"{ollama_url}/api/ps").json().get("models", []):
            if item.get("name") == model or item.get("model") == model:
                size, vram = item.get("size") or 0, item.get("size_vram") or 0
                return {"size": size, "size_vram": vram,
                        "context_length": item.get("context_length"),
                        "gpu_fraction": round(vram / size, 3) if size else None}
    except Exception:
        pass
    return {}


def gc_telemetry(client: httpx.Client, proxy_url: str, run_id: str) -> dict:
    """What the proxy did during one run, from its telemetry ring buffer."""
    try:
        events = client.get(f"{proxy_url}/telemetry/recent", params={"n": 500}).json()["events"]
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    mine = [e for e in events if e.get("run_id") == run_id]
    stage_saved: dict[str, int] = {}
    for e in mine:
        for stage, n in (e.get("stage_tokens_saved") or {}).items():
            stage_saved[stage] = stage_saved.get(stage, 0) + int(n or 0)
    return {"requests": len(mine),
            "compactions": sum(1 for e in mine if e.get("compaction")),
            "fidelity_fallbacks": sum(1 for e in mine
                                      if ((e.get("meta") or {}).get("fidelity_fallback") or {}).get("triggered")),
            "raw_tokens_est": sum(e.get("raw_tokens") or 0 for e in mine),
            "clean_tokens_est": sum(e.get("clean_tokens") or 0 for e in mine),
            "stage_tokens_saved": stage_saved}


def parse_lanes(spec: str, windows: dict, args) -> dict:
    """'a1,a2,b,b-no_semantic' -> {lane: (key, base_url, window, proxied, stages_preset)}."""
    from app.core.gc import _STAGE_PRESETS

    lanes = {}
    for lane in (x.strip() for x in spec.split(",") if x.strip()):
        if lane == "a1":
            lanes[lane] = ("A1", args.ollama_url, windows["a1"], False, None)
        elif lane == "a2":
            lanes[lane] = ("A2", args.ollama_url, windows["small"], False, None)
        elif lane == "b":
            lanes[lane] = ("B", args.proxy_url, windows["small"], True, None)
        elif lane.startswith("b-") and lane[2:] in _STAGE_PRESETS:
            lanes[lane] = (f"B:{lane[2:]}", args.proxy_url, windows["small"], True, lane[2:])
        else:
            sys.exit(f"[tasks] unknown lane {lane!r}: use a1, a2, b or b-<preset> "
                     f"with a preset from {sorted(_STAGE_PRESETS)}")
    return lanes


def resolve_windows(args) -> dict:
    """Turn 'auto' settings into numbers for this machine.

    The small window (A2 and B) is what the capacity probe picks: VRAM-sized on
    a GPU (CONTEXTGC_VRAM_OVERRIDE_MB simulates a smaller card), the configured
    default on CPU. Padding defaults to one small window of estimated tokens,
    which is ~1.5x that in real tokens for log text, so the window overflows.
    """
    settings = config.get_settings()
    window, probe = capacity.choose_window(args.model, settings.numctx_default,
                                           settings.numctx_min, settings.numctx_max)
    small = window if args.small_num_ctx == "auto" else int(args.small_num_ctx)
    a1 = settings.numctx_max if args.a1_num_ctx == "auto" else int(args.a1_num_ctx)
    pad = small if args.pad_tokens == "auto" else int(args.pad_tokens)
    if small >= a1:
        print(f"[tasks] warning: small window {small} >= A1 window {a1}; this machine fits "
              "the big window, so set CONTEXTGC_VRAM_OVERRIDE_MB to simulate a smaller card.")
    return {"small": small, "a1": a1, "pad": pad, "probe": probe}


def summarize(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {"runs": len(rows), "passed": sum(r["passed"] for r in rows),
            "success_rate": round(sum(r["passed"] for r in rows) / len(rows), 3) if rows else None,
            "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
            "reloads": sum(r["reloads"] for r in rows), "errors": sum(r["errors"] for r in rows),
            "avg_steps": mean("steps"), "avg_cache_hit_ratio": mean("cache_hit_ratio"),
            "min_gpu_fraction": min((r["residency"].get("gpu_fraction") for r in rows
                                     if r.get("residency", {}).get("gpu_fraction") is not None),
                                    default=None),
            "prefilled_tokens_est": sum(r["prefilled_tokens_est"] or 0 for r in rows),
            "prompt_eval_tokens": sum(r["prompt_eval_tokens"] or 0 for r in rows),
            "compactions": sum((r.get("gc") or {}).get("compactions") or 0 for r in rows),
            "fidelity_fallbacks": sum((r.get("gc") or {}).get("fidelity_fallbacks") or 0 for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="ContextGC three-lane task-success benchmark")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--proxy-url", default="http://localhost:11400")
    ap.add_argument("--lanes", default="a1,a2,b")
    ap.add_argument("--tasks", default="", help="comma list; default all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--a1-num-ctx", default="auto",
                    help="big window; auto = CONTEXTGC_NUMCTX_MAX (32768)")
    ap.add_argument("--small-num-ctx", default="auto",
                    help="A2 and B window; auto = capacity probe for this machine")
    ap.add_argument("--pad-tokens", default="auto",
                    help="earlier-session noise, chars/4 estimate; auto = one small window")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--tag", default="tasks")
    args = ap.parse_args()

    windows = resolve_windows(args)
    machine = {"platform": platform.platform(), "python": platform.python_version(),
               "windows": {k: windows[k] for k in ("small", "a1", "pad")},
               "capacity_probe": windows["probe"],
               "vram_override_mb": os.getenv("CONTEXTGC_VRAM_OVERRIDE_MB")}
    print(f"[tasks] windows: small={windows['small']} a1={windows['a1']} pad={windows['pad']} "
          f"(vram source: {windows['probe'].get('source')})", flush=True)
    lanes = parse_lanes(args.lanes, windows, args)
    order = list(lanes)
    keys = [lanes[lane][0] for lane in order]
    wanted = {x.strip() for x in args.tasks.split(",") if x.strip()}
    tasks = [t for t in TASKS if not wanted or t.name in wanted]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}.json"
    runs: list[dict] = []
    with httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0)) as client:
        for rep in range(args.repeats):
            rotated = order[rep % len(order):] + order[:rep % len(order)]
            for task in tasks:
                for lane in rotated:
                    key, base, window, proxy, stages = lanes[lane]
                    # Straight to Ollama: through the proxy a header-less warm-up gets a
                    # derived run id that may already be pinned to another window.
                    warmup(client, args.ollama_url, args.model, window)
                    run_id = f"{args.tag}-{key}-{task.name}-r{rep}-{time.time_ns()}"
                    result = run_task(client, base, args.model, task, window, run_id,
                                      proxy, args.max_steps, windows["pad"], stages)
                    result.update(lane=key, repeat=rep + 1, window=window,
                                  residency=residency(client, args.ollama_url, args.model))
                    if proxy:
                        result["gc"] = gc_telemetry(client, args.proxy_url, run_id)
                    runs.append(result)
                    gc = result.get("gc") or {}
                    print(f"[r{rep + 1}] {key:2} {task.name:24} "
                          f"{'PASS' if result['passed'] else 'fail'} steps={result['steps']} "
                          f"wall={result['wall_s']}s reloads={result['reloads']} "
                          f"cache={result['cache_hit_ratio']} "
                          f"gpu={result['residency'].get('gpu_fraction')}"
                          + (f" compactions={gc.get('compactions')} fidelity={gc.get('fidelity_fallbacks')}"
                             if proxy else ""), flush=True)
                    summary = {k: summarize([r for r in runs if r["lane"] == k])
                               for k in keys if any(r["lane"] == k for r in runs)}
                    out.write_text(json.dumps({"args": vars(args), "machine": machine,
                                               "summary": summary, "runs": runs},
                                              indent=2, default=str), encoding="utf-8")
    print(f"\n{'lane':16}{'pass':>10}{'wall s':>10}{'reloads':>9}{'cache':>8}{'prefilled':>11}"
          f"{'prompt tok':>12}{'fidelity':>10}")
    for key, s in summary.items():
        print(f"{key:16}{s['passed']:>5}/{s['runs']:<4}{s['total_wall_s']:>10}{s['reloads']:>9}"
              f"{str(s['avg_cache_hit_ratio']):>8}{s['prefilled_tokens_est']:>11}"
              f"{s['prompt_eval_tokens']:>12}{s['fidelity_fallbacks']:>10}")
    print(f"[tasks] results written: {out}")


if __name__ == "__main__":
    main()
