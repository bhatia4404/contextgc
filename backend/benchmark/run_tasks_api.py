"""The task benchmark against a hosted OpenAI-compatible API (e.g. Meta's Model API).

Same 8 tasks, tools, earlier-session noise and pytest grading as run_tasks.py;
only the transport differs: /v1/chat/completions with OpenAI message shapes.
Two lanes, same model:

    raw        straight to the API; the model sees the whole history
    contextgc  through the ContextGC proxy (CONTEXTGC_SINK=<api base>), which
               keeps the prompt near a budget (--budget, sent as num_ctx and
               stripped by the proxy before it reaches the API)

The API bills by tokens, so the question is: same pass rate, fewer tokens?
Every response's `usage` is recorded and priced; the run stops at --max-cost.

    MODEL_API_KEY=... python benchmark/run_tasks_api.py --model muse-spark-1.3-contributor \
        --api-url https://api.meta.ai --proxy-url http://localhost:11401 --tasks fix_path_join --repeats 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tasks import (SYSTEM, TOOLS, gc_telemetry, padding_messages, parse_calls,  # noqa: E402
                       run_tool, _noise_dump)
from task_suite import TASKS  # noqa: E402

RESULTS_DIR = Path(os.getenv("CONTEXTGC_RESULTS_DIR") or Path(__file__).resolve().parent / "results")
# $ per 1M tokens (dev.meta.ai/docs/pricing-rate-limits, 2026-11-04): cached input, input, output
PRICES = {"muse-spark-1.3-contributor": (0.002, 0.10, 0.20), "muse-spark-1.3": (0.15, 1.25, 4.25),
          "muse-wonder-1.0": (0.20, 4.00, 20.00)}


def to_openai(messages: list[dict]) -> list[dict]:
    """run_tasks.py builds Ollama-shaped messages; the API wants OpenAI shapes."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                "name": c["function"]["name"],
                "arguments": c["function"]["arguments"] if isinstance(c["function"]["arguments"], str)
                else json.dumps(c["function"]["arguments"])}} for c in m["tool_calls"]]
            m["content"] = m.get("content") or None
        m.pop("tool_name", None)
        out.append(m)
    return out


def cost(usage: dict, model: str) -> float:
    cached_p, in_p, out_p = PRICES.get(model, PRICES["muse-spark-1.3"])
    prompt = usage.get("prompt_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    out = usage.get("completion_tokens") or 0
    return ((prompt - cached) * in_p + cached * cached_p + out * out_p) / 1e6


def run_task(client: httpx.Client, base: str, key: str, model: str, task, run_id: str,
             proxied: bool, budget: int, max_steps: int, pad_tokens: int) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"cgc-api-{task.name}-") as directory:
        root = Path(directory)
        task.materialize(root)
        for c in range(3 if pad_tokens else 0):
            (root / "logs").mkdir(exist_ok=True)
            (root / "logs" / f"ci_run_{c}.log").write_text(_noise_dump(c + 1, 200), encoding="utf-8")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task.prompt}] + padding_messages(pad_tokens)
        headers = {"Authorization": f"Bearer {key}"}
        if proxied:
            headers["x-contextgc-run"] = run_id
        steps, usage_rows, errors, done = [], [], 0, False
        t0 = time.perf_counter()
        for step in range(max_steps):
            payload = {"model": model, "messages": to_openai(messages), "tools": TOOLS,
                       "temperature": 0, "max_tokens": 2048}
            if proxied:
                payload["num_ctx"] = budget
            try:
                r = client.post(f"{base}/v1/chat/completions", json=payload, headers=headers)
                if r.status_code == 429:
                    time.sleep(5)
                    r = client.post(f"{base}/v1/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
            except Exception as exc:
                errors += 1
                body = getattr(getattr(exc, "response", None), "text", "")[:300]
                steps.append({"step": step + 1, "error": f"{type(exc).__name__}: {exc} {body}"[:400]})
                if errors >= 3:
                    break
                continue
            usage = data.get("usage") or {}
            usage_rows.append(usage)
            message = (data.get("choices") or [{}])[0].get("message") or {}
            calls = parse_calls(message)[:1]
            assistant = {"role": "assistant", "content": message.get("content") or ""}
            if calls:
                assistant["tool_calls"] = [{"id": f"s{step}_{i}", "function": {"name": n, "arguments": a}}
                                           for i, (n, a) in enumerate(calls)]
            messages.append(assistant)
            steps.append({"step": step + 1, "tools": [n for n, _ in calls],
                          "args": [json.dumps(a)[:300] for _, a in calls],
                          "prompt_tokens": usage.get("prompt_tokens"),
                          "completion_tokens": usage.get("completion_tokens"),
                          "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")})
            if not calls:
                if "DONE" in (assistant["content"] or "").upper():
                    ok, output = task.run_checker(root)
                    if ok:
                        done = True
                        break
                    messages.append({"role": "user", "content": "pytest still fails:\n" + output[-1500:] + "\nKeep fixing."})
                    continue
                messages.append({"role": "user", "content": "Use a tool call to continue, or reply DONE if pytest passes."})
                continue
            for i, (name, args) in enumerate(calls):
                messages.append({"role": "tool", "tool_call_id": f"s{step}_{i}", "tool_name": name,
                                 "content": run_tool(root, task, name, args)})
        passed, output = task.run_checker(root)
    return {"task": task.name, "passed": passed, "said_done": done, "steps": len(steps), "errors": errors,
            "wall_s": round(time.perf_counter() - t0, 1),
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage_rows),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage_rows),
            "cost_usd": round(sum(cost(u, model) for u in usage_rows), 5),
            "checker_tail": output[-600:], "trace": steps}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {"runs": n, "passed": sum(r["passed"] for r in rows),
            "success_rate": round(sum(r["passed"] for r in rows) / n, 3) if n else None,
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
            "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
            "fidelity_fallbacks": sum((r.get("gc") or {}).get("fidelity_fallbacks") or 0 for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="muse-spark-1.3-contributor")
    ap.add_argument("--api-url", default="https://api.meta.ai")
    ap.add_argument("--proxy-url", default="http://localhost:11401")
    ap.add_argument("--key-env", default="MODEL_API_KEY")
    ap.add_argument("--lanes", default="raw,contextgc")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--budget", type=int, default=8192, help="ContextGC prompt budget (tokens)")
    ap.add_argument("--pad-tokens", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-cost", type=float, default=1.0, help="stop once this many dollars are spent")
    ap.add_argument("--tag", default="api")
    args = ap.parse_args()
    key = os.environ.get(args.key_env) or sys.exit(f"set {args.key_env}")
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    wanted = {x.strip() for x in args.tasks.split(",") if x.strip()}
    tasks = [t for t in TASKS if not wanted or t.name in wanted]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}.json"
    runs, spent = [], 0.0
    with httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0)) as client:
        for rep in range(args.repeats):
            for task in tasks:
                for lane in (lanes if rep % 2 == 0 else lanes[::-1]):
                    if spent >= args.max_cost:
                        print(f"[api] stopping: ${spent:.3f} spent (cap ${args.max_cost})", flush=True)
                        break
                    proxied = lane == "contextgc"
                    run_id = f"{args.tag}-{lane}-{task.name}-r{rep}-{time.time_ns()}"
                    r = run_task(client, args.proxy_url if proxied else args.api_url, key, args.model, task,
                                 run_id, proxied, args.budget, args.max_steps, args.pad_tokens)
                    r.update(lane=lane, repeat=rep + 1)
                    if proxied:
                        r["gc"] = gc_telemetry(client, args.proxy_url, run_id)
                    runs.append(r)
                    spent += r["cost_usd"]
                    print(f"[r{rep + 1}] {lane:9} {task.name:24} {'PASS' if r['passed'] else 'fail'} "
                          f"steps={r['steps']} prompt={r['prompt_tokens']} out={r['completion_tokens']} "
                          f"${r['cost_usd']:.4f} errors={r['errors']} (total ${spent:.3f})", flush=True)
                    out.write_text(json.dumps({"args": vars(args), "summary": {
                        l: summarize([x for x in runs if x["lane"] == l]) for l in lanes if any(x["lane"] == l for x in runs)},
                        "runs": runs}, indent=2, default=str), encoding="utf-8")
    print(f"[api] results written: {out}  (spent ${spent:.3f})")


if __name__ == "__main__":
    main()
