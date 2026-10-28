r""" ContextGC — multi-agent-style benchmark harness (Phase 2.1, minimal lane).

Simulates a deterministic multi-turn agent session and drives it through BOTH
lanes, measuring the failure modes plan v2 named:

    Lane A — straight to Ollama (unoptimized control)
    Lane B — via the ContextGC proxy (the product under test)

Per turn, each lane records:
    prompt_tokens_est  (what enters prefill — the thing that grows every turn)
    ttft_s             (time to first streamed token)
    total_latency_s    end-to-end
    reply_chars        (output sanity — comparable behavior check)
    http status        (Lane A's silent truncation shows up as degraded
                        replies/errors once history exceeds its num_ctx)

Console table + JSON export under benchmark/results/.

Usage (two terminals):
    # lane A target is plain Ollama; lane B needs the proxy with GC on:
    #   CONTEXTGC_GC_ENABLED=true python -m uvicorn app.main:app --port 11400
    python scripts\..\benchmark\run_simulation.py --turns 15 --lane both
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Deterministic agent-style scenario: alternating instruction turns; every 4th
# turn injects a big synthetic "tool dump" (the episodic bloat trigger).
_USER_TURNS = [
    "Summarize the current task state in one sentence and continue.",
    "Check the failing tests again and propose the next fix.",
    "Refactor the parser helper we discussed; keep it minimal.",
    "Report progress briefly, then continue with the plan.",
    "Verify the last change compiles; fix if not, then continue.",
    "We need to finalize docs for the parser change; draft them.",
]
_DUMP_LINES = 220   # ~3,300 chars ≈ 800 est tokens per "tool result"


def scenario_turns(n_turns: int, dumps_every: int = 4) -> list[dict]:
    turns = []
    for t in range(n_turns):
        turns.append({"role": "user", "content": _USER_TURNS[t % len(_USER_TURNS)]})
        if dumps_every and t % dumps_every == 2:
            turns.append({
                "role": "tool",
                "name": "bash",
                "tool_call_id": f"tc_{t}",
                "content": f"[pytest output turn {t}]\n"
                           + "CheckError: assertion mismatch on line %d\n" % t * _DUMP_LINES
                           + f"FAILED: {t} tests failed.",
            })
    return turns


def estimate_tokens(payload: dict) -> int:
    from app.core.num_ctx import estimate_tokens as est
    return est(payload)


async def run_lane(client: httpx.AsyncClient, base_url: str, n_turns: int,
                   model: str, telemetry_base: str | None = None,
                   run_id: str = "default", dumps_every: int = 4,
                   stages_preset: str | None = None,
                   replay: bool = False,
                   request_num_ctx: int | None = None) -> list[dict]:
    """One lane: grows a real conversation history turn by turn.

    telemetry_base: for Lane B, the proxy base — after each turn we pull the
    turn_telemetry event the proxy recorded, giving what ACTUALLY entered
    prefill (clean_tokens, per-stage savings, negotiated num_ctx).

    run_id isolates cross-request GC state per experiment (unique per run);
    stages_preset drives the proxy's per-request stage filter header.
    """
    history: list[dict] = [{"role": "system",
                            "content": "You are a terse coding agent. Answer in <= 2 sentences."}]
    script = scenario_turns(n_turns, dumps_every=dumps_every)
    metrics = []
    seen_event_ids: set[str] = set()
    for t, incoming in enumerate(script):
        history.append(incoming)
        payload = {"model": model, "messages": history, "stream": True}
        if request_num_ctx:
            payload["options"] = {"num_ctx": request_num_ctx}
        t0 = time.perf_counter()
        ttft = None
        chunks = 0
        reply_parts: list[str] = []
        last_chunk: dict = {}
        transport_error = None
        try:
            req = client.build_request(
                "POST", f"{base_url}/api/chat", json=payload,
                headers={"content-type": "application/json",
                         "x-contextgc-run": run_id,
                         **({"x-contextgc-replay": "true"} if replay else {}),
                         **({"x-contextgc-stages": stages_preset} if stages_preset else {})})
            resp = await client.send(req, stream=True)
            status = resp.status_code
            if status == 200:
                async for line in resp.aiter_lines():
                    line = line.strip()
                    # Ollama native: newline-delimited JSON; OpenAI-compat: SSE.
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if line == "[DONE]":
                            continue
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    delta = ""
                    if isinstance(obj, dict):
                        last_chunk = obj
                        delta = (obj.get("message") or {}).get("content") or ""
                    if delta:
                        reply_parts.append(delta)
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                    chunks += 1
            await resp.aclose()
        except Exception as exc:
            status, transport_error = -1, f"{type(exc).__name__}: {exc}"
        total = time.perf_counter() - t0
        reply = "".join(reply_parts)
        metric = {
            "turn": t + 1,
            "stage": "tool_dump" if incoming.get("role") == "tool" else "chat",
            "prompt_tokens_est": estimate_tokens(payload),
            "http_status": status,
            "ttft_s": round(ttft, 4) if ttft else None,
            "total_latency_s": round(total, 4),
            "reply_chars": len(reply),
        }
        if transport_error:
            metric["transport_error"] = transport_error
        if not reply and status == 200:
            metric["empty_reply"] = True
            history.append({"role": "assistant", "content":
                            f"<<empty reply turn {t + 1}: {transport_error or 'no deltas parsed'}>>"})
        else:
            history.append({"role": "assistant", "content":
                            "Replay reply: continue with the requested task." if replay
                            else (reply or "<<no reply>>")})
        if last_chunk:
            metric["ollama"] = {
                key: last_chunk.get(key)
                for key in ("load_duration", "prompt_eval_count", "prompt_eval_duration",
                            "eval_count", "eval_duration")
                if key in last_chunk
            }
            load_ns = last_chunk.get("load_duration")
            prompt_ns = last_chunk.get("prompt_eval_duration")
            # Ollama reports nanoseconds; warm model bookkeeping is commonly
            # single-digit milliseconds and must not be called a reload.
            if load_ns is not None:
                metric["reload"] = load_ns > 500_000_000
            if prompt_ns is not None:
                metric["prompt_cache_candidate"] = prompt_ns < 100_000_000
        # Lane B: pull what the proxy itself sent downstream this turn
        if telemetry_base:
            ev = await _pull_next_telemetry(client, telemetry_base, seen_event_ids)
            if ev:
                metric["proxy"] = {
                    "turn_id": ev.get("turn_id"),
                    "clean_tokens": ev.get("clean_tokens"),
                    "tokens_saved": ev.get("tokens_saved"),
                    "pct": ev.get("tokens_saved_pct"),
                    "stages": ev.get("stage_tokens_saved"),
                    "num_ctx": (ev.get("num_ctx") or {}).get("chosen_num_ctx"),
                    "fidelity": bool((ev.get("meta") or {}).get("fidelity_fallback", {}).get("triggered")),
                }
        metrics.append(metric)
        last = metric.get("proxy") or {}
        print(f"  turn {t + 1:>2} [{metric['stage']}] sent_est={metric['prompt_tokens_est']:>5}"
              + (f" clean={last.get('clean_tokens'):>5} saved={last.get('tokens_saved'):>5} "
                 f"({last.get('pct')}%) stages={last.get('stages')}" if last else "")
              + (f" ttft={metric['ttft_s']}s" if metric["ttft_s"] else "")
              + (f" EMPTY/ERR status={status}" if not reply else ""))
    return metrics


async def _pull_next_telemetry(client: httpx.AsyncClient, proxy_base: str,
                               seen: set[str]) -> dict | None:
    """Next unseen turn_telemetry event from the proxy's ring buffer."""
    r = await client.get(f"{proxy_base}/telemetry/recent")
    for e in reversed((r.json() or {}).get("events", [])):
        if e.get("type") == "turn_telemetry" and e.get("id") not in seen:
            seen.add(e["id"])
            return e
    return None


def summarize(name: str, metrics: list[dict]) -> dict:
    """Most meaningful lane-B number is what ENTERED prefill (proxy clean)."""
    tok = [m["prompt_tokens_est"] for m in metrics]
    prefill_last = [
        m["proxy"]["clean_tokens"] for m in metrics
        if (m.get("proxy") or {}).get("clean_tokens") is not None
    ] or tok
    ttft = [m["ttft_s"] for m in metrics if m["ttft_s"] is not None]
    lat = [m["total_latency_s"] for m in metrics]
    saved = [
        m["proxy"]["tokens_saved"] for m in metrics
        if (m.get("proxy") or {}).get("tokens_saved")
    ]
    ollama = [m.get("ollama", {}) for m in metrics]
    prompt_eval = [x.get("prompt_eval_count") for x in ollama if x.get("prompt_eval_count") is not None]
    prompt_dur = [x.get("prompt_eval_duration") for x in ollama if x.get("prompt_eval_duration") is not None]
    return {
        "lane": name,
        "turns": len(metrics),
        "tokens_first": tok[0] if tok else 0,
        "tokens_sent_last": prefill_last[-1] if prefill_last else 0,
        "raw_grew_to": tok[-1] if tok else 0,
        "tokens_saved_total": sum(saved) if saved else 0,
        "ttft_avg_s": round(sum(ttft) / len(ttft), 4) if ttft else None,
        "ttft_last_s": ttft[-1] if ttft else None,
        "latency_avg_s": round(sum(lat) / len(lat), 4) if lat else None,
        "errors": sum(1 for m in metrics if m["http_status"] != 200),
        "reload_count": sum(1 for m in metrics if m.get("reload") is True),
        "cache_hit_ratio": round(
            sum(1 for m in metrics if m.get("prompt_cache_candidate") is True) / len(metrics), 4
        ) if metrics else 0.0,
        "prompt_eval_tokens": sum(prompt_eval),
        "prompt_eval_duration_ns": sum(prompt_dur),
    }


def print_table(rows: list[dict]) -> None:
    print(f"\n{'lane':24}{'turns':7}{'sent last (est)':17}{'raw grew to':13}"
          f"{'saved total':13}{'ttft avg':10}{'lat avg':10}errors")
    for s in rows:
        print(f"{str(s['lane']):24}{s['turns']:<7}"
              f"{s['tokens_sent_last']:<17}{s['raw_grew_to']:<13}"
              f"{s['tokens_saved_total']:<13}"
              f"{str(s['ttft_avg_s'] if s['ttft_avg_s'] else '—'):10}"
              f"{str(s['latency_avg_s'] if s['latency_avg_s'] else '—'):10}"
              f"{s['errors']}")


async def run_session(client: httpx.AsyncClient, args: argparse.Namespace,
                      order: str, run_id: str) -> tuple[list[dict], dict[str, list]]:
    """One alternating session: run the requested lanes in the given order.

    order: 'ab' or 'ba' — swapping which lane goes first per session separates
    a real lane effect from warm-up/churn drift (Iteration 13 item 2).
    """
    lane_desc = {
        "a": ("A1", "raw ollama (large window)", args.ollama_url, False, None, args.a1_num_ctx),
        "a1": ("A1", "raw ollama (large window)", args.ollama_url, False, None, args.a1_num_ctx),
        "a2": ("A2", "raw ollama (small window)", args.ollama_url, False, None, args.a2_num_ctx),
        "b": ("B", "contextgc (small window)", args.proxy_url, True, args.stages, args.b_num_ctx),
    }
    summaries = []
    lane_metrics: dict[str, list] = {}
    for lane in order:
        key, label, url, with_telemetry, stages, window = lane_desc[lane]
        print(f"[harness] Lane {key} ({label}, {args.turns} turns) ...")
        await _warmup(client, url, args.model, window, run_id,
                      replay=args.replay, proxy=with_telemetry)
        m = await run_lane(client, url, args.turns, args.model,
                           telemetry_base=url if with_telemetry else None,
                           run_id=run_id, dumps_every=args.dumps_every,
                           stages_preset=stages, replay=args.replay,
                           request_num_ctx=window)
        lane_metrics[key] = m
        s = summarize(f"{key}: {label}", m)
        summaries.append(s)
        print_table([s])
    return summaries, lane_metrics


async def _warmup(client: httpx.AsyncClient, url: str, model: str, window: int,
                  run_id: str, replay: bool, proxy: bool) -> None:
    payload = {"model": model, "messages": [{"role": "user", "content": "warm up"}],
               "stream": False, "options": {"num_ctx": window}}
    headers = {"content-type": "application/json"}
    if proxy:
        headers["x-contextgc-run"] = run_id + "-warmup"
        if replay:
            headers["x-contextgc-replay"] = "true"
    try:
        response = await client.post(f"{url}/api/chat", json=payload, headers=headers)
        response.raise_for_status()
    except Exception:
        pass


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return (sum(vals) / len(vals)) if vals else None


def lane_avg(rows: list[dict]) -> dict:
    """Cross-session mean of a lane's per-session summary rows."""
    out = {"lane": rows[0]["lane"], "turns": rows[0]["turns"],
           "sessions": len(rows)}
    for key in ("tokens_first", "tokens_sent_last", "raw_grew_to",
                 "tokens_saved_total", "ttft_avg_s", "latency_avg_s",
                 "errors", "reload_count", "prompt_eval_tokens",
                 "prompt_eval_duration_ns", "cache_hit_ratio"):
        m = _mean(rows, key)
        out[f"avg_{key}"] = round(m, 4) if m is not None else None
    return out


async def main() -> None:
    ap = argparse.ArgumentParser(description="ContextGC two-lane benchmark harness (minimal)")
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--proxy-url", default="http://localhost:11400")
    ap.add_argument("--lane", choices=["a", "a1", "a2", "b", "both", "three"], default="both")
    ap.add_argument("--dumps-every", type=int, default=4,
                    help="inject a tool dump every N turns (0 = never)")
    ap.add_argument("--stages", default="full",
                    choices=["full", "none", "anchor_only", "no_semantic", "no_evict"],
                    help="Lane-B stage-filter preset (control experiments)")
    ap.add_argument("--rotate", type=int, default=1,
                    help="run the lane pair N times, alternating which lane goes "
                         "first (A/B then B/A) to cancel warm-up/churn drift; "
                         "per-session summaries + cross-session averages in JSON")
    ap.add_argument("--tag", default="run", help="output filename tag")
    ap.add_argument("--a1-num-ctx", type=int, default=32768,
                    help="raw quality-ceiling context window")
    ap.add_argument("--a2-num-ctx", type=int, default=4096,
                    help="raw constrained context window")
    ap.add_argument("--b-num-ctx", type=int, default=4096,
                    help="ContextGC constrained context window")
    ap.add_argument("--replay", action="store_true",
                    help="use the same fixed assistant reply in both lanes")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    n_rotate = max(1, args.rotate)

    sessions, agg = [], {"A1": [], "A2": [], "B": []}
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        for i in range(n_rotate):
            order = "ab" if i % 2 == 0 else "ba"
            if args.lane in {"a", "a1"}:
                order = "a"
            elif args.lane == "a2":
                order = "a2"
            elif args.lane == "b":
                order = "b"
            elif args.lane == "three":
                order = ["a1", "a2", "b"]
            run_id = f"{args.tag}-s{i + 1}-{time.time_ns()}"  # unique per session
            print(f"[harness] session {i + 1}/{n_rotate} "
                  f"(run_id={run_id}, lanes {'+'.join(order).upper()}) ...")
            summaries, lane_metrics = await run_session(client, args, order, run_id)
            for s in summaries:
                agg.setdefault(s["lane"].split(":")[0], []).append(s)
            sessions.append({
                "session": i + 1,
                "lane_order": " then ".join(s["lane"] for s in summaries),
                "summary": summaries,
                "turn_metrics": lane_metrics,
            })

        # Cross-session averages (the churn-resistant numbers).
        averages = {key: lane_avg(rows) for key, rows in agg.items() if rows}
        if len(avg_rows := list(averages.values())) > 0:
            print(f"\n[averages across {n_rotate} session(s)]")
            header = (f"{'lane':24}{'sessions':10}{'sent last (est)':17}"
                      f"{'saved total':13}{'ttft avg':10}{'lat avg':10}{'errors':7}")
            print(header)
            for a in avg_rows:
                print(f"{str(a['lane']):24}{a['sessions']:<10}"
                      f"{a['avg_tokens_sent_last']:<17.1f}"
                      f"{a['avg_tokens_saved_total']:<13.1f}"
                      f"{str(a['avg_ttft_avg_s']):10}"
                      f"{str(a['avg_latency_avg_s']):10}"
                      f"{a['avg_errors']:.1f}")

        out_file = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}.json"
        out_file.write_text(json.dumps({
            "args": vars(args), "summary": summaries, "summary_avg": averages,
            "sessions": sessions,
        }, indent=2, ensure_ascii=False))
        print(f"[harness] metrics written: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
