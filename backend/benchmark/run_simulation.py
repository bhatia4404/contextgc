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


def scenario_turns(n_turns: int) -> list[dict]:
    turns = []
    for t in range(n_turns):
        turns.append({"role": "user", "content": _USER_TURNS[t % len(_USER_TURNS)]})
        if t % 4 == 2:
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
                   model: str, telemetry_base: str | None = None) -> list[dict]:
    """One lane: grows a real conversation history turn by turn.

    telemetry_base: for Lane B, the proxy base — after each turn we pull the
    turn_telemetry event the proxy recorded, giving what ACTUALLY entered
    prefill (clean_tokens, per-stage savings, negotiated num_ctx).
    """
    history: list[dict] = [{"role": "system",
                            "content": "You are a terse coding agent. Answer in <= 2 sentences."}]
    script = scenario_turns(n_turns)
    metrics = []
    seen_event_ids: set[str] = set()
    for t, incoming in enumerate(script):
        history.append(incoming)
        payload = {"model": model, "messages": history, "stream": True}
        t0 = time.perf_counter()
        ttft = None
        chunks = 0
        reply_parts: list[str] = []
        transport_error = None
        try:
            req = client.build_request(
                "POST", f"{base_url}/api/chat", json=payload,
                headers={"content-type": "application/json"})
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
            history.append({"role": "assistant", "content": reply or "<<no reply>>"})
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


async def main() -> None:
    ap = argparse.ArgumentParser(description="ContextGC two-lane benchmark harness (minimal)")
    ap.add_argument("--turns", type=int, default=15)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--proxy-url", default="http://localhost:11400")
    ap.add_argument("--lane", choices=["a", "b", "both"], default="both")
    ap.add_argument("--tag", default="run", help="output filename tag")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        summaries = []
        lane_metrics: dict[str, list] = {}
        if args.lane in ("a", "both"):
            print(f"[harness] Lane A (raw Ollama, {args.turns} turns) ...")
            m = await run_lane(client, args.ollama_url, args.turns, args.model)
            lane_metrics["A"] = m
            s = summarize("A: raw ollama", m)
            summaries.append(s)
            print_table([s])
        if args.lane in ("b", "both"):
            print(f"[harness] Lane B (ContextGC proxy, {args.turns} turns) ...")
            m = await run_lane(client, args.proxy_url, args.turns, args.model,
                               telemetry_base=args.proxy_url)
            lane_metrics["B"] = m
            s = summarize("B: contextgc", m)
            summaries.append(s)
            print_table([s])

        out_file = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}.json"
        out_file.write_text(json.dumps({
            "args": vars(args), "summary": summaries,
            "turn_metrics": lane_metrics,
        }, indent=2, ensure_ascii=False))
        print(f"[harness] metrics written: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
