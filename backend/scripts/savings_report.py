"""Savings report from the append-only store: tokens + KV-cache VRAM.

Comparisons (state of this run's data):
  raw  = what the agent framework sent (pre-GC)      -> est tokens = chars/4 heuristic
  clean= what ContextGC forwarded                     -> est tokens same heuristic
  KV VRAM per token for llama3.2:3b:
      n_layers=28, kv_heads=8, head_dim=128 -> 2*8*128*28*2B = 114,688 B/token
  Window comparison: with ContextGC the window follows the cleaned estimate
      (negotiated); without it the window stays the model's default
      (assumed 4096, the common Ollama default) regardless of payload size.

Usage: python scripts\savings_report.py
"""
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo/backend on path

KV_BYTES_PER_TOKEN = 114_688          # llama3.2:3b KV footprint per token
DEFAULT_WINDOW = int(__import__("os").environ.get("CONTEXTGC_REPORT_DEFAULT_WINDOW", "4096"))


def est_tokens(p: dict) -> int:
    from app.core import num_ctx
    return num_ctx.estimate_tokens(p)


def window(est: int, floor=2048, cap=32768) -> int:
    chosen = max(int(math.ceil(est * 1.25)), floor)
    chosen -= chosen % 1024
    return min(max(chosen, floor), cap)


def main() -> None:
    conn = sqlite3.connect("data/contextgc.db")
    conn.row_factory = sqlite3.Row
    turns = conn.execute(
        "SELECT turn_id, run_id, path, raw_body, clean_body, gc_report FROM turns ORDER BY turn_id"
    ).fetchall()

    tot_raw = tot_clean = 0
    kv_with_gc = kv_without_gc = 0.0
    pointered = deduped = reinjected = 0
    rows = []
    for t in turns:
        try:
            raw, clean, rep = (
                json.loads(t["raw_body"]), json.loads(t["clean_body"]), json.loads(t["gc_report"]),
            )
        except Exception:
            continue
        e_raw, e_clean = est_tokens(raw), est_tokens(clean)
        tot_raw += e_raw; tot_clean += e_clean
        # worst-turn-window booked with GC = negotiated on cleaned payload;
        # without GC: window stays the default regardless of the huge payload
        w_gc = window(e_clean)
        w_no = DEFAULT_WINDOW
        kv_with_gc += w_gc * KV_BYTES_PER_TOKEN
        kv_without_gc += max(w_no, window(e_raw) if False else w_no) * KV_BYTES_PER_TOKEN
        rr = rep or {}
        pointered += len(((rr.get("tool_offload") or {}).get("offloads")) or [])
        deduped += ((rr.get("stage1_dedup") or {}).get("removed") or []) and len(rr["stage1_dedup"]["removed"]) or 0
        reinjected += ((rr.get("ref_reinject") or {}).get("restored") or []) and 1 or 0
        rows.append((t["turn_id"], t["run_id"], e_raw, e_clean, w_gc, rr.get("pipeline", {}).get("modified")))

    print(f"turns analyzed (JSON-decodable chat turns): {len(rows)}")
    print()
    print("=== TOKENS ===")
    print(f"raw (what agents sent):        {tot_raw:>12,} tokens")
    print(f"clean (what the LLM saw):      {tot_clean:>12,} tokens")
    print(f"tokens saved:                  {tot_raw - tot_clean:>12,}  ({100*(1 - tot_clean/max(tot_raw,1)):.1f}% cut)")
    print(f"GC actions: dedup removed x{deduped}, pointered dumps x{pointered}, reinjects x{reinjected}")
    print()
    print("=== KV-CACHE VRAM (llama3.2:3b, 114.6 KB/token; assume no-GC window stays at default %d) ===" % DEFAULT_WINDOW)
    print(f"cleaned-turn windows summed:   {kv_with_gc/1e9:8.2f} GB booked (ContextGC)")
    print(f"default-window booked:         {kv_without_gc/1e9:8.2f} GB booked (no GC)")
    print(f"peak KV per turn avoided in the runs that ended pointered:")
    for tid, run, er, ec, wg, modified in rows:
        if wg > DEFAULT_WINDOW / 2048 and er > 3000:  # only turns where cleaning mattered
            print(f"   turn {tid} ({run}): raw est {er:,} tok | window {wg} | clean est {ec:,} tok"
                  f" | KV delta = {(window(er)-wg)*KV_BYTES_PER_TOKEN/1e6:,.0f} MB vs raw-sized window")
    conn.close()


if __name__ == "__main__":
    main()
