"""Probe 2: does returning to the ORIGINAL num_ctx recover the old prefill?

Sequence (same unedited prompt throughout, num_predict=5):
  1. ctx=4096, cold            -> full prefill baseline
  2. ctx=4096, repeat          -> cache hit (hot)
  3. ctx=5120, repeat          -> realloc bust (cold again)
  4. ctx=4096, repeat          <- THE QUESTION:
       ~2s   => the old 4096 prefill survived the 5120 detour
       ~25s  => the detour DESTROYED it (must recompute from scratch)
"""
import json
import time

import httpx

FILLER = "alpha beta gamma delta epsilon zeta eta theta iota kappa\n" * 110


def make_prompt() -> dict:
    return {
        "model": "llama3.2:3b",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Remember this constant: K-774. " + FILLER},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What constant did I give you?"},
        ],
        "options": {"num_predict": 5},
    }


def run(client: httpx.Client, num_ctx: int) -> float:
    p = make_prompt()
    p["options"]["num_ctx"] = num_ctx
    t0 = time.perf_counter()
    with client.stream("POST", "http://localhost:11434/api/chat", json=p) as r:
        for line in r.iter_lines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if (obj.get("message") or {}).get("content"):
                return time.perf_counter() - t0
    return float("nan")


with httpx.Client(timeout=600) as c:
    cases = [
        ("1 ctx=4096 cold", 4096),
        ("2 ctx=4096 repeat (hot)", 4096),
        ("3 ctx=5120 detour (bust)", 5120),
        ("4 ctx=4096 again: old prefill back?", 4096),
    ]
    print(f"{'case':45}{'ttft_s':>9}")
    for label, ctx in cases:
        print(f"{label:45}{run(c, ctx):9.2f}")
