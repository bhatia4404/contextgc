"""Controlled Ollama prefix-cache probe (one-off diagnostic).

Sends 5 nearly-identical prompts and measures TTFT for each:
  1. baseline large prompt, num_ctx=4096        -> full prefill (cold)
  2. identical repeat, num_ctx=4096             -> cache hit expected (tiny TTFT)
  3. identical, num_ctx=5120 (CHANGED)          -> realloc bust expected (full prefill)
  4. num_ctx back to 4096, one word CHANGED in
     the FIRST user message (top of history)    -> prefix bust expected (full prefill)
  5. identical to #4                            -> cache hit expected (tiny TTFT)

num_predict=5 keeps generation negligible; TTFT here ~= prefill time.
"""
import json
import time

import httpx

FILLER = "alpha beta gamma delta epsilon zeta eta theta iota kappa\n" * 110  # ~2k chars


def make_prompt(edited: bool) -> dict:
    u1 = "Remember this constant: K-774. " + FILLER
    if edited:
        u1 = u1.replace("alpha", "ALPHA", 1)
    return {
        "model": "llama3.2:3b",
        "stream": True,
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": u1},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What constant did I give you?"},
        ],
        "options": {"num_predict": 5},
    }


def run(client: httpx.Client, num_ctx: int, edited: bool) -> float:
    p = make_prompt(edited)
    p["options"]["num_ctx"] = num_ctx
    t0 = time.perf_counter()
    ttft = None
    with client.stream("POST", "http://localhost:11434/api/chat", json=p) as r:
        for line in r.iter_lines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if (obj.get("message") or {}).get("content"):
                ttft = time.perf_counter() - t0
                break
    return ttft


with httpx.Client(timeout=600) as c:
    cases = [
        ("1 baseline (cold, ctx=4096)", 4096, False),
        ("2 identical repeat (ctx=4096)", 4096, False),
        ("3 SAME text, num_ctx CHANGED to 5120", 5120, False),
        ("4 ctx back to 4096, first msg EDITED", 4096, True),
        ("5 identical repeat of #4", 4096, True),
    ]
    print(f"{'case':45}{'ttft_s':>9}")
    for label, ctx, edited in cases:
        ttft = run(c, ctx, edited)
        print(f"{label:45}{ttft:9.2f}")
