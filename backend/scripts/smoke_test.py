r"""One-line smoke test for the ContextGC proxy.

Usage:
    python scripts\smoke_test.py                # non-streaming, prints the reply
    python scripts\smoke_test.py --stream       # streaming, prints tokens live

Exits 0 with [PASS] when everything works.
"""
import argparse
import json
import os
import sys

import httpx

PROXY = "http://127.0.0.1:11400"
MODEL = os.environ.get("CONTEXTGC_MODEL", "llama3")  # switch models via env, no code edits


def check_health() -> dict:
    r = httpx.get(f"{PROXY}/healthz", timeout=10)
    r.raise_for_status()
    return r.json()


def send_non_stream() -> str | None:
    r = httpx.post(
        f"{PROXY}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "what do you say? Just say: hello world"}],
            "stream": False,
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"], data["usage"]["total_tokens"]


def send_stream() -> str | None:
    text = []
    with httpx.stream(
        "POST",
        f"{PROXY}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "what do you say? Just say: hello world"}],
            "stream": True,
        },
        timeout=300,
    ) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        assert ctype.startswith("text/event-stream"), f"not streaming: {ctype}"
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                text.append(piece)
                print(piece, end="", flush=True)
    print()
    return "".join(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    health = check_health()
    print(f"[health] service={health['service']} gc_enabled={health['gc_enabled']} sink={health['sink']}")

    try:
        if args.stream:
            reply = send_stream()
            print(f"\n[PASS] streamed reply: {reply!r}")
        else:
            reply, tokens = send_non_stream()
            print(f"reply: {reply!r}   (total_tokens={tokens})")
            print("[PASS] non-streaming passthrough works")
    except httpx.ConnectError:
        print("[FAIL] proxy not reachable at", PROXY, "- start it with:")
        print("       python -m uvicorn app.main:app --host 127.0.0.1 --port 11400")
        sys.exit(2)


if __name__ == "__main__":
    main()
