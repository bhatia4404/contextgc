r"""Bring up Ollama + the ContextGC proxy on this machine, then run a benchmark.

One entry point for every target: a CPU laptop, a remote GPU box, or a Modal
container (deploy/modal_app.py calls this script).

  * Ollama: reused if it already answers at --ollama-url; otherwise started
    with `ollama serve` (local URLs only) and stopped afterwards.
  * Models: pulled if missing.
  * Proxy: reused if it answers with GC on; otherwise started with
    CONTEXTGC_GC_ENABLED=true and stopped afterwards.
  * Benchmark: everything after `--` runs with this Python, with
    --ollama-url / --proxy-url / --model appended when not given.

Examples (from backend/):
    python scripts/run_stack.py --model qwen2.5:3b -- benchmark/run_tasks.py --repeats 3
    python scripts/run_stack.py --model qwen2.5:3b -- benchmark/run_simulation.py --lane three
    # remote Ollama on a GPU box, proxy local:
    python scripts/run_stack.py --ollama-url http://gpu-box:11434 --model llama3.1:8b -- benchmark/run_tasks.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

BACKEND = Path(__file__).resolve().parents[1]


def _up(url: str) -> dict | None:
    try:
        r = httpx.get(url, timeout=2.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def _wait(url: str, what: str, proc: subprocess.Popen | None, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (body := _up(url)) is not None:
            return body
        if proc is not None and proc.poll() is not None:
            sys.exit(f"[stack] {what} exited with code {proc.returncode}")
        time.sleep(1)
    sys.exit(f"[stack] {what} did not come up at {url} within {timeout_s:.0f}s")


def _is_local(url: str) -> bool:
    return urlparse(url).hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def ensure_ollama(url: str, log_dir: Path) -> subprocess.Popen | None:
    if _up(f"{url}/api/version") is not None:
        print(f"[stack] using running Ollama at {url}")
        return None
    if not _is_local(url):
        sys.exit(f"[stack] remote Ollama at {url} is not reachable")
    binary = shutil.which("ollama")
    if not binary:
        sys.exit("[stack] Ollama is not running and `ollama` is not on PATH")
    port = urlparse(url).port or 11434
    env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{port}"}
    env.setdefault("OLLAMA_KEEP_ALIVE", "30m")
    print(f"[stack] starting ollama serve on :{port}")
    proc = subprocess.Popen([binary, "serve"], env=env,
                            stdout=open(log_dir / "ollama.log", "ab"), stderr=subprocess.STDOUT)
    version = _wait(f"{url}/api/version", "ollama serve", proc, 120)
    print(f"[stack] Ollama {version.get('version')} up")
    return proc


def ensure_models(url: str, models: list[str]) -> None:
    present = set()
    for item in (_up(f"{url}/api/tags") or {}).get("models", []):
        present.update({item.get("name"), item.get("model")})
    for model in models:
        if model in present or f"{model}:latest" in present:
            continue
        print(f"[stack] pulling {model} ...", flush=True)
        r = httpx.post(f"{url}/api/pull", json={"model": model, "stream": False}, timeout=3600)
        r.raise_for_status()


def ensure_proxy(url: str, ollama_url: str, log_dir: Path) -> subprocess.Popen | None:
    health = _up(f"{url}/healthz")
    if health is not None:
        if not health.get("gc_enabled"):
            sys.exit(f"[stack] proxy at {url} is running with GC off; stop it or use another port")
        print(f"[stack] using running proxy at {url}")
        return None
    if not _is_local(url):
        sys.exit(f"[stack] remote proxy at {url} is not reachable")
    port = urlparse(url).port or 11400
    env = {**os.environ, "CONTEXTGC_GC_ENABLED": "true", "CONTEXTGC_OLLAMA_BASE_URL": ollama_url}
    print(f"[stack] starting ContextGC proxy on :{port}")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=BACKEND, env=env,
        stdout=open(log_dir / "proxy.log", "ab"), stderr=subprocess.STDOUT)
    _wait(f"{url}/healthz", "proxy", proc, 60)
    return proc


def _with_defaults(cmd: list[str], args: argparse.Namespace) -> list[str]:
    flags = {"--ollama-url": args.ollama_url, "--proxy-url": args.proxy_url,
             "--model": args.models[0]}
    extra = [x for flag, value in flags.items() if flag not in cmd for x in (flag, value)]
    return cmd + extra


def main() -> None:
    argv = sys.argv[1:]
    bench = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description="Run Ollama + ContextGC + a benchmark on this machine")
    ap.add_argument("--ollama-url", default=os.getenv("CONTEXTGC_OLLAMA_BASE_URL", "http://localhost:11434"))
    ap.add_argument("--proxy-url", default="http://localhost:11400")
    ap.add_argument("--model", dest="models", action="append",
                    help="model to pull (repeatable); the first is passed to the benchmark")
    ap.add_argument("--log-dir", default=os.getenv("CONTEXTGC_RESULTS_DIR") or str(BACKEND / "benchmark" / "results"))
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    args.models = args.models or ["qwen2.5:3b"]
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    started: list[subprocess.Popen] = []
    code = 0
    try:
        if p := ensure_ollama(args.ollama_url, log_dir):
            started.append(p)
        ensure_models(args.ollama_url, args.models)
        if p := ensure_proxy(args.proxy_url, args.ollama_url, log_dir):
            started.append(p)
        if bench:
            cmd = [sys.executable] + _with_defaults(bench, args)
            print(f"[stack] running: {' '.join(cmd[1:])}", flush=True)
            code = subprocess.call(cmd, cwd=BACKEND)
        else:
            print("[stack] stack is up; press Ctrl+C to stop")
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in reversed(started):
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
    sys.exit(code)


if __name__ == "__main__":
    main()
