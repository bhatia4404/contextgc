"""A/B test: the same CrewAI crew against raw Ollama and through ContextGC.

Both lanes use the same model, the same context window (num_ctx, enforced via
the native /api/chat endpoint) and a fresh copy of ../demo_repository per run.
Only the base URL differs:

    raw        Ollama directly      (http://localhost:11434)
    contextgc  through the proxy    (http://localhost:11400, GC on)

Real incidents come with large logs, and the demo repository is tiny, so each
run's copy gets a deterministic production-sized checkout log (--log-lines).
That is what puts the window under pressure.

Success is decided here, not by the agent: the repository's own pytest suite
must pass and the test files must be unchanged.

Usage (Ollama and a GC-enabled proxy already running, or via run_stack.py):
    python crewai_agent/run_ab.py --agents incident --repeats 3
    python backend/scripts/run_stack.py --model qwen2.5:7b -- ../crewai_agent/run_ab.py --repeats 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo_repository"
AGENTS = {"incident": "incident_response_agent.py", "code": "code_maintenance_agent.py"}
LOG_PATH = Path("data/incidents/checkout-2026-10-28.log")
RESULTS_DIR = Path(os.getenv("CONTEXTGC_RESULTS_DIR") or HERE / "results")


def outage_log(lines: int) -> str:
    """Deterministic checkout log for the INC-1042 window, original lines kept."""
    original = (DEMO / LOG_PATH).read_text(encoding="utf-8").splitlines()
    out, every = [], max(1, lines // max(1, len(original)))
    for i in range(lines):
        if i % every == 0 and original:
            out.append(original.pop(0))
            continue
        ts = f"2026-10-28T09:{14 + i // 600:02d}:{(i // 10) % 60:02d}Z"
        req = f"{(i * 2654435761) % 0xFFFFFF:06x}"
        kind = i % 7
        if kind in (0, 3):
            out.append(f"{ts} ERROR request_id={req} route=checkout status=503 error=provider_timeout "
                       f"provider=acme attempt={1 + i % 3} upstream_ms=3000")
        elif kind == 1:
            out.append(f"{ts} WARN request_id={req} payment_provider=acme timeout_ms=3000 retry_bucket=checkout")
        elif kind == 2:
            out.append(f"{ts} INFO request_id={req} route=search status=200 latency_ms={120 + i % 90}")
        elif kind == 4:
            out.append(f"{ts} INFO request_id={req} route=checkout status=200 latency_ms={900 + i % 400} cached_cart=true")
        elif kind == 5:
            out.append(f"{ts} DEBUG worker=gw{i % 4} queue=payments-platform depth={300 + i % 600} lag_ms={i % 5000}")
        else:
            out.append(f"{ts} INFO request_id={req} route=mobile status=200 latency_ms={200 + i % 150}")
    return "\n".join(out + original) + "\n"


def prepare_repo(root: Path, log_lines: int) -> Path:
    repo = root / "repo"
    shutil.copytree(DEMO, repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if log_lines:
        (repo / LOG_PATH).write_text(outage_log(log_lines), encoding="utf-8")
    return repo


def tests_digest(repo: Path) -> str:
    h = hashlib.sha256()
    for path in sorted((repo / "tests").rglob("*.py")):
        h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()


def pytest_passes(repo: Path) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=repo,
                       env=env, capture_output=True, text=True, timeout=120)
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


def warm(ollama_url: str, model: str, num_ctx: int) -> None:
    """Load the model at the lane's window directly in Ollama, outside the timing."""
    httpx.post(f"{ollama_url}/api/chat", timeout=600, json={
        "model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False,
        "keep_alive": "30m", "options": {"num_ctx": num_ctx, "num_predict": 1}}).raise_for_status()


def residency(ollama_url: str, model: str) -> dict:
    try:
        for m in httpx.get(f"{ollama_url}/api/ps", timeout=5).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size = m.get("size") or 0
                return {"context_length": m.get("context_length"),
                        "gpu_fraction": round((m.get("size_vram") or 0) / size, 3) if size else None}
    except Exception:
        pass
    return {}


def gc_telemetry(proxy_url: str, run_id: str) -> dict:
    try:
        events = httpx.get(f"{proxy_url}/telemetry/recent", params={"n": 500}, timeout=10).json()["events"]
    except Exception as exc:
        return {"error": str(exc)}
    mine = [e for e in events if str(e.get("run_id", "")).startswith(run_id)]
    raw = sum(e.get("raw_tokens") or 0 for e in mine)
    clean = sum(e.get("clean_tokens") or 0 for e in mine)
    return {"requests": len(mine), "raw_tokens_est": raw, "clean_tokens_est": clean,
            "saved_pct": round((raw - clean) / raw * 100, 1) if raw else 0.0,
            "fidelity_fallbacks": sum(1 for e in mine
                                      if ((e.get("meta") or {}).get("fidelity_fallback") or {}).get("triggered"))}


def run_one(agent: str, lane: str, rep: int, args, base_url: str) -> dict:
    run_id = f"{args.tag}-{agent}-{lane}-r{rep}-{time.time_ns() % 10**8}"
    with tempfile.TemporaryDirectory(prefix=f"cgc-crew-{agent}-") as tmp:
        repo = prepare_repo(Path(tmp), args.log_lines)
        before = tests_digest(repo)
        result_file = Path(tmp) / "result.json"
        log_file = RESULTS_DIR / f"{args.tag}_{agent}_{lane}_r{rep}.log"
        env = {**os.environ, "AGENT_BASE_URL": base_url, "AGENT_MODEL": args.model,
               "AGENT_NUM_CTX": str(args.num_ctx), "AGENT_RUN_ID": run_id,
               "AGENT_REPOSITORY": str(repo), "AGENT_RESULT_FILE": str(result_file),
               "AGENT_CREW_CONTEXT": "0" if lane.endswith("-nocrew") else "1",
               "CREWAI_DISABLE_TELEMETRY": "true", "OTEL_SDK_DISABLED": "true",
               "LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        t0 = time.perf_counter()
        timed_out = False
        with open(log_file, "w", encoding="utf-8") as log:
            try:
                code = subprocess.run([sys.executable, AGENTS[agent]], cwd=HERE, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        wall = round(time.perf_counter() - t0, 1)
        agent_result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
        passed, pytest_tail = pytest_passes(repo)
        tampered = tests_digest(repo) != before
    row = {"agent": agent, "lane": lane, "repeat": rep, "run_id": run_id,
           "passed": passed and not tampered, "tests_tampered": tampered, "timed_out": timed_out,
           "exit_code": code, "wall_s": wall, "usage": agent_result.get("usage", {}),
           "residency": residency(args.ollama_url, args.model), "pytest_tail": pytest_tail,
           "crew_output_tail": str(agent_result.get("crew_output", ""))[-800:], "log": log_file.name}
    row["crew_context"] = not lane.endswith("-nocrew")
    if lane.startswith("contextgc"):
        row["contextgc"] = gc_telemetry(args.proxy_url, run_id)
    return row


def summarize(runs: list[dict]) -> dict:
    out: dict = {}
    for r in runs:
        s = out.setdefault(f"{r['agent']}/{r['lane']}", {"runs": 0, "passed": 0, "wall_s": 0.0, "prompt_tokens": 0})
        s["runs"] += 1
        s["passed"] += int(r["passed"])
        s["wall_s"] += r["wall_s"]
        s["prompt_tokens"] += int((r.get("usage") or {}).get("prompt_tokens") or 0)
    for s in out.values():
        s["wall_s"] = round(s["wall_s"] / s["runs"], 1)
        s["prompt_tokens"] = s["prompt_tokens"] // s["runs"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agents", default="incident,code", help=f"comma list of {sorted(AGENTS)}")
    ap.add_argument("--lanes", default="raw,contextgc",
                    help="raw, contextgc; add '-nocrew' (e.g. raw-nocrew) to switch off CrewAI's own context handling")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--num-ctx", type=int, default=8192, help="same window for both lanes")
    ap.add_argument("--log-lines", type=int, default=2000, help="size of the generated outage log (0 = original)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per crew run")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--proxy-url", default="http://localhost:11400")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M%S"))
    args = ap.parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    base = {"raw": args.ollama_url, "contextgc": args.proxy_url}
    urls = {lane: base[lane.removesuffix("-nocrew")] for lane in lanes}
    if any(lane.startswith("contextgc") for lane in lanes) and not httpx.get(f"{args.proxy_url}/healthz", timeout=5).json().get("gc_enabled"):
        sys.exit("[crew-ab] the proxy is running with GC off; start it with CONTEXTGC_GC_ENABLED=true")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"crew_ab_{args.tag}.json"
    report = {"args": vars(args), "machine": {"platform": platform.platform()}, "runs": []}
    print(f"[crew-ab] model={args.model} num_ctx={args.num_ctx} log_lines={args.log_lines} -> {out_path}")

    warm(args.ollama_url, args.model, args.num_ctx)
    for rep in range(1, args.repeats + 1):
        order = lanes[rep % len(lanes):] + lanes[:rep % len(lanes)]   # alternate who goes first
        for agent in agents:
            for lane in order:
                row = run_one(agent, lane, rep, args, urls[lane])
                report["runs"].append(row)
                report["summary"] = summarize(report["runs"])
                out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                gc = row.get("contextgc") or {}
                extra = f" gc_saved={gc.get('saved_pct')}% req={gc.get('requests')}" if gc else ""
                print(f"[r{rep}] {agent:<8} {lane:<16} {'PASS' if row['passed'] else 'fail'} "
                      f"wall={row['wall_s']}s prompt_tokens={(row['usage'] or {}).get('prompt_tokens')}"
                      f"{' TIMEOUT' if row['timed_out'] else ''}{' TESTS-TAMPERED' if row['tests_tampered'] else ''}"
                      f"{extra}", flush=True)

    print(f"\n{'agent/lane':<27} {'pass':>6} {'wall s':>8} {'prompt tok':>11}")
    for key, s in report["summary"].items():
        print(f"{key:<27} {s['passed']}/{s['runs']:<4} {s['wall_s']:>8} {s['prompt_tokens']:>11}")
    print(f"[crew-ab] results written: {out_path}")


if __name__ == "__main__":
    main()
