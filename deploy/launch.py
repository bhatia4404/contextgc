"""Launch long benchmark campaigns that survive a closed laptop or session.

`modal run --detach` still dies when the local process is killed on Windows.
This script instead calls functions of the *deployed* app with `.spawn()`:
each call runs server-side until it finishes, and results land in the
`contextgc-results` Volume. One container per task, so a campaign runs in
parallel and one crash loses one task, not the campaign.

    modal deploy deploy/modal_app.py                 # once, and after code changes
    python deploy/launch.py main --gpu t4 --model qwen2.5:7b --repeats 3 --campaign final_qwen7b_t4
    python deploy/launch.py ablation --gpu t4 --model qwen2.5:7b --repeats 3 --campaign ablation3_qwen7b_t4
    python deploy/launch.py crewai --gpu l4 --model qwen2.5:14b --repeats 3 --campaign crew_14b_l4
    python deploy/launch.py status                    # what is still running

Fetch results:  modal volume get contextgc-results <campaign> backend/benchmark/results/modal/
Merge them:     python backend/benchmark/merge_results.py backend/benchmark/results/modal/<campaign>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import modal

APP = "contextgc-bench"
TASKS = ["fix_failing_sum", "repair_json_reader", "fix_path_join", "preserve_false_values",
         "fix_retry_counter", "implement_stable_sort", "fix_cache_key", "fix_csv_parser"]
ABLATION = "b,b-no_semantic,b-no_state,b-no_fidelity,b-no_evict,b-no_offload"
LEDGER = Path(__file__).resolve().parent / "launched.jsonl"


def bench_args(lanes: str, repeats: int, task: str, args) -> list[str]:
    return ["--lanes", lanes, "--repeats", str(repeats), "--max-steps", str(args.max_steps),
            "--small-num-ctx", str(args.small_num_ctx), "--a1-num-ctx", str(args.a1_num_ctx),
            "--pad-tokens", "auto", "--tasks", task]


def spawn(gpu: str, models: list[str], bench: list[str], tag: str, script: str) -> str:
    fn = modal.Function.from_name(APP, f"bench_{gpu}")
    call = fn.spawn(models, bench, tag, 0, script)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"call_id": call.object_id, "tag": tag, "gpu": gpu, "models": models,
                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    print(f"spawned {tag} -> {call.object_id}")
    return call.object_id


def status() -> None:
    if not LEDGER.exists():
        print("nothing launched yet")
        return
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        call = modal.FunctionCall.from_id(row["call_id"])
        try:
            result = call.get(timeout=0)
            state = f"done (exit {result['exit_code']})"
        except TimeoutError:
            state = "running"
        except Exception as exc:  # noqa: BLE001 - report any remote failure
            state = f"failed: {type(exc).__name__}: {str(exc)[:80]}"
        print(f"{row['at']}  {row['tag']:<48} {state}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["main", "ablation", "crewai", "status"])
    ap.add_argument("--gpu", default="t4", choices=["t4", "l4", "a10g"])
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--small-num-ctx", type=int, default=8192)
    ap.add_argument("--a1-num-ctx", type=int, default=32768)
    ap.add_argument("--crew-lanes", default="raw,contextgc,raw-nocrew,contextgc-nocrew")
    ap.add_argument("--campaign", default="")
    args = ap.parse_args()
    if args.kind == "status":
        return status()

    campaign = args.campaign or f"{time.strftime('%Y%m%d_%H%M')}_{args.kind}_{args.gpu}_{args.model.replace(':', '-')}"
    if args.kind == "crewai":
        # one container per lane: each runs both crews x repeats
        for lane in args.crew_lanes.split(","):
            crew = ["--agents", "incident,code", "--lanes", lane, "--repeats", str(args.repeats),
                    "--num-ctx", str(args.small_num_ctx), "--log-lines", "2000", "--timeout", "900",
                    "--tag", f"{campaign}_{lane}"]
            spawn(args.gpu, [args.model], crew, f"{campaign}/{lane}", "../crewai_agent/run_ab.py")
        return
    lanes = "a1,a2,b" if args.kind == "main" else ABLATION
    for task in args.tasks.split(","):
        spawn(args.gpu, [args.model], bench_args(lanes, args.repeats, task, args),
              f"{campaign}/{task}", "benchmark/run_tasks.py")


if __name__ == "__main__":
    main()
