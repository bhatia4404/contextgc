"""Merge one campaign's per-task result files into a single results file.

deploy/launch.py runs one container per task, so a campaign folder holds one
<task>/<stamp>_<tag>.json per task (run_tasks.py) or per CrewAI lane
(run_ab.py). This writes <campaign>.json next to the folder with every run, a
summary per lane, and a per-task x lane table, in the same shape the dashboard
already reads ({"args", "machine", "summary", "runs"}).

    python benchmark/merge_results.py benchmark/results/modal/final_qwen7b_t4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tasks import summarize  # noqa: E402


def crew_summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {"runs": n, "passed": sum(r["passed"] for r in rows),
            "success_rate": round(sum(r["passed"] for r in rows) / n, 3) if n else None,
            "total_wall_s": round(sum(r["wall_s"] for r in rows), 1),
            "timeouts": sum(bool(r.get("timed_out")) for r in rows),
            "prompt_tokens": sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in rows),
            "gc_saved_pct_avg": (round(sum((r.get("contextgc") or {}).get("saved_pct") or 0 for r in rows) / n, 1)
                                 if any(r.get("contextgc") for r in rows) else None)}


def merge(folder: Path) -> dict:
    runs, args, machine = [], None, None
    # Newest file per task folder only: a preempted container is restarted by
    # Modal and leaves the partial file of its first attempt behind.
    newest: dict[Path, Path] = {}
    for path in sorted(folder.rglob("*.json")):
        newest[path.parent] = path
    for path in newest.values():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "runs" not in data:
            continue
        runs += data["runs"]
        args = args or data.get("args")
        machine = machine or data.get("machine")
    if not runs:
        raise SystemExit(f"no result files under {folder}")
    crew = "agent" in runs[0]
    lane_of = (lambda r: f"{r['agent']}/{r['lane']}") if crew else (lambda r: r["lane"])
    task_of = (lambda r: r["agent"]) if crew else (lambda r: r["task"])
    summ = crew_summary if crew else summarize
    lanes = list(dict.fromkeys(lane_of(r) for r in runs))
    tasks = list(dict.fromkeys(task_of(r) for r in runs))
    per_task = {t: {lane: summ([r for r in runs if task_of(r) == t and lane_of(r) == lane])
                    for lane in lanes if any(task_of(r) == t and lane_of(r) == lane for r in runs)}
                for t in tasks}
    return {"campaign": folder.name, "kind": "crewai" if crew else "tasks", "args": args, "machine": machine,
            "summary": {lane: summ([r for r in runs if lane_of(r) == lane]) for lane in lanes},
            "per_task": per_task, "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    merged = merge(a.folder)
    out = a.out or a.folder.with_suffix(".json")
    out.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"{merged['campaign']}: {len(merged['runs'])} runs -> {out}")
    for lane, s in merged["summary"].items():
        print(f"  {lane:28} {s['passed']}/{s['runs']}")


if __name__ == "__main__":
    main()
