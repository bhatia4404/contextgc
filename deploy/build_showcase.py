"""Build deploy/data/showcase.json, the data the API serves to the judge site.

Reads merged campaign files (backend/benchmark/merge_results.py output) and
keeps what the site shows: per-lane totals for every model, the per-task grid,
the stage ablation, the CrewAI comparison, and three highlighted tasks.

Highlights are picked by a fixed rule, not by hand, and the site says so:
ContextGC's pass rate minus the raw small window's, then minus the raw big
window's, then token savings. Every other task stays visible in the full grid.

    python deploy/build_showcase.py \
        --main backend/benchmark/results/modal/final_qwen7b_t4.json \
        --main backend/benchmark/results/modal/final_qwen14b_l4.json \
        --main backend/benchmark/results/modal/final_llama8b_l4.json \
        --ablation backend/benchmark/results/modal/ablation3_qwen7b_t4.json \
        --crewai backend/benchmark/results/modal/crew_14b_l4.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
LANES = ("A1", "A2", "B")
REPO_URL = "https://github.com/bhatia4404/contextgc"


def gpu_of(data: dict) -> str:
    name = (data.get("campaign") or "").lower()
    return next((g.upper() for g in ("t4", "l4", "a10g") if f"_{g}" in name), "GPU")


def lane_row(s: dict) -> dict:
    return {"runs": s["runs"], "passed": s["passed"], "rate": s.get("success_rate"),
            "prompt_tokens": s.get("prompt_eval_tokens", s.get("prompt_tokens")),
            "wall_s": s.get("total_wall_s"), "reloads": s.get("reloads"),
            "fidelity_fallbacks": s.get("fidelity_fallbacks"), "compactions": s.get("compactions")}


def trace(run: dict) -> list[dict]:
    return [{"step": t["step"], "prompt": t.get("prompt_eval_count"), "tools": t.get("tools") or [],
             "error": bool(t.get("error"))} for t in run.get("trace") or []]


def pick_run(runs: list[dict], prefer_pass: bool) -> dict | None:
    ordered = sorted(runs, key=lambda r: (r["passed"] != prefer_pass, r.get("repeat", 0)))
    return ordered[0] if ordered else None


def campaign(data: dict) -> dict:
    windows = (data.get("machine") or {}).get("windows") or {}
    return {"id": data["campaign"], "model": (data.get("args") or {}).get("model"), "gpu": gpu_of(data),
            "windows": {"big": windows.get("a1"), "small": windows.get("small")},
            "repeats": (data.get("args") or {}).get("repeats"),
            "summary": {k: lane_row(v) for k, v in data["summary"].items() if k in LANES},
            "per_task": {t: {k: lane_row(v) for k, v in lanes.items() if k in LANES}
                         for t, lanes in data["per_task"].items()}}


def highlights(mains: list[dict], n: int = 3) -> list[dict]:
    scored = []
    for data in mains:
        for task, lanes in data["per_task"].items():
            if not all(k in lanes for k in LANES):
                continue
            rate = {k: lanes[k]["success_rate"] or 0 for k in LANES}
            tokens = {k: lanes[k]["prompt_eval_tokens"] or 0 for k in LANES}
            saving = 1 - tokens["B"] / tokens["A1"] if tokens["A1"] else 0
            # rounded so 1.0 - 0.667 ties with 0.333 - 0 and the next key decides
            scored.append((tuple(round(x, 2) for x in (rate["B"] - rate["A2"], rate["B"] - rate["A1"], saving)),
                           data, task))
    scored.sort(key=lambda x: x[0], reverse=True)
    out, seen = [], set()
    for score, data, task in scored:
        if task in seen or score[0] <= 0:      # one card per task; only real wins
            continue
        seen.add(task)
        runs = [r for r in data["runs"] if r["task"] == task]
        reps = {k: pick_run([r for r in runs if r["lane"] == k], prefer_pass=(k == "B")) for k in LANES}
        out.append({"campaign": data["campaign"], "model": data["args"]["model"], "gpu": gpu_of(data),
                    "task": task, "score": [round(x, 3) for x in score],
                    "lanes": {k: lane_row(v) for k, v in data["per_task"][task].items() if k in LANES},
                    "traces": {k: {"passed": r["passed"], "repeat": r.get("repeat"), "window": r.get("window"),
                                   "steps": trace(r)} for k, r in reps.items() if r}})
        if len(out) == n:
            break
    return out


def ablation(data: dict) -> dict:
    return {"id": data["campaign"], "model": data["args"]["model"], "gpu": gpu_of(data),
            "repeats": data["args"].get("repeats"),
            "summary": {k: lane_row(v) for k, v in data["summary"].items()},
            "per_task": {t: {k: {"passed": v["passed"], "runs": v["runs"]} for k, v in lanes.items()}
                         for t, lanes in data["per_task"].items()}}


def crewai(data: dict) -> dict:
    return {"id": data["campaign"], "model": (data.get("args") or {}).get("model"), "gpu": gpu_of(data),
            "summary": data["summary"]}


def hosted_api(data: dict, provider: str = "Meta Model API") -> dict:
    """run_tasks_api.py output -> per-run averages per lane, plus the per-task grid."""
    lanes = {}
    for lane, s in data["summary"].items():
        n = s["runs"] or 1
        lanes[lane] = {"runs": s["runs"], "passed": s["passed"],
                       "prompt_tokens_per_run": round(s["prompt_tokens"] / n),
                       "completion_tokens_per_run": round(s["completion_tokens"] / n),
                       "cost_per_run": round(s["cost_usd"] / n, 5), "time_per_run": round(s["total_wall_s"] / n, 1),
                       "cost_total": s["cost_usd"]}
    tasks: dict = {}
    for r in data["runs"]:
        t = tasks.setdefault(r["task"], {}).setdefault(r["lane"], {"passed": 0, "runs": 0})
        t["runs"] += 1
        t["passed"] += int(r["passed"])
    return {"provider": provider, "model": data["args"]["model"], "repeats": data["args"]["repeats"],
            "budget": data["args"]["budget"], "lanes": lanes, "per_task": tasks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--main", action="append", type=Path, default=[])
    ap.add_argument("--ablation", type=Path)
    ap.add_argument("--crewai", type=Path)
    ap.add_argument("--meta", type=Path, help="run_tasks_api.py result against Meta's Model API")
    ap.add_argument("--out", type=Path, default=DATA / "showcase.json")
    a = ap.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    mains = [load(p) for p in a.main]
    show = {"generated": time.strftime("%Y-%m-%d"), "repo": REPO_URL,
            "campaigns": [campaign(d) for d in mains],
            "highlights": highlights(mains),
            "ablation": ablation(load(a.ablation)) if a.ablation else None,
            "crewai": crewai(load(a.crewai)) if a.crewai else None,
            "hosted_api": hosted_api(load(a.meta)) if a.meta else None}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(show, indent=1), encoding="utf-8")
    print(f"wrote {a.out}: {len(show['campaigns'])} campaigns, "
          f"highlights: {[h['task'] + ' (' + h['model'] + ')' for h in show['highlights']]}")


if __name__ == "__main__":
    main()
