"""Backend API for the judge site (the React app in frontend/ is the front end).

    python deploy/build_showcase.py ...      # refresh deploy/data/showcase.json
    modal deploy deploy/api_app.py           # prints the API URL

Endpoints: GET /api/showcase (measured results), POST /api/run (start a live
run), GET /api/run/{id} (its progress), GET /api/recent, GET /api/status.
The live run is one GPU container at a time (max_containers=1) with a daily
cap, so a public page can never run up more than a few dollars a day:
LIVE_DAILY_CAP runs x ~6 min of T4 (~$0.06 each).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BACKEND = REPO / "backend"
REMOTE_BACKEND = "/root/backend"
MODEL = "qwen2.5:7b"
TASKS = ["fix_failing_sum", "repair_json_reader", "fix_path_join", "preserve_false_values",
         "fix_retry_counter", "implement_stable_sort", "fix_cache_key", "fix_csv_parser"]
LANE_ARG = {"A1": "a1", "A2": "a2", "B": "b"}
LIVE_DAILY_CAP = int(os.getenv("CONTEXTGC_LIVE_DAILY_CAP", "25"))
STALE_S = 40 * 60          # a "current" run older than this no longer blocks new ones
OLLAMA_VERSION = os.getenv("CONTEXTGC_OLLAMA_VERSION", "0.40.13")

# Same layers as deploy/modal_app.py (shared in Modal's image cache), minus CrewAI.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "zstd", "pciutils")
    .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh")
    .pip_install_from_requirements(str(BACKEND / "requirements.txt"))
    .pip_install("pynvml")
    .pip_install("sentence-transformers")
    .run_commands("python -c \"from sentence_transformers import SentenceTransformer; "
                  "SentenceTransformer('all-MiniLM-L6-v2')\"")
    .env({"OLLAMA_MODELS": "/models", "OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_NUM_PARALLEL": "1",
          "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(BACKEND), REMOTE_BACKEND,
                   ignore=["**/__pycache__", "**/.pytest_cache", "data", "benchmark/results", "*.log", "tmp_tel.json"])
)
web_image = (modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")
             .add_local_file(str(HERE / "data" / "showcase.json"), "/data/showcase.json"))
# Origins allowed to call the API: the hosted React app and the Vite dev server.
ALLOWED_ORIGINS = ["https://shivambhatia4404--contextgc-showcase-web.us-east.modal.run",
                   "http://localhost:5173", "http://127.0.0.1:5173"]

app = modal.App("contextgc-api")
models_vol = modal.Volume.from_name("contextgc-models", create_if_missing=True)
runs = modal.Dict.from_name("contextgc-live-runs", create_if_missing=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _update(key: str, **fields) -> dict:
    run = runs.get(key) or {}
    run.update(fields)
    runs[key] = run
    return run


@app.function(image=gpu_image, gpu="T4", volumes={"/models": models_vol}, timeout=30 * 60, max_containers=1)
def live_run(key: str, task: str, lanes: list[str]) -> None:
    """Run one task in the chosen lanes and stream every step into the shared Dict."""
    _update(key, status="starting", started=_now(), message="GPU is up; starting Ollama and loading the model.")
    out_dir = Path("/tmp/live")
    out_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, "scripts/run_stack.py", "--model", MODEL, "--",
           "benchmark/run_tasks.py", "--lanes", ",".join(LANE_ARG[k] for k in lanes), "--repeats", "1",
           "--tasks", task, "--max-steps", "12", "--small-num-ctx", "8192", "--a1-num-ctx", "32768",
           "--pad-tokens", "auto", "--tag", "live"]
    env = {**os.environ, "CONTEXTGC_STEP_LOG": "1", "CONTEXTGC_RESULTS_DIR": str(out_dir)}
    steps: dict[str, list] = {k: [] for k in lanes}
    log: list[str] = []
    last_push = 0.0
    proc = subprocess.Popen(cmd, cwd=REMOTE_BACKEND, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("[step] "):
            row = json.loads(line[7:])
            lane = row["run"].split("-")[1]           # run id: live-<lane>-<task>-r0-<ns>
            steps.setdefault(lane, []).append({"step": row["step"], "prompt": row.get("prompt_eval_count"),
                                               "tools": row.get("tools") or [], "error": bool(row.get("error"))})
            log.append(f"{LANE_ARG.get(lane, lane).upper():3} step {row['step']:>2}: "
                       f"{', '.join(row.get('tools') or []) or ('error' if row.get('error') else 'reply')}"
                       f" · {row.get('prompt_eval_count')} prompt tokens")
            msg = f"Running lane {lane}, step {row['step']}."
        elif re.match(r"\[r\d+\]", line) or line.startswith(("[stack]", "[tasks]")):
            log.append(line)
            msg = "Model loaded; agent is working." if "running:" in line else None
        else:
            continue
        if time.time() - last_push > 1.5 or line.startswith("[r"):
            _update(key, status="running", steps=steps, log=log[-200:], **({"message": msg} if msg else {}))
            last_push = time.time()
    code = proc.wait()
    results = {}
    for path in out_dir.glob("*.json"):
        for r in json.loads(path.read_text(encoding="utf-8")).get("runs", []):
            results[r["lane"]] = {"passed": r["passed"], "steps": r["steps"], "wall_s": r["wall_s"],
                                  "prompt_tokens": r.get("prompt_eval_tokens"),
                                  "compactions": (r.get("gc") or {}).get("compactions")}
    ok = code == 0 and results
    _update(key, status="done" if ok else "error", finished=_now(), steps=steps, log=log[-200:], results=results,
            message="Finished." if ok else f"The run stopped early (exit code {code}).")


@app.function(image=web_image, max_containers=2)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    api = FastAPI(docs_url=None, redoc_url=None)
    api.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["content-type"])

    def busy() -> str | None:
        key = runs.get("_current")
        run = runs.get(key) if key else None
        if not run or run.get("status") in ("done", "error"):
            return None
        created = time.mktime(time.strptime(run["created"], "%Y-%m-%dT%H:%M:%S"))
        return key if time.mktime(time.gmtime()) - created < STALE_S else None

    @api.post("/api/run")
    def start(req: dict = Body(...)):
        task, wanted = req.get("task"), req.get("lanes") or ["A2", "B"]
        lanes = [k for k in ("A2", "B", "A1") if k in wanted]
        if task not in TASKS or not lanes:
            raise HTTPException(400, "unknown task or no lanes")
        if (key := busy()):
            return {"id": key, "joined": True}
        day = "_count_" + time.strftime("%Y-%m-%d", time.gmtime())
        if runs.get(day, 0) >= LIVE_DAILY_CAP:
            raise HTTPException(429, f"Today's limit of {LIVE_DAILY_CAP} live runs is used up. "
                                     "The results page has every measured run.")
        runs[day] = runs.get(day, 0) + 1
        key = uuid.uuid4().hex[:10]
        runs[key] = {"id": key, "task": task, "lanes": lanes, "status": "queued", "created": _now(),
                     "message": "Waiting for a GPU.", "steps": {}, "results": {}, "log": []}
        runs["_current"] = key
        runs["_recent"] = ([key] + (runs.get("_recent") or []))[:15]
        live_run.spawn(key, task, lanes)
        return {"id": key, "joined": False}

    @api.get("/api/run/{key}")
    def status(key: str):
        if key.startswith("_") or not (run := runs.get(key)):
            raise HTTPException(404, "no such run")
        return run

    @api.get("/api/recent")
    def recent():
        rows = []
        for key in runs.get("_recent") or []:
            run = runs.get(key)
            if run:
                rows.append({k: run.get(k) for k in ("id", "task", "status", "created", "results")})
        return rows

    @api.get("/api/status")
    def quota():
        day = "_count_" + time.strftime("%Y-%m-%d", time.gmtime())
        return {"today": runs.get(day, 0), "cap": LIVE_DAILY_CAP, "busy": bool(busy())}

    @api.get("/api/showcase")
    def showcase():
        return FileResponse("/data/showcase.json", headers={"cache-control": "no-cache"})

    return api
