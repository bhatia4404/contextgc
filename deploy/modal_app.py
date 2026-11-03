"""ContextGC benchmarks on Modal GPUs (plan v3, Phase 3.5).

The container runs the same launcher as a laptop or any remote box,
backend/scripts/run_stack.py: Ollama + the ContextGC proxy + a benchmark, all
on localhost inside one container. Model weights live in a Modal Volume so they
are pulled once; results go to a second Volume and are also copied back to
backend/benchmark/results/modal/<tag>/ on your machine.

Setup (once):
    pip install modal
    modal token new

Long campaigns (survive a closed session): `modal deploy deploy/modal_app.py`, then
deploy/launch.py (one spawned container per task; see its docstring).

Run (from the repo root):
    modal run deploy/modal_app.py::pull --models qwen2.5:7b,llama3.1:8b          # CPU-only, cheap
    modal run deploy/modal_app.py::main --gpu t4 --models qwen2.5:7b --tasks fix_path_join --repeats 1 --small-num-ctx 8192  # smoke
    modal run deploy/modal_app.py::main --gpu t4 --models qwen2.5:7b --small-num-ctx 8192
    modal run deploy/modal_app.py::crewai --gpu t4 --model qwen2.5:7b --repeats 3      # CrewAI raw vs ContextGC
    modal run deploy/modal_app.py::sweep --gpus t4,l4 --models qwen2.5:7b,llama3.1:8b --overrides 0,8192,6144

Deploy-time knobs (environment of the `modal run` shell):
    CONTEXTGC_OLLAMA_VERSION   Ollama release baked into the image (default 0.40.13; empty = latest)
    CONTEXTGC_MODAL_SEMANTIC   1 (default) installs sentence-transformers so the semantic stage runs
    CONTEXTGC_MODAL_TIMEOUT_H  container timeout in hours (default 4) — the cost ceiling per run
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
REMOTE_BACKEND = "/root/backend"
OLLAMA_VERSION = os.getenv("CONTEXTGC_OLLAMA_VERSION", "0.40.13")
SEMANTIC = os.getenv("CONTEXTGC_MODAL_SEMANTIC", "1") == "1"
TIMEOUT_S = int(float(os.getenv("CONTEXTGC_MODAL_TIMEOUT_H", "4")) * 3600)

install_ollama = "curl -fsSL https://ollama.com/install.sh | " + (
    f"OLLAMA_VERSION={OLLAMA_VERSION} sh" if OLLAMA_VERSION else "sh")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "zstd", "pciutils")
    .run_commands(install_ollama)
    .pip_install_from_requirements(str(BACKEND / "requirements.txt"))
    .pip_install("pynvml")
)
if SEMANTIC:
    image = image.pip_install("sentence-transformers").run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer('all-MiniLM-L6-v2')\"")
# CrewAI A/B (crewai_agent/run_ab.py): LiteLLM routes ollama_chat to the native
# /api/chat, so the window is enforced. Versions are the ones it was tested with.
image = image.pip_install("crewai[litellm]==1.15.34", "litellm==1.105.3")
image = image.env({
    "OLLAMA_MODELS": "/models",
    "OLLAMA_KEEP_ALIVE": "30m",
    "OLLAMA_NUM_PARALLEL": "1",
    "PYTHONUNBUFFERED": "1",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
}).add_local_dir(
    str(BACKEND), REMOTE_BACKEND,
    ignore=["**/__pycache__", "**/.pytest_cache", "data", "benchmark/results", "*.log", "tmp_tel.json"],
).add_local_dir(
    str(REPO / "crewai_agent"), "/root/crewai_agent", ignore=["**/__pycache__", "results", ".venv", ".env"],
).add_local_dir(
    str(REPO / "demo_repository"), "/root/demo_repository", ignore=["**/__pycache__", "**/.pytest_cache"],
)

app = modal.App("contextgc-bench", image=image)
models_vol = modal.Volume.from_name("contextgc-models", create_if_missing=True)
results_vol = modal.Volume.from_name("contextgc-results", create_if_missing=True)
VOLUMES = {"/models": models_vol, "/results": results_vol}


def _stack(models: list[str], bench: list[str], env_extra: dict[str, str] | None = None) -> int:
    cmd = [sys.executable, "scripts/run_stack.py"]
    for model in models:
        cmd += ["--model", model]
    if bench:
        cmd += ["--"] + bench
    return subprocess.call(cmd, cwd=REMOTE_BACKEND, env={**os.environ, **(env_extra or {})})


def _bench(models: list[str], bench_args: list[str], tag: str, vram_override_mb: int,
           script: str = "benchmark/run_tasks.py") -> dict:
    out_dir = Path("/results") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"CONTEXTGC_RESULTS_DIR": str(out_dir)}
    if vram_override_mb:
        env["CONTEXTGC_VRAM_OVERRIDE_MB"] = str(vram_override_mb)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
    (out_dir / "gpu.txt").write_text(smi.stdout or "no GPU\n", encoding="utf-8")
    # a campaign tag "<campaign>/<task>" is a folder; the script's own tag is a file name
    code = _stack(models, [script, *bench_args, "--tag", tag.replace("/", "__")], env)
    models_vol.commit()   # keep any newly pulled weights
    results_vol.commit()
    files = {p.name: p.read_text(encoding="utf-8", errors="replace")
             for p in out_dir.iterdir() if p.is_file() and p.suffix in {".json", ".log", ".txt"}}
    return {"exit_code": code, "tag": tag, "files": files}


FN = dict(volumes=VOLUMES, timeout=TIMEOUT_S)


@app.function(cpu=2, memory=4096, timeout=3600, volumes=VOLUMES)
def pull_only(models: list[str]) -> list[str]:
    """Populate the weights Volume without paying for a GPU."""
    env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"}
    server = subprocess.Popen(["ollama", "serve"], env=env)
    try:
        for _ in range(60):
            if subprocess.run(["ollama", "list"], capture_output=True).returncode == 0:
                break
            time.sleep(1)
        for model in models:
            subprocess.run(["ollama", "pull", model], check=True)
        listing = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    finally:
        server.terminate()
    models_vol.commit()
    return listing.splitlines()


@app.function(gpu="T4", **FN)
def bench_t4(models: list[str], bench_args: list[str], tag: str, vram_override_mb: int = 0,
             script: str = "benchmark/run_tasks.py") -> dict:
    return _bench(models, bench_args, tag, vram_override_mb, script)


@app.function(gpu="L4", **FN)
def bench_l4(models: list[str], bench_args: list[str], tag: str, vram_override_mb: int = 0,
             script: str = "benchmark/run_tasks.py") -> dict:
    return _bench(models, bench_args, tag, vram_override_mb, script)


@app.function(gpu="A10G", **FN)
def bench_a10g(models: list[str], bench_args: list[str], tag: str, vram_override_mb: int = 0,
               script: str = "benchmark/run_tasks.py") -> dict:
    return _bench(models, bench_args, tag, vram_override_mb, script)


@app.function(cpu=8, memory=16384, **FN)
def bench_cpu(models: list[str], bench_args: list[str], tag: str, vram_override_mb: int = 0,
              script: str = "benchmark/run_tasks.py") -> dict:
    return _bench(models, bench_args, tag, vram_override_mb, script)


BENCH = {"t4": bench_t4, "l4": bench_l4, "a10g": bench_a10g, "cpu": bench_cpu}


def _split(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _bench_args(tasks: str, lanes: str, repeats: int, max_steps: int,
                small_num_ctx: str, a1_num_ctx: str, pad_tokens: str) -> list[str]:
    args = ["--lanes", lanes, "--repeats", str(repeats), "--max-steps", str(max_steps),
            "--small-num-ctx", small_num_ctx, "--a1-num-ctx", a1_num_ctx, "--pad-tokens", pad_tokens]
    return args + (["--tasks", tasks] if tasks else [])


def _save(result: dict) -> None:
    local = BACKEND / "benchmark" / "results" / "modal" / result["tag"]
    local.mkdir(parents=True, exist_ok=True)
    for name, text in result["files"].items():
        (local / name).write_text(text, encoding="utf-8")
    print(f"[modal] {result['tag']}: exit {result['exit_code']}, files in {local}")


@app.local_entrypoint()
def pull(models: str = "qwen2.5:3b"):
    for line in pull_only.remote(_split(models)):
        print(line)


@app.local_entrypoint()
def main(gpu: str = "t4", models: str = "qwen2.5:3b", tasks: str = "", lanes: str = "a1,a2,b",
         repeats: int = 3, max_steps: int = 12, vram_override_mb: int = 0,
         small_num_ctx: str = "auto", a1_num_ctx: str = "auto", pad_tokens: str = "auto",
         tag: str = ""):
    if gpu not in BENCH:
        raise SystemExit(f"--gpu must be one of {sorted(BENCH)}")
    model_list = _split(models)
    tag = tag or f"{time.strftime('%Y%m%d_%H%M%S')}_{gpu}_{model_list[0].replace(':', '-')}" + (
        f"_v{vram_override_mb}" if vram_override_mb else "")
    args = _bench_args(tasks, lanes, repeats, max_steps, small_num_ctx, a1_num_ctx, pad_tokens)
    _save(BENCH[gpu].remote(model_list, args, tag, vram_override_mb))


@app.local_entrypoint()
def crewai(gpu: str = "t4", model: str = "qwen2.5:7b", agents: str = "incident,code",
           lanes: str = "raw,contextgc", repeats: int = 3, num_ctx: int = 8192,
           log_lines: int = 2000, timeout: int = 900, tag: str = ""):
    """CrewAI A/B: the same crew against raw Ollama and through ContextGC, same window."""
    if gpu not in BENCH:
        raise SystemExit(f"--gpu must be one of {sorted(BENCH)}")
    tag = tag or f"{time.strftime('%Y%m%d_%H%M%S')}_crewai_{gpu}_{model.replace(':', '-')}"
    args = ["--agents", agents, "--lanes", lanes, "--repeats", str(repeats), "--num-ctx", str(num_ctx),
            "--log-lines", str(log_lines), "--timeout", str(timeout)]
    _save(BENCH[gpu].remote([model], args, tag, 0, "../crewai_agent/run_ab.py"))


@app.local_entrypoint()
def sweep(gpus: str = "t4,l4", models: str = "llama3.2:3b,llama3.1:8b", overrides: str = "0,8192,6144",
          tasks: str = "", lanes: str = "a1,a2,b", repeats: int = 3, max_steps: int = 12):
    """One container per (GPU, model, simulated VRAM) cell, all in parallel."""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    args = _bench_args(tasks, lanes, repeats, max_steps, "auto", "auto", "auto")
    calls = []
    for gpu in _split(gpus):
        for model in _split(models):
            for override in (int(x) for x in _split(overrides)):
                tag = f"{stamp}_sweep_{gpu}_{model.replace(':', '-')}" + (f"_v{override}" if override else "")
                calls.append(BENCH[gpu].spawn([model], args, tag, override))
    print(f"[modal] {len(calls)} containers launched")
    for call in calls:
        try:
            _save(call.get())
        except Exception as exc:
            print(f"[modal] a sweep cell failed: {type(exc).__name__}: {exc}")
