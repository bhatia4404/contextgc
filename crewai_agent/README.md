# ContextGC CrewAI Agent Demo

This is a separate, realistic CrewAI application. It is intentionally not
inside `backend/`: the backend is the reusable middleware and this folder is
the customer agent project using it. The target is the sibling
`../demo_repository`, just like a real agent operates on an existing repo.

Two crews, each with two agents and real tools (list, read, search, run pytest,
write source files; tests are read-only):

| Crew | File | Job |
|---|---|---|
| Incident response | `incident_response_agent.py` | investigate incident INC-1042 from logs, code and tests, then fix the routing regression |
| Code maintenance | `code_maintenance_agent.py` | diagnose the failing test and make the smallest source fix |

A run passes only when the repository's original pytest suite passes and the
test files are unchanged.

## How the crews reach the model

Only the base URL changes between the two lanes:

```text
raw Ollama:  http://localhost:11434
ContextGC:   http://localhost:11400
```

Both crews build their LLM in `llm_config.py`: LiteLLM's `ollama_chat` provider,
which calls Ollama's native `/api/chat` with `options.num_ctx`. CrewAI's default
Ollama client calls `/v1`, where Ollama ignores `num_ctx`, so the context window
would never be enforced and there would be nothing for ContextGC to do. Each
agent also sends its own `x-contextgc-run` header.

| Variable | Default | Meaning |
|---|---|---|
| `AGENT_BASE_URL` | `http://localhost:11400` | raw Ollama or the ContextGC proxy |
| `AGENT_MODEL` | `qwen2.5:7b` | Ollama model (`llama3.2:3b` cannot do these tasks) |
| `AGENT_NUM_CTX` | `8192` | context window for every request |
| `AGENT_REPOSITORY` | `../demo_repository` | repository the crew works on |

## Setup

```powershell
cd crewai_agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt     # includes crewai[litellm]
```

Make sure Ollama is running and the model is pulled (`ollama pull qwen2.5:7b`).
For the ContextGC lane, start the proxy with GC on:

```powershell
cd ..\backend
$env:CONTEXTGC_GC_ENABLED="true"
python -m uvicorn app.main:app --host 127.0.0.1 --port 11400
```

## The A/B test (recommended)

`run_ab.py` runs each crew against both lanes with the same model and window, on a
fresh copy of the demo repository per run, and records pass/fail, wall time,
prompt tokens and ContextGC's raw-vs-clean telemetry:

```powershell
python run_ab.py --agents incident,code --repeats 3
```

The demo repository is tiny, so each run's copy gets a deterministic,
production-sized checkout log (`--log-lines`, default 2000 lines, ~200 KB). A
single search for `provider_timeout` then returns several thousand tokens, which
is what puts an 8k window under pressure, as a real incident log would.
Results go to `results/crew_ab_<tag>.json`, with one agent log per run.

On a GPU this is one command from the repo root (see the main README):

```powershell
modal run deploy/modal_app.py::crewai --gpu t4 --model qwen2.5:7b --repeats 3
```

## Running one crew by hand

```powershell
$env:AGENT_BASE_URL="http://localhost:11400"     # or http://localhost:11434 for raw Ollama
python incident_response_agent.py
python code_maintenance_agent.py
```

The result is printed as JSON and written to `results/`. A manual run edits
`../demo_repository` in place; reset it afterwards with
`git checkout -- ../demo_repository`.
