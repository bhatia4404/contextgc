# ContextGC CrewAI Agent Demo

This is a separate, realistic CrewAI application. It is intentionally not
inside `backend/`: the backend is the reusable middleware and this folder is
the customer agent project using it. The target is the sibling
`../demo_repository`, just like a real agent operates on an existing repo.

The agent fixes a failing Python repository using real tools:

- list files;
- read source and tests;
- run pytest;
- write source files;
- run the final deterministic checker.

Only the LLM endpoint changes between lanes:

```text
raw Ollama:  http://localhost:11434/v1
ContextGC:   http://localhost:11400/v1
```

## Setup

```powershell
cd crewai_agent
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Make sure Ollama is running. Start ContextGC separately for the proxy lane:

```powershell
cd ..\backend
$env:CONTEXTGC_GC_ENABLED="true"
python -m uvicorn app.main:app --host 127.0.0.1 --port 11400
```

## Agent 1: code-maintenance agent

Run the raw control lane:

```powershell
cd crewai_agent
$env:AGENT_BASE_URL="http://localhost:11434/v1"
python code_maintenance_agent.py
```

## Run through ContextGC

```powershell
cd crewai_agent
$env:AGENT_BASE_URL="http://localhost:11400/v1"
python code_maintenance_agent.py
```

The final result is printed as JSON and written to `results/last_run.json`.

The project deliberately does not modify tests through its write tool. A run
passes only when the agent changes the source and the original pytest suite
passes afterward.

## Agent 2: incident-response use case

The same project includes a more realistic production workflow in
the sibling `../demo_repository/`: an incident agent correlates checkout logs, routing code,
configuration, and tests, then fixes the confirmed routing regression.

Run it through ContextGC:

```powershell
$env:AGENT_BASE_URL="http://localhost:11400/v1"
python incident_response_agent.py
```

Run the control lane against Ollama directly:

```powershell
$env:AGENT_BASE_URL="http://localhost:11434/v1"
python incident_response_agent.py
```

This use case is where ContextGC is useful in practice: an incident agent can
accumulate repeated log excerpts, source reads, test output, and handoff
messages while investigating a long-running production issue.
