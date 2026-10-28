# Real CrewAI evaluation

This is an isolated, real repository test rather than a synthetic prompt test.
Each run:

1. creates a temporary buggy Python repository;
2. starts a CrewAI investigator agent;
3. passes its diagnosis to a CrewAI implementer agent;
4. gives agents real `list_files`, `read_file`, `run_tests`, and source-writing tools;
5. prevents modification of test files;
6. runs pytest after the crew finishes;
7. writes a JSON pass/fail result.

Install the optional dependency:

```powershell
cd backend
python -m pip install crewai
```

Run one task through raw Ollama:

```powershell
python benchmark/crewai_eval/run_crewai.py `
  --lane raw --task fix_failing_sum
```

Run the same task through ContextGC:

```powershell
python benchmark/crewai_eval/run_crewai.py `
  --lane proxy --task fix_failing_sum
```

Run all eight tasks in each lane:

```powershell
python benchmark/crewai_eval/run_crewai.py --lane raw --all
python benchmark/crewai_eval/run_crewai.py --lane proxy --all
```

Compare `benchmark/results/crewai_raw.json` and
`benchmark/results/crewai_proxy.json` by task pass rate. Run each lane at least
three times with the same model and task order. Keep latency/context telemetry
from ContextGC separate from the deterministic pytest result.
