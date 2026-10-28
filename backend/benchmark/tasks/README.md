# ContextGC deterministic task suite

The suite contains eight small coding/debugging tasks in
`backend/benchmark/task_suite.py`.

Each case includes:

- a user task prompt;
- a small repository fixture;
- expected tool work (`grep`, file reads, and `pytest`);
- a deterministic `pytest` checker.

## How to use a case

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from benchmark.task_suite import TASKS

case = TASKS[0]
with TemporaryDirectory() as directory:
    root = Path(directory)
    case.materialize(root)
    # Point the agent's working directory at root and send case.prompt.
    passed, output = case.run_checker(root)
    print(case.name, passed, output)
```

Verify that every starter fixture executes and initially fails with:

```powershell
cd backend
python benchmark/task_suite.py
```

For the actual A1/A2/B experiment, run each case three times per lane and
record checker pass/fail separately from latency and context metrics. Do not
let the agent read or modify the checker files except through normal tool use.
