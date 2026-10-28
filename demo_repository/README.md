# OpsDesk Incident Service

OpsDesk is a small internal service used by a support team to classify incoming
customer incidents and route them to the correct engineering queue.

The repository contains:

- ticket normalization and priority calculation;
- team routing rules;
- a CLI-style incident summary module;
- configuration and production-like incident logs;
- pytest coverage for normal and edge-case incidents.

The current exercise is an intentionally introduced routing regression. A real
incident-response agent should inspect the logs, reproduce the failure, fix
source code only, and run the existing tests.

Run locally:

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
```
