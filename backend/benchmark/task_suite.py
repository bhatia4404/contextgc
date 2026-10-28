"""Deterministic task definitions for the A1/A2/B quality benchmark.

Each task is intentionally small but tool-heavy: the agent must inspect files,
run a command, make a change, and pass a deterministic checker. The runner can
materialize ``files`` into an isolated workspace before starting an agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class TaskCase:
    name: str
    prompt: str
    files: dict[str, str]
    check: str

    def materialize(self, root: Path) -> None:
        for relative, content in self.files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def run_checker(self, root: Path) -> tuple[bool, str]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0, (result.stdout + result.stderr)[-4000:]


TASKS = [
    TaskCase(
        "fix_failing_sum",
        "Inspect the repository, run pytest, and fix the failing sum function. Do not change the tests.",
        {
            "math_utils.py": "def total(values):\n    return values[0]\n",
            "test_math_utils.py": "from math_utils import total\n\ndef test_total():\n    assert total([2, 3, 5]) == 10\n",
        },
        "pytest",
    ),
    TaskCase(
        "repair_json_reader",
        "Run the tests and fix the JSON reader so a missing optional field uses an empty list.",
        {
            "reader.py": "import json\n\ndef names(text):\n    return json.loads(text)['names']\n",
            "test_reader.py": "from reader import names\n\ndef test_missing_names():\n    assert names('{}') == []\n\ndef test_names():\n    assert names('{\"names\": [\"a\"]}') == ['a']\n",
        },
        "pytest",
    ),
    TaskCase(
        "fix_path_join",
        "Find and fix the path builder. It must work when the directory ends with a slash.",
        {
            "paths.py": "def config_path(directory):\n    return directory + '/config.json'\n",
            "test_paths.py": "from paths import config_path\n\ndef test_no_double_slash():\n    assert config_path('/tmp/app/') == '/tmp/app/config.json'\n",
        },
        "pytest",
    ),
    TaskCase(
        "preserve_false_values",
        "Fix configuration lookup so false, zero, and empty-string values are preserved.",
        {
            "config.py": "def get_value(values, key, default):\n    return values.get(key) or default\n",
            "test_config.py": "from config import get_value\n\ndef test_false(): assert get_value({'x': False}, 'x', True) is False\n\ndef test_zero(): assert get_value({'x': 0}, 'x', 9) == 0\n",
        },
        "pytest",
    ),
    TaskCase(
        "fix_retry_counter",
        "Run pytest and fix the retry counter so it resets after a successful operation.",
        {
            "retry.py": "class Retry:\n    def __init__(self): self.count = 0\n    def record(self, success):\n        self.count += 1\n        return self.count\n",
            "test_retry.py": "from retry import Retry\n\ndef test_success_resets():\n    r = Retry(); r.record(False); assert r.record(True) == 0\n",
        },
        "pytest",
    ),
    TaskCase(
        "implement_stable_sort",
        "Implement sort_records by timestamp, preserving original order for equal timestamps.",
        {
            "records.py": "def sort_records(records):\n    raise NotImplementedError\n",
            "test_records.py": "from records import sort_records\n\ndef test_stable():\n    rows = [{'id': 'a', 'ts': 2}, {'id': 'b', 'ts': 1}, {'id': 'c', 'ts': 2}]\n    assert [r['id'] for r in sort_records(rows)] == ['b', 'a', 'c']\n",
        },
        "pytest",
    ),
    TaskCase(
        "fix_cache_key",
        "Fix the cache so values from different namespaces do not collide.",
        {
            "cache.py": "class Cache:\n    def __init__(self): self.data = {}\n    def put(self, ns, key, value): self.data[key] = value\n    def get(self, ns, key): return self.data.get(key)\n",
            "test_cache.py": "from cache import Cache\n\ndef test_namespaces():\n    c = Cache(); c.put('a', 'x', 1); c.put('b', 'x', 2)\n    assert c.get('a', 'x') == 1 and c.get('b', 'x') == 2\n",
        },
        "pytest",
    ),
    TaskCase(
        "fix_csv_parser",
        "Fix CSV parsing so commas inside quoted fields remain part of the field.",
        {
            "csv_reader.py": "def parse(line):\n    return line.split(',')\n",
            "test_csv_reader.py": "from csv_reader import parse\n\ndef test_quoted_comma():\n    assert parse('a,\"b,c\",d') == ['a', 'b,c', 'd']\n",
        },
        "pytest",
    ),
]


def task_names() -> list[str]:
    return [task.name for task in TASKS]


if __name__ == "__main__":  # quick fixture sanity check
    for task in TASKS:
        with tempfile.TemporaryDirectory(prefix=f"contextgc-{task.name}-") as directory:
            root = Path(directory)
            task.materialize(root)
            ok, output = task.run_checker(root)
            # Every starter fixture is intentionally failing; the agent must
            # make it pass. This checks that the fixture and checker execute.
            print(f"{task.name}: ready (initial checker {'unexpectedly passed' if ok else 'fails as expected'})")
