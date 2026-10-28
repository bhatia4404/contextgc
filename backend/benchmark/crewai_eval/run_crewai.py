"""Run a real CrewAI coding crew against raw Ollama or ContextGC.

CrewAI is optional in the main project requirements. Install it separately:

    python -m pip install crewai

Usage:
    python benchmark/crewai_eval/run_crewai.py --lane proxy --task fix_failing_sum
    python benchmark/crewai_eval/run_crewai.py --lane raw --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.task_suite import TASKS, TaskCase


def _tools(root: Path):
    try:
        from crewai.tools import BaseTool
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Install CrewAI with: python -m pip install crewai") from exc

    class ListFilesTool(BaseTool):
        name: str = "list_files"
        description: str = "List source and test files in the isolated workspace."

        def _run(self) -> str:
            return "\n".join(str(p.relative_to(root)) for p in sorted(root.rglob("*")) if p.is_file())

    class ReadFileTool(BaseTool):
        name: str = "read_file"
        description: str = "Read a text file. Input must be a relative path in the workspace."

        def _run(self, path: str) -> str:
            target = (root / path).resolve()
            if root.resolve() not in target.parents:
                return "ERROR: path outside workspace"
            try:
                return target.read_text(encoding="utf-8")
            except OSError as exc:
                return f"ERROR: {exc}"

    class RunTestsTool(BaseTool):
        name: str = "run_tests"
        description: str = "Run pytest in the isolated workspace and return stdout/stderr."

        def _run(self) -> str:
            import subprocess
            result = subprocess.run(
                ["python", "-m", "pytest", "-q"], cwd=root,
                capture_output=True, text=True, timeout=30,
            )
            return f"exit_code={result.returncode}\n{result.stdout}\n{result.stderr}"[-12000:]

    class WriteFileTool(BaseTool):
        name: str = "write_source_file"
        description: str = "Write a source file. Test files are read-only and cannot be changed."

        def _run(self, path: str, content: str) -> str:
            target = (root / path).resolve()
            if root.resolve() not in target.parents:
                return "ERROR: path outside workspace"
            if target.name.startswith("test_") or target.name.endswith("_test.py"):
                return "ERROR: tests are read-only"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"wrote {target.relative_to(root)}"

    return [ListFilesTool(), ReadFileTool(), RunTestsTool(), WriteFileTool()]


def run_case(case: TaskCase, lane: str, model: str, base_url: str) -> dict:
    try:
        from crewai import Agent, Crew, LLM, Process, Task
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install CrewAI with: python -m pip install crewai") from exc

    with tempfile.TemporaryDirectory(prefix=f"contextgc-crewai-{case.name}-") as directory:
        root = Path(directory)
        case.materialize(root)
        tools = _tools(root)
        llm = LLM(
            model=f"ollama/{model}",
            base_url=base_url,
            api_key="contextgc",
            temperature=0,
        )
        analyst = Agent(
            role="Repository investigator",
            goal="Understand the failing repository and give precise evidence to the implementer.",
            backstory="You inspect files and run tests. You never modify files.",
            tools=tools[:3], llm=llm, verbose=False,
        )
        implementer = Agent(
            role="Test-driven maintainer",
            goal="Fix the source code without changing tests, then prove the fix with pytest.",
            backstory="You make the smallest correct source change and verify it.",
            tools=tools, llm=llm, verbose=False,
        )
        investigate = Task(
            description=(
                f"Workspace: {root}\n{case.prompt}\n"
                "First list files, read the relevant source and tests, run pytest, "
                "and report the failure and likely fix. Do not edit files."
            ),
            expected_output="A concise diagnosis with failing test output and proposed source change.",
            agent=analyst,
        )
        fix = Task(
            description=(
                f"Workspace: {root}\nImplement the fix for this task: {case.prompt}\n"
                "Use the diagnosis, modify source files only, run pytest, and iterate until it passes."
            ),
            expected_output="A summary of source changes and the final pytest result.",
            agent=implementer,
            context=[investigate],
        )
        crew = Crew(agents=[analyst, implementer], tasks=[investigate, fix],
                    process=Process.sequential, verbose=False)
        result = crew.kickoff()
        passed, checker_output = case.run_checker(root)
        return {
            "task": case.name, "lane": lane, "passed": passed,
            "checker_output": checker_output,
            "crew_output": str(result)[-4000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("raw", "proxy"), required=True)
    parser.add_argument("--task", choices=[task.name for task in TASKS])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--raw-url", default="http://localhost:11434/v1")
    parser.add_argument("--proxy-url", default="http://localhost:11400/v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.all and not args.task:
        parser.error("provide --task or --all")
    cases = TASKS if args.all else [next(task for task in TASKS if task.name == args.task)]
    base_url = args.raw_url if args.lane == "raw" else args.proxy_url
    results = [run_case(case, args.lane, args.model, base_url) for case in cases]
    output = args.output or Path(f"benchmark/results/crewai_{args.lane}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"lane": args.lane, "passed": sum(r["passed"] for r in results),
                      "total": len(results), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
