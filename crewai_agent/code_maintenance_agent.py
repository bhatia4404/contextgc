"""Code-maintenance CrewAI agent operating on the external demo repository."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.getenv("AGENT_REPOSITORY", str(ROOT.parent / "demo_repository"))).resolve()


def build_tools():
    try:
        from crewai.tools import BaseTool
        from llm_config import resolve_in_repo
    except ImportError as exc:
        raise RuntimeError("CrewAI is not installed. Run: pip install -r requirements.txt") from exc

    class ListFiles(BaseTool):
        name: str = "list_files"
        description: str = "List files in the agent workspace."

        def _run(self) -> str:
            return "\n".join(str(path.relative_to(WORKSPACE))
                               for path in sorted(WORKSPACE.rglob("*")) if path.is_file())

    class ReadFile(BaseTool):
        name: str = "read_file"
        description: str = "Read a relative source or test file from the workspace."

        def _run(self, path: str) -> str:
            target = resolve_in_repo(WORKSPACE, path)
            if WORKSPACE.resolve() not in target.parents:
                return "ERROR: path outside workspace"
            try:
                return target.read_text(encoding="utf-8")
            except OSError as exc:
                return f"ERROR: {exc}"

    class RunPytest(BaseTool):
        name: str = "run_pytest"
        description: str = "Run pytest in the workspace and return its output."

        def _run(self) -> str:
            result = subprocess.run(
                ["python", "-m", "pytest", "-q"], cwd=WORKSPACE,
                capture_output=True, text=True, timeout=30,
            )
            return f"exit_code={result.returncode}\n{result.stdout}\n{result.stderr}"[-10000:]

    class WriteSource(BaseTool):
        name: str = "write_source"
        description: str = "Write a source file. Test files are read-only."

        def _run(self, path: str, content: str) -> str:
            target = resolve_in_repo(WORKSPACE, path)
            if WORKSPACE.resolve() not in target.parents:
                return "ERROR: path outside workspace"
            if target.name.startswith("test_"):
                return "ERROR: tests are read-only"
            target.write_text(content, encoding="utf-8")
            return f"wrote {path}"

    return [ListFiles(), ReadFile(), RunPytest(), WriteSource()]


def run() -> dict:
    try:
        from crewai import Agent, Crew, Process, Task
    except ImportError as exc:
        raise RuntimeError("CrewAI is not installed. Run: pip install -r requirements.txt") from exc

    from llm_config import make_llm, settings, usage

    s = settings()
    tools = build_tools()

    investigator = Agent(
        role="Senior repository investigator",
        goal="Diagnose the failing repository using evidence from files and pytest.",
        backstory="You inspect carefully and never edit source files.",
        tools=tools[:3], llm=make_llm("investigator"), verbose=True, max_iter=15,
    )
    maintainer = Agent(
        role="Test-driven Python maintainer",
        goal="Make the smallest correct source-only fix and prove it with pytest.",
        backstory="You never change tests and always run the final checker.",
        tools=tools, llm=make_llm("maintainer"), verbose=True, max_iter=15,
    )
    diagnosis = Task(
        description=(
            f"The isolated repository is at {WORKSPACE}. List files, read the source "
            "and tests, run pytest, and explain the exact bug. Do not modify files."
        ),
        expected_output="A concise diagnosis with failing test evidence and a proposed fix.",
        agent=investigator,
    )
    implementation = Task(
        description=(
            f"Fix the repository at {WORKSPACE}. Modify source files only. The goal is "
            "for all original pytest tests to pass. Run pytest after editing and report it."
        ),
        expected_output="Changed source files and a passing pytest result.",
        agent=maintainer,
        context=[diagnosis],
    )
    crew = Crew(agents=[investigator, maintainer], tasks=[diagnosis, implementation],
                process=Process.sequential, verbose=True)
    output = crew.kickoff()
    check = subprocess.run(["python", "-m", "pytest", "-q"], cwd=WORKSPACE,
                           capture_output=True, text=True, timeout=30)
    result = {
        "base_url": s["base_url"],
        "model": s["model"],
        "num_ctx": s["num_ctx"],
        "run_id": s["run_id"],
        "workspace": str(WORKSPACE),
        "passed": check.returncode == 0,
        "usage": usage(crew, output),
        "pytest": (check.stdout + check.stderr)[-4000:],
        "crew_output": str(output)[-4000:],
    }
    from llm_config import write_result
    write_result(result, "last_run.json", ROOT / "results")
    return result


def main() -> None:
    argparse.ArgumentParser(
        description="Run a maintenance crew against the external demo repository"
    ).parse_args()
    if not WORKSPACE.exists():
        raise SystemExit(f"Repository does not exist: {WORKSPACE}")
    print(json.dumps(run(), indent=2))
    from llm_config import hard_exit
    hard_exit()


if __name__ == "__main__":
    main()
