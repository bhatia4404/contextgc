"""CrewAI incident-response agent for the OpsDesk demo repository.

This is a separate real-life use case for ContextGC: an incident agent may
repeatedly inspect source, tests, configuration, and large logs while handing
diagnosis from one specialist to another.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
REPOSITORY = Path(os.getenv("AGENT_REPOSITORY", str(ROOT.parent / "demo_repository"))).resolve()


def build_tools():
    from crewai.tools import BaseTool

    class ListFiles(BaseTool):
        name: str = "list_repository_files"
        description: str = "List files in the OpsDesk repository."

        def _run(self) -> str:
            return "\n".join(str(path.relative_to(REPOSITORY))
                               for path in sorted(REPOSITORY.rglob("*")) if path.is_file())

    class ReadFile(BaseTool):
        name: str = "read_repository_file"
        description: str = "Read a source, test, configuration, or incident log file."

        def _run(self, path: str) -> str:
            target = (REPOSITORY / path).resolve()
            if REPOSITORY.resolve() not in target.parents:
                return "ERROR: path outside repository"
            try:
                return target.read_text(encoding="utf-8")[-16000:]
            except OSError as exc:
                return f"ERROR: {exc}"

    class SearchRepository(BaseTool):
        name: str = "search_repository"
        description: str = "Search source and logs for an incident ID, error, or queue name."

        def _run(self, query: str) -> str:
            matches = []
            for path in REPOSITORY.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if query.lower() in line.lower():
                            matches.append(f"{path.relative_to(REPOSITORY)}:{number}: {line}")
                except UnicodeDecodeError:
                    continue
            return "\n".join(matches[-200:]) or "no matches"

    class RunTests(BaseTool):
        name: str = "run_repository_tests"
        description: str = "Run the OpsDesk pytest suite."

        def _run(self) -> str:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPOSITORY / "src")
            result = subprocess.run(["python", "-m", "pytest", "-q"], cwd=REPOSITORY,
                                    env=env, capture_output=True, text=True, timeout=30)
            return f"exit_code={result.returncode}\n{result.stdout}\n{result.stderr}"[-12000:]

    class WriteSource(BaseTool):
        name: str = "write_source_file"
        description: str = "Write a source file only. Tests are read-only."

        def _run(self, path: str, content: str) -> str:
            target = (REPOSITORY / path).resolve()
            if REPOSITORY.resolve() not in target.parents:
                return "ERROR: path outside repository"
            if "test" in target.name.lower():
                return "ERROR: tests are read-only"
            target.write_text(content, encoding="utf-8")
            return f"wrote {path}"

    return [ListFiles(), ReadFile(), SearchRepository(), RunTests(), WriteSource()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpsDesk incident-response crew")
    parser.add_argument("--base-url", default=os.getenv("AGENT_BASE_URL", "http://localhost:11400/v1"))
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL", "llama3.2:3b"))
    args = parser.parse_args()

    try:
        from crewai import Agent, Crew, LLM, Process, Task
    except ImportError as exc:
        raise SystemExit("Install CrewAI first: pip install -r requirements.txt") from exc

    llm = LLM(model=f"ollama/{args.model}", base_url=args.base_url,
              api_key="contextgc-incident-demo", temperature=0)
    tools = build_tools()
    investigator = Agent(
        role="Production incident investigator",
        goal="Correlate logs, source, configuration, and tests to explain the incident.",
        backstory="You are an on-call engineer. Preserve evidence and do not edit code.",
        tools=tools[:4], llm=llm, verbose=True,
    )
    responder = Agent(
        role="Incident remediation engineer",
        goal="Fix the confirmed source regression, run tests, and produce a concise incident report.",
        backstory="You make minimal source-only changes and never alter tests or historical logs.",
        tools=tools, llm=llm, verbose=True,
    )
    investigate = Task(
        description=(
            f"Inspect {REPOSITORY}. Investigate incident INC-1042 using the checkout log, "
            "routing code, configuration, and pytest. Search for the incident ID and "
            "provider_timeout. Do not modify files. Explain customer impact and root cause."
        ),
        expected_output="Evidence-backed incident diagnosis with affected queue and root cause.",
        agent=investigator,
    )
    remediate = Task(
        description=(
            f"Using the investigator's evidence, fix the source regression in {REPOSITORY}. "
            "Do not change tests or logs. Run the full pytest suite and report the exact "
            "source change, final test result, and remaining operational risk."
        ),
        expected_output="Root cause, source fix, pytest result, and incident handoff summary.",
        agent=responder,
        context=[investigate],
    )
    result = Crew(agents=[investigator, responder], tasks=[investigate, remediate],
                  process=Process.sequential, verbose=True).kickoff()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY / "src")
    check = subprocess.run(["python", "-m", "pytest", "-q"], cwd=REPOSITORY,
                           env=env, capture_output=True, text=True)
    output = {
        "use_case": "production_incident_response",
        "base_url": args.base_url,
        "model": args.model,
        "passed": check.returncode == 0,
        "pytest": (check.stdout + check.stderr)[-4000:],
        "crew_output": str(result)[-6000:],
    }
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "incident_last_run.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
