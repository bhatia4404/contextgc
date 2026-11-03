"""One LLM factory for both crews, configured from the environment.

CrewAI 1.15 sends ``ollama/`` and ``ollama_chat/`` models through its
OpenAI-compatible client, which calls Ollama's ``/v1`` endpoint, and Ollama
ignores ``num_ctx`` there. ``is_litellm=True`` routes through LiteLLM's
``ollama_chat`` provider instead, which calls the native ``/api/chat`` with
``options.num_ctx``, so the context window is really enforced.

Environment:
    AGENT_BASE_URL   Ollama (http://localhost:11434) or ContextGC (http://localhost:11400)
    AGENT_MODEL      default qwen2.5:7b
    AGENT_NUM_CTX    context window for every request, default 8192
    AGENT_RUN_ID     run id sent as x-contextgc-run (one per agent: "<id>-<role>")
"""
from __future__ import annotations

import os
import uuid


def settings() -> dict:
    base_url = os.getenv("AGENT_BASE_URL", "http://localhost:11400").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]          # native API, see module docstring
    return {
        "base_url": base_url,
        "model": os.getenv("AGENT_MODEL", "qwen2.5:7b"),
        "num_ctx": int(os.getenv("AGENT_NUM_CTX", "8192")),
        "run_id": os.getenv("AGENT_RUN_ID") or f"crewai-{uuid.uuid4().hex[:8]}",
    }


def make_llm(role: str):
    from crewai import LLM

    s = settings()
    return LLM(
        model=f"ollama_chat/{s['model']}", base_url=s["base_url"], is_litellm=True,
        temperature=0, seed=7, max_tokens=1024, num_ctx=s["num_ctx"],
        extra_headers={"x-contextgc-run": f"{s['run_id']}-{role}"},
    )


def resolve_in_repo(repo, path: str):
    """A tool path inside the repository, however the model wrote it.

    Models pass '/data/x.log', '/tmp/.../repo/src/a.py' or 'src/a.py' for the
    same file; all map into the repository. The result is still checked by the
    caller to stay inside it.
    """
    from pathlib import Path

    repo = Path(repo).resolve()
    raw = str(path).strip().strip("'\"").replace("\\", "/")
    root = repo.as_posix()
    if raw.startswith(root):
        raw = raw[len(root):]
    return (repo / raw.lstrip("/")).resolve()


def usage(crew, output) -> dict:
    """Token usage CrewAI recorded for the run (field names vary by version)."""
    metrics = getattr(output, "token_usage", None) or getattr(crew, "usage_metrics", None)
    if metrics is None:
        return {}
    if hasattr(metrics, "model_dump"):
        metrics = metrics.model_dump()
    return {k: metrics.get(k) for k in ("total_tokens", "prompt_tokens", "completion_tokens",
                                        "successful_requests") if isinstance(metrics, dict)}


def write_result(result: dict, default_name: str, results_dir) -> None:
    import json
    from pathlib import Path

    target = Path(os.getenv("AGENT_RESULT_FILE") or Path(results_dir) / default_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")


def hard_exit(code: int = 0) -> None:
    """CrewAI can leave a non-daemon thread running after kickoff, which keeps
    the process alive forever; the result is already written, so leave now."""
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
