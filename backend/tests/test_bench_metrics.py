"""Benchmark metric helpers: cache-hit estimate and tool-call parsing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))

from run_simulation import cache_stats  # noqa: E402
from run_tasks import parse_calls  # noqa: E402


def test_cache_stats_uses_prefill_time_not_reported_count():
    # Ollama reports the full prompt even when the prefix cache served it.
    rows = [
        {"prompt_eval_count": 2000, "prompt_eval_duration": 60_000_000_000},  # cold: 30 ms/token
        {"prompt_eval_count": 2040, "prompt_eval_duration": 1_200_000_000},   # ~40 new tokens
    ]
    stats = cache_stats(rows)
    assert stats["prefilled_tokens_est"] == 2040
    assert stats["cache_hit_ratio"] == round(1 - 2040 / 4040, 4)


def test_cache_stats_without_a_cold_reference():
    assert cache_stats([{"prompt_eval_count": 40, "prompt_eval_duration": 10}])["cache_hit_ratio"] is None


def test_parse_native_tool_calls():
    msg = {"tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]}
    assert parse_calls(msg) == [("read_file", {"path": "a.py"})]


def test_parse_json_in_content_tool_call():
    msg = {"content": 'Sure. {"name": "run_pytest", "parameters": {}}'}
    assert parse_calls(msg) == [("run_pytest", {})]


def test_plain_text_is_not_a_tool_call():
    assert parse_calls({"content": "DONE"}) == []


def test_cache_stats_prefers_ollama_cached_count():
    rows = [{"prompt_eval_count": 100, "prompt_eval_cached_count": 0, "prompt_eval_duration": 9},
            {"prompt_eval_count": 120, "prompt_eval_cached_count": 100, "prompt_eval_duration": 1}]
    stats = cache_stats(rows)
    assert stats["cache_source"] == "ollama"
    assert stats["prefilled_tokens_est"] == 120
