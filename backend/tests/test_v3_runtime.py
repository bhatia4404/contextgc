"""Regression tests for the v3 fixed-window and incremental runtime."""
from __future__ import annotations

import json

from app.core.gc import process_payload
from app.core import run_state


def test_append_only_view_keeps_cleaned_prefix(settings_factory, tmp_path):
    settings_factory(
        gc_enabled=True,
        numctx_default=8192,
        offload_enabled=True,
        scratchpad_dir=str(tmp_path / "scratch"),
    )
    first = {"model": "m", "messages": [
        {"role": "user", "content": "remember the goal: preserve this task state"},
        {"role": "tool", "content": "short output"},
    ]}
    second = {"model": "m", "messages": first["messages"] + [
        {"role": "user", "content": "continue with the next step"},
    ]}
    out1, r1 = process_payload("/api/chat", json.dumps(first).encode(), run_id="append")
    out2, r2 = process_payload("/api/chat", json.dumps(second).encode(), run_id="append")
    assert r1["incremental_view"] is False
    assert r2["incremental_view"] is True
    assert r2["compaction"] is False
    messages = json.loads(out2)["messages"]
    assert any(m.get("content") == "preserve this task state" or
               "preserve this task state" in str(m.get("content")) for m in messages)
    assert messages[-1]["content"] == "continue with the next step"
    run_state.reset()


def test_capacity_override_is_reported(monkeypatch, settings_factory):
    settings_factory(gc_enabled=True, sink="http://mock-sink")
    monkeypatch.setenv("CONTEXTGC_VRAM_OVERRIDE_MB", "4096")
    from app.core import vram
    event = vram.sample()
    assert event["vram_total_mb"] == 4096
    assert event["override"] is True


def test_dedup_keeps_newest_copy(settings_factory):
    settings_factory(gc_enabled=True, state_enabled=False)
    repeated = "Check the failing tests again and propose the next fix."
    payload = {"model": "m", "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": repeated},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": repeated},
        {"role": "assistant", "content": "second reply"},
        {"role": "user", "content": "latest instruction"},
    ]}
    out, report = process_payload("/api/chat", json.dumps(payload).encode(), run_id="newest")
    assert [r["index"] for r in report["stage1_dedup"]["removed"]] == [1]
    contents = [m["content"] for m in json.loads(out)["messages"]]
    assert contents.index(repeated) == contents.index("first reply") + 1


def test_capacity_reads_weight_size_from_tags(monkeypatch, settings_factory):
    settings_factory(gc_enabled=True, sink="ollama")
    monkeypatch.setenv("CONTEXTGC_VRAM_OVERRIDE_MB", "4096")
    from app.core import capacity
    info = {"llama.block_count": 28, "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128}

    class _Resp:
        def __init__(self, data):
            self.status_code, self._data = 200, data

        def json(self):
            return self._data

    class _Client:
        closed = False

        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            _Client.closed = True

        def _check(self):
            assert not _Client.closed, "request made on a closed client"

        def post(self, url, json=None):
            self._check()
            return _Resp({"model_info": info})

        def get(self, url):
            self._check()
            return _Resp({"models": [{"name": "llama3.2:3b", "size": 2_019_393_189}]})

    monkeypatch.setattr(capacity.httpx, "Client", _Client)
    window, details = capacity.choose_window("llama3.2:3b", 8192, 2048, 32768)
    assert details["weight_bytes"] == 2_019_393_189
    assert details["kv_bytes_per_token"] == 114_688
    # (4096 MB - 1926 MB weights - 256 MB) * 0.85 / 112 KiB per token ~= 15k tokens
    assert window == 8192


def test_lower_window_at_most_once_per_run():
    from app.core import num_ctx
    num_ctx._RUN_WINDOWS["once"] = 16384
    assert num_ctx.lower_window("once") == 8192
    assert num_ctx.lower_window("once") == 8192
    assert num_ctx.pinned_window("once") == 8192


def test_compaction_keeps_recent_turns_with_big_live_dump(settings_factory, tmp_path):
    """The live raw dump must not consume the compaction allowance."""
    settings_factory(gc_enabled=True, numctx_default=8192, state_enabled=False,
                     scratchpad_dir=str(tmp_path / "scratch"))
    history = [{"role": "system", "content": "s"}]
    compacted = None
    for t in range(40):
        history.append({"role": "user", "content": f"instruction {t}: work on module_{t}.py now"})
        if t % 3 == 2:
            history.append({"role": "tool", "name": "bash", "tool_call_id": f"tc{t}",
                            "content": f"[pytest turn {t}]\n" + "E assert mismatch\n" * 400})
        out, report = process_payload("/api/chat", json.dumps({"model": "m", "messages": history}).encode(),
                                      run_id="compact")
        if report.get("compaction"):
            compacted = json.loads(out)["messages"]
            break
        history.append({"role": "assistant", "content": f"reply {t}: done with module_{t}.py"})
    assert compacted is not None, "scenario must reach a compaction epoch"
    users = [m["content"] for m in compacted if m["role"] == "user"]
    assert len(users) >= 3, f"recent instructions lost in compaction: {users}"
    # oldest-first: survivors are the most recent instructions, in order
    idx = [int(u.split()[1].rstrip(":")) for u in users]
    assert idx == sorted(idx) and idx[-1] == max(idx)
