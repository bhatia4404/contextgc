"""Per-turn story of the tool-offload + re-inject loop, read from the store.

Maps every turn of the demo runs to what RAW came in, what CLEAN went to the
LLM (raw / pointered / restored), and which GC stages acted. This is the
"explain from the DB" companion to Milestone 1.5/1.5b.

Usage: python scripts\explain_offload_turns.py [run_id]
"""
import json
import sqlite3
import sys


def state_of(content: str) -> str:
    if content.lstrip().startswith("[Tool Output Ref #"):
        if "FULL OUTPUT RESTORED" in content:
            return "RESTORED (full dump back this turn)"
        return "POINTERED"
    return "RAW"


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "coding-agent-final"
    conn = sqlite3.connect("data/contextgc.db")
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        "SELECT turn_id, path, raw_body, clean_body, gc_report FROM turns"
        " WHERE run_id = ? ORDER BY turn_id", (run_id,)))
    print(f"run: {run_id} | turns: {len(rows)}\n")
    for row in rows:
        raw = json.loads(bytes(row[2])); clean = json.loads(bytes(row[3]))
        rep = json.loads(row[4])
        tool_msgs = [m for m in raw["messages"] if m.get("role") == "tool"]
        clean_state = state_of(clean["messages"][1]["content"] if len(clean["messages"]) > 1 else "")
        print(f"== TURN {row[0]} ({row[1]}) - {human_bytes(len(row[2]), len(row[3]))}")
        for i, m in enumerate(clean["messages"]):
            c = m.get("content") or ""
            if m.get("role") == "tool":
                print(f"   tool msg [{i}]: {state_of(c)}")
            elif "[Tool Output Ref #" in c or "ref " in c.lower():
                print(f"   [{m['role']} msg {i}] mentions a ref: {c[:60]!r}")
        t = rep.get("tool_offload", {})
        if t.get("offloads"):
            print("   offload:", [(f"Ref #{o['ref']}", f"idx {o['index']}",
                                     f"{o['tokens_before']} tok -> {o['tokens_after']} tok") for o in t["offloads"]])
        ri = rep.get("ref_reinject")
        if ri:
            print("   reinject: mentioned", ri["mentioned"], "| restored",
                  [r["ref"] for r in ri["restored"]], "| unknown", ri["unknown_or_missing"])
        neg = rep.get("numctx_negotiation") or {}
        if neg.get("estimated_tokens") is not None:
            print("   num_ctx:", neg["estimated_tokens"], "->", neg["chosen_num_ctx"],
                  f"({neg['injected_where']})")
    conn.close()


def human_bytes(a: int, b: int) -> str:
    return f"raw {a:,} B -> clean {b:,} B"


if __name__ == "__main__":
    main()
