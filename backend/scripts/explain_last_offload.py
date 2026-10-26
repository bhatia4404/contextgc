"""Explains the last tool-offload turn by reading the append-only store.

Usage: python scripts\explain_last_offload.py
"""
import glob
import json
import sqlite3


def human(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    conn = sqlite3.connect("data/contextgc.db")
    conn.row_factory = sqlite3.Row

    run_id = "offload-live"
    print(f"=== online turn history for run '{run_id}' (store view) ===")
    for row in conn.execute(
        "SELECT turn_id, path, LENGTH(raw_body) AS raw_len, LENGTH(clean_body) AS clean_len"
        " FROM turns WHERE run_id = ? ORDER BY turn_id", (run_id,)
    ):
        print("turn %d | %-22s | raw %8s B | clean %8s B" % (row[0], row[1], human(row[2]), human(row[3])))

    row = conn.execute(
        "SELECT turn_id, raw_body, clean_body, gc_report FROM turns WHERE run_id = ?"
        " ORDER BY turn_id DESC LIMIT 1", (run_id,)
    ).fetchone()
    turn_id = row[0]
    raw = json.loads(bytes(row[1]))
    clean = json.loads(bytes(row[2]))
    rep = json.loads(row[3])

    print(f"\n=== TURN {turn_id} mechanics (dump -> pointered) ===")
    offl = rep.get("tool_offload", {})
    print("threshold:", offl.get("threshold"), "| newest N kept raw:", offl.get("kept_recent_raw"))
    for o in offl.get("offloads", []):
        rmsg = raw["messages"][o["index"]]
        cut = 100 * (1 - o["tokens_after"] / max(o["tokens_before"], 1))
        print(f"- dump id={rmsg.get('tool_call_id')} at index {o['index']}:"
              f" {human(len(rmsg['content']))} chars / {human(o['tokens_before'])} tokens"
              f" -> pointer {o['tokens_after']} tokens ({cut:.1f}% cut)")
        print("  pointer text the LLM now sees:")
        for line in clean["messages"][o["index"]]["content"].splitlines():
            print("    " + line)
        print("  file kept:", o["file"])

    freshest = [
        (i, m) for i, m in enumerate(clean["messages"])
        if m.get("role") == "tool" and not m["content"].lstrip().startswith("[Tool Output Ref #")
    ]
    print("\nfreshest tool results kept RAW (freshness rule):",
          [(i, m['content'][:60]) for i, m in freshest])

    neg = rep.get("numctx_negotiation", {})
    print("\nnum_ctx negotiation (sized AFTER offload):",
          f"estimated {human(neg['estimated_tokens'])} -> chosen {neg['chosen_num_ctx']}"
          f" (headroom {neg['headroom']}, floor {neg['floor']})")

    files = sorted(glob.glob("data/scratchpad/ref_*_*.txt"))
    print("\n=== scratchpad on disk ===")
    for f in files:
        content = open(f, encoding="utf-8").read()
        print(f"{f}: {human(len(content))} bytes | header: {content.splitlines()[1]}")
        print("   first line of dump:", content.splitlines()[3][:70])
        print("   last line of dump:", content.splitlines()[-1][:70])

    conn.close()


if __name__ == "__main__":
    main()
