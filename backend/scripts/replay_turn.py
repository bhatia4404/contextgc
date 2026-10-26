"""Replays a stored turn's CLEAN payload to Ollama and prints the actual reply.

Usage:  python scripts\replay_turn.py 14
Proves the cleaned prompt still gets a coherent answer (GC quality check).
"""
import json
import sqlite3
import sys
import urllib.request

tid = int(sys.argv[1]) if len(sys.argv) > 1 else None
conn = sqlite3.connect("data/contextgc.db")
row = (conn.execute("SELECT turn_id, clean_body FROM turns WHERE turn_id = ?", (tid,)).fetchone()
       if tid else conn.execute("SELECT turn_id, clean_body FROM turns ORDER BY turn_id DESC LIMIT 1").fetchone())
conn.close()
turn_id, payload = row[0], json.loads(bytes(row[1]))
payload["stream"] = False
payload.pop("num_ctx", None)
payload["options"] = {"num_ctx": 2048}

req = urllib.request.Request("http://localhost:11434/api/chat",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
r = urllib.request.urlopen(req, timeout=300)
d = json.loads(r.read())
print(f"[turn {turn_id}] LLM reply to CLEANED payload:")
print(d["message"]["content"][:300])
