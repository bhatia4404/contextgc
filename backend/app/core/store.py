"""Append-only store (Milestone 1.2).

The non-destructive GC invariant lives here: every incoming request payload
(raw, pre-GC) plus what we sent after GC is persisted, keyed by turn.
Nothing is ever mutated or deleted; later GC stages are *views* over this
log (fallback replay 2.3, cross-agent dedup state 1.4).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from app import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    path       TEXT NOT NULL,
    run_id     TEXT NOT NULL DEFAULT 'default',
    raw_body   BLOB NOT NULL,
    clean_body BLOB NOT NULL,
    gc_report  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_turns_run ON turns(run_id, turn_id);
"""


class Store:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Single writer connection + lock: appends are tiny; this avoids
        # blocking the event loop beyond microseconds while staying stdlib.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:  # schema init
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def append(self, path: str, run_id: str, raw_body: bytes,
               clean_body: bytes, gc_report: Optional[dict]) -> int:
        """Persist one request. Returns turn_id."""
        report = json.dumps(gc_report or {}, ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turns (ts, path, run_id, raw_body, clean_body, gc_report)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), path, run_id, raw_body, clean_body, report),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_raw(self, turn_id: int) -> Optional[bytes]:
        return self._fetch_col(turn_id, "raw_body")

    def get_clean(self, turn_id: int) -> Optional[bytes]:
        return self._fetch_col(turn_id, "clean_body")

    def get_report(self, turn_id: int) -> Optional[dict]:
        row = self._fetch_col(turn_id, "gc_report")
        return json.loads(row) if row is not None else None

    def _fetch_col(self, turn_id: int, col: str):
        with self._lock:
            row = self._conn.execute(
                f"SELECT {col} FROM turns WHERE turn_id = ?", (turn_id,)
            ).fetchone()
        return row[0] if row else None

    def raw_json(self, turn_id: int) -> Optional[Any]:
        raw = self.get_raw(turn_id)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw.decode("utf-8", errors="replace")

    def recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_id, ts, path, run_id, LENGTH(raw_body), LENGTH(clean_body)"
                " FROM turns ORDER BY turn_id DESC LIMIT ?", (n,)
            ).fetchall()
        return [
            {"turn_id": r[0], "ts": r[1], "path": r[2], "run_id": r[3],
             "raw_bytes": r[4], "clean_bytes": r[5]}
            for r in rows
        ]

    def stats(self) -> dict:
        with self._lock:
            n, raw_sum, clean_sum = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(raw_body)),0),"
                " COALESCE(SUM(LENGTH(clean_body)),0) FROM turns"
            ).fetchone()
        return {
            "turns": n,
            "raw_bytes_total": raw_sum,
            "clean_bytes_total": clean_sum,
            "db_path": str(self.db_path),
            "db_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@lru_cache
def get_store() -> Store:
    return Store(config.get_settings().db_path)
