"""Stage 1.5 — Pointer-based tool offload (Milestone 1.5).

The single biggest token consumer in agent runs: tool outputs (terminal logs,
file greps, test dumps) that accumulate turn after turn and are NEVER
duplicate-like enough for MinHash dedup (timestamps/interleaving differ).

Mechanism (plan v2 "pointer offload", matches user's Iteration-11 restatement):
  1. DUMP-SIZED tool results (> offload_threshold_tokens) are written verbatim
     to the disk scratchpad — nothing is destroyed (scratchpad + store both).
  2. The prompt keeps a pointer:  [Tool Output Ref #N — <tool> | ~X lines
     (~Y tokens) on disk. Key result: <line>. Deep-dive on demand.]
     — informative, not a dead pointer: the model usually works off the
     summary without reading the file at all.
  3. gc_report records the exchange for the diff viewer / fallback.

Guards (conservative, Iteration-1 style):
  - the FRESHEST tool results (last `offload_keep_recent`) stay raw — the
    agent is actively reasoning over them on this turn
  - the last message of the list is never offloaded (freshness)
  - tool_call_id / name envelope fields are PRESERVED (pairing integrity)
  - pointer content is recognized and never re-offloaded (idempotence)
  - user/assistant/system messages are never offloaded (tool results only)

Runs BETWEEN dedup and num_ctx negotiation: the pointer's small size lets
1.7 estimate the *post-offload* payload for the window sizing.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from app import config
from app.core import num_ctx as num_ctx_mod

_POINTER_MARKER = "[Tool Output Ref #"
_TOOL_ROLES = {"tool", "function", "tool_result"}
_KEY_RESULT_RE = re.compile(
    r"failed|error|success|passed|exit code|exception|warning|completed|skipped|✓|✗|❌|⚠",
    re.IGNORECASE,
)


class _RefCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._n = 0

    def next(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def remember(self, ref_id: int, filepath: str) -> None:
        """ref -> scratchpad path map: enables on-demand re-injection (path 2)."""
        with self._lock:
            _REF_MAP[ref_id] = filepath


_REF_MAP: dict[int, str] = {}          # ref_id -> scratchpad file (path 2 readback)
_refs = _RefCounter()

# Content-addressed refs (Iteration-12 fix, accurate-contract backbone):
# the same dump re-sent by a framework in a later turn must map to the SAME
# ref id — otherwise the agent/model mentions "Ref #2" while this turn's
# fresh pointer says "Ref #5" and re-injection cannot resolve. Registry is
# per run: (run_id, content_hash) -> ref_id. Rebuilt from the scratchpad
# after a proxy restart (self-healing).
_CONTENT_TO_REF: dict[tuple[str, str], int] = {}


def _content_hash(text: str) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(text.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _restore_registry_from_disk() -> None:
    """After a proxy restart, re-derive ref ids from scratchpad headers so
    old refs stay mentionable. Files carry a '# ... | run: <id>' header."""
    settings = config.get_settings()
    path = Path(settings.scratchpad_dir)
    if not path.exists():
        return
    for f in path.glob("ref_*.txt"):
        try:
            content = f.read_text(encoding="utf-8")
            lines = content.splitlines()
            hdr = next((l for l in lines if l.startswith("# ref: ")), None)
            run_m = next((l for l in lines if l.startswith("# ... |") or " run: " in l), None)
            if not hdr or not run_m:
                continue
            ref_id = int(hdr.split("ref: ")[1].split(" ")[0])
            run_id = run_m.split(" run: ")[1].split(" ")[0]
            body = "\n".join(lines[3:])
            _CONTENT_TO_REF.setdefault((run_id, _content_hash(body)), ref_id)
            if not content.strip():
                continue
            _REF_MAP.setdefault(ref_id, str(f))
            if ref_id > _refs._n:
                _refs._n = ref_id      # resume the counter, never re-issue old ids
        except Exception:
            continue  # damaged file: skip; other refs stay restorable


_RESTORE_DONE = False


def _ensure_registry():
    global _RESTORE_DONE
    if not _RESTORE_DONE:
        _restore_registry_from_disk()
        _RESTORE_DONE = True


def _is_pointer(text: str) -> bool:
    return text.lstrip().startswith(_POINTER_MARKER)


def _extract_summary(text: str) -> str:
    """One decision-relevant line from a dump (status-bearing line first)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "(empty output)"
    for ln in lines:
        if _KEY_RESULT_RE.search(ln):
            return (ln[:120] + "…") if len(ln) > 120 else ln
    first = lines[0]
    return (first[:100] + "…") if len(first) > 100 else first


def _run_to_scratchpad(ref_id: int, tool_label: str, text: str, run_id: str) -> str:
    settings = config.get_settings()
    path = Path(settings.scratchpad_dir)
    path.mkdir(parents=True, exist_ok=True)
    fname = f"ref_{ref_id}_{int(time.time())}.txt"
    header = (
        f"# ContextGC tool-offload original\n# ref: {ref_id} | tool: {tool_label}"
        f" | run: {run_id} | ts: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    (path / fname).write_text(header + text, encoding="utf-8")
    return str(path / fname)


def run_offload(payload: dict, report: dict, run_id: str) -> bool:
    """Offload dump-sized tool results in place. Returns True if modified."""
    if not config.get_settings().offload_enabled:
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    # tool-result indexes that are legally offloadable
    tool_idx = [
        i for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get("role") in _TOOL_ROLES
    ]
    if not tool_idx:
        return False

    settings = config.get_settings()
    keep_recent_from = tool_idx[-settings.offload_keep_recent:] if settings.offload_keep_recent else []
    last_index = len(messages) - 1

    offloads: list[dict] = []
    for i in tool_idx:
        msg = messages[i]
        if i == last_index:
            continue                              # freshness: never the last
        if i in keep_recent_from:                 # freshness: newest N stay raw
            continue
        content = msg.get("content")
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        if _is_pointer(text):
            continue                              # idempotence: already a pointer
        est_before = num_ctx_mod.estimate_tokens({"messages": [msg]})
        if est_before < settings.offload_threshold_tokens:
            continue                              # dump-sized only (precision-first)

        est_before = num_ctx_mod.estimate_tokens({"messages": [msg]})
        if est_before < settings.offload_threshold_tokens:
            continue                              # dump-sized only (precision-first)

        _ensure_registry()                        # survive proxy restarts (self-heal)
        chash = (run_id, _content_hash(text))
        tool_label = msg.get("name") or (msg.get("tool_call_id") and f"tool_call {msg['tool_call_id']}") or "tool"
        existing = _CONTENT_TO_REF.get(chash)
        filepath = _REF_MAP.get(existing) if existing is not None else None
        if existing is None or not filepath or not Path(filepath).exists():
            # fresh (or self-heal: registry entry whose file vanished/dropped on reset)
            ref_id = existing if (existing is not None and not filepath) else _refs.next()
            filepath = _run_to_scratchpad(ref_id, tool_label, text, run_id)
            _CONTENT_TO_REF[chash] = ref_id
            _refs.remember(ref_id, filepath)
        else:
            ref_id = existing          # SAME content, SAME ref
        summary = _extract_summary(text)
        lines = text.count("\n") + 1

        pointer = (
            f"[Tool Output Ref #{ref_id} — {tool_label} | {lines:,} lines"
            f" (~{est_before:,} tokens) saved to scratchpad: {filepath}\n"
            f"Key result: {summary}\n"
            f'Content trimmed to save context. Retrieve via "read_ref {ref_id}" if the full output is needed.]'
        )
        # interchangeable content only; envelope fields stay untouched
        msg["content"] = pointer
        _refs.remember(ref_id, filepath)
        offloads.append({
            "index": i,
            "ref": ref_id,
            "file": filepath,
            "tool": tool_label,
            "tool_call_id": msg.get("tool_call_id"),
            "tokens_before": est_before,
            "tokens_after": num_ctx_mod.estimate_tokens({"messages": [msg]}),
        })

    if offloads:
        saved = sum(o["tokens_before"] - o["tokens_after"] for o in offloads)
        report["tool_offload"] = {
            "offloads": offloads,
            "tokens_saved_estimate": max(saved, 0),
            "threshold": settings.offload_threshold_tokens,
            "kept_recent_raw": len(keep_recent_from),
        }
        report.setdefault("counts", {})["offloads"] = sum(
            1 for _ in offloads
        )
        return True
    return False


# --- Path 2: on-demand re-injection (accuracy contract) ---
# If the agent's CURRENT turn explicitly asks about a pointered ref
# ("Ref #12", "read_ref 12", "what's the exact stack trace from ref 12?"),
# we restore that dump's FULL original for exactly this one turn. This is
# the "GC is a view" invariant turned into a live contract: precision is
# recoverable on demand instead of permanently lost (ContextGC quality bar).
_MENTION_RE = re.compile(r"(?:ref\s*#|read_ref\s*)(\d{1,6})", re.IGNORECASE)


def scan_for_ref_mentions(text: str) -> list[int]:
    refs = []
    for m in _MENTION_RE.finditer(text):
        try:
            refs.append(int(m.group(1)))
        except ValueError:
            continue
    return refs


def maybe_reinject(payload: dict, report: dict) -> bool:
    """Restore FULL originals for refs the current turn explicitly mentions.

    Blast radius: only messages that currently ARE pointers (only refs that
    previously went through this module); unknown refs are reported, never
    crash-run. Returns True if the payload was modified.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    mentioned: set[int] = set()
    for m in messages:
        if not isinstance(m, dict):
            continue
        text = m.get("content")
        text = text if isinstance(text, str) else ""
        if _is_pointer(text):
            continue  # pointers themselves don't count as a *request*
        mentioned.update(scan_for_ref_mentions(text))
    if not mentioned:
        return False

    restored: list[dict] = []
    misses: list[int] = []
    for ref_id in sorted(mentioned):
        filepath = _REF_MAP.get(ref_id)
        pointer_idx = next((i for i, m in enumerate(messages)
                            if isinstance(m, dict) and _is_pointer(str(m.get("content") or ""))
                            and f"Ref #{ref_id}" in m["content"]), None)
        if filepath is None or pointer_idx is None:
            misses.append(ref_id)
            continue
        try:
            full = (Path(filepath).read_text(encoding="utf-8"))
        except OSError:
            misses.append(ref_id)
            continue
        body = "\n".join(full.splitlines()[3:])  # drop the 3-line scratchpad header
        messages[pointer_idx]["content"] = (
            f"[Tool Output Ref #{ref_id} — FULL OUTPUT RESTORED FOR THIS TURN (on explicit request)]\n"
            + body
        )
        restored.append({"ref": ref_id, "index": pointer_idx,
                         "tokens_restored_estimate": num_ctx_mod.estimate_tokens({"messages": [{"role": "tool", "content": body}]})})

    if restored or misses:
        report["ref_reinject"] = {
            "restored": restored,
            "unknown_or_missing": misses,
            "mentioned": sorted(mentioned),
        }
        report.setdefault("counts", {})["reinjects"] = len(restored)
        return bool(restored)
    return False
