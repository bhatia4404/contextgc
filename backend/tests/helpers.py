"""Index-independent helpers for pipeline tests (post-anchor-insertion).

The 1.6 anchor prepends a synthetic system message, so hard-coded message
indexes [0]/[1] shift by one. Tests should address messages by ROLE/CONTENT
through these helpers, never by position.
"""
from __future__ import annotations

import json

from app.core.anchored_state import _ANCHOR_MARKER


def messages(body: bytes):
    return json.loads(body)["messages"]


def without_anchor(out: bytes):
    """All messages except the ContextGC anchor pin (comparable to raw payloads)."""
    return [m for m in messages(out)
            if not (isinstance(m.get("content"), str)
                    and m["content"].lstrip().startswith(_ANCHOR_MARKER))]


def anchor_block(out: bytes):
    """The pinned 4-key dict, or None when no anchor message exists."""
    for m in messages(out):
        c = m.get("content")
        if isinstance(c, str) and c.lstrip().startswith(_ANCHOR_MARKER):
            after = c.split(_ANCHOR_MARKER, 1)[1].lstrip()
            return json.loads(after)
    return None


def find(out: bytes, role: str = None, contains: str = None, all: bool = False):
    """Messages filtered by role and/or substring (applied to str content)."""
    hits = []
    for m in messages(out):
        if role and m.get("role") != role:
            continue
        c = m.get("content")
        c = c if isinstance(c, str) else json.dumps(c, default=str)
        if contains and contains not in c:
            continue
        hits.append(m)
    return hits if all else (hits[-1] if hits else None)
