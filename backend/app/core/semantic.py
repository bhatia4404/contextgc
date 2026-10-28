"""S4 — Semantic Contraction (Tier 2, Milestone 2.1; plan v2; Iteration-1/L5).

Stage 1's MinHash/LSH dedup only catches STRUCTURALLY identical text. When two
agents restate the same fact in different words ("the parser fails on
comments" vs "comments break the parser"), both survive and bloat the window.
This stage merges near-duplicate *meaning* while making safe-merge guarantees
it CANNOT enforce structurally:

  - Same-namespace rule: messages only merge within the same (role, name)
    scope — cross-agent statements are never merged (no cross-talk).
  - Negation / polarity guard: contradictory phrasings are near-identical in
    embedding space; if polarity flips, NEVER merge.
  - Number guard: "3 tests failed" vs "2 tests failed" stays unmerged.
  - Precision over recall (L5): when in doubt, don't merge.

Merge action: keep the MOST RECENT phrasing as canonical, drop older near-
duplicates (GC is a view — originals live in the append-only store).

Embeddings: `sentence-transformers` MiniLM, lazy-loaded; per-message LRU cache
(agent loops re-send history, so repeats cost a hash). NO LLM in the hot path
— a local encoder runs offline-cacheable; without the lib the stage skips
cleanly (reported, never fatal).
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Optional

from app import config
from app.core.caches import LRU

# Content-hash -> embedding vector. History repeats across turns.
_EMBED_CACHE = LRU(maxsize=4096)

# Injectable encoder (tests inject a deterministic fake; prod uses MiniLM).
_encoder_fn: Optional[Callable[[str], list[float]]] = None
_default_encoder_instance: Optional[Callable[[str], list[float]]] = None

# Candidate filters (precision-first).
_CANDIDATE_MIN_CHARS = 40
_CANDIDATE_MAX = 128          # cap embeddings per request (perf guardrail)

# Safety guards (Iteration-1/L5: merge only when polarity & numbers agree).
#
# FLIP words change an assertion's truth value. "fails" appearing in BOTH
# texts is fine (both describe the problem); but "fails" in one and "does
# NOT fail" in the other is a contradiction. Guard: flip-presence must
# MATCH on both sides for a merge to be allowed.
_FLIP_MARKERS = (
    " not ", " never ", " cannot ", " can't ", " won't ",
    " isn't ", " aren't ", " doesn't ", " didn't ", " don't ", " wasn't ",
    " will not ", " is not ", " are not ", " does not ", " did not ",
    " do not ",
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def has_flip(text: str) -> bool:
    """True when a truth-value negator is present in the text."""
    lowered = " %s " % text.lower()
    return any(marker in lowered for marker in _FLIP_MARKERS)


def _numbers(text: str) -> frozenset:
    return frozenset(_NUMBER_RE.findall(text))

# Content-hash -> embedding vector. History repeats across turns.
_EMBED_CACHE = LRU(maxsize=4096)


def set_encoder(fn: Optional[Callable[[str], list[float]]]) -> None:
    """Inject an embedding function (tests); None restores the default."""
    global _encoder_fn
    _encoder_fn = fn


def reset_state() -> None:
    """Test isolation: clear the embedding cache + injected encoder."""
    global _default_encoder_instance
    _EMBED_CACHE.clear()
    _default_encoder_instance = None
    set_encoder(None)


def _default_encoder() -> Optional[Callable[[str], list[float]]]:
    """Lazy sentence-transformers MiniLM; None when the lib is missing."""
    global _default_encoder_instance
    if _default_encoder_instance is not None:
        return _default_encoder_instance
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    _default_encoder_instance = lambda text: [float(v) for v in model.encode(text)]
    return _default_encoder_instance


def _encoder():
    if _encoder_fn is not None:
        return _encoder_fn
    return _default_encoder()


def _embed(text: str) -> Optional[list[float]]:
    enc = _encoder()
    if enc is None:
        return None
    key = hashlib.blake2b(
        text.encode("utf-8", errors="replace")[:8192], digest_size=10).hexdigest()
    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    vec = enc(text)
    _EMBED_CACHE.put(key, vec)
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


# --- Safety guards (Iteration-1/L5: merge only when polarity & numbers agree) ---

# FLIP words change an assertion's truth value. "fails" appearing in BOTH texts
# is fine (both describe the problem); but "fails" in one and "does NOT fail"
# in the other is a contradiction. Guard: flip-presence must MATCH to merge.
_FLIP_MARKERS = (
    " not ", " never ", " cannot ", " can't ", " won't ",
    " isn't ", " aren't ", " doesn't ", " didn't ", " don't ", " wasn't ",
    " will not ", " is not ", " are not ", " does not ", " did not ", " do not ",
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


# --- Stage entry ---


def run_semantic(payload: dict, report: dict, run_id: str = "default") -> bool:
    """Contract like the other stages: True if the payload was modified."""
    settings = config.get_settings()
    messages = payload.get("messages")
    if not isinstance(messages, list) or report.get("skipped"):
        return False

    candidates = []            # (index, text) eligible for contraction
    annotations = {
        str(id(msg)): report.get("_annotations_by_object", {}).get(str(id(msg)), {})
        for msg in messages
    }
    for i, msg in enumerate(messages):
        if i == len(messages) - 1:          # the live turn is always kept
            continue
        if not isinstance(msg, dict):
            continue
        ann = annotations.get(str(id(msg))) or {}
        mtype = ann.get("type")
        if mtype not in ("semantic", "episodic"):        # procedural: locked
            continue
        role = str(msg.get("role", ""))
        if role not in ("user", "assistant") or "tool_calls" in msg:
            continue                                    # tool results: S2's domain
        text = msg.get("content")
        if not isinstance(text, str) or len(text) < _CANDIDATE_MIN_CHARS:
            continue
        candidates.append((i, text))
        if len(candidates) >= _CANDIDATE_MAX:
            break

    stage = report.setdefault("semantic_contraction", {
        "cosine_threshold": settings.semantic_threshold,
    })
    if not candidates:
        stage["merged"] = 0
        return False

    # Embed (cache-backed)
    vecs: dict[int, list[float]] = {}
    for i, text in candidates:
        v = _embed(text)
        if v is None:
            stage["skipped_reason"] = "encoder unavailable (install sentence-transformers)"
            stage["merged"] = 0
            return False
        vecs[i] = v

    # Namespace = (role, name). Iterate NEWEST -> OLDEST; a message similar to
    # an already-kept message is dropped in favor of the canonical newer one.
    kept_by_ns: dict[tuple, list[int]] = {}
    dropped: list[dict] = []
    chars_saved = 0
    for i, text in reversed(candidates):
        ns = (str(messages[i].get("role")), messages[i].get("name"))
        vec = vecs[i]
        pol_i, nums_i = has_flip(text), _numbers(text)
        similar_to = None
        for j in kept_by_ns.get(ns, []):
            if _cosine(vec, vecs[j]) <= settings.semantic_threshold:
                continue
            text_j = messages[j]["content"]
            if has_flip(text_j) != pol_i:        # negation guard: NEVER merge
                continue
            if _numbers(text_j) != nums_i:       # number guard: facts differ
                continue
            similar_to = j
            break
        if similar_to is None:
            kept_by_ns.setdefault(ns, []).insert(0, i)
        else:
            dropped.append({
                "index": i, "merged_into": similar_to,
                "chars_saved": len(text),
            })
            chars_saved += len(text)

    if not dropped:
        stage["merged"] = 0
        return False

    drop_set = {d["index"] for d in dropped}
    payload["messages"] = [
        m for i, m in enumerate(messages)
        if i not in drop_set or i == len(messages) - 1
    ]

    stage.update({
        "merged": len(dropped),
        "dropped": dropped,
        "chars_saved": chars_saved,
        "tokens_saved_estimate": chars_saved // 4,
    })
    counts = report.setdefault("counts", {})
    counts["semantically_merged"] = len(dropped)
    return True
