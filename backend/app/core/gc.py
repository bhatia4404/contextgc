"""GC pipeline orchestrator.

Contract (final_plan_v2.md): body bytes in -> (body bytes possibly cleaned, report).

Stage status:
  1.1  identity passthrough when CONTEXTGC_GC_ENABLED=false (byte-faithful)
  1.3  Stage 0 classifier: annotates messages, never modifies.
  1.4  Stage 1 MinHash/LSH dedup: removes near-duplicate episodic messages
       (same-request and cross-request per run). First stage that shrinks
       the payload -> sets modified -> payload re-serialized compactly.
"""
from __future__ import annotations

import json
from typing import Any

from app import config
from app.core import classifier, hash_filter, num_ctx, offload, semantic
from app.core import anchored_state
from app.core import evict
from app.core import fidelity


def process_payload(path: str, body: bytes, run_id: str = "default") -> tuple[bytes, dict[str, Any]]:
    settings = config.get_settings()
    if not settings.gc_enabled:
        return body, {}

    # --- parse (guard: anything we cannot understand is forwarded unchanged) ---
    try:
        payload, report = classifier.annotate(path, body)
    except Exception:  # undecodable / non-chat -> identity
        return body, {"pipeline": {"skipped": "undecodable or non-chat payload"}}

    # --- Stage 2.3: deterministic fidelity fallback — if the RAW history
    #     shows the model looping/derailing, this request IS the recovery
    #     turn: replay the raw view byte-identically, no GC at all. Evidence
    #     lives in the arriving history, so no cross-request state is needed.
    if settings.fidelity_enabled and not report.get("skipped"):
        sig = fidelity.detect(payload)
        if sig:
            report["fidelity_fallback"] = {"triggered": True, "signals": sig}
            report["pipeline"] = {"stages": ["fidelity_fallback"], "modified": False}
            return body, report  # ← recovery turn: the raw view wins

    modified = False
    stages_run = ["stage0_classifier"]

    # --- Stage 1: episodic dedup ---
    if settings.gc_stage1_enabled and report.get("counts", {}):
        if hash_filter.run_stage1(payload, run_id, report):
            modified = True
        stages_run.append("stage1_dedup")

    # --- Stage 1.5: pointer-based tool offload (bigger cuts than dedup; runs
    #     BEFORE num_ctx so the window is sized to the post-offload payload) ---
    if settings.offload_enabled and not report.get("skipped"):
        if offload.run_offload(payload, report, run_id):
            modified = True
        stages_run.append("tool_offload")

    # --- Stage 1.5b: on-demand re-injection (accuracy contract; MUST precede
    #     num_ctx because restored dumps grow the estimate back) ---
    if settings.offload_enabled and not report.get("skipped"):
        if offload.maybe_reinject(payload, report):
            modified = True
        stages_run.append("ref_reinject")

    # --- Stage 2.1: semantic contraction (same-namespace + negation guard;
    #     runs AFTER reinject so a pointer-restored dump isn't re-merged away,
    #     BEFORE anchor/num_ctx which see the post-contraction payload) ---
    if settings.semantic_enabled and not report.get("skipped"):
        if semantic.run_semantic(payload, report, run_id):
            modified = True
        stages_run.append("semantic_contraction")

    # --- Stage 1.6: anchored state (pinned AFTER all cleaning; num_ctx runs after) ---
    if settings.state_enabled and not report.get("skipped"):
        if anchored_state.run_anchor(payload, report, run_id):
            modified = True
        stages_run.append("anchored_state")

    # --- Stage 2.2: importance eviction (pressure-triggered; runs AFTER the
    #     anchor pin so the state block was extracted from the full history,
    #     BEFORE num_ctx which sizes the post-eviction window) ---
    if settings.evict_enabled and not report.get("skipped"):
        if evict.run_evict(payload, report, run_id):
            modified = True
        stages_run.append("importance_eviction")

    # --- Stage 1.7: dynamic num_ctx negotiation (LAST: sizes what remains) ---
    if settings.numctx_enabled and not report.get("skipped"):
        family = "api" if path.startswith("/api/") else "v1"
        if num_ctx.negotiate(payload, report, family):
            modified = True
        stages_run.append("numctx_negotiation")

    report["pipeline"] = {"stages": stages_run, "modified": modified}

    if not modified:
        return body, report  # byte-faithful even with GC enabled

    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), report
