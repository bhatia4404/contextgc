"""GC pipeline orchestrator.

Contract (final_plan_v2.md): body bytes in -> (body bytes possibly cleaned, report).

Stage status:
  1.1  identity passthrough when CONTEXTGC_GC_ENABLED=false (byte-faithful)
  1.3  Stage 0 classifier: annotates messages, never modifies.
  1.4  Stage 1 MinHash/LSH dedup: removes near-duplicate episodic messages
       (same-request and cross-request per run). First stage that shrinks
       the payload -> sets modified -> payload re-serialized compactly.
  1.5  Stage 2 pointer offload: dump-sized tool results -> disk + ref token.
  1.5b re-inject: on-demand restore when the agent explicitly asks a ref.
  2.1  S4 semantic contraction: embedding merge, same-namespace, neg guard.
  1.6  S3 anchored state pin.
  2.2  S5 importance eviction: pressure-triggered, hard-protected classes.
  2.3  fidelity fallback: deterministic loop/confusion rule engine -> raw.
"""
from __future__ import annotations

import json
import copy
from typing import Any, Optional

from app import config
from app.core import classifier, hash_filter, num_ctx, offload, semantic
from app.core import anchored_state
from app.core import evict
from app.core import fidelity
from app.core import run_state

# Per-request stage overrides (benchmark control experiments). A preset name
# maps to the set of stages allowed to run; None = default (no filter). This
# is how the harness isolates e.g. "proxy overhead only" or "anchor prepend
# only" — measured, not assumed.
_STAGE_PRESETS: dict[str, Optional[set[str]]] = {
    "full": None,
    "none": set(),                                    # proxy overhead only
    "anchor_only": {"state", "numctx"},               # prepend cost isolated
    "no_semantic": {"dedup", "offload", "reinject", "state", "evict", "numctx"},
    "no_evict": {"dedup", "offload", "reinject", "semantic", "state", "numctx"},
}


def parse_stages_header(value: str | None) -> Optional[set[str]]:
    """Header value: 'full' | 'none' | 'anchor_only' | 'no_semantic' | 'no_evict'."""
    if not value:
        return None
    value = value.strip().lower()
    if value == "":
        return None
    if value not in _STAGE_PRESETS:
        raise ValueError(f"unknown stages preset: {value!r} "
                         f"(one of {sorted(_STAGE_PRESETS)})")
    return _STAGE_PRESETS[value]


def process_payload(path: str, body: bytes, run_id: str = "default",
                    stages_preset: str | None = None,
                    replay: bool = False) -> tuple[bytes, dict[str, Any]]:
    settings = config.get_settings()
    stage_filter = parse_stages_header(stages_preset)

    def stage_on(name: str) -> bool:
        return stage_filter is None or name in stage_filter

    if not settings.gc_enabled:
        return body, {}

    # --- parse (guard: anything we cannot understand is forwarded unchanged) ---
    try:
        raw_payload = json.loads(body)
        if not isinstance(raw_payload, dict):
            raise ValueError("not a JSON object")
        pinned = num_ctx.pinned_window(run_id)
        if pinned:
            raw_payload["_contextgc_pinned_window"] = pinned
        payload, prior_state, append_only, compaction, retry = run_state.prepare(
            run_id, raw_payload, settings.compaction_trigger, num_ctx.estimate_tokens,
        )
        if compaction:
            payload = copy.deepcopy(raw_payload)
            append_only = False
        payload.pop("_contextgc_pinned_window", None)
        raw_payload.pop("_contextgc_pinned_window", None)
        report = classifier.classify_payload(payload)
        report["incremental_view"] = append_only
        report["compaction"] = compaction
        report["retry"] = retry
        if compaction:
            pinned_for_compaction = num_ctx.pinned_window(run_id)
            if pinned_for_compaction:
                report["compaction_budget"] = int(pinned_for_compaction * settings.compaction_target)
                report["compaction_window"] = pinned_for_compaction
    except Exception:  # undecodable / non-chat -> identity
        return body, {"pipeline": {"skipped": "undecodable or non-chat payload"}}

    if stages_preset:
        report["stages_preset"] = stages_preset

    # --- Stage 2.3: deterministic fidelity fallback — if the RAW history
    #     shows the model looping/derailing, this request IS the recovery
    #     turn: replay the raw view byte-identically, no GC at all. Evidence
    #     lives in the arriving history, so no cross-request state is needed.
    fidelity_raw = False
    if settings.fidelity_enabled and not replay and not retry and stage_on("fidelity") and not report.get("skipped"):
        sig = fidelity.detect(raw_payload)
        if sig:
            report["fidelity_fallback"] = {"triggered": True, "signals": sig}
            fidelity_raw = True
            payload = copy.deepcopy(raw_payload)
            append_only = False
            compaction = False

    modified = False
    stages_run = ["fidelity_fallback"] if fidelity_raw else ["stage0_classifier"]

    # --- Stage 1: episodic dedup ---
    if not fidelity_raw and not retry and settings.gc_stage1_enabled and stage_on("dedup") and report.get("counts", {}):
        if hash_filter.run_stage1(payload, run_id, report):
            modified = True
        stages_run.append("stage1_dedup")

    # --- Stage 1.5: pointer-based tool offload (bigger cuts than dedup; runs
    #     BEFORE num_ctx so the window is sized to the post-offload payload) ---
    if not fidelity_raw and not retry and settings.offload_enabled and stage_on("offload") and not report.get("skipped"):
        if offload.run_offload(payload, report, run_id):
            modified = True
        stages_run.append("tool_offload")

    # --- Stage 1.5b: on-demand re-injection (accuracy contract; MUST precede
    #     num_ctx because restored dumps grow the estimate back) ---
    if not fidelity_raw and not retry and settings.offload_enabled and stage_on("reinject") and not report.get("skipped"):
        if offload.maybe_reinject(payload, report):
            modified = True
        stages_run.append("ref_reinject")

    # --- Stage 2.1: semantic contraction (same-namespace + negation guard;
    #     runs AFTER reinject so a pointer-restored dump isn't re-merged away,
    #     BEFORE anchor/num_ctx which see the post-contraction payload) ---
    if not fidelity_raw and not retry and settings.semantic_enabled and stage_on("semantic") and not report.get("skipped"):
        if semantic.run_semantic(payload, report, run_id):
            modified = True
        stages_run.append("semantic_contraction")

    # --- Stage 1.6: anchored state (pinned AFTER all cleaning; num_ctx runs after) ---
    if not fidelity_raw and not retry and settings.state_enabled and stage_on("state") and not append_only and not report.get("skipped"):
        if anchored_state.run_anchor(payload, report, run_id):
            modified = True
        stages_run.append("anchored_state")

    # --- Stage 2.2: importance eviction (pressure-triggered; runs AFTER the
    #     anchor pin so the state block was extracted from the full history,
    #     BEFORE num_ctx which sizes the post-eviction payload) ---
    if not fidelity_raw and not retry and settings.evict_enabled and stage_on("evict") and not append_only and not report.get("skipped"):
        if evict.run_evict(payload, report, run_id):
            modified = True
            stages_run.append("importance_eviction")

    if append_only:
        # Existing cleaned messages are immutable within an epoch. Only the
        # newly appended suffix was transformed above.
        payload = run_state.combine(prior_state, payload, append_only=True)
        # The incoming request is raw history; the forwarded view is different
        # even when no suffix stage changed its own bytes.
        modified = True
        if compaction:
            report["compaction"] = True

    if fidelity_raw or retry:
        modified = True

    # --- Stage 1.7: dynamic num_ctx negotiation (LAST: sizes what remains) ---
    if settings.numctx_enabled and stage_on("numctx") and not report.get("skipped"):
        family = "api" if path.startswith("/api/") else "v1"
        if num_ctx.negotiate(payload, report, family, run_id=run_id):
            modified = True
        stages_run.append("numctx_negotiation")

    # keep_alive is a request setting, not a GC transformation. It is applied
    # consistently so idle gaps do not evict the loaded model.
    if settings.sink_is_ollama and settings.keep_alive:
        if payload.get("keep_alive") != settings.keep_alive:
            payload["keep_alive"] = settings.keep_alive
            modified = True
        report["keep_alive"] = settings.keep_alive

    report["pipeline"] = {"stages": stages_run, "modified": modified}

    if not modified:
        report.pop("_annotations_by_object", None)
        return body, report  # byte-faithful even with GC enabled

    run_state.commit(run_id, raw_payload, payload, epoch=int(compaction), append_only=append_only)
    report.pop("_annotations_by_object", None)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), report
