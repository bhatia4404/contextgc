"""ContextGC configuration (Milestone 1.1).

All runtime knobs are environment variables so the sink/backend can be swapped
with zero code changes (Iteration 6 decision).
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONTEXTGC_", env_file=".env", extra="ignore")

    # --- Downstream inference sink ---
    # "ollama" -> uses ollama_base_url (native + /v1 endpoints on same host).
    # any other value is treated as a full OpenAI-compatible base URL
    # (e.g. Baseten/Fireworks/Modal Endpoints), which always speaks /v1/*.
    sink: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"

    # --- GC master switch (Milestone 1.1 ships with GC layers OFF) ---
    # Each later GC stage will get its own flag (gc_stage1, gc_stage2, ...).
    gc_enabled: bool = False

    # --- Persistence (Milestone 1.2: append-only store) ---
    db_path: str = "data/contextgc.db"
    cache_max_entries: int = 4096

    # --- Stage flags (Milestone 1.4: Stage 1 dedup) ---
    gc_stage1_enabled: bool = True
    gc_stage1_threshold: float = 0.92

    # --- Milestone 1.7: dynamic num_ctx negotiation ---
    numctx_enabled: bool = True
    numctx_headroom: float = 1.25     # chosen window = 1.25 x estimated tokens
    numctx_min: int = 2048            # Ollama's historic tiny default — never below
    numctx_max: int = 32768           # VRAM guardrail
    numctx_v1_mode: str = "auto"      # auto | report  (auto: inject top-level num_ctx on Ollama sinks only)

    # --- Milestone 1.5: pointer-based tool offload ---
    offload_enabled: bool = True
    offload_threshold_tokens: int = 800   # dump-sized results only (precision-first)
    offload_keep_recent: int = 2          # newest N tool results stay raw (freshness)
    scratchpad_dir: str = "data/scratchpad"

    # --- Milestone 1.6: anchored state (task-drift pin) ---
    state_enabled: bool = True
    state_max_completed: int = 5          # rolling completed-actions list cap

    # --- Milestone 1.8: telemetry bus / VRAM sampling ---
    telemetry_vram_enabled: bool = True    # sampler loop during lifespan
    telemetry_vram_interval_s: float = 3.0  # sampler poll period
    kv_bytes_per_token: int = 114_688      # llama3.2:3b KV/token (28L, 8 kvH,
                                           # 128 hd, fp16 K+V) — CPU-only gauge

    # --- Proxy behavior ---
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 11400
    request_timeout_s: float = 600.0

    @property
    def sink_base_url(self) -> str:
        if self.sink == "ollama":
            return self.ollama_base_url.rstrip("/")
        return self.sink.rstrip("/")

    @property
    def sink_is_ollama(self) -> bool:
        return self.sink == "ollama"


@lru_cache
def get_settings() -> Settings:
    return Settings()
