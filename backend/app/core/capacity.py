"""Best-effort machine/model capacity probe for fixed run windows.

The probe is intentionally conservative.  If Ollama metadata or VRAM cannot
be read, the configured default tier is used rather than pretending that the
model is fully resident.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from app import config
from app.core import vram


def _number(info: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = info.get(name)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def _suffix_number(info: dict[str, Any], suffixes: tuple[str, ...]) -> int | None:
    for key, value in info.items():
        if any(str(key).endswith(suffix) for suffix in suffixes):
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def _kv_bytes_per_token(info: dict[str, Any]) -> int | None:
    layers = _number(info, "llama.block_count", "block_count") or _suffix_number(info, (".block_count",))
    kv_heads = (_number(info, "llama.attention.head_count_kv", "attention.head_count_kv")
                or _suffix_number(info, ("attention.head_count_kv", ".head_count_kv")))
    head_dim = (_number(info, "llama.attention.key_length", "attention.key_length")
                or _suffix_number(info, ("attention.key_length", ".key_length")))
    if not all((layers, kv_heads, head_dim)):
        return None
    # K and V, two bytes each for the default f16 KV cache.
    kv_type = os.getenv("CONTEXTGC_KV_TYPE", "f16").lower()
    bytes_per_value = 1 if kv_type in {"q8_0", "q8", "int8"} else 2
    return layers * kv_heads * head_dim * 2 * bytes_per_value


def probe(model: str | None) -> dict[str, Any]:
    settings = config.get_settings()
    override = os.getenv("CONTEXTGC_VRAM_OVERRIDE_MB")
    sample = vram.sample()
    total = sample.get("vram_total_mb")
    used = sample.get("vram_used_mb") or 0
    if override:
        try:
            total = int(override)
            used = 0
        except ValueError:
            pass

    result: dict[str, Any] = {
        "model": model,
        "vram_total_mb": total,
        "vram_used_mb": used,
        "free_vram_mb": max(int(total) - int(used), 0) if total else None,
        "source": sample.get("source"),
        "weight_bytes": None,
        "kv_bytes_per_token": settings.kv_bytes_per_token,
        "metadata_available": False,
    }
    if not model or not settings.sink_is_ollama:
        return result
    try:
        base = settings.ollama_base_url.rstrip('/')
        with httpx.Client(timeout=0.5) as client:
            response = client.post(f"{base}/api/show", json={"name": model})
            if response.status_code != 200:
                return result
            data = response.json()
            info = data.get("model_info") or {}
            # /api/show on Ollama 0.40.x exposes architecture metadata but not the
            # on-disk weight size. /api/tags is the authoritative size source.
            try:
                tags = client.get(f"{base}/api/tags").json().get("models", [])
                names = {model, model if ":" in model else f"{model}:latest"}
                match = next((item for item in tags if item.get("name") in names), None)
                result["weight_bytes"] = (match or {}).get("size") or (match or {}).get("size_bytes")
            except Exception:
                result["weight_bytes"] = data.get("size") or data.get("size_bytes")
        result["kv_bytes_per_token"] = _kv_bytes_per_token(info) or result["kv_bytes_per_token"]
        result["metadata_available"] = bool(info)
    except Exception as exc:
        result["probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def choose_window(model: str | None, configured_default: int,
                  minimum: int, maximum: int) -> tuple[int, dict[str, Any]]:
    details = probe(model)
    free = details.get("free_vram_mb")
    weights = details.get("weight_bytes")
    kv = int(details.get("kv_bytes_per_token") or 0)
    if not free or not kv or not weights:
        return max(minimum, min(configured_default, maximum)), details

    # Keep buffers and 15% headroom for runtime allocations and fragmentation.
    available = max(int(free * 1024**2) - int(weights) - 256 * 1024**2, 0)
    available = int(available * 0.85)
    raw = available // (kv * max(1, int(config.get_settings().num_parallel)))
    tiers = (4096, 8192, 16384, 32768)
    candidates = [tier for tier in tiers if minimum <= tier <= maximum and tier <= raw]
    details["calculated_context"] = raw
    return (max(candidates) if candidates else minimum), details


def post_load_residency(model: str | None) -> dict[str, Any]:
    """Inspect Ollama's loaded model after a request.

    ``size_vram < size`` means the model is partly on CPU. The caller decides
    whether to lower the next request's pinned tier and reload once.
    """
    settings = config.get_settings()
    result: dict[str, Any] = {"checked": False, "fully_on_gpu": None}
    if not model or not settings.sink_is_ollama:
        return result
    try:
        with httpx.Client(timeout=0.5) as client:
            response = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/ps")
        models = response.json().get("models", []) if response.status_code == 200 else []
        item = next((m for m in models if m.get("name") == model), None)
        if item:
            size = int(item.get("size") or 0)
            vram = int(item.get("size_vram") or 0)
            result.update({"checked": True, "size": size, "size_vram": vram,
                           "fully_on_gpu": bool(size and vram >= size),
                           # size_vram == 0 is a CPU-only runtime, not a spill:
                           # a smaller window cannot move the model onto a GPU.
                           "cpu_only": vram == 0,
                           "partial_spill": bool(size and 0 < vram < size)})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
