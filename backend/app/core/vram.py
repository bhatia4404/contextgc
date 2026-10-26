"""VRAM sampler (Milestone 1.8 — plan v2 Tier-1, Iteration-1/L3 fix).

`psutil` cannot see VRAM, so gauges need one of:
  1. `pynvml` (NVML binding, preferred when installed)
  2. `nvidia-smi` subprocess (same driver, no Python dep)
  3. Ollama's own `/api/ps` (reports model VRAM residency) — polled async
     in main.py's sampler loop, not here (this module is sync/cheap)
  4. graceful "none" — CPU-only rigs degrade to the KV-token gauge:
     estimated_tokens x kv_bytes_per_token (config), which the dashboard
     still shows as a comparable "KV footprint" bar.

No exceptions ever escape sample(); every failure becomes a field.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

# Source detection is done once (changes under the app's feet are rare);
# guarded so concurrent first calls can't race.
_lock = threading.Lock()
_detected: str | None = None


def detect_source() -> str:
    """Resolve once: 'pynvml' > 'nvidia_smi' > 'none'."""
    global _detected
    with _lock:
        if _detected is None:
            try:
                import pynvml  # noqa: F401
                _detected = "pynvml"
            except Exception:
                _detected = "nvidia_smi" if shutil.which("nvidia-smi") else "none"
        return _detected


def _sample_pynvml() -> dict:
    """Aggregate across every detected device (multi-GPU rigs included)."""
    import pynvml

    pynvml.nvmlInit()
    try:
        used = total = 0
        name = None
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            used += mem.used
            total += mem.total
            if name is None:
                name = pynvml.nvmlDeviceGetName(h)
        return {
            "vram_used_mb": int(used / 1024**2),
            "vram_total_mb": int(total / 1024**2),
            "gpu_name": name.decode() if isinstance(name, bytes) else name,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _sample_nvidia_smi() -> dict:
    """Fallback: one subprocess call, 5s timeout, aggregate all GPUs."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"nvidia-smi rc={proc.returncode}")
    used = total = 0
    name = None
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name = parts[0] if name is None else name
        used += int(parts[1])
        total += int(parts[2])
    return {"vram_used_mb": used, "vram_total_mb": total, "gpu_name": name}


def sample() -> dict:
    """One host-VRAM snapshot. Never raises."""
    event = {"ts": time.time(), "source": detect_source(),
             "vram_used_mb": None, "vram_total_mb": None,
             "gpu_name": None, "error": None}
    try:
        if event["source"] == "pynvml":
            event.update(_sample_pynvml())
        elif event["source"] == "nvidia_smi":
            event.update(_sample_nvidia_smi())
        else:
            event["error"] = "no NVIDIA source (CPU-only or driver missing)"
    except FileNotFoundError as exc:  # nvidia-smi vanished between checks
        event.update({"error": str(exc), "source": "none"})
    except Exception as exc:
        event["error"] = f"{type(exc).__name__}: {exc}"
    return event
