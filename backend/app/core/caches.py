"""Generic bounded LRU cache (Milestone 1.2).

Cross-request caches live here: MinHash signatures (1.4) and embedding
vectors (Tier 2) are keyed by content hash. Small and dependency-free.
"""
from __future__ import annotations

import threading
from collections import OrderedDict


class LRU:
    """Thread-safe dict-based LRU with O(1) get/put."""

    def __init__(self, maxsize: int = 4096):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._data: OrderedDict = OrderedDict()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


def maybe_prune(cache: LRU, maxsize: int) -> None:
    """Resets caches whose maxsize config changed (rare, dev-time helper)."""
    if cache.maxsize != maxsize:
        cache.maxsize = maxsize
        cache.clear()
