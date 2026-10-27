"""Pre-cache the S4 embedding model weights (offline-demo insurance).

Semantic contraction (Tier 2, S4) encodes messages with all-MiniLM-L6-v2.
The weights live on HuggingFace and download once, on FIRST use — which, on
venue Wi-Fi or an offline demo machine, is exactly the wrong moment. This
script forces that download NOW, into the local HF cache:

    python scripts\warm_embeddings.py

Run it after installing sentence-transformers and before any offline demo /
benchmark / video recording. Exit codes: 0 = ready; 2 = not installed; 1 = download failed.
"""
from __future__ import annotations

import sys

MODEL_NAME = "all-MiniLM-L6-v2"


def main() -> int:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # ImportError + wheel issues both end here
        print(f"[warm] sentence-transformers not installed ({exc.__class__.__name__})")
        print("[warm] install it with:  pip install sentence-transformers --index-url https://download.pytorch.org/whl/cpu")
        return 2

    print(f"[warm] loading/verifying {MODEL_NAME} ...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        vec = model.encode(" ContextGC warm-up probe ")
        assert getattr(vec, "size", len(vec)) > 0
    except Exception as exc:
        print(f"[warm] FAILED to fetch/load weights: {exc}")
        print("[warm] (offline machine? run this once on a connected machine,")
        print("[warm]  then copy %LOCALAPPDATA%\\torch\\sentence_transformers + the HF cache over)")
        return 1

    print("[warm] weights cached and encoder verified — S4 is safe for offline demos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
