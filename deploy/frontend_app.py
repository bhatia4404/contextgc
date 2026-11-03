"""Static host for the React front end (frontend/). It has no logic of its own:
the pages call the backend API (deploy/api_app.py) from the browser.

    cd frontend && npm ci && npm run build      # writes frontend/dist
    modal deploy deploy/frontend_app.py         # same URL as before

Unknown paths fall back to index.html so /live works on a reload.
"""
from __future__ import annotations

from pathlib import Path

import modal

DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

image = (modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")
         .add_local_dir(str(DIST), "/dist"))
app = modal.App("contextgc-showcase", image=image)


@app.function(max_containers=2)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse

    site = FastAPI(docs_url=None, redoc_url=None)
    root = Path("/dist").resolve()

    @site.get("/{path:path}")
    def serve(path: str):
        target = (root / path).resolve()
        if path and target.is_file() and root in target.parents:
            # hashed assets never change; index.html must always be fresh
            cache = "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
            return FileResponse(target, headers={"cache-control": cache})
        return FileResponse(root / "index.html", headers={"cache-control": "no-cache"})

    return site
