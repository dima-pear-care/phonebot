#!/usr/bin/env python3
"""Entry point shim.

The application lives in the `phonebot` package; this module only re-exports the
FastAPI `app` so the existing deploy path (`uvicorn server:app`, the Dockerfile,
docker-compose, deploy.sh) keeps working unchanged, and provides the
`python server.py [--transcript]` script entry point.
"""

import os

import uvicorn

from phonebot.app import app  # noqa: F401  (re-exported for `uvicorn server:app`)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
