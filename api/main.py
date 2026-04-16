"""
grow-ai FastAPI server.

Serves:
  /api/*       — REST API (insights, growth log, system)
  /            — Single-page dashboard (static/index.html)

Start:
  python -m api.main
  # or
  grow-ai-serve
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routers import insights, growth, system

app = FastAPI(
    title="grow-ai",
    description="Personal AI that grows from your Claude Code sessions",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(insights.router)
app.include_router(growth.router)
app.include_router(system.router)

# Serve static dashboard
_STATIC_DIR = Path(__file__).parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(str(_STATIC_DIR / "index.html"))


def main() -> None:
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
