"""
CreditScope FastAPI application entry point.
Handles startup, routing, CORS, logging, and static file serving.
"""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

# ─── Logging Setup ─────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# ─── Database Setup ────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./creditscope.db")

engine = create_async_engine(
    DATABASE_URL,
    echo=os.environ.get("DB_ECHO", "false").lower() == "true",
    pool_pre_ping=True,
)

SessionFactory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    """Dependency: yield an async database session."""
    async with SessionFactory() as session:
        yield session


# ─── App Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown tasks."""
    log.info("creditscope.startup", database_url=DATABASE_URL)

    # Import here to avoid circular imports at module level
    from db.models import Base
    from db.seed import seed

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    log.info("creditscope.db_tables_created")

    # Seed database if empty
    try:
        await seed(engine)
        log.info("creditscope.db_seeded")
    except Exception as e:
        log.warning("creditscope.seed_skipped", reason=str(e))

    log.info("creditscope.ready")
    yield

    # Shutdown
    await engine.dispose()
    log.info("creditscope.shutdown")


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CreditScope API",
    description="AI-powered credit analysis platform with MoE observability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
cors_origins = [o.strip() for o in cors_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include Routers ───────────────────────────────────────────────────────────

# Import routers after app creation to avoid circular deps
from routers import chat, customers, observability, thinking

app.include_router(chat.router, prefix="/api")
app.include_router(customers.router, prefix="/api")
app.include_router(observability.router, prefix="/api")
app.include_router(thinking.router, prefix="/api")


# ─── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "creditscope-backend",
        "version": "1.0.0",
    }


@app.get("/api/health")
async def api_health():
    return {"status": "ok"}


# ─── Static Files (Frontend) ───────────────────────────────────────────────────

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React SPA for all non-API routes."""
        if full_path.startswith("api/"):
            return JSONResponse({"error": "Not found"}, status_code=404)
        index = frontend_dist / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse({"error": "Frontend not built"}, status_code=404)
else:
    log.info("creditscope.frontend_not_built", path=str(frontend_dist))


# ─── Global Exception Handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error("creditscope.unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ─── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
