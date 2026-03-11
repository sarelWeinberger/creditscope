"""
FastAPI application entry point for CreditScope.
"""

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.models import init_db
from backend.db.seed import seed_database
from backend.routers import chat, customers, observability, thinking

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("starting_creditscope")

    # Initialize and seed database
    init_db()
    if os.getenv("SEED_DB", "true").lower() == "true":
        seed_database()
        logger.info("database_seeded")

    yield

    logger.info("shutting_down_creditscope")


app = FastAPI(
    title="CreditScope",
    description="Agentic Credit Scoring with MoE Observability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(customers.router, prefix="/api", tags=["customers"])
app.include_router(observability.router, prefix="/api", tags=["observability"])
app.include_router(thinking.router, prefix="/api", tags=["thinking"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "creditscope"}
