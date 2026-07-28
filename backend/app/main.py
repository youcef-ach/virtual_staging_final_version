"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401
from app.api.auth import router as auth_router
from app.api.catalog import router as catalog_router
from app.api.pipeline import router as pipeline_router
from app.api.health import router as health_router
from app.api.properties import router as properties_router
from app.api.scans import router as scans_router
from app.api.share import router as share_router
from app.api.sse import router as sse_router
from app.config import get_settings
from app.db import Base, get_engine


def create_app() -> FastAPI:
    settings = get_settings()
    fastapi_app = FastAPI(
        title="Virtual Staging Platform API",
        version=settings.version,
    )

    # CORS for dashboard dev server (localhost:5174) and viewer (localhost:5173).
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auto-create tables for local development
    try:
        Base.metadata.create_all(get_engine())
    except Exception:
        pass

    fastapi_app.include_router(health_router)
    fastapi_app.include_router(auth_router, prefix="/api")
    fastapi_app.include_router(properties_router, prefix="/api")
    fastapi_app.include_router(scans_router, prefix="/api")
    fastapi_app.include_router(catalog_router, prefix="/api")
    fastapi_app.include_router(pipeline_router, prefix="/api")
    fastapi_app.include_router(sse_router, prefix="/api")

    # Wire Prometheus monitoring (metrics available at /metrics)
    try:
        from app.monitoring import setup_monitoring
        setup_monitoring(fastapi_app)
    except ImportError:
        pass # prometheus_fastapi_instrumentator might not be in dev deps

    return fastapi_app


app = create_app()
