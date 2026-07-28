"""Prometheus monitoring integration for the FastAPI application.

Provides system and pipeline-level metrics, exposed at /metrics.
"""

from fastapi import FastAPI
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

# ── Custom Pipeline Metrics ────────────────────────────────────────────────

pipeline_runs_total = Counter(
    "vs_pipeline_runs_total",
    "Total number of pipeline runs started",
    ["status"]
)

pipeline_stage_duration_seconds = Histogram(
    "vs_pipeline_stage_duration_seconds",
    "Duration of pipeline stages in seconds",
    ["stage_name"]
)

pipeline_stage_failures_total = Counter(
    "vs_pipeline_stage_failures_total",
    "Total number of pipeline stage failures",
    ["stage_name"]
)

active_pipeline_runs = Gauge(
    "vs_active_pipeline_runs",
    "Number of currently running pipelines"
)

# ── Setup ──────────────────────────────────────────────────────────────────

def setup_monitoring(app: FastAPI) -> None:
    """Initialize Prometheus metrics and instrument the FastAPI app."""
    # The instrumentator adds default metrics (requests, latency, etc.)
    # and mounts the /metrics endpoint.
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*admin.*", "/metrics"],
        env_var_name="ENABLE_METRICS",
    )
    
    instrumentator.instrument(app).expose(app, include_in_schema=False, tags=["monitoring"])
