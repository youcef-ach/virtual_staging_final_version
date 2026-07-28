"""Pipeline orchestration REST API.

Endpoints:
  POST /api/pipeline/run              — start a new pipeline run
  GET  /api/pipeline/runs             — list all runs (optional ?scan_version_id filter)
  GET  /api/pipeline/runs/{run_id}    — detailed run with all stage statuses
  POST /api/pipeline/runs/{run_id}/stages/{stage_number}/retry — retry a failed stage
  POST /api/pipeline/runs/{run_id}/cancel — cancel a running pipeline
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ScanVersion
from app.pipeline import orchestrator
from app.pipeline.executor import SimulatedExecutor
from app.schemas import (
    PipelineRunCreate,
    PipelineRunDetail,
    PipelineRunOut,
    StageExecutionOut,
)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _get_db_factory():
    """Return a callable that creates new DB sessions for background tasks."""
    from app.db import get_session_factory

    factory = get_session_factory()
    return factory


@router.post("/run", response_model=PipelineRunOut, status_code=201)
async def start_pipeline_run(
    body: PipelineRunCreate,
    db: Session = Depends(get_db),
):
    """Create and start a new pipeline run for a scan version."""
    # Validate scan version exists
    sv = db.get(ScanVersion, body.scan_version_id)
    if sv is None:
        raise HTTPException(status_code=404, detail="Scan version not found")

    # Use LocalExecutor to run the real pipeline subprocesses
    from app.pipeline.executor import LocalExecutor
    executor = LocalExecutor()
    db_factory = _get_db_factory()

    run = orchestrator.start_run(
        db=db,
        scan_version_id=body.scan_version_id,
        db_factory=db_factory,
        executor=executor,
    )
    return run


@router.get("/runs", response_model=list[PipelineRunOut])
async def list_pipeline_runs(
    scan_version_id: uuid.UUID | None = None,
    db: Session = Depends(get_db),
):
    """List pipeline runs, optionally filtered by scan_version_id."""
    if scan_version_id:
        return orchestrator.get_runs_for_scan(db, scan_version_id)
    return orchestrator.get_all_runs(db)


@router.get("/runs/{run_id}", response_model=PipelineRunDetail)
async def get_pipeline_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Get detailed pipeline run status with all stage executions."""
    run = orchestrator.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    stages = orchestrator.get_run_stages(db, run_id)
    stage_outs = [StageExecutionOut.model_validate(s) for s in stages]

    return PipelineRunDetail(
        id=run.id,
        scan_version_id=run.scan_version_id,
        status=run.status.value if hasattr(run.status, "value") else run.status,
        created_at=run.created_at,
        updated_at=run.updated_at,
        stages=stage_outs,
    )


@router.post("/runs/{run_id}/stages/{stage_number}/retry", response_model=StageExecutionOut)
async def retry_stage(
    run_id: uuid.UUID,
    stage_number: int,
    db: Session = Depends(get_db),
):
    """Retry a failed stage. Resets downstream dependents to pending."""
    run = orchestrator.get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    from app.pipeline.executor import LocalExecutor
    executor = LocalExecutor()
    db_factory = _get_db_factory()

    stage = orchestrator.retry_stage(
        db=db,
        run_id=run_id,
        stage_number=stage_number,
        db_factory=db_factory,
        executor=executor,
    )
    if stage is None:
        raise HTTPException(
            status_code=400,
            detail=f"Stage {stage_number} cannot be retried (not failed/cancelled)",
        )
    return stage


@router.post("/runs/{run_id}/cancel", response_model=PipelineRunOut)
async def cancel_pipeline_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Cancel a running or pending pipeline run."""
    run = orchestrator.cancel_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    return run
