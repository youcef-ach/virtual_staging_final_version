"""Server-Sent Events (SSE) endpoint for real-time pipeline status streaming.

Streams stage status changes for a pipeline run as JSON events. The client
connects via ``EventSource`` (browser-native, no WebSocket needed) and
receives updates every time a stage transitions state.

Event format::

    data: {"run_id": "...", "status": "running", "stages": [...]}

"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import PipelineRun, PipelineStatus
from app.pipeline.orchestrator import get_run, get_run_stages
from app.schemas import StageExecutionOut

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


async def _event_generator(run_id: uuid.UUID, db_factory):
    """Yield SSE events as stage statuses change."""
    last_snapshot: str | None = None
    terminal = {PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.CANCELLED}
    poll_interval = 1.0  # seconds

    while True:
        db: Session = db_factory()
        try:
            run = get_run(db, run_id)
            if run is None:
                yield f"data: {json.dumps({'error': 'run not found'})}\n\n"
                return

            stages = get_run_stages(db, run_id)
            stage_data = [
                StageExecutionOut.model_validate(s).model_dump(mode="json")
                for s in stages
            ]
            snapshot = {
                "run_id": str(run.id),
                "scan_version_id": str(run.scan_version_id),
                "status": run.status.value if hasattr(run.status, "value") else run.status,
                "stages": stage_data,
            }
            snapshot_json = json.dumps(snapshot, default=str)

            # Only emit when something changed
            if snapshot_json != last_snapshot:
                last_snapshot = snapshot_json
                yield f"data: {snapshot_json}\n\n"

            # Stop streaming once the run is terminal
            if run.status in terminal:
                yield f"data: {json.dumps({'event': 'done', 'status': snapshot['status']})}\n\n"
                return
        finally:
            db.close()

        await asyncio.sleep(poll_interval)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Stream pipeline run status as Server-Sent Events.

    Connect via ``new EventSource('/api/pipeline/runs/{id}/events')``.
    Emits JSON snapshots on every state change, closes when terminal.
    """
    run = get_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    from app.db import get_session_factory

    db_factory = get_session_factory()

    return StreamingResponse(
        _event_generator(run_id, db_factory),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx passthrough
        },
    )


@router.get("/runs/active")
async def list_active_runs(
    db: Session = Depends(get_db),
):
    """List pipeline runs in PENDING or RUNNING state."""
    from sqlalchemy import select

    stmt = (
        select(PipelineRun)
        .where(PipelineRun.status.in_([PipelineStatus.PENDING, PipelineStatus.RUNNING]))
        .order_by(PipelineRun.created_at.desc())
    )
    runs = list(db.scalars(stmt).all())
    return runs
