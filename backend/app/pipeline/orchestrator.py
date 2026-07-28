"""Pipeline orchestrator: manages the lifecycle of a pipeline run.

Creates PipelineRun + StageExecution rows, drives the DAG loop via the
executor, handles retries, and updates statuses in the database.

The orchestrator is designed to run as a background asyncio task started
by the API layer. It is safe to restart (idempotent stage status checks).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    PipelineRun,
    PipelineStatus,
    ScanVersion,
    StageExecution,
    StageStatus,
)
from app.pipeline.dag import STAGES, get_downstream, get_ready_stages, is_dag_complete
from app.pipeline.executor import SimulatedExecutor, StageExecutorProtocol

logger = logging.getLogger(__name__)

# Global registry of active DAG loop tasks so they can be cancelled.
_active_loops: dict[str, asyncio.Task] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Database helpers ─────────────────────────────────────────────────────


def create_run(db: Session, scan_version_id: uuid.UUID) -> PipelineRun:
    """Create a new PipelineRun with 10 pending StageExecution rows."""
    run = PipelineRun(
        scan_version_id=scan_version_id,
        status=PipelineStatus.PENDING,
    )
    db.add(run)
    db.flush()  # get run.id

    for stage_def in STAGES:
        se = StageExecution(
            run_id=run.id,
            stage_number=stage_def.number,
            stage_name=stage_def.name,
            status=StageStatus.PENDING,
        )
        db.add(se)

    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: uuid.UUID) -> PipelineRun | None:
    return db.get(PipelineRun, run_id)


def get_run_stages(db: Session, run_id: uuid.UUID) -> list[StageExecution]:
    stmt = (
        select(StageExecution)
        .where(StageExecution.run_id == run_id)
        .order_by(StageExecution.stage_number)
    )
    return list(db.scalars(stmt).all())


def get_runs_for_scan(db: Session, scan_version_id: uuid.UUID) -> list[PipelineRun]:
    stmt = (
        select(PipelineRun)
        .where(PipelineRun.scan_version_id == scan_version_id)
        .order_by(PipelineRun.created_at.desc())
    )
    return list(db.scalars(stmt).all())


def get_all_runs(db: Session) -> list[PipelineRun]:
    stmt = select(PipelineRun).order_by(PipelineRun.created_at.desc())
    return list(db.scalars(stmt).all())


def _stage_status_sets(
    stages: list[StageExecution],
) -> tuple[set[int], set[int], set[int]]:
    """Partition stage numbers into completed, running, and failed sets."""
    completed: set[int] = set()
    running: set[int] = set()
    failed: set[int] = set()
    for se in stages:
        if se.status == StageStatus.COMPLETED:
            completed.add(se.stage_number)
        elif se.status == StageStatus.RUNNING:
            running.add(se.stage_number)
        elif se.status == StageStatus.FAILED:
            failed.add(se.stage_number)
    return completed, running, failed


# ── DAG execution loop ───────────────────────────────────────────────────


async def run_dag_loop(
    run_id: uuid.UUID,
    db_factory,  # callable returning a new Session
    executor: StageExecutorProtocol | None = None,
    property_dir: str = "",
) -> None:
    """Drive the DAG to completion.

    Runs stages sequentially (ready stages one at a time to keep the local
    machine responsive). Parallelism can be added via `asyncio.gather` on
    ready stages when using a CeleryExecutor.
    """
    if executor is None:
        executor = SimulatedExecutor(speed_factor=0.3)

    logger.info("[orchestrator] DAG loop started for run %s", run_id)

    while True:
        db = db_factory()
        try:
            run = get_run(db, run_id)
            if run is None:
                logger.error("[orchestrator] Run %s not found — aborting", run_id)
                return

            if run.status in (PipelineStatus.CANCELLED,):
                logger.info("[orchestrator] Run %s is cancelled — exiting", run_id)
                return

            stages = get_run_stages(db, run_id)
            completed, running, failed = _stage_status_sets(stages)

            # Check completion
            if is_dag_complete(completed):
                run.status = PipelineStatus.COMPLETED
                run.updated_at = _utcnow()
                db.commit()
                logger.info("[orchestrator] Run %s COMPLETED ✅", run_id)
                return

            # Check for unrecoverable failure (failed stage with no retry)
            if failed and not running:
                ready = get_ready_stages(completed, running, failed)
                if not ready:
                    run.status = PipelineStatus.FAILED
                    run.updated_at = _utcnow()
                    db.commit()
                    failed_names = [
                        s.stage_name for s in stages if s.status == StageStatus.FAILED
                    ]
                    logger.info(
                        "[orchestrator] Run %s FAILED — blocked stages: %s",
                        run_id,
                        failed_names,
                    )
                    return

            # Find next stage to run
            ready = get_ready_stages(completed, running, failed)
            if not ready:
                # All stages either completed, running, or failed — wait for runners
                logger.debug("[orchestrator] No ready stages — waiting")
                await asyncio.sleep(0.5)
                continue

            # Pick the first ready stage (by number)
            stage_def = ready[0]
            stage_exec = next(
                se for se in stages if se.stage_number == stage_def.number
            )

            # Mark running
            run.status = PipelineStatus.RUNNING
            stage_exec.status = StageStatus.RUNNING
            stage_exec.started_at = _utcnow()
            run.updated_at = _utcnow()
            db.commit()
        finally:
            db.close()

        # Execute outside of DB session
        result = await executor.execute(stage_def, str(run_id), property_dir)

        # Update result
        db = db_factory()
        try:
            stage_exec = db.get(StageExecution, stage_exec.id)
            run = get_run(db, run_id)
            if stage_exec is None or run is None:
                return

            stage_exec.finished_at = _utcnow()
            stage_exec.log_path = result.log_path

            if result.success:
                stage_exec.status = StageStatus.COMPLETED
                logger.info(
                    "[orchestrator] Stage %d (%s) → COMPLETED (%.1fs)",
                    stage_def.number,
                    stage_def.name,
                    result.duration_s,
                )
            else:
                stage_exec.status = StageStatus.FAILED
                stage_exec.error_message = result.error_message
                logger.warning(
                    "[orchestrator] Stage %d (%s) → FAILED: %s",
                    stage_def.number,
                    stage_def.name,
                    result.error_message,
                )

            run.updated_at = _utcnow()
            db.commit()
        finally:
            db.close()


# ── Public API ───────────────────────────────────────────────────────────


def start_run(
    db: Session,
    scan_version_id: uuid.UUID,
    db_factory,
    executor: StageExecutorProtocol | None = None,
    property_dir: str = "",
) -> PipelineRun:
    """Create a pipeline run and start the DAG loop as a background task."""
    run = create_run(db, scan_version_id)
    run_id_str = str(run.id)

    task = asyncio.ensure_future(
        run_dag_loop(run.id, db_factory, executor, property_dir)
    )
    _active_loops[run_id_str] = task

    # Cleanup when done
    def _cleanup(t: asyncio.Task) -> None:
        _active_loops.pop(run_id_str, None)

    task.add_done_callback(_cleanup)

    return run


def retry_stage(
    db: Session,
    run_id: uuid.UUID,
    stage_number: int,
    db_factory,
    executor: StageExecutorProtocol | None = None,
    property_dir: str = "",
) -> StageExecution | None:
    """Reset a failed stage (and its downstream dependents) to pending,
    then restart the DAG loop.
    """
    stages = get_run_stages(db, run_id)
    target = next((se for se in stages if se.stage_number == stage_number), None)
    if target is None:
        return None
    if target.status not in (StageStatus.FAILED, StageStatus.CANCELLED):
        return None  # can only retry failed/cancelled stages

    # Reset target and all downstream stages
    downstream = get_downstream(stage_number)
    for se in stages:
        if se.stage_number == stage_number or se.stage_number in downstream:
            if se.status in (StageStatus.FAILED, StageStatus.CANCELLED, StageStatus.SKIPPED):
                se.status = StageStatus.PENDING
                se.started_at = None
                se.finished_at = None
                se.error_message = None
                se.log_path = None

    # Reset run to running
    run = get_run(db, run_id)
    if run:
        run.status = PipelineStatus.RUNNING
        run.updated_at = _utcnow()

    db.commit()

    # Cancel any existing DAG loop and start a new one
    run_id_str = str(run_id)
    old_task = _active_loops.pop(run_id_str, None)
    if old_task and not old_task.done():
        old_task.cancel()

    task = asyncio.ensure_future(
        run_dag_loop(run_id, db_factory, executor, property_dir)
    )
    _active_loops[run_id_str] = task

    def _cleanup(t: asyncio.Task) -> None:
        _active_loops.pop(run_id_str, None)

    task.add_done_callback(_cleanup)

    db.refresh(target)
    return target


def cancel_run(db: Session, run_id: uuid.UUID) -> PipelineRun | None:
    """Cancel a running pipeline run."""
    run = get_run(db, run_id)
    if run is None:
        return None
    if run.status not in (PipelineStatus.PENDING, PipelineStatus.RUNNING):
        return run  # already terminal

    run.status = PipelineStatus.CANCELLED
    run.updated_at = _utcnow()

    # Cancel any running stages
    stages = get_run_stages(db, run_id)
    for se in stages:
        if se.status in (StageStatus.PENDING, StageStatus.RUNNING):
            se.status = StageStatus.CANCELLED
            se.finished_at = _utcnow()

    db.commit()

    # Cancel background task
    run_id_str = str(run_id)
    old_task = _active_loops.pop(run_id_str, None)
    if old_task and not old_task.done():
        old_task.cancel()

    db.refresh(run)
    return run
