"""Stage executor: runs a pipeline stage as a subprocess or in simulated mode.

The executor is a thin abstraction so we can swap between:
  - SimulatedExecutor: fake execution with configurable delays (for UI testing)
  - LocalExecutor: real subprocess invocation (for local development)
  - (Future) CeleryExecutor: dispatches to Redis-backed worker pods

Both executors implement the same async interface.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Protocol

from app.pipeline.dag import StageDef

logger = logging.getLogger(__name__)


class ExecutionResult:
    """Result of executing a single stage."""

    __slots__ = ("success", "error_message", "log_path", "duration_s")

    def __init__(
        self,
        success: bool,
        error_message: str | None = None,
        log_path: str | None = None,
        duration_s: float = 0.0,
    ) -> None:
        self.success = success
        self.error_message = error_message
        self.log_path = log_path
        self.duration_s = duration_s


class StageExecutorProtocol(Protocol):
    """Protocol for stage executors."""

    async def execute(
        self,
        stage: StageDef,
        run_id: str,
        property_dir: str,
    ) -> ExecutionResult: ...


class SimulatedExecutor:
    """Fake executor that simulates stage execution with configurable delays.

    Used for testing the dashboard UI and orchestrator logic without running
    real pipeline stages. Optionally injects random failures for stress testing.
    """

    def __init__(
        self,
        fail_probability: float = 0.0,
        speed_factor: float = 1.0,
    ) -> None:
        self.fail_probability = fail_probability
        self.speed_factor = speed_factor

    async def execute(
        self,
        stage: StageDef,
        run_id: str,
        property_dir: str,
    ) -> ExecutionResult:
        duration = stage.sim_duration * self.speed_factor
        logger.info(
            "[sim] Stage %d (%s) starting — %.1fs simulated",
            stage.number,
            stage.name,
            duration,
        )
        await asyncio.sleep(duration)

        # Optional random failure injection
        if self.fail_probability > 0 and random.random() < self.fail_probability:
            msg = f"Simulated failure in stage {stage.number} ({stage.name})"
            logger.warning("[sim] %s", msg)
            return ExecutionResult(success=False, error_message=msg, duration_s=duration)

        logger.info("[sim] Stage %d (%s) completed", stage.number, stage.name)
        return ExecutionResult(success=True, duration_s=duration)


class LocalExecutor:
    """Runs pipeline stages as local subprocesses.

    Captures stdout/stderr to per-stage log files under
    ``{log_root}/{run_id}/{stage_name}.log``.
    """

    def __init__(self, log_root: Path | None = None, python_exe: str | None = None) -> None:
        self.log_root = log_root or Path("data/pipeline_logs")
        self.python_exe = python_exe or "python"

    async def execute(
        self,
        stage: StageDef,
        run_id: str,
        property_dir: str,
    ) -> ExecutionResult:
        if not stage.cli_template:
            return ExecutionResult(
                success=False,
                error_message=f"No CLI command configured for stage {stage.number}",
            )

        cmd = stage.cli_template.format(property_dir=property_dir)
        # Replace generic 'python' with configured executable
        if cmd.startswith("python "):
            cmd = f"{self.python_exe} {cmd[7:]}"

        log_dir = self.log_root / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = stage.name.lower().replace(" ", "_").replace("&", "and")
        log_file = log_dir / f"stage_{stage.number:02d}_{safe_name}.log"

        logger.info(
            "[local] Stage %d (%s) executing: %s", stage.number, stage.name, cmd
        )
        start = time.monotonic()

        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=lf,
                    stderr=asyncio.subprocess.STDOUT,
                )
                returncode = await proc.wait()
        except Exception as exc:
            duration = time.monotonic() - start
            return ExecutionResult(
                success=False,
                error_message=f"Subprocess launch failed: {exc}",
                log_path=str(log_file),
                duration_s=duration,
            )

        duration = time.monotonic() - start

        if returncode == 0:
            logger.info(
                "[local] Stage %d (%s) completed in %.1fs",
                stage.number,
                stage.name,
                duration,
            )
            return ExecutionResult(
                success=True, log_path=str(log_file), duration_s=duration
            )
        else:
            # Read last 500 chars of log for error summary
            try:
                tail = log_file.read_text(encoding="utf-8")[-500:]
            except Exception:
                tail = "(could not read log)"
            msg = f"Exit code {returncode}. Log tail: {tail}"
            logger.error("[local] Stage %d (%s) FAILED: %s", stage.number, stage.name, msg)
            return ExecutionResult(
                success=False,
                error_message=msg,
                log_path=str(log_file),
                duration_s=duration,
            )
