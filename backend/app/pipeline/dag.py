"""DAG definition for the 10-stage per-property pipeline.

Maps stage numbers to names, dependencies, and CLI commands. Provides
dependency-resolution helpers consumed by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class StageDef:
    """Immutable definition of a single pipeline stage."""

    number: int
    name: str
    #: Stage numbers that must be COMPLETED before this stage can start.
    depends_on: tuple[int, ...] = ()
    #: CLI template (not used in simulated mode). Placeholders: {property_dir}
    cli_template: str = ""
    #: Simulated duration in seconds (for demo/test mode).
    sim_duration: float = 2.0


# ── 10-stage DAG definition ──────────────────────────────────────────────
#
# Mirrors PLAN.md §3 "Offline Pipeline Stage Reference":
#
#   Stage 1: Ingest & Validate        → root
#   Stage 2: Pose Extraction & QA     → 1
#   Stage 3: Point Cloud Preprocess   → 1
#   Stage 4: Semantic Segmentation    → 3
#   Stage 5: Plane Fitting            → 3
#   Stage 6: Proxy Mesh Synthesis     → 5, (4 optional → required here)
#   Stage 7: G-Buffer Baking          → 6
#   Stage 8: IBL Pre-computation      → 1
#   Stage 9: Navigation Graph         → 5, 6
#   Stage 10: Compress & Publish      → all (2,4,7,8,9 are the leaf deps)

STAGES: tuple[StageDef, ...] = (
    StageDef(
        number=1,
        name="Ingest & Validate",
        depends_on=(),
        cli_template="python -m vs_pipeline.stages.pose --input {property_dir}",
        sim_duration=1.5,
    ),
    StageDef(
        number=2,
        name="Pose Extraction & QA",
        depends_on=(1,),
        cli_template="python -m vs_pipeline.stages.pose_qa --input {property_dir}",
        sim_duration=3.0,
    ),
    StageDef(
        number=3,
        name="Point Cloud Preprocess",
        depends_on=(1,),
        cli_template="python -m vs_pipeline.stages.pointcloud --input {property_dir}",
        sim_duration=4.0,
    ),
    StageDef(
        number=4,
        name="Semantic Segmentation",
        depends_on=(3,),
        cli_template="python -m vs_pipeline.stages.segmentation --input {property_dir}",
        sim_duration=5.0,
    ),
    StageDef(
        number=5,
        name="Plane Fitting",
        depends_on=(3,),
        cli_template="python -m vs_pipeline.stages.planes fit --input {property_dir}",
        sim_duration=3.5,
    ),
    StageDef(
        number=6,
        name="Proxy Mesh Synthesis",
        depends_on=(5, 4),
        cli_template="python -m vs_pipeline.stages.proxy --input {property_dir}",
        sim_duration=4.0,
    ),
    StageDef(
        number=7,
        name="G-Buffer Baking",
        depends_on=(6,),
        cli_template="python -m vs_pipeline.stages.gbuffer --input {property_dir}",
        sim_duration=6.0,
    ),
    StageDef(
        number=8,
        name="IBL Pre-computation",
        depends_on=(1,),
        cli_template="python -m vs_pipeline.stages.ibl --input {property_dir}",
        sim_duration=5.0,
    ),
    StageDef(
        number=9,
        name="Navigation Graph",
        depends_on=(5, 6),
        cli_template="python -m vs_pipeline.stages.navigation --input {property_dir}",
        sim_duration=2.0,
    ),
    StageDef(
        number=10,
        name="Compress & Publish",
        depends_on=(2, 4, 7, 8, 9),
        cli_template="python -m vs_pipeline.stages.compress publish --input {property_dir} -o published",
        sim_duration=3.0,
    ),
)

# Lookup helpers
STAGE_BY_NUMBER: dict[int, StageDef] = {s.number: s for s in STAGES}
STAGE_NAMES: dict[int, str] = {s.number: s.name for s in STAGES}


def get_stage(number: int) -> StageDef:
    """Return the stage definition for a given stage number."""
    if number not in STAGE_BY_NUMBER:
        raise ValueError(f"Unknown stage number: {number}")
    return STAGE_BY_NUMBER[number]


def get_ready_stages(
    completed: set[int],
    running: set[int],
    failed: set[int],
) -> list[StageDef]:
    """Return stages whose dependencies are all completed and which are
    not yet completed, running, or failed.

    This is the core scheduling algorithm: it determines which stages
    can be started next.
    """
    ready: list[StageDef] = []
    blocked = completed | running | failed
    for stage in STAGES:
        if stage.number in blocked:
            continue
        if all(dep in completed for dep in stage.depends_on):
            ready.append(stage)
    return ready


def is_dag_complete(completed: set[int]) -> bool:
    """Return True when all 10 stages are completed."""
    return len(completed) == len(STAGES)


def get_downstream(stage_number: int) -> list[int]:
    """Return all stage numbers that transitively depend on a given stage."""
    downstream: list[int] = []
    direct = [s.number for s in STAGES if stage_number in s.depends_on]
    for d in direct:
        downstream.append(d)
        downstream.extend(get_downstream(d))
    return sorted(set(downstream))
