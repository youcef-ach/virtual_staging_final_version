"""Stage 5 (G2) — plane fitting & boundary extraction (local-to-global).

Design (locked in PLAN.md changelog 2026-07-12):
1. Per-sweep plane proposals: each cached sweep cloud (15 mm, sensor-oriented
   normals) is voxeled to a ~5 cm coarse copy; sequential RANSAC extracts
   dominant planes, gated by point-normal agreement, oriented toward the
   sensor via the stored normals, classified floor/ceiling/wall by normal.
2. Local-to-global merge: proposals are clustered with union-find using
   normal angle + plane offset + PADDED BBOX OVERLAP — distant coplanar
   surfaces never merge (the “plane bleeding” guard).
3. Coarse-to-fine refinement: for every global plane, the 15 mm points of
   all member sweeps within threshold are gathered (capped) and a
   least-squares SVD fit produces the authoritative coefficients.
4. Wall thickness: parallel planes with opposing normals < 300 mm apart are
   paired into one wall entity with an explicit thickness.
5. Boundaries: inliers projected into the plane frame -> occupancy grid ->
   external contour -> Douglas-Peucker simplification (OpenCV). First-pass
   precision; exact corner sealing by plane intersection happens in G3.
6. Residuals: coarse points unclaimed by any plane are clustered (DBSCAN)
   into non-structural clusters with bounding boxes — consumed by G3 detail
   meshes now, SAM prompts (G6) and Year-2 room emptying later.

Outputs (runtime Y-up frame, Coordinate Lock preserved end to end):
    structural_planes.json    plane class/coefficients/boundary/thickness
    residual_clusters.json    non-planar cluster boxes
    planes_review.ply         color-coded inliers + gray residuals (review)

CLI:
    python -m vs_pipeline.stages.planes fit --cache preprocessed/sweep_cache -o planes_out
    python -m vs_pipeline.stages.planes fit --cache preprocessed/sweep_cache \
        --sweeps "30-40" -o planes_room    # one-room validation slice
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PLANES_SCHEMA_VERSION = 1

#: Coarse detection resolution (proposals + residuals). Refinement uses 15 mm.
COARSE_VOXEL_M = 0.05
RANSAC_DISTANCE_M = 0.02
RANSAC_ITERATIONS = 1000
#: ~1 m^2 of surface at 5 cm grid density.
MIN_PROPOSAL_INLIERS = 400
MAX_PLANES_PER_SWEEP = 12
#: Point normals must agree with the plane normal at least this much.
NORMAL_DOT_MIN = 0.7
#: |n_y| above -> horizontal (floor/ceiling); below WALL_NY_MAX -> wall.
HORIZONTAL_NY = 0.92
WALL_NY_MAX = 0.35
#: Proposal clustering: max normal angle, max offset, bbox pad.
MERGE_MAX_ANGLE_DEG = 5.0
MERGE_MAX_OFFSET_M = 0.03
MERGE_BBOX_PAD_M = 0.3
#: Refinement gathering.
REFINE_DISTANCE_M = 0.02
REFINE_MAX_POINTS_PER_PLANE = 150_000
#: Wall pairing (opposing normals).
WALL_PAIR_MAX_GAP_M = 0.3
WALL_PAIR_MIN_OPPOSE = 0.985
#: Second merge pass on refined coefficients (accurate after the 15 mm refit).
REFINED_MERGE_MAX_ANGLE_DEG = 3.0
REFINED_MERGE_MAX_OFFSET_M = 0.025
#: Storey-level detection: large horizontals define floor/ceiling levels;
#: up-facing planes further than this above a floor level are furniture tops.
LEVEL_MIN_AREA_M2 = 8.0
LEVEL_TOLERANCE_M = 0.25
#: Storey-level interiority: a level-defining floor needs (a) plan-view UNION
#: ceiling cover >= this fraction of its bbox at a plausible height above and
#: (b) tripod evidence when sweep heights are available -- excludes outdoor
#: terrain on facade-inclusive scans.
INTERIOR_COVER_MIN = 0.6
CEILING_ABOVE_MIN_M = 1.8
CEILING_ABOVE_MAX_M = 4.5
COVER_CELL_M = 0.25
#: Matterport tripods stand ~1.4 m above the floor they rest on.
TRIPOD_ABOVE_MIN_M = 1.0
TRIPOD_ABOVE_MAX_M = 2.0
#: Boundary rasterization.
BOUNDARY_GRID_M = 0.05
BOUNDARY_DP_EPS_PX = 1.5
#: Residual clustering.
RESIDUAL_DISTANCE_M = 0.05
RESIDUAL_DBSCAN_EPS_M = 0.15
RESIDUAL_DBSCAN_MIN_POINTS = 20
RESIDUAL_MIN_CLUSTER_POINTS = 40


@dataclass
class PlaneProposal:
    sweep_index: int
    cls: str
    normal: np.ndarray  # unit, oriented toward the sensor (interior side)
    d: float  # plane: normal . p + d = 0
    inlier_count: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray


@dataclass
class GlobalPlane:
    plane_id: str
    cls: str
    normal: np.ndarray
    d: float
    inlier_count: int
    area_m2: float
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    member_sweeps: list[int]
    elevation_y: float | None = None  # horizontal planes only
    thickness_m: float | None = None  # paired walls only
    paired_with: str | None = None
    frame_origin: np.ndarray | None = None
    frame_u: np.ndarray | None = None
    frame_v: np.ndarray | None = None
    boundary_2d: list[list[float]] = field(default_factory=list)
    boundary_3d: list[list[float]] = field(default_factory=list)
    points: np.ndarray | None = None  # transient (review export only)


def fit_plane_lsq(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through points via SVD. Returns (unit normal, d)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("need at least 3 points to fit a plane")
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    d = -float(normal @ centroid)
    return normal, d


def classify_normal(normal: np.ndarray) -> str:
    """floor / ceiling / wall / other from an interior-oriented unit normal."""
    ny = float(normal[1])
    if ny > HORIZONTAL_NY:
        return "floor"  # surface faces up toward the sensor
    if ny < -HORIZONTAL_NY:
        return "ceiling"  # surface faces down toward the sensor
    if abs(ny) < WALL_NY_MAX:
        return "wall"
    return "other"


def plane_frame(
    normal: np.ndarray, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed in-plane (u, v) frame; v is the up-most in-plane axis."""
    ref = (
        np.array([0.0, 1.0, 0.0])
        if abs(float(normal[1])) < 0.9
        else np.array([0.0, 0.0, 1.0])
    )
    v = ref - normal * float(ref @ normal)
    v = v / np.linalg.norm(v)
    u = np.cross(v, normal)
    return origin.astype(np.float64), u, v


def _bboxes_overlap(
    min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray, pad: float
) -> bool:
    return bool(np.all(min_a - pad <= max_b) and np.all(min_b - pad <= max_a))


def propose_sweep_planes(cloud: Any, sweep_index: int) -> list[PlaneProposal]:
    """Sequential RANSAC on a coarse copy of one sweep cloud (with normals)."""
    coarse = cloud.voxel_down_sample(COARSE_VOXEL_M)
    proposals: list[PlaneProposal] = []
    remaining = coarse
    for _ in range(MAX_PLANES_PER_SWEEP):
        if len(remaining.points) < MIN_PROPOSAL_INLIERS:
            break
        model, inlier_idx = remaining.segment_plane(
            distance_threshold=RANSAC_DISTANCE_M,
            ransac_n=3,
            num_iterations=RANSAC_ITERATIONS,
        )
        if len(inlier_idx) < MIN_PROPOSAL_INLIERS:
            break
        normal = np.asarray(model[:3], dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        d = float(model[3])

        pts = np.asarray(remaining.points)[inlier_idx]
        nrm = np.asarray(remaining.normals)[inlier_idx]
        # Orient the plane toward the sensor using the stored point normals.
        if float(np.mean(nrm @ normal)) < 0.0:
            normal, d = -normal, -d
        # Gate: keep only inliers whose own normal agrees with the plane.
        agree = (nrm @ normal) > NORMAL_DOT_MIN
        kept = pts[agree]
        if kept.shape[0] >= MIN_PROPOSAL_INLIERS:
            proposals.append(
                PlaneProposal(
                    sweep_index=sweep_index,
                    cls=classify_normal(normal),
                    normal=normal,
                    d=d,
                    inlier_count=int(kept.shape[0]),
                    centroid=kept.mean(axis=0),
                    bbox_min=kept.min(axis=0),
                    bbox_max=kept.max(axis=0),
                )
            )
        remaining = remaining.select_by_index(inlier_idx, invert=True)
    return proposals


def _compatible(a: PlaneProposal, b: PlaneProposal) -> bool:
    if float(a.normal @ b.normal) < np.cos(np.radians(MERGE_MAX_ANGLE_DEG)):
        return False  # different orientation (opposing wall sides stay apart)
    if abs(a.d - b.d) > MERGE_MAX_OFFSET_M:
        return False
    return _bboxes_overlap(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max, MERGE_BBOX_PAD_M)


def cluster_proposals(proposals: list[PlaneProposal]) -> list[list[PlaneProposal]]:
    """Union-find clustering; the bbox-overlap term is the plane-bleeding guard."""
    n = len(proposals)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _compatible(proposals[i], proposals[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, list[PlaneProposal]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(proposals[i])
    return list(groups.values())


def _load_sweep_cloud(cache_dir: Path, index: int) -> Any:
    import open3d as o3d  # local import

    return o3d.io.read_point_cloud(str(cache_dir / f"scan_{index:03d}.ply"))


def refine_planes(
    clusters: list[list[PlaneProposal]], cache_dir: Path
) -> list[GlobalPlane]:
    """Coarse-to-fine: refit every global plane on 15 mm member-sweep inliers."""
    rng = np.random.default_rng(0)
    accum: list[list[np.ndarray]] = [[] for _ in clusters]
    counts = [0] * len(clusters)

    sweep_to_clusters: dict[int, list[int]] = {}
    for ci, cluster in enumerate(clusters):
        for prop in cluster:
            sweep_to_clusters.setdefault(prop.sweep_index, []).append(ci)

    for sweep_index, cluster_ids in sorted(sweep_to_clusters.items()):
        cloud = _load_sweep_cloud(cache_dir, sweep_index)
        pts = np.asarray(cloud.points)
        nrm = np.asarray(cloud.normals)
        for ci in set(cluster_ids):
            cluster = clusters[ci]
            # Cluster-average plane (weighted by inliers).
            w = np.array([p.inlier_count for p in cluster], dtype=np.float64)
            navg = np.sum([p.normal * wi for p, wi in zip(cluster, w)], axis=0)
            navg = navg / np.linalg.norm(navg)
            davg = float(np.average([p.d for p in cluster], weights=w))
            bmin = np.min([p.bbox_min for p in cluster], axis=0) - MERGE_BBOX_PAD_M
            bmax = np.max([p.bbox_max for p in cluster], axis=0) + MERGE_BBOX_PAD_M

            near = np.abs(pts @ navg + davg) < REFINE_DISTANCE_M
            inside = np.all((pts >= bmin) & (pts <= bmax), axis=1)
            agree = (nrm @ navg) > NORMAL_DOT_MIN
            sel = pts[near & inside & agree]
            if sel.shape[0] == 0:
                continue
            budget = REFINE_MAX_POINTS_PER_PLANE - counts[ci]
            if budget <= 0:
                continue
            if sel.shape[0] > budget:
                sel = sel[rng.choice(sel.shape[0], budget, replace=False)]
            accum[ci].append(sel)
            counts[ci] += sel.shape[0]

    planes: list[GlobalPlane] = []
    class_counters: dict[str, int] = {}
    for ci, cluster in enumerate(clusters):
        if not accum[ci]:
            continue
        points = np.vstack(accum[ci])
        normal, d = fit_plane_lsq(points)
        # Preserve the interior orientation voted by the proposals.
        w = np.array([p.inlier_count for p in cluster], dtype=np.float64)
        navg = np.sum([p.normal * wi for p, wi in zip(cluster, w)], axis=0)
        if float(normal @ navg) < 0.0:
            normal, d = -normal, -d
        cls = classify_normal(normal)
        k = class_counters.get(cls, 0)
        class_counters[cls] = k + 1
        plane = GlobalPlane(
            plane_id=f"{cls}_{k:02d}",
            cls=cls,
            normal=normal,
            d=d,
            inlier_count=int(points.shape[0]),
            area_m2=0.0,  # filled by boundary extraction
            centroid=points.mean(axis=0),
            bbox_min=points.min(axis=0),
            bbox_max=points.max(axis=0),
            member_sweeps=sorted({p.sweep_index for p in cluster}),
            points=points,
        )
        if cls in ("floor", "ceiling"):
            plane.elevation_y = float(plane.centroid[1])
        planes.append(plane)
    return planes


def merge_refined_planes(planes: list[GlobalPlane]) -> list[GlobalPlane]:
    """Second merge pass on refined (accurate) coefficients.

    Proposal-level merging is deliberately conservative because raw per-sweep
    RANSAC estimates wobble (especially in glass-heavy rooms), which
    under-merges. After the 15 mm refit the coefficients are trustworthy, so
    compatible planes merge here — with the same padded-bbox bleeding guard.
    """
    n = len(planes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cos_max = np.cos(np.radians(REFINED_MERGE_MAX_ANGLE_DEG))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = planes[i], planes[j]
            if float(a.normal @ b.normal) < cos_max:
                continue
            if abs(a.d - b.d) > REFINED_MERGE_MAX_OFFSET_M:
                continue
            if not _bboxes_overlap(
                a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max, MERGE_BBOX_PAD_M
            ):
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

    rng = np.random.default_rng(1)
    groups: dict[int, list[GlobalPlane]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(planes[i])

    merged: list[GlobalPlane] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        points = np.vstack([p.points for p in group if p.points is not None])
        if points.shape[0] > REFINE_MAX_POINTS_PER_PLANE:
            points = points[
                rng.choice(points.shape[0], REFINE_MAX_POINTS_PER_PLANE, replace=False)
            ]
        normal, d = fit_plane_lsq(points)
        navg = np.sum([p.normal * p.inlier_count for p in group], axis=0)
        if float(normal @ navg) < 0.0:
            normal, d = -normal, -d
        base = max(group, key=lambda p: p.inlier_count)
        base.normal, base.d = normal, d
        base.cls = classify_normal(normal)
        base.points = points
        base.inlier_count = int(sum(p.inlier_count for p in group))
        base.centroid = points.mean(axis=0)
        base.bbox_min = points.min(axis=0)
        base.bbox_max = points.max(axis=0)
        base.member_sweeps = sorted({s for p in group for s in p.member_sweeps})
        base.elevation_y = (
            float(base.centroid[1]) if base.cls in ("floor", "ceiling") else None
        )
        merged.append(base)
    return merged


def _has_ceiling_cover(floor: GlobalPlane, ceilings: list[GlobalPlane]) -> bool:
    """Interiority test: the plan-view UNION of ceilings CEILING_ABOVE_MIN/MAX_M
    above must cover >= INTERIOR_COVER_MIN of the floor's bbox. Union on a
    coarse grid -- summing per-ceiling intersections double-counts overlapping
    ceiling patches, which is exactly how outdoor terrain slipped through v1."""
    fx0, fz0 = float(floor.bbox_min[0]), float(floor.bbox_min[2])
    fx1, fz1 = float(floor.bbox_max[0]), float(floor.bbox_max[2])
    nx = max(int(np.ceil((fx1 - fx0) / COVER_CELL_M)), 1)
    nz = max(int(np.ceil((fz1 - fz0) / COVER_CELL_M)), 1)
    covered = np.zeros((nx, nz), dtype=bool)
    for c in ceilings:
        dy = float(c.centroid[1]) - float(floor.centroid[1])
        if not (CEILING_ABOVE_MIN_M <= dy <= CEILING_ABOVE_MAX_M):
            continue
        ix0 = max(int(np.floor((float(c.bbox_min[0]) - fx0) / COVER_CELL_M)), 0)
        iz0 = max(int(np.floor((float(c.bbox_min[2]) - fz0) / COVER_CELL_M)), 0)
        ix1 = min(int(np.ceil((float(c.bbox_max[0]) - fx0) / COVER_CELL_M)), nx)
        iz1 = min(int(np.ceil((float(c.bbox_max[2]) - fz0) / COVER_CELL_M)), nz)
        if ix1 > ix0 and iz1 > iz0:
            covered[ix0:ix1, iz0:iz1] = True
    return float(covered.mean()) >= INTERIOR_COVER_MIN


def storey_floor_levels(
    planes: list[GlobalPlane], sweep_heights: list[float] | None = None
) -> list[float]:
    """Storey floor elevations from INTERIOR floors only.

    Interiority = union ceiling cover (_has_ceiling_cover) AND, when sweep
    heights are supplied, tripod evidence: some sweep position stands
    TRIPOD_ABOVE_MIN/MAX_M above the floor (scanners always stand on real
    interior floors; outdoor terrain has no such sweep). The 30%-of-largest
    area bar is computed over interior floors, so terrain can neither define
    a level nor inflate the bar. Falls back to all floors when nothing passes
    (e.g. ceiling-less synthetic data).
    """
    floors = [p for p in planes if p.cls == "floor"]
    ceilings = [p for p in planes if p.cls == "ceiling"]

    def tripod_ok(f: GlobalPlane) -> bool:
        if not sweep_heights:
            return True
        e = f.elevation_y if f.elevation_y is not None else float(f.centroid[1])
        return any(
            TRIPOD_ABOVE_MIN_M <= h - e <= TRIPOD_ABOVE_MAX_M
            for h in sweep_heights
        )

    interior = [
        f for f in floors if tripod_ok(f) and _has_ceiling_cover(f, ceilings)
    ]
    candidates = interior if interior else floors
    max_area = max((p.area_m2 for p in candidates), default=0.0)
    area_min = max(LEVEL_MIN_AREA_M2, 0.3 * max_area)
    return [
        p.elevation_y
        for p in candidates
        if p.area_m2 >= area_min and p.elevation_y is not None
    ]


def reclassify_and_rename(
    planes: list[GlobalPlane], sweep_heights: list[float] | None = None
) -> None:
    """Storey-aware reclassification, then stable ids.

    Interior floors define the storey levels (storey_floor_levels). Up-facing
    planes further than LEVEL_TOLERANCE_M from every level are demoted:
    above the lowest level -> "surface" (furniture tops); below it -> "other"
    (outdoor terrain). Down-facing planes far from every ceiling level ->
    "other". Ids are rebuilt afterwards (area order).
    """
    levels = storey_floor_levels(planes, sweep_heights)
    if levels:
        lowest = min(levels)
        for p in [q for q in planes if q.cls == "floor"]:
            if p.elevation_y is None:
                continue
            if min(abs(p.elevation_y - lv) for lv in levels) > LEVEL_TOLERANCE_M:
                p.cls = "surface" if p.elevation_y > lowest else "other"
    ceilings = [p for p in planes if p.cls == "ceiling"]
    max_ceiling_area = max((p.area_m2 for p in ceilings), default=0.0)
    clevel_area_min = max(LEVEL_MIN_AREA_M2, 0.3 * max_ceiling_area)
    clevels = [
        p.elevation_y
        for p in ceilings
        if p.area_m2 >= clevel_area_min and p.elevation_y is not None
    ]
    if clevels:
        for p in ceilings:
            if p.elevation_y is None:
                continue
            if min(abs(p.elevation_y - lv) for lv in clevels) > LEVEL_TOLERANCE_M:
                p.cls = "other"
    counters: dict[str, int] = {}
    for p in sorted(planes, key=lambda q: -q.area_m2):
        k = counters.get(p.cls, 0)
        counters[p.cls] = k + 1
        p.plane_id = f"{p.cls}_{k:02d}"


def pair_walls(planes: list[GlobalPlane]) -> None:
    """Group parallel, opposing-normal planes < 300 mm apart into wall entities."""
    walls = [p for p in planes if p.cls == "wall"]
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            a, b = walls[i], walls[j]
            if a.paired_with or b.paired_with:
                continue
            if float(a.normal @ b.normal) > -WALL_PAIR_MIN_OPPOSE:
                continue  # not opposing
            gap = abs(float(a.normal @ b.centroid + a.d))
            if gap > WALL_PAIR_MAX_GAP_M:
                continue
            if not _bboxes_overlap(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max, MERGE_BBOX_PAD_M):
                continue
            a.thickness_m = b.thickness_m = round(gap, 4)
            a.paired_with, b.paired_with = b.plane_id, a.plane_id


def extract_boundary(plane: GlobalPlane) -> None:
    """Occupancy-grid contour + Douglas-Peucker boundary in the plane frame."""
    if plane.points is None or plane.points.shape[0] < 3:
        return
    origin, u, v = plane_frame(plane.normal, plane.centroid)
    plane.frame_origin, plane.frame_u, plane.frame_v = origin, u, v
    rel = plane.points - origin
    uu = rel @ u
    vv = rel @ v
    umin, vmin = float(uu.min()), float(vv.min())
    iu = ((uu - umin) / BOUNDARY_GRID_M).astype(np.int32)
    iv = ((vv - vmin) / BOUNDARY_GRID_M).astype(np.int32)
    grid = np.zeros((int(iv.max()) + 3, int(iu.max()) + 3), dtype=np.uint8)
    grid[iv + 1, iu + 1] = 255
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    plane.area_m2 = round(float(np.count_nonzero(grid)) * BOUNDARY_GRID_M**2, 3)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    largest = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(largest, BOUNDARY_DP_EPS_PX, True).reshape(-1, 2)
    boundary_2d: list[list[float]] = []
    boundary_3d: list[list[float]] = []
    for px, py in poly:
        bu = umin + (float(px) - 1 + 0.5) * BOUNDARY_GRID_M
        bv = vmin + (float(py) - 1 + 0.5) * BOUNDARY_GRID_M
        p3 = origin + u * bu + v * bv
        # Snap exactly onto the refined plane.
        p3 = p3 - plane.normal * float(plane.normal @ p3 + plane.d)
        boundary_2d.append([round(bu, 4), round(bv, 4)])
        boundary_3d.append([round(float(c), 4) for c in p3])
    plane.boundary_2d = boundary_2d
    plane.boundary_3d = boundary_3d


def find_residual_clusters(
    coarse_points: np.ndarray, planes: list[GlobalPlane]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Coarse points unclaimed by any plane -> DBSCAN clusters with boxes."""
    import open3d as o3d  # local import

    claimed = np.zeros(coarse_points.shape[0], dtype=bool)
    for plane in planes:
        near = np.abs(coarse_points @ plane.normal + plane.d) < RESIDUAL_DISTANCE_M
        inside = np.all(
            (coarse_points >= plane.bbox_min - MERGE_BBOX_PAD_M)
            & (coarse_points <= plane.bbox_max + MERGE_BBOX_PAD_M),
            axis=1,
        )
        claimed |= near & inside
    residual = coarse_points[~claimed]
    if residual.shape[0] < RESIDUAL_MIN_CLUSTER_POINTS:
        return [], residual
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(residual))
    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=RESIDUAL_DBSCAN_EPS_M, min_points=RESIDUAL_DBSCAN_MIN_POINTS
        )
    )
    clusters: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist()) - {-1}):
        pts = residual[labels == label]
        if pts.shape[0] < RESIDUAL_MIN_CLUSTER_POINTS:
            continue
        clusters.append(
            {
                "cluster_id": f"cluster_{len(clusters):03d}",
                "point_count": int(pts.shape[0]),
                "centroid": [round(float(c), 4) for c in pts.mean(axis=0)],
                "bbox_min": [round(float(c), 4) for c in pts.min(axis=0)],
                "bbox_max": [round(float(c), 4) for c in pts.max(axis=0)],
            }
        )
    return clusters, residual


def _palette(index: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(index * 7919 + 13)
    c = rng.uniform(0.25, 1.0, size=3)
    return float(c[0]), float(c[1]), float(c[2])


def _write_review_ply(
    planes: list[GlobalPlane], residual: np.ndarray, out_path: Path
) -> None:
    import open3d as o3d  # local import

    pts_all: list[np.ndarray] = []
    col_all: list[np.ndarray] = []
    for i, plane in enumerate(planes):
        if plane.points is None:
            continue
        pts_all.append(plane.points)
        col_all.append(np.tile(_palette(i), (plane.points.shape[0], 1)))
    if residual.shape[0]:
        pts_all.append(residual)
        col_all.append(np.tile((0.35, 0.35, 0.35), (residual.shape[0], 1)))
    if not pts_all:
        return
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts_all)))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack(col_all))
    o3d.io.write_point_cloud(str(out_path), cloud, write_ascii=False, compressed=True)


def render_review_images(
    planes: list[GlobalPlane],
    residual: np.ndarray,
    out_dir: Path,
    size_px: int = 1600,
    positions: list[np.ndarray] | None = None,
) -> None:
    """Orthographic 2D review renders (no 3D viewer needed).

    Combined views plus per-class top-downs: the walls-only render reads as
    the floor plan (under-merge = parallel double lines in two colors); the
    floors-only and ceilings-only renders expose level structure without
    cross-class overdraw. Sweep positions are drawn as white crosses.
    """
    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i, plane in enumerate(planes):
        if plane.points is None:
            continue
        r, g, b = _palette(i)
        entries.append((plane.cls, plane.points, np.array([b, g, r]) * 255.0))
    gray = np.array([90.0, 90.0, 90.0])

    def render(
        classes: set[str] | None,
        include_residual: bool,
        h_axis: int,
        v_axis: int,
        flip_v: bool,
        name: str,
    ) -> None:
        pts_all = [p for cls, p, _ in entries if classes is None or cls in classes]
        col_all = [
            np.tile(c, (p.shape[0], 1))
            for cls, p, c in entries
            if classes is None or cls in classes
        ]
        if include_residual and residual.shape[0]:
            pts_all.append(residual)
            col_all.append(np.tile(gray, (residual.shape[0], 1)))
        if not pts_all:
            return
        pts = np.vstack(pts_all)
        cols = np.vstack(col_all).astype(np.uint8)
        h, v = pts[:, h_axis], pts[:, v_axis]
        hmin, vmin = float(h.min()), float(v.min())
        extent = max(float(h.max()) - hmin, float(v.max()) - vmin, 1e-6)
        scale = (size_px - 40) / extent
        width = int((float(h.max()) - hmin) * scale) + 40
        height = int((float(v.max()) - vmin) * scale) + 40
        iu = ((h - hmin) * scale).astype(np.int32) + 20
        iv = ((v - vmin) * scale).astype(np.int32) + 20
        if flip_v:
            iv = height - 1 - iv
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[iv, iu] = cols
        for pos in positions or []:
            mu = int((float(pos[h_axis]) - hmin) * scale) + 20
            mv = int((float(pos[v_axis]) - vmin) * scale) + 20
            if flip_v:
                mv = height - 1 - mv
            if 0 <= mu < width and 0 <= mv < height:
                cv2.drawMarker(
                    image, (mu, mv), (255, 255, 255), cv2.MARKER_CROSS, 14, 2
                )
        cv2.imwrite(str(out_dir / name), image)

    render(None, True, 0, 2, False, "planes_review_topdown.png")
    render(None, True, 0, 1, True, "planes_review_front.png")
    render({"wall"}, False, 0, 2, False, "planes_review_topdown_walls.png")
    render({"floor"}, False, 0, 2, False, "planes_review_topdown_floors.png")
    render({"ceiling"}, False, 0, 2, False, "planes_review_topdown_ceilings.png")


def parse_sweep_spec(spec: str | None, available: list[int]) -> list[int]:
    """Parse "3", "3,7", "30-40", "3,30-40" against available indices."""
    if not spec:
        return available
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            wanted.update(range(int(lo), int(hi) + 1))
        elif part:
            wanted.add(int(part))
    return [i for i in available if i in wanted]


def fit_property(
    cache_dir: Path,
    out_dir: Path,
    sweep_spec: str | None = None,
    poses_path: Path | None = None,
    crop_margin: float | None = None,
) -> list[GlobalPlane]:
    """Full G2 pass: propose -> cluster -> refine -> pair -> boundaries -> residuals.

    With crop_margin set (requires poses_path), every sweep cloud is cropped
    to the selected sweep positions' bbox + margin before proposing — the
    one-room slice mode: geometry seen through windows/doorways is excluded.
    """
    available = sorted(
        int(p.stem.split("_")[1]) for p in cache_dir.glob("scan_*.ply")
    )
    indices = parse_sweep_spec(sweep_spec, available)
    if not indices:
        raise ValueError(f"no cached sweeps selected under {cache_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    positions: list[np.ndarray] = []
    crop_min: np.ndarray | None = None
    crop_max: np.ndarray | None = None
    if poses_path is not None:
        from vs_pipeline.stages.pose import read_poses_json

        pose_by_index = {s.index: s.pose_rt[:3, 3] for s in read_poses_json(poses_path)}
        positions = [pose_by_index[i] for i in indices if i in pose_by_index]
    if crop_margin is not None:
        if not positions:
            raise ValueError("--crop-margin requires --poses (sweep positions)")
        pos = np.vstack(positions)
        pad = np.array([crop_margin, 3.0, crop_margin])
        crop_min = pos.min(axis=0) - pad
        crop_max = pos.max(axis=0) + pad
        print(
            "room crop enabled: "
            f"{np.round(crop_min, 2).tolist()} .. {np.round(crop_max, 2).tolist()}"
        )

    print(f"proposing planes on {len(indices)} sweep(s)…")
    proposals: list[PlaneProposal] = []
    coarse_parts: list[np.ndarray] = []
    for index in indices:
        started = time.perf_counter()
        cloud = _load_sweep_cloud(cache_dir, index)
        if crop_min is not None and crop_max is not None:
            import open3d as o3d  # local import

            cloud = cloud.crop(o3d.geometry.AxisAlignedBoundingBox(crop_min, crop_max))
            if len(cloud.points) == 0:
                print(f"  [{index:03d}] empty after crop -> skip")
                continue
        props = propose_sweep_planes(cloud, index)
        proposals.extend(props)
        coarse_parts.append(
            np.asarray(cloud.voxel_down_sample(COARSE_VOXEL_M).points)
        )
        print(
            f"  [{index:03d}] {len(props)} proposal(s)  "
            f"({time.perf_counter() - started:.1f}s)"
        )

    clusters = cluster_proposals(proposals)
    print(f"{len(proposals)} proposal(s) -> {len(clusters)} global plane cluster(s)")

    planes = refine_planes(clusters, cache_dir)
    planes = merge_refined_planes(planes)
    print(f"second-pass merge on refined coefficients -> {len(planes)} plane(s)")
    for plane in planes:
        extract_boundary(plane)
    reclassify_and_rename(
        planes,
        sweep_heights=[float(p[1]) for p in positions] if positions else None,
    )
    pair_walls(planes)
    planes.sort(key=lambda p: -p.area_m2)

    coarse_points = np.vstack(coarse_parts)
    residual_clusters, residual_points = find_residual_clusters(coarse_points, planes)

    payload = {
        "schema_version": PLANES_SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "sweeps_used": indices,
        "plane_count": len(planes),
        "planes": [
            {
                "plane_id": p.plane_id,
                "class": p.cls,
                "coefficients": [round(float(c), 6) for c in (*p.normal, p.d)],
                "inlier_count": p.inlier_count,
                "area_m2": p.area_m2,
                "centroid": [round(float(c), 4) for c in p.centroid],
                "bbox_min": [round(float(c), 4) for c in p.bbox_min],
                "bbox_max": [round(float(c), 4) for c in p.bbox_max],
                "elevation_y": p.elevation_y,
                "thickness_m": p.thickness_m,
                "paired_with": p.paired_with,
                "member_sweeps": p.member_sweeps,
                "frame": (
                    {
                        "origin": [round(float(c), 6) for c in p.frame_origin],
                        "u": [round(float(c), 6) for c in p.frame_u],
                        "v": [round(float(c), 6) for c in p.frame_v],
                    }
                    if p.frame_origin is not None
                    else None
                ),
                "boundary_2d": p.boundary_2d,
                "boundary_3d": p.boundary_3d,
            }
            for p in planes
        ],
    }
    (out_dir / "structural_planes.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "residual_clusters.json").write_text(
        json.dumps(
            {
                "schema_version": PLANES_SCHEMA_VERSION,
                "coordinate_frame": "runtime_yup",
                "cluster_count": len(residual_clusters),
                "clusters": residual_clusters,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_review_ply(planes, residual_points, out_dir / "planes_review.ply")
    render_review_images(planes, residual_points, out_dir, positions=positions)

    print(f"\n{len(planes)} global plane(s):")
    for p in planes:
        extra = ""
        if p.elevation_y is not None:
            extra = f"  y = {p.elevation_y:+.3f} m"
        if p.thickness_m is not None:
            extra += f"  thickness = {p.thickness_m * 1000:.0f} mm (pair: {p.paired_with})"
        print(
            f"  {p.plane_id:<12} area {p.area_m2:7.2f} m^2  "
            f"inliers {p.inlier_count:>7,}  sweeps {len(p.member_sweeps):>3}{extra}"
        )
    print(f"{len(residual_clusters)} residual cluster(s) (furniture/detail candidates)")
    print(f"outputs -> {out_dir}")
    return planes


def _planes_from_json(data: dict[str, Any]) -> list[GlobalPlane]:
    """Lightweight GlobalPlane shims from a structural_planes.json payload
    (points=None; enough for classification/level logic; order preserved)."""
    shims: list[GlobalPlane] = []
    for p in data["planes"]:
        coeff = p["coefficients"]
        shims.append(
            GlobalPlane(
                plane_id=p["plane_id"],
                cls=p["class"],
                normal=np.asarray(coeff[:3], dtype=np.float64),
                d=float(coeff[3]),
                inlier_count=int(p["inlier_count"]),
                area_m2=float(p["area_m2"]),
                centroid=np.asarray(p["centroid"], dtype=np.float64),
                bbox_min=np.asarray(p["bbox_min"], dtype=np.float64),
                bbox_max=np.asarray(p["bbox_max"], dtype=np.float64),
                member_sweeps=list(p["member_sweeps"]),
                elevation_y=p["elevation_y"],
                thickness_m=p["thickness_m"],
                paired_with=p["paired_with"],
            )
        )
    return shims


def _sweep_heights_from_poses(poses_path: Path | None) -> list[float] | None:
    """Sweep-position Y values from poses.json (tripod evidence), or None."""
    if poses_path is None:
        return None
    from vs_pipeline.stages.pose import read_poses_json

    return [float(s.pose_rt[1, 3]) for s in read_poses_json(poses_path)]


def reclassify_planes_json(planes_path: Path, poses_path: Path | None = None) -> None:
    """Re-apply storey reclassification to an existing structural_planes.json.

    Classes are reset from the stored normals first (undoing any prior
    demotion), then the current storey rule runs and ids are rebuilt --
    idempotent, runs in seconds, no refit. Wall ids are stable (the wall set
    is untouched) so paired_with references stay valid. Review PNG/PLY are
    NOT regenerated; they still reflect the original run's classes.
    """
    data = json.loads(planes_path.read_text(encoding="utf-8"))
    shims = _planes_from_json(data)
    for s in shims:
        s.cls = classify_normal(s.normal)
        s.elevation_y = (
            float(s.centroid[1]) if s.cls in ("floor", "ceiling") else None
        )
    heights = _sweep_heights_from_poses(poses_path)
    reclassify_and_rename(shims, sweep_heights=heights)

    levels = sorted(storey_floor_levels(shims, heights))
    if heights is None:
        print("NOTE: no --poses given -> tripod evidence skipped (less robust)")
    print("storey floor levels: " + ", ".join(f"{lv:+.3f}" for lv in levels))
    print("class changes:")
    changes = 0
    for s, p in zip(shims, data["planes"]):
        if p["class"] != s.cls:
            print(
                f"  {p['plane_id']:<14} {p['class']:<8} -> {s.cls:<8}"
                f"  (y = {float(s.centroid[1]):+7.3f}, {p['area_m2']:.2f} m^2)"
            )
            changes += 1
        p["class"] = s.cls
        p["plane_id"] = s.plane_id
        p["elevation_y"] = (
            round(s.elevation_y, 4) if s.elevation_y is not None else None
        )
    counts: dict[str, int] = {}
    for p in data["planes"]:
        counts[p["class"]] = counts.get(p["class"], 0) + 1
    print(
        f"{changes} change(s); now: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    planes_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"saved -> {planes_path}")


def inspect_planes(planes_path: Path, poses_path: Path | None = None) -> None:
    """Read-only diagnostics on an existing structural_planes.json (no refit).

    Prints: per-class counts; every horizontal plane sorted by elevation;
    elevation clusters (storey candidates) with per-cluster areas; the exact
    storey levels the reclassifier used (reconstructed from the same
    deterministic rule); and suspicious room-sized `surface` / down-facing
    `other` planes that may be true floors/ceilings demoted by a storey
    anchor polluted with outdoor terrain.
    """
    data = json.loads(planes_path.read_text(encoding="utf-8"))
    planes: list[dict[str, Any]] = data["planes"]

    counts: dict[str, int] = {}
    for p in planes:
        counts[p["class"]] = counts.get(p["class"], 0) + 1
    print(
        f"{data['plane_count']} plane(s): "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )

    horizontals: list[tuple[float, str, dict[str, Any]]] = []
    for p in planes:
        ny = float(p["coefficients"][1])
        if abs(ny) > HORIZONTAL_NY:
            horizontals.append(
                (float(p["centroid"][1]), "up" if ny > 0 else "down", p)
            )
    horizontals.sort(key=lambda t: -t[0])

    print("\nhorizontal planes (top to bottom):")
    print(
        f"  {'y (m)':>8}  {'faces':<5}  {'class':<8}  {'plane_id':<14}"
        f"  {'area m^2':>9}  sweeps"
    )
    for y, faces, p in horizontals:
        print(
            f"  {y:+8.3f}  {faces:<5}  {p['class']:<8}  {p['plane_id']:<14}"
            f"  {p['area_m2']:>9.2f}  {len(p['member_sweeps']):>3}"
        )

    print("\nelevation clusters (a gap > 0.15 m starts a new cluster):")
    ordered = sorted(horizontals, key=lambda t: t[0])
    groups: list[list[tuple[float, str, dict[str, Any]]]] = []
    for item in ordered:
        if groups and item[0] - groups[-1][-1][0] <= 0.15:
            groups[-1].append(item)
        else:
            groups.append([item])
    for group in groups:
        ys = [g[0] for g in group]
        up_area = sum(g[2]["area_m2"] for g in group if g[1] == "up")
        down_area = sum(g[2]["area_m2"] for g in group if g[1] == "down")
        biggest = max(group, key=lambda g: g[2]["area_m2"])[2]
        print(
            f"  y {min(ys):+7.3f} .. {max(ys):+7.3f}  ({len(group):>3} planes)"
            f"  up {up_area:8.2f} m^2  down {down_area:8.2f} m^2"
            f"  biggest: {biggest['plane_id']} ({biggest['area_m2']:.2f} m^2)"
        )

    shims = _planes_from_json(data)
    for s in shims:
        s.cls = classify_normal(s.normal)
        s.elevation_y = (
            float(s.centroid[1]) if s.cls in ("floor", "ceiling") else None
        )
    levels = sorted(storey_floor_levels(shims, _sweep_heights_from_poses(poses_path)))
    print(
        "\nstorey FLOOR levels under the current interiority-aware rule "
        "(classes reset from normals first; pass --poses for tripod evidence):"
    )
    for lv in levels:
        print(f"  y = {lv:+7.3f}")

    ceilings = [p for p in planes if p["class"] == "ceiling"]
    max_ceiling_area = max((p["area_m2"] for p in ceilings), default=0.0)
    clevel_area_min = max(LEVEL_MIN_AREA_M2, 0.3 * max_ceiling_area)
    print(f"\nstorey CEILING levels (ceilings with area >= {clevel_area_min:.1f} m^2):")
    for p in sorted(ceilings, key=lambda q: -q["area_m2"]):
        if p["area_m2"] >= clevel_area_min and p["elevation_y"] is not None:
            print(
                f"  y = {p['elevation_y']:+7.3f}  {p['plane_id']:<14}"
                f"  {p['area_m2']:.2f} m^2"
            )

    big_surfaces = [
        p
        for p in planes
        if p["class"] == "surface" and p["area_m2"] >= LEVEL_MIN_AREA_M2
    ]
    big_down_other = [
        p
        for p in planes
        if p["class"] == "other"
        and p["area_m2"] >= LEVEL_MIN_AREA_M2
        and float(p["coefficients"][1]) < -HORIZONTAL_NY
    ]
    print("\nsuspicious demotions (room-sized planes classed as furniture/other):")
    if not big_surfaces and not big_down_other:
        print("  none - no surface/other plane reaches room scale.")
    for p in big_surfaces:
        print(
            f"  SURFACE {p['plane_id']:<14} y = {float(p['centroid'][1]):+7.3f}"
            f"  {p['area_m2']:.2f} m^2  <- possible true floor demoted by a bad"
            " storey anchor"
        )
    for p in big_down_other:
        print(
            f"  OTHER   {p['plane_id']:<14} y = {float(p['centroid'][1]):+7.3f}"
            f"  {p['area_m2']:.2f} m^2  <- possible true ceiling demoted"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 5 (G2): plane fitting & boundary extraction."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_fit = sub.add_parser("fit", help="propose -> merge -> refine -> boundaries -> residuals")
    p_fit.add_argument(
        "--cache", type=Path, required=True, help="G1 sweep_cache directory"
    )
    p_fit.add_argument("--output", "-o", type=Path, default=Path("planes_out"))
    p_fit.add_argument(
        "--sweeps",
        type=str,
        default=None,
        help='sweep selection, e.g. "30-40" or "3,7,30-40" (default: all cached)',
    )
    p_fit.add_argument(
        "--poses",
        type=Path,
        default=None,
        help="poses.json (enables sweep markers in renders and --crop-margin)",
    )
    p_fit.add_argument(
        "--crop-margin",
        type=float,
        default=None,
        help="one-room slice mode: crop clouds to the selected sweep positions' "
        "bbox + margin (m); requires --poses",
    )
    p_inspect = sub.add_parser(
        "inspect", help="read-only diagnostics on an existing structural_planes.json"
    )
    p_inspect.add_argument(
        "--planes", type=Path, required=True, help="path to structural_planes.json"
    )
    p_inspect.add_argument(
        "--poses", type=Path, default=None, help="poses.json (tripod evidence)"
    )
    p_reclass = sub.add_parser(
        "reclassify",
        help="re-apply the storey rule to an existing structural_planes.json "
        "(no refit; classes reset from stored normals first)",
    )
    p_reclass.add_argument(
        "--planes", type=Path, required=True, help="path to structural_planes.json"
    )
    p_reclass.add_argument(
        "--poses", type=Path, default=None, help="poses.json (tripod evidence)"
    )
    args = parser.parse_args(argv)

    if args.command == "inspect":
        if not args.planes.is_file():
            parser.error(f"planes file not found: {args.planes}")
        inspect_planes(args.planes, poses_path=args.poses)
        return 0

    if args.command == "reclassify":
        if not args.planes.is_file():
            parser.error(f"planes file not found: {args.planes}")
        reclassify_planes_json(args.planes, poses_path=args.poses)
        return 0

    if not args.cache.is_dir():
        parser.error(f"cache directory not found: {args.cache}")
    fit_property(
        args.cache,
        args.output,
        args.sweeps,
        poses_path=args.poses,
        crop_margin=args.crop_margin,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
