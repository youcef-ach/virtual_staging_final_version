"""Stage 4 (G6) slice 3 -- 2D->3D fusion (dual-direction, per the locked brief).

Core constraint: the G4 G-buffers store PROXY depth (shell + detail boxes),
not true scene depth. Fusion direction therefore depends on class:

1. Wall-borne classes 2D->3D (window/door): mask pixels lift by DIRECT
   RAY-PLANE INTERSECTION -- camera + pixel ray -> nearest wall plane whose
   boundary bbox contains the hit -- with proxy depth demoted to an
   OCCLUSION GATE: the ray must reach the plane at or before the proxy hit
   (+ slack). Doors pass because the G3 proxy has door HOLES (the proxy hit
   lies beyond the plane); windows pass because the proxy has no window
   holes yet (hit ON the plane); occluding walls block. Lifting through
   proxy depth was wrong exactly for doors: rays pass through the G3 holes
   and land on geometry in the room behind (first run: 0/54 door match).
   Occupancy accumulates per wall plane (u, v) across sweeps -> opening
   rectangles (openings.py-compatible schema).
2. Volumetric classes 3D->2D (furniture): lifting furniture pixels through
   proxy depth would land BEHIND the furniture wherever no detail proxy
   exists. Instead the G2 residual-cluster boxes are the 3D evidence: each
   box projects into every face that sees it and mask class fractions vote
   -> labeled 3D boxes (the Year-2 SAM-prompt / room-emptying schema).
3. Mirror detector (OPEN_PROBLEMS #2 candidate): coherent components where
   the 2D masks say room content (floor/ceiling/furniture) but proxy depth
   agrees with a wall plane => mirror. v2 NORMAL GATE: the rendered proxy
   normal must belong to a VERTICAL surface ALIGNED with the assigned wall
   plane -- kills wall-top ceiling strips (normal down) and horizontal rims
   (normal up), the dominant 152/157 false positives. Normal maps are
   AUTO-DECODED when stored encoded ((n+1)/2 in 0..1 -- common through
   image pipelines); a one-time diagnostic line reports the verdict.
   Residual FP mode: furniture tight against a wall WITHOUT a detail
   proxy. Real mirrors labeled `wall` by Mask2Former are structurally
   MISSED (never enter the candidate set) -- needs raw-ADE
   reflected-content evidence (next iteration).

Review: accepted opening rects are REPROJECTED onto the pano faces
(occlusion-gated; the F5 ghosting-proof philosophy -- evidence drawn over
the photograph). The per-plane (u, v) grids remain as secondary evidence
maps.

Door cross-check vs G3 is GEOMETRIC (match_doors): with G2's under-merged
duplicate wall candidates, the fused door routinely lands on a
duplicate/pair-face plane with a DIFFERENT id than the one G3 notched --
plane_id equality measured that, not geometry.

Pixel alignment: masks are stored in the canonical (stored) face layout,
pixel-aligned with the extracted faces. The LOCKED F5 calibration maps each
extracted face to a rendered G-buffer face + orientation op; applying the op
to the worldpos ARRAY (pure rot90/flip permutations) yields world positions
pixel-aligned with the masks. Positions convert Matterport Z-up -> runtime
Y-up once (Coordinate Lock rule).

CLI:
    python -m vs_pipeline.stages.fusion run \
        --panos extracted_cubemaps --seg-dir seg_out \
        --renders gbuffer_renders --jobs gbuffer_jobs.json \
        --calibration qa_output/calibration.json \
        --planes planes_full/structural_planes.json \
        --clusters planes_full/residual_clusters.json \
        --sweeps "1,50,60,152,157,175" -o fusion_out
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vs_pipeline.stages.gbuffer import (
    _load_depth,
    depth_to_worldpos,
    face_ray_directions,
)
from vs_pipeline.stages.pose_qa import _ROT_K, apply_orientation
from vs_pipeline.stages.segmentation import CLASS_IDS, CLASSES, FACE_NAMES

FUSION_SCHEMA_VERSION = 1

#: Occupancy grid resolution on wall planes (matches openings.py GRID_M).
GRID_M = 0.05
#: (u, v) slack outside the wall boundary bbox.
PLANE_UV_PAD_M = 0.2
#: Proxy depth validity window (Cycles no-hit depth is huge).
DEPTH_VALID_MAX_M = 100.0
#: Ray-plane lift: minimum hit distance (self-hit guard) and the occlusion
#: slack (proxy hit may be up to this much SHORT of the plane and still pass;
#: door holes put the proxy hit BEYOND the plane, which always passes).
RAY_MIN_T_M = 0.3
RAY_BLOCK_SLACK_M = 0.25
#: Opening acceptance gates on the extracted rect.
OPENING_MIN_DIM_M = 0.30
OPENING_MIN_AREA_M2 = 0.15
OPENING_MIN_FILL = 0.30
#: A grid cell needs at least this many mask pixels to count as occupied.
MIN_PIXELS_PER_CELL = 3
#: Cap on lifted points per face/class (subsampled beyond; keeps it fast).
MAX_POINTS_PER_FACE = 300_000

#: Residual-cluster box filter (mirrors details.py gates).
BOX_MIN_POINTS = 100
BOX_MIN_DIM_M = 0.15
BOX_MIN_HEIGHT_M = 0.2
BOX_MAX_DIM_M = 6.0
#: Voting gates.
VOTE_MIN_FOOTPRINT_PX = 400
FURNITURE_CONFIRM_FRACTION = 0.35
OCCLUSION_SLACK_M = 0.3

#: Mirror detector.
MIRROR_CANDIDATE_CLASSES = ("floor", "ceiling", "furniture")
MIRROR_WALL_DIST_M = 0.12
MIRROR_MIN_AREA_FRAC = 0.005  # of the face area
#: v2 normal gate: proxy normal must be vertical-surface and wall-aligned.
MIRROR_NORMAL_MAX_Y = 0.3
MIRROR_NORMAL_ALIGN_MIN = 0.7

#: Geometric door cross-check gates (plane_id-agnostic; see match_doors).
DOOR_MATCH_HORIZ_M = 0.75
DOOR_MATCH_V_OVERLAP_M = 0.2


# ------------------------------------------------------------------ frames & alignment


def invert_orientation(image: np.ndarray, op: str) -> np.ndarray:
    """Exact inverse of pose_qa.apply_orientation (rotNN [+ _flip])."""
    base, _, flip = op.partition("_")
    if base not in _ROT_K or (flip and flip != "flip"):
        raise ValueError(f"unknown orientation op: {op}")
    out = np.asarray(image)
    if flip:
        out = np.fliplr(out)
    return np.rot90(out, k=-_ROT_K[base])


def zup_to_yup_points(pts: np.ndarray) -> np.ndarray:
    """Matterport Z-up -> runtime Y-up (Coordinate Lock rule), any leading shape."""
    p = np.asarray(pts, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def yup_to_zup_points(pts: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`zup_to_yup_points`."""
    p = np.asarray(pts, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def aligned_worldpos_yup(
    depth_raw: np.ndarray, cam_matrix: np.ndarray, op: str
) -> tuple[np.ndarray, np.ndarray]:
    """Raw render depth -> (worldpos Y-up, valid), pixel-aligned with the EXTRACTED face.

    depth_to_worldpos works in render pixel space (Matterport frame); the F5
    orientation op then permutes the worldpos array into the extracted-face
    layout -- lossless, so masks and positions correspond pixel for pixel.
    """
    world_mp = depth_to_worldpos(depth_raw, cam_matrix)
    valid = (
        np.isfinite(depth_raw) & (depth_raw > 0.0) & (depth_raw < DEPTH_VALID_MAX_M)
    )
    world_mp = np.ascontiguousarray(apply_orientation(world_mp, op))
    valid = np.ascontiguousarray(apply_orientation(valid, op))
    return zup_to_yup_points(world_mp), valid


# ------------------------------------------------------------------ wall planes


@dataclass
class WallPlane:
    plane_id: str
    normal: np.ndarray  # unit, runtime Y-up; plane: normal . p + d = 0
    d: float
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    u_range: tuple[float, float]
    v_range: tuple[float, float]


def load_wall_planes(planes_path: Path) -> list[WallPlane]:
    data = json.loads(planes_path.read_text(encoding="utf-8"))
    walls: list[WallPlane] = []
    for p in data["planes"]:
        if p.get("class") != "wall":
            continue
        frame = p.get("frame")
        b2 = p.get("boundary_2d") or []
        if not frame or len(b2) < 3:
            continue
        origin = np.asarray(frame["origin"], dtype=np.float64)
        u_axis = np.asarray(frame["u"], dtype=np.float64)
        v_axis = np.asarray(frame["v"], dtype=np.float64)
        if "normal" in p and "d" in p:
            normal = np.asarray(p["normal"], dtype=np.float64)
            d = float(p["d"])
        else:  # derive from the orthonormal frame (robust to schema detail)
            normal = np.cross(u_axis, v_axis)
            normal = normal / np.linalg.norm(normal)
            d = -float(normal @ origin)
        arr = np.asarray(b2, dtype=np.float64)
        walls.append(
            WallPlane(
                plane_id=str(p["plane_id"]),
                normal=normal,
                d=d,
                origin=origin,
                u_axis=u_axis,
                v_axis=v_axis,
                u_range=(float(arr[:, 0].min()), float(arr[:, 0].max())),
                v_range=(float(arr[:, 1].min()), float(arr[:, 1].max())),
            )
        )
    return walls


def assign_to_walls(
    pts: np.ndarray, walls: list[WallPlane], max_dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest wall plane per point: (index into walls or -1, distance).

    A point is assignable when its plane distance is within ``max_dist`` AND
    its (u, v) projection falls inside the wall's boundary bbox (+ pad) --
    the same bleeding guard family as G2 (distant coplanar walls never grab
    each other's pixels).
    """
    n = pts.shape[0]
    best = np.full(n, -1, dtype=np.int64)
    best_d = np.full(n, np.inf, dtype=np.float64)
    for i, w in enumerate(walls):
        dist = np.abs(pts @ w.normal + w.d)
        rel = pts - w.origin
        u = rel @ w.u_axis
        v = rel @ w.v_axis
        ok = (
            (dist <= max_dist)
            & (dist < best_d)
            & (u >= w.u_range[0] - PLANE_UV_PAD_M)
            & (u <= w.u_range[1] + PLANE_UV_PAD_M)
            & (v >= w.v_range[0] - PLANE_UV_PAD_M)
            & (v <= w.v_range[1] + PLANE_UV_PAD_M)
        )
        best[ok] = i
        best_d[ok] = dist[ok]
    return best, best_d


def ray_plane_lift(
    cam_matrix: np.ndarray,
    cls_render: np.ndarray,
    depth_raw: np.ndarray,
    walls: list[WallPlane],
    class_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift class pixels onto wall planes by DIRECT ray-plane intersection.

    Works in RENDER pixel space (Matterport-frame camera, 90-deg pinhole,
    matching gbuffer.face_ray_directions exactly). For every pixel of
    ``class_id``, cast its ray and intersect with each wall plane; keep the
    NEAREST plane whose hit passes all gates:

    - finite t >= RAY_MIN_T_M, non-parallel ray;
    - (u, v) inside the wall boundary bbox (+ pad) -- bleeding guard; a
      nearer plane failing this gate does NOT block a farther passing one
      (the physical ray continues past a wall's lateral extent);
    - occlusion gate: the proxy depth along the ray must not be clearly
      SHORTER than t. Proxy hits AT the plane (windows: the proxy has no
      window holes yet) and BEYOND it (doors: the G3 door hole lets the ray
      through) both pass; an occluding wall in front blocks.

    Returns (points_yup (N, 3), wall_index (N,)).
    """
    empty = (np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
    sel = np.asarray(cls_render) == class_id
    if not bool(sel.any()) or not walls:
        return empty
    size = int(cls_render.shape[0])
    dirs = face_ray_directions(size)[sel]
    depth = np.asarray(depth_raw, dtype=np.float64)[sel]
    if dirs.shape[0] > MAX_POINTS_PER_FACE:
        step = dirs.shape[0] // MAX_POINTS_PER_FACE + 1
        dirs = dirs[::step]
        depth = depth[::step]
    cam = np.asarray(cam_matrix, dtype=np.float64)
    d_yup = zup_to_yup_points(dirs @ cam[:3, :3].T)
    p0 = zup_to_yup_points(cam[:3, 3])
    depth_ok = np.isfinite(depth) & (depth > 0.0) & (depth < DEPTH_VALID_MAX_M)
    n = d_yup.shape[0]
    best_t = np.full(n, np.inf)
    best_w = np.full(n, -1, dtype=np.int64)
    for i, w in enumerate(walls):
        denom = d_yup @ w.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -(float(p0 @ w.normal) + w.d) / denom
        ok = (
            (np.abs(denom) > 1e-6)
            & np.isfinite(t)
            & (t >= RAY_MIN_T_M)
            & (t < best_t)
        )
        if not bool(ok.any()):
            continue
        idx = np.where(ok)[0]
        pts = p0 + d_yup[idx] * t[idx, None]
        rel = pts - w.origin
        u = rel @ w.u_axis
        v = rel @ w.v_axis
        inside = (
            (u >= w.u_range[0] - PLANE_UV_PAD_M)
            & (u <= w.u_range[1] + PLANE_UV_PAD_M)
            & (v >= w.v_range[0] - PLANE_UV_PAD_M)
            & (v <= w.v_range[1] + PLANE_UV_PAD_M)
        )
        unblocked = (~depth_ok[idx]) | (depth[idx] >= t[idx] - RAY_BLOCK_SLACK_M)
        keep = idx[inside & unblocked]
        best_t[keep] = t[keep]
        best_w[keep] = i
    hit = best_w >= 0
    if not bool(hit.any()):
        return empty
    return p0 + d_yup[hit] * best_t[hit, None], best_w[hit]


# ------------------------------------------------------------------ opening accumulation


class PlaneAccumulator:
    """Per-wall (u, v) occupancy grids for window/door evidence."""

    KINDS = ("window", "door")

    def __init__(self, wall: WallPlane) -> None:
        self.wall = wall
        self.u0, u1 = wall.u_range
        self.v0, v1 = wall.v_range
        self.nu = max(int(np.ceil((u1 - self.u0) / GRID_M)), 1)
        self.nv = max(int(np.ceil((v1 - self.v0) / GRID_M)), 1)
        self.grids = {
            k: np.zeros((self.nu, self.nv), dtype=np.int32) for k in self.KINDS
        }
        self.sweeps: dict[str, set[str]] = {k: set() for k in self.KINDS}

    def add(self, kind: str, pts: np.ndarray, sweep_uuid: str) -> int:
        rel = pts - self.wall.origin
        u = rel @ self.wall.u_axis
        v = rel @ self.wall.v_axis
        iu = np.floor((u - self.u0) / GRID_M).astype(np.int64)
        iv = np.floor((v - self.v0) / GRID_M).astype(np.int64)
        ok = (iu >= 0) & (iu < self.nu) & (iv >= 0) & (iv < self.nv)
        if not bool(ok.any()):
            return 0
        np.add.at(self.grids[kind], (iu[ok], iv[ok]), 1)
        self.sweeps[kind].add(sweep_uuid)
        return int(ok.sum())


def extract_rects(
    grid: np.ndarray, min_cells_hit: int = MIN_PIXELS_PER_CELL
) -> list[tuple[float, float, float, float, int]]:
    """Occupancy grid -> accepted rects as (ua, ub, vb, vt, support_cells).

    Grid layout: axis 0 = u, axis 1 = v (cell size GRID_M, offsets applied
    by the caller). Morphological close+open kills salt noise and bridges
    single-cell gaps; components pass min-dim / min-area / fill-ratio gates.
    """
    occ = (grid >= min_cells_hit).astype(np.uint8)
    if not bool(occ.any()):
        return []
    kernel = np.ones((3, 3), np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, kernel)
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, kernel)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(occ, connectivity=8)
    rects: list[tuple[float, float, float, float, int]] = []
    for i in range(1, n):
        x, y, w, h, area = (int(s) for s in stats[i])
        # cv2: x/w run along axis 1 (v), y/h along axis 0 (u).
        u_ext = h * GRID_M
        v_ext = w * GRID_M
        if min(u_ext, v_ext) < OPENING_MIN_DIM_M:
            continue
        if u_ext * v_ext < OPENING_MIN_AREA_M2:
            continue
        if area / float(max(w * h, 1)) < OPENING_MIN_FILL:
            continue
        rects.append((y * GRID_M, (y + h) * GRID_M, x * GRID_M, (x + w) * GRID_M, area))
    return rects


def opening_dict(
    wall: WallPlane,
    kind: str,
    ua: float,
    ub: float,
    vb: float,
    vt: float,
    support_cells: int,
    sweep_count: int,
) -> dict[str, Any]:
    """openings.py-compatible rect (+ fusion provenance fields)."""
    corners = [
        wall.origin + wall.u_axis * u + wall.v_axis * v
        for u, v in ((ua, vb), (ub, vb), (ub, vt), (ua, vt))
    ]
    return {
        "plane_id": wall.plane_id,
        "kind": kind,
        "u_range": [round(ua, 4), round(ub, 4)],
        "v_range": [round(vb, 4), round(vt, 4)],
        "width_m": round(ub - ua, 3),
        "height_m": round(vt - vb, 3),
        "corners_3d": [[round(float(c), 4) for c in p] for p in corners],
        "support_cells": int(support_cells),
        "sweep_count": int(sweep_count),
        "source": "fusion",
    }


def match_doors(
    fused_doors: list[dict[str, Any]],
    g3_doors: list[dict[str, Any]],
    max_horiz_dist_m: float = DOOR_MATCH_HORIZ_M,
    min_v_overlap_m: float = DOOR_MATCH_V_OVERLAP_M,
) -> dict[str, int]:
    """GEOMETRIC door cross-check (plane_id-agnostic).

    G2's under-merged duplicate wall candidates mean the fused door
    routinely lands on a duplicate/pair-face plane with a DIFFERENT id
    than the one G3 notched -- plane_id equality measured id luck, not
    geometry. Here two door rects match when their world-space centers are
    within ``max_horiz_dist_m`` horizontally (XZ) AND their vertical (Y)
    intervals overlap by >= ``min_v_overlap_m`` (G3 doors reach the floor
    while fused masks may only cover the visible leaf, so vertical extents
    legitimately differ). Reports both directions.
    """

    def _geom(o: dict[str, Any]) -> tuple[np.ndarray, float, float]:
        corners = np.asarray(o["corners_3d"], dtype=np.float64)
        return (
            corners.mean(axis=0),
            float(corners[:, 1].min()),
            float(corners[:, 1].max()),
        )

    fused_matched = 0
    g3_hit: set[int] = set()
    for o in fused_doors:
        c, ylo, yhi = _geom(o)
        hit_any = False
        for gi, g in enumerate(g3_doors):
            gc, gylo, gyhi = _geom(g)
            horiz = float(np.hypot(c[0] - gc[0], c[2] - gc[2]))
            v_overlap = min(yhi, gyhi) - max(ylo, gylo)
            if horiz <= max_horiz_dist_m and v_overlap >= min_v_overlap_m:
                hit_any = True
                g3_hit.add(gi)
        fused_matched += int(hit_any)
    return {
        "fused_doors": len(fused_doors),
        "g3_doors_on_property": len(g3_doors),
        "fused_matched": fused_matched,
        "g3_recovered": len(g3_hit),
    }


# ------------------------------------------------------------------ furniture voting


def load_boxes(clusters_path: Path) -> list[dict[str, Any]]:
    """Residual clusters -> filtered, merged candidate boxes (runtime Y-up)."""
    from vs_pipeline.stages.details import merge_cluster_boxes

    data = json.loads(clusters_path.read_text(encoding="utf-8"))
    kept = []
    for c in data.get("clusters", []):
        if int(c.get("point_count", 0)) < BOX_MIN_POINTS:
            continue
        bmin = np.asarray(c["bbox_min"], dtype=np.float64)
        bmax = np.asarray(c["bbox_max"], dtype=np.float64)
        dims = bmax - bmin
        if float(dims.min()) < BOX_MIN_DIM_M or float(dims.max()) > BOX_MAX_DIM_M:
            continue
        if float(dims[1]) < BOX_MIN_HEIGHT_M:
            continue
        kept.append(c)
    merged = merge_cluster_boxes(kept)
    return [
        {
            "box_id": f"box_{i:03d}",
            "bbox_min": bmin,
            "bbox_max": bmax,
            "source_clusters": int(count),
        }
        for i, (bmin, bmax, count) in enumerate(
            sorted(merged, key=lambda b: -float(np.prod(b[1] - b[0])))
        )
    ]


def project_box_footprint(
    cam_matrix: np.ndarray, bmin_yup: np.ndarray, bmax_yup: np.ndarray, size: int
) -> np.ndarray | None:
    """Convex-hull footprint of a Y-up box in RENDER pixel space, or None.

    Camera matrices live in the Matterport frame (gbuffer_jobs.json), 90-deg
    pinhole, Blender convention (X=right, Y=up, -Z=view), matching
    gbuffer.face_ray_directions exactly.
    """
    lo, hi = np.asarray(bmin_yup, dtype=np.float64), np.asarray(bmax_yup, dtype=np.float64)
    corners_yup = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    corners_mp = yup_to_zup_points(corners_yup)
    inv = np.linalg.inv(np.asarray(cam_matrix, dtype=np.float64))
    cam_pts = corners_mp @ inv[:3, :3].T + inv[:3, 3]
    front = cam_pts[:, 2] < -1e-6
    if int(front.sum()) < 3:
        return None
    pts = cam_pts[front]
    xn = pts[:, 0] / -pts[:, 2]
    yn = pts[:, 1] / -pts[:, 2]
    col = (xn + 1.0) / 2.0 * size
    row = (1.0 - yn) / 2.0 * size
    lim = 8 * size
    pix = np.stack([np.clip(col, -lim, lim), np.clip(row, -lim, lim)], axis=-1)
    hull = cv2.convexHull(pix.astype(np.int32))
    canvas = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(canvas, hull, 1)
    return canvas.astype(bool)


def vote_box(
    cam_matrix: np.ndarray,
    depth_raw: np.ndarray,
    cls_render: np.ndarray,
    bmin_yup: np.ndarray,
    bmax_yup: np.ndarray,
) -> np.ndarray | None:
    """Class-pixel counts (len(CLASSES),) inside the box footprint, or None.

    Occlusion guard: when the median proxy depth in the footprint is clearly
    SHORTER than the camera-to-box distance (a wall in front), the view does
    not see the box and casts no vote.
    """
    size = depth_raw.shape[0]
    foot = project_box_footprint(cam_matrix, bmin_yup, bmax_yup, size)
    if foot is None or int(foot.sum()) < VOTE_MIN_FOOTPRINT_PX:
        return None
    cam_pos_mp = np.asarray(cam_matrix, dtype=np.float64)[:3, 3]
    center_mp = yup_to_zup_points((np.asarray(bmin_yup) + np.asarray(bmax_yup)) / 2.0)
    dist = float(np.linalg.norm(center_mp - cam_pos_mp))
    half_diag = float(np.linalg.norm(np.asarray(bmax_yup) - np.asarray(bmin_yup))) / 2.0
    d = depth_raw[foot]
    d = d[np.isfinite(d) & (d > 0) & (d < DEPTH_VALID_MAX_M)]
    if d.size and float(np.median(d)) < dist - half_diag - OCCLUSION_SLACK_M:
        return None  # occluded by proxy geometry
    return np.bincount(
        np.asarray(cls_render, dtype=np.uint8)[foot].reshape(-1), minlength=len(CLASSES)
    ).astype(np.int64)


# ------------------------------------------------------------------ mirror detector


def detect_mirror_mask(
    cand_mask: np.ndarray,
    world_yup: np.ndarray,
    walls: list[WallPlane],
    min_area_px: int,
    normals_yup: np.ndarray | None = None,
    stats_out: dict[str, int] | None = None,
) -> np.ndarray:
    """Room-content pixels whose proxy hit sits ON a wall plane => mirror.

    v2 normal gate (active when the normal G-buffer is available): the
    rendered proxy normal at the pixel must belong to a VERTICAL surface
    (|n_y| <= MIRROR_NORMAL_MAX_Y) ALIGNED with the assigned wall's normal
    (|dot| >= MIRROR_NORMAL_ALIGN_MIN). This kills the two dominant false
    positives from the first 152/157 run: wall-top ceiling strips (normal
    points down) and horizontal rims like bathtub edges / toilet bases
    (normal points up). Wall-tight furniture without a detail proxy remains
    the known residual false-positive mode.

    ``stats_out`` (optional) receives cand_px / on_wall_px_v1 / passed_px
    so the caller can report exactly what the gate kills.
    """
    out = np.zeros(cand_mask.shape, dtype=bool)
    if not bool(cand_mask.any()) or not walls:
        if stats_out is not None:
            stats_out.update(
                cand_px=int(cand_mask.sum()), on_wall_px_v1=0, passed_px=0
            )
        return out
    pts = world_yup[cand_mask]
    best, _ = assign_to_walls(pts, walls, MIRROR_WALL_DIST_M)
    on_wall = best >= 0
    v1_count = int(on_wall.sum())
    if normals_yup is not None:
        nrm = np.asarray(normals_yup[cand_mask], dtype=np.float64)
        length = np.linalg.norm(nrm, axis=-1)
        ok_len = length > 0.5
        unit = np.zeros_like(nrm)
        unit[ok_len] = nrm[ok_len] / length[ok_len, None]
        vertical = np.abs(unit[:, 1]) <= MIRROR_NORMAL_MAX_Y
        wall_normals = np.zeros_like(nrm)
        assigned = best >= 0
        if bool(assigned.any()):
            wn = np.stack([w.normal for w in walls])
            wall_normals[assigned] = wn[best[assigned]]
        aligned = (
            np.abs(np.einsum("ij,ij->i", unit, wall_normals))
            >= MIRROR_NORMAL_ALIGN_MIN
        )
        on_wall = on_wall & ok_len & vertical & aligned
    if stats_out is not None:
        stats_out.update(
            cand_px=int(cand_mask.sum()),
            on_wall_px_v1=v1_count,
            passed_px=int(on_wall.sum()),
        )
    flat = np.zeros(cand_mask.shape, dtype=np.uint8)
    flat[cand_mask] = on_wall.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area_px:
            out[labels == i] = True
    return out


# ------------------------------------------------------------------ run


def _load_cls(seg_dir: Path, uuid: str, face: str) -> np.ndarray | None:
    for name in (f"{face}_cls_refined.png", f"{face}_cls.png"):
        p = seg_dir / uuid / name
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    return None


def _load_normal(path: Path) -> np.ndarray | None:
    """Load a world-space normal map (H, W, 3), XYZ channel order.

    Mirrors gbuffer._load_depth's fallback chain: .npy sibling first (the
    renderer writes RGB via its bpy round-trip), then cv2 (EXR decodes as
    BGR -> reversed to RGB), then imageio (RGB). The verticality gate
    (|n_y|) only reads the middle channel and is therefore channel-order
    robust; the alignment gate assumes XYZ.
    """
    npy = path.with_suffix(".npy")
    if npy.is_file():
        arr = np.load(str(npy))
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return np.asarray(arr[..., :3], dtype=np.float64)
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None and img.ndim == 3 and img.shape[2] >= 3:
        return np.asarray(img[..., 2::-1], dtype=np.float64)  # BGR -> RGB
    try:
        import imageio.v3 as iio

        img = iio.imread(str(path))
    except BaseException:
        return None
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        return None
    return np.asarray(img[..., :3], dtype=np.float64)


def _decode_normals(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Auto-detect ENCODED normal maps ((n+1)/2 in 0..1) and decode to signed.

    Raw world-space normals contain negative components and/or have ~unit
    lengths; encoded maps are non-negative with lengths spread away from 1
    (an encoded straight-down normal (0,0,-1) becomes (0.5, 0.5, 0), length
    0.707 -- which is exactly how a ceiling can pass a verticality gate).
    Applies n = 2v - 1 when encoding is detected. Returns (array, decoded).
    """
    finite = np.isfinite(arr).all(axis=-1)
    if not bool(finite.any()):
        return arr, False
    vals = arr[finite]
    lengths = np.linalg.norm(vals, axis=-1)
    nonzero = lengths > 1e-6
    if not bool(nonzero.any()):
        return arr, False
    has_negative = bool((vals < -0.05).any())
    med_len = float(np.median(lengths[nonzero]))
    if has_negative or 0.9 <= med_len <= 1.1:
        return arr, False
    return arr * 2.0 - 1.0, True


def _grid_review(acc: PlaneAccumulator, rects_by_kind: dict[str, list], path: Path) -> None:
    total = sum(acc.grids[k] for k in acc.KINDS)
    vis = np.zeros((acc.nu, acc.nv, 3), dtype=np.uint8)
    peak = max(float(total.max()), 1.0)
    base = np.clip(total / peak * 255.0, 0, 255).astype(np.uint8)
    vis[..., 0] = vis[..., 1] = vis[..., 2] = base
    color = {"window": (0, 220, 250), "door": (40, 140, 245)}  # BGR
    for kind, rects in rects_by_kind.items():
        for ua, ub, vb, vt, _sup in rects:
            r0 = int((ua) / GRID_M)
            r1 = int((ub) / GRID_M)
            c0 = int((vb) / GRID_M)
            c1 = int((vt) / GRID_M)
            cv2.rectangle(vis, (c0, r0), (c1, r1), color[kind], 1)
    scale = max(1, 512 // max(acc.nu, acc.nv))
    vis = cv2.resize(
        vis, (acc.nv * scale, acc.nu * scale), interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(str(path), vis)


def _reproject_openings_review(
    openings: list[dict[str, Any]],
    metas: list[dict[str, Any]],
    cams_by_uuid: dict[str, Any],
    mapping: dict[str, Any],
    renders_dir: Path,
    panos_dir: Path,
    review_dir: Path,
) -> int:
    """Draw accepted opening rects over the pano faces (ghosting-proof style).

    Rect corners project through each face camera in RENDER space (occlusion
    gate at the rect center, same rule as the lift); the drawn canvas is then
    oriented into the stored face layout and overlaid on the photograph --
    correctness is judged against the actual window/door in the photo.
    Returns the number of review images written.
    """
    color = {"window": (0, 220, 250), "door": (40, 140, 245)}  # BGR
    written = 0
    for meta in metas:
        uuid = str(meta["sweep_uuid"])
        idx = int(meta["index"])
        cam_entry = cams_by_uuid.get(uuid)
        if cam_entry is None:
            continue
        for face in FACE_NAMES:
            m = mapping.get(face)
            if not m:
                continue
            render_face = str(m["render_face"])
            op = str(m["op"])
            depth_raw = _load_depth(renders_dir / uuid / f"{render_face}_depth.exr")
            if depth_raw is None:
                continue
            size = int(depth_raw.shape[0])
            cam = np.asarray(cam_entry["faces"][render_face], dtype=np.float64)
            inv = np.linalg.inv(cam)
            canvas = np.zeros((size, size, 3), dtype=np.uint8)
            drawn = 0
            for o in openings:
                corners = np.asarray(o["corners_3d"], dtype=np.float64)
                c = yup_to_zup_points(corners) @ inv[:3, :3].T + inv[:3, 3]
                if not bool((c[:, 2] < -1e-6).all()):
                    continue
                col = (c[:, 0] / -c[:, 2] + 1.0) / 2.0 * size
                row = (1.0 - c[:, 1] / -c[:, 2]) / 2.0 * size
                if (
                    col.max() < 0
                    or col.min() >= size
                    or row.max() < 0
                    or row.min() >= size
                ):
                    continue
                cc = int(np.clip(round(float(col.mean())), 0, size - 1))
                rr = int(np.clip(round(float(row.mean())), 0, size - 1))
                dist = float(np.linalg.norm(c.mean(axis=0)))
                d = float(depth_raw[rr, cc])
                if (
                    np.isfinite(d)
                    and 0.0 < d < DEPTH_VALID_MAX_M
                    and d < dist - RAY_BLOCK_SLACK_M
                ):
                    continue  # occluded from this face
                lim = 8 * size
                pts = np.stack(
                    [np.clip(col, -lim, lim), np.clip(row, -lim, lim)], axis=-1
                ).astype(np.int32)
                cv2.polylines(
                    canvas, [pts.reshape(-1, 1, 2)], True, color[o["kind"]], 2
                )
                drawn += 1
            if not drawn:
                continue
            canvas_ext = np.ascontiguousarray(apply_orientation(canvas, op))
            pano = cv2.imread(str(panos_dir / uuid / f"{face}.jpg"), cv2.IMREAD_COLOR)
            if pano is None:
                continue
            pano = cv2.resize(pano, (size, size))
            mask = canvas_ext.any(axis=-1)
            out = pano.copy()
            out[mask] = canvas_ext[mask]
            cv2.putText(
                out,
                f"[{idx:03d}] {face}: {drawn} opening(s) (window=yellow, door=orange)",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            review_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(review_dir / f"openings_{idx:03d}_{face}.png"), out)
            written += 1
    return written


def run(
    panos_dir: Path,
    seg_dir: Path,
    renders_dir: Path,
    jobs_path: Path,
    calibration_path: Path,
    planes_path: Path,
    clusters_path: Path,
    out_dir: Path,
    g3_openings_path: Path | None = None,
    sweep_indices: set[int] | None = None,
    write_review: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    cams_by_uuid = {str(s["sweep_uuid"]): s for s in jobs["sweeps"]}
    cal = json.loads(calibration_path.read_text(encoding="utf-8"))
    mapping = cal.get("mapping", cal)
    walls = load_wall_planes(planes_path)
    boxes = load_boxes(clusters_path)
    print(f"{len(walls)} wall plane(s), {len(boxes)} candidate box(es)")

    metas = []
    for meta_path in sorted(seg_dir.glob("*/seg_meta.json")):
        try:
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if sweep_indices is not None:
        metas = [m for m in metas if int(m["index"]) in sweep_indices]
    metas.sort(key=lambda m: int(m["index"]))

    accs: dict[str, PlaneAccumulator] = {}
    box_votes = np.zeros((len(boxes), len(CLASSES)), dtype=np.int64)
    box_views = np.zeros(len(boxes), dtype=np.int64)
    mirror_stats: list[dict[str, Any]] = []
    normals_missing = 0
    normals_decoded = 0
    normal_diag_printed = False
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror_dir = out_dir / "mirror_masks"
    review_dir = out_dir / "review"

    kind_ids = {"window": CLASS_IDS["window"], "door": CLASS_IDS["door"]}
    cand_ids = np.asarray([CLASS_IDS[c] for c in MIRROR_CANDIDATE_CLASSES], dtype=np.uint8)

    for meta in metas:
        uuid = str(meta["sweep_uuid"])
        idx = int(meta["index"])
        cam_entry = cams_by_uuid.get(uuid)
        if cam_entry is None:
            print(f"[{idx:03d}] {uuid}: not in jobs file -> skipped")
            continue
        sweep_mirror: dict[str, float] = {}
        faces_done = 0
        gate_v1_px = 0
        gate_passed_px = 0
        for face in FACE_NAMES:
            m = mapping.get(face)
            if not m:
                continue
            render_face = str(m["render_face"])
            op = str(m["op"])
            depth_raw = _load_depth(renders_dir / uuid / f"{render_face}_depth.exr")
            cls = _load_cls(seg_dir, uuid, face)
            if depth_raw is None or cls is None:
                continue
            size = int(depth_raw.shape[0])
            cam = np.asarray(cam_entry["faces"][render_face], dtype=np.float64)
            cls_r = cv2.resize(cls, (size, size), interpolation=cv2.INTER_NEAREST)
            cls_render = np.ascontiguousarray(invert_orientation(cls_r, op))
            world, valid = aligned_worldpos_yup(depth_raw, cam, op)

            # --- wall-borne 2D->3D: ray-plane lift (proxy depth = occlusion gate)
            for kind, cid in kind_ids.items():
                pts, widx = ray_plane_lift(cam, cls_render, depth_raw, walls, cid)
                for wi in np.unique(widx):
                    w = walls[int(wi)]
                    acc = accs.setdefault(w.plane_id, PlaneAccumulator(w))
                    acc.add(kind, pts[widx == wi], uuid)

            # --- mirror detector (extracted-face space, v2 normal gate)
            normals_yup = None
            nrm_render = _load_normal(
                renders_dir / uuid / f"{render_face}_normal.exr"
            )
            if nrm_render is not None and nrm_render.shape[0] == size:
                if not normal_diag_printed:
                    fin = nrm_render[np.isfinite(nrm_render).all(axis=-1)]
                    # Skip degenerate faces (empty outdoor renders are all
                    # zeros) -- wait for real data before spending the
                    # one-time print.
                    if fin.size and float(np.abs(fin).max()) > 1e-3:
                        lens = np.linalg.norm(fin, axis=-1)
                        lens = lens[lens > 1e-6]
                        med = float(np.median(lens)) if lens.size else 0.0
                        print(
                            f"normal-map diagnostic ({uuid}/{render_face}): "
                            f"min {fin.min():+.3f} max {fin.max():+.3f} "
                            f"median|n| {med:.3f}"
                        )
                        normal_diag_printed = True
                nrm_render, decoded = _decode_normals(nrm_render)
                normals_decoded += int(decoded)
                nrm_ext = np.ascontiguousarray(apply_orientation(nrm_render, op))
                normals_yup = zup_to_yup_points(nrm_ext)
            else:
                normals_missing += 1
            cand = np.isin(cls_r, cand_ids) & valid
            min_area = int(MIRROR_MIN_AREA_FRAC * size * size)
            gate_stats: dict[str, int] = {}
            mirror = detect_mirror_mask(
                cand, world, walls, min_area, normals_yup, gate_stats
            )
            gate_v1_px += gate_stats.get("on_wall_px_v1", 0)
            gate_passed_px += gate_stats.get("passed_px", 0)
            frac = float(mirror.mean())
            sweep_mirror[face] = frac
            if bool(mirror.any()):
                (mirror_dir / uuid).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(mirror_dir / uuid / f"{face}_mirror.png"),
                    mirror.astype(np.uint8) * 255,
                )
                if write_review:
                    pano = cv2.imread(
                        str(panos_dir / uuid / f"{face}.jpg"), cv2.IMREAD_COLOR
                    )
                    if pano is not None:
                        pano = cv2.resize(pano, (size, size))
                        overlay = pano.copy()
                        overlay[mirror] = (0, 0, 255)
                        blend = cv2.addWeighted(pano, 0.5, overlay, 0.5, 0)
                        cv2.putText(
                            blend,
                            f"{face} mirror {frac:.1%}",
                            (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        review_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(
                            str(review_dir / f"mirror_{idx:03d}_{face}.png"), blend
                        )

            # --- volumetric 3D->2D: box voting (render space)
            for b_i, box in enumerate(boxes):
                votes = vote_box(
                    cam, depth_raw, cls_render, box["bbox_min"], box["bbox_max"]
                )
                if votes is not None:
                    box_votes[b_i] += votes
                    box_views[b_i] += 1
            faces_done += 1
        mirror_faces = {f: v for f, v in sweep_mirror.items() if v > 0}
        mirror_stats.append(
            {"sweep_uuid": uuid, "index": idx, "mirror_face_fractions": mirror_faces}
        )
        print(
            f"[{idx:03d}] {uuid}: {faces_done} face(s) fused, "
            f"mirror on {len(mirror_faces)} face(s) "
            f"(gate: {gate_v1_px} on-wall px -> {gate_passed_px} passed)"
        )
    if normals_missing:
        print(
            f"WARNING: {normals_missing} face(s) had no readable normal map -> "
            "mirror normal gate skipped there (expect more false positives)"
        )
    if normals_decoded:
        print(
            f"normal maps auto-decoded ((n+1)/2 -> signed) on "
            f"{normals_decoded} face(s)"
        )

    # --- finalize openings
    openings: list[dict[str, Any]] = []
    for plane_id, acc in sorted(accs.items()):
        rects_by_kind: dict[str, list] = {}
        for kind in acc.KINDS:
            rects = extract_rects(acc.grids[kind])
            rects_by_kind[kind] = rects
            for ua, ub, vb, vt, sup in rects:
                openings.append(
                    opening_dict(
                        acc.wall,
                        kind,
                        acc.u0 + ua,
                        acc.u0 + ub,
                        acc.v0 + vb,
                        acc.v0 + vt,
                        sup,
                        len(acc.sweeps[kind]),
                    )
                )
        if write_review and any(rects_by_kind.values()):
            review_dir.mkdir(parents=True, exist_ok=True)
            _grid_review(acc, rects_by_kind, review_dir / f"plane_{plane_id}.png")

    # --- reproject accepted rects onto the pano faces (primary review)
    if write_review and openings:
        n_imgs = _reproject_openings_review(
            openings, metas, cams_by_uuid, mapping, renders_dir, panos_dir, review_dir
        )
        print(f"opening reprojection review: {n_imgs} image(s) -> {review_dir}")

    # --- optional cross-check vs G3 jamb-detected doors (GEOMETRIC)
    door_matches = None
    if g3_openings_path is not None and g3_openings_path.is_file():
        g3 = json.loads(g3_openings_path.read_text(encoding="utf-8"))
        g3_doors = [o for o in g3.get("openings", []) if o.get("kind") == "door"]
        fused_doors = [o for o in openings if o["kind"] == "door"]
        door_matches = match_doors(fused_doors, g3_doors)

    # --- finalize furniture boxes
    box_entries: list[dict[str, Any]] = []
    confirmed = 0
    for b_i, box in enumerate(boxes):
        total = int(box_votes[b_i].sum())
        fractions = {
            name: (float(box_votes[b_i][ci]) / total if total else 0.0)
            for ci, name in enumerate(CLASSES)
        }
        label = CLASSES[int(np.argmax(box_votes[b_i]))] if total else "unseen"
        is_furniture = fractions.get("furniture", 0.0) >= FURNITURE_CONFIRM_FRACTION
        confirmed += int(is_furniture)
        box_entries.append(
            {
                "box_id": box["box_id"],
                "bbox_min": [round(float(v), 4) for v in box["bbox_min"]],
                "bbox_max": [round(float(v), 4) for v in box["bbox_max"]],
                "source_clusters": box["source_clusters"],
                "views": int(box_views[b_i]),
                "label": label,
                "furniture_confirmed": bool(is_furniture),
                "class_fractions": {
                    k: round(v, 4) for k, v in fractions.items() if v > 0
                },
            }
        )

    elapsed = time.perf_counter() - t0
    params = {
        "grid_m": GRID_M,
        "wall_lift": "ray_plane_v2",
        "ray_block_slack_m": RAY_BLOCK_SLACK_M,
        "mirror_wall_dist_m": MIRROR_WALL_DIST_M,
        "mirror_normal_gate": True,
        "normal_decode": "auto",
        "door_match": "geometric",
        "furniture_confirm_fraction": FURNITURE_CONFIRM_FRACTION,
    }
    windows = sum(1 for o in openings if o["kind"] == "window")
    doors = sum(1 for o in openings if o["kind"] == "door")
    payload = {
        "schema_version": FUSION_SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "params": params,
        "sweep_count": len(metas),
        "opening_count": len(openings),
        "openings": openings,
        "door_cross_check": door_matches,
        "mirror_stats": mirror_stats,
        "furniture_summary": {
            "boxes": len(box_entries),
            "furniture_confirmed": confirmed,
            "seen": int((box_views > 0).sum()),
        },
        "seconds": round(elapsed, 1),
    }
    (out_dir / "fusion.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "openings_fused.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_frame": "runtime_yup",
                "opening_count": len(openings),
                "openings": openings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "furniture_boxes.json").write_text(
        json.dumps(
            {
                "schema_version": FUSION_SCHEMA_VERSION,
                "coordinate_frame": "runtime_yup",
                "boxes": box_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"openings: {windows} window(s), {doors} door(s) on {len(accs)} wall plane(s)"
        + (f"  door cross-check: {door_matches}" if door_matches else "")
    )
    print(
        f"boxes: {len(box_entries)} candidates, {int((box_views > 0).sum())} seen, "
        f"{confirmed} furniture-confirmed"
    )
    print(f"done in {elapsed:.1f} s -> {out_dir}")
    return payload


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4 (G6) slice 3: 2D->3D fusion."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="fuse segmentation masks with G-buffers")
    p.add_argument("--panos", type=Path, required=True, help="extracted_cubemaps dir")
    p.add_argument("--seg-dir", type=Path, default=Path("seg_out"))
    p.add_argument("--renders", type=Path, required=True, help="G-buffer renders dir")
    p.add_argument("--jobs", type=Path, required=True, help="gbuffer_jobs.json")
    p.add_argument(
        "--calibration", type=Path, required=True, help="F5 calibration.json"
    )
    p.add_argument(
        "--planes", type=Path, required=True, help="structural_planes.json"
    )
    p.add_argument(
        "--clusters", type=Path, required=True, help="residual_clusters.json"
    )
    p.add_argument(
        "--g3-openings", type=Path, default=None, help="G3 openings.json (cross-check)"
    )
    p.add_argument("--sweeps", type=str, default=None)
    p.add_argument("--output", "-o", type=Path, default=Path("fusion_out"))
    p.add_argument("--no-review", action="store_true")
    args = parser.parse_args(argv)

    from vs_pipeline.stages.ibl import parse_sweep_selection

    for path, name in (
        (args.panos, "panos dir"),
        (args.seg_dir, "seg dir"),
        (args.renders, "renders dir"),
        (args.jobs, "jobs file"),
        (args.calibration, "calibration file"),
        (args.planes, "planes file"),
        (args.clusters, "clusters file"),
    ):
        if not path.exists():
            parser.error(f"{name} not found: {path}")
    run(
        args.panos,
        args.seg_dir,
        args.renders,
        args.jobs,
        args.calibration,
        args.planes,
        args.clusters,
        args.output,
        g3_openings_path=args.g3_openings,
        sweep_indices=parse_sweep_selection(args.sweeps) if args.sweeps else None,
        write_review=not args.no_review,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
