"""Stage 6 (G3) slice 1 -- wall-line graph, room polygonization, election.

Design (locked in PLAN.md changelog 2026-07-12, brief steps 1-3):
1. Wall-line graph per storey: wall planes project to plan segments via
   endpoint PCA; paired walls collapse into one centerline with an explicit
   thickness; collinear duplicate candidates merge into canonical segments
   (the dedup for G2's tolerated under-merge).
2. Room polygonization: segments are extended slightly to close corner
   gaps, noded (shapely unary_union) and polygonized; a closed face is a
   room when it is floor-covered (>= 30%) or sweep-supported with a lower
   floor cover (>= 10%). The wall-enclosure test falls out of this: floor
   patches mostly OUTSIDE every room are exterior walkable ground (garden).
3. Per-room plane election: the floor/ceiling patch with maximal plan
   overlap wins the room; walls are elected by boundary proximity;
   duplicate/fragment candidates lose the election.

Pure JSON -> JSON: consumes structural_planes.json (+ poses.json for tripod
evidence). No point clouds, runs in seconds, reviewable in 2D renders.

Outputs (runtime Y-up frame, plan view = (x, z)):
    rooms.json                    storeys, rooms, per-room elections
    rooms_review_storey<k>.png    wall lines + room polygons + sweeps

CLI:
    python -m vs_pipeline.stages.rooms build \
        --planes planes_full/structural_planes.json --poses poses.json -o rooms_out
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import polygonize, unary_union

ROOMS_SCHEMA_VERSION = 2

#: Storey detection (same rule family as G2 storey levels v3).
STOREY_BIN_GAP_M = 0.3
MIN_STOREY_PATCH_M2 = 8.0
MIN_STOREY_SWEEPS = 3
TRIPOD_MIN_M = 1.0
TRIPOD_MAX_M = 2.0
CEILING_MIN_ABOVE_M = 1.8
CEILING_MAX_ABOVE_M = 4.5
CEILING_BIN_GAP_M = 0.2
DEFAULT_STOREY_HEIGHT_M = 3.0
#: Walls join a storey graph when their vertical span overlaps it enough.
WALL_MIN_OVERLAP_M = 0.8
#: Canonical segment merging (dedup of under-merged duplicate walls).
SEG_MERGE_ANGLE_DEG = 3.0
SEG_MERGE_OFFSET_M = 0.08
SEG_MERGE_GAP_M = 0.3
MIN_SEGMENT_LENGTH_M = 0.3
#: Segment ends are extended to close corner gaps before polygonization.
SNAP_EXTEND_M = 0.3
#: Dangling wall ends (doorways interrupt wall inliers) are extended toward
#: the nearest wall line, up to a doorway-sized distance.
DANGLE_TOUCH_TOL_M = 0.35
DANGLE_MAX_EXTEND_M = 1.2
#: Room acceptance.
MIN_ROOM_AREA_M2 = 1.5
#: Faces narrower than this are slivers (unpaired opposing faces), not rooms.
MIN_ROOM_WIDTH_M = 0.6
MIN_FLOOR_COVER = 0.3
MIN_FLOOR_COVER_SWEPT = 0.1
#: Floor patches with less of their area inside rooms are exterior ground.
EXTERIOR_INSIDE_MAX = 0.3
#: Wall election: min shared length with the (buffered) room boundary.
WALL_ELECT_BUFFER_M = 0.15
WALL_ELECT_MIN_LENGTH_M = 0.4
#: Exported room outlines are simplified and annotated per edge for the
#: shell stage (interior-face offsetting needs per-edge wall thickness).
POLYGON_SIMPLIFY_M = 0.02
EDGE_MATCH_MAX_DIST_M = 0.3


@dataclass
class Storey:
    storey_id: int
    floor_y: float
    floor_span: tuple[float, float]
    ceiling_y: float | None
    sweep_indices: list[int]


@dataclass
class WallSegment:
    p0: np.ndarray  # plan (x, z)
    p1: np.ndarray
    thickness: float | None
    plane_ids: list[str]
    weight: float


def _plan_points(plane: dict[str, Any]) -> np.ndarray:
    """Plan-view (x, z) points of a plane: boundary if present, else bbox."""
    pts = plane.get("boundary_3d") or []
    if len(pts) >= 2:
        return np.asarray(pts, dtype=np.float64)[:, [0, 2]]
    bmin = np.asarray(plane["bbox_min"], dtype=np.float64)
    bmax = np.asarray(plane["bbox_max"], dtype=np.float64)
    return np.array(
        [
            [bmin[0], bmin[2]],
            [bmax[0], bmin[2]],
            [bmax[0], bmax[2]],
            [bmin[0], bmax[2]],
        ]
    )


def _fit_segment(
    points: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted PCA line fit through plan points -> segment endpoints."""
    w = weights / max(float(weights.sum()), 1e-12)
    mu = (points * w[:, None]).sum(axis=0)
    d = points - mu
    cov = (d * w[:, None]).T @ d
    _, evecs = np.linalg.eigh(cov)
    d2 = evecs[:, -1]
    t = d @ d2
    return mu + d2 * float(t.min()), mu + d2 * float(t.max())


def wall_segments(walls: list[dict[str, Any]]) -> list[WallSegment]:
    """Plan segments from wall planes; paired walls collapse to a centerline.

    The endpoint-PCA fit over BOTH faces of a pair lands on the centerline
    naturally (two parallel point sets -> principal axis along the wall,
    weighted mean between the faces), with the pair's measured thickness.
    """
    by_id = {w["plane_id"]: w for w in walls}
    consumed: set[str] = set()
    segments: list[WallSegment] = []
    for w in walls:
        pid = w["plane_id"]
        if pid in consumed:
            continue
        group = [w]
        thickness = None
        mate_id = w.get("paired_with")
        if mate_id and mate_id in by_id and mate_id not in consumed:
            group.append(by_id[mate_id])
            consumed.add(mate_id)
            thickness = w.get("thickness_m")
        consumed.add(pid)
        pts = np.vstack([_plan_points(g) for g in group])
        p0, p1 = _fit_segment(pts, np.ones(pts.shape[0]))
        if float(np.linalg.norm(p1 - p0)) < MIN_SEGMENT_LENGTH_M:
            continue
        segments.append(
            WallSegment(
                p0=p0,
                p1=p1,
                thickness=thickness,
                plane_ids=[g["plane_id"] for g in group],
                weight=float(sum(g["area_m2"] for g in group)),
            )
        )
    return segments


def _seg_dir(s: WallSegment) -> np.ndarray:
    d = s.p1 - s.p0
    return d / max(float(np.linalg.norm(d)), 1e-9)


def _compatible_segments(a: WallSegment, b: WallSegment) -> bool:
    da, db = _seg_dir(a), _seg_dir(b)
    if abs(float(da @ db)) < np.cos(np.radians(SEG_MERGE_ANGLE_DEG)):
        return False
    na = np.array([-da[1], da[0]])
    if abs(float(na @ (b.p0 - a.p0))) > SEG_MERGE_OFFSET_M:
        return False
    if abs(float(na @ (b.p1 - a.p0))) > SEG_MERGE_OFFSET_M:
        return False
    ta = sorted([0.0, float(da @ (a.p1 - a.p0))])
    tb = sorted([float(da @ (b.p0 - a.p0)), float(da @ (b.p1 - a.p0))])
    return tb[0] <= ta[1] + SEG_MERGE_GAP_M and ta[0] <= tb[1] + SEG_MERGE_GAP_M


def merge_segments(segments: list[WallSegment]) -> list[WallSegment]:
    """Union-find merge of collinear duplicate segments -> canonical walls."""
    n = len(segments)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _compatible_segments(segments[i], segments[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups: dict[int, list[WallSegment]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(segments[i])

    merged: list[WallSegment] = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        pts = np.array([[s.p0, s.p1] for s in group], dtype=np.float64).reshape(-1, 2)
        wts = np.repeat([s.weight for s in group], 2).astype(np.float64)
        p0, p1 = _fit_segment(pts, wts)
        thicknesses = [s.thickness for s in group if s.thickness]
        merged.append(
            WallSegment(
                p0=p0,
                p1=p1,
                thickness=max(thicknesses) if thicknesses else None,
                plane_ids=sorted({pid for s in group for pid in s.plane_ids}),
                weight=float(sum(s.weight for s in group)),
            )
        )
    return merged


def close_dangling_ends(segments: list[WallSegment]) -> list[WallSegment]:
    """Extend dangling segment ends to the nearest wall line (doorway seal).

    Doorway openings interrupt wall inliers, but the wall physically
    continues above the door (lintel), so the wall LINE must continue
    through the opening for room polygonization. An endpoint is dangling
    when no other segment passes within DANGLE_TOUCH_TOL_M; a ray cast
    outward extends it to the first wall hit within DANGLE_MAX_EXTEND_M.
    """
    lines = [LineString([tuple(s.p0), tuple(s.p1)]) for s in segments]
    out: list[WallSegment] = []
    for i, s in enumerate(segments):
        p0, p1 = s.p0.copy(), s.p1.copy()
        d = _seg_dir(s)
        for end in (0, 1):
            p = p0 if end == 0 else p1
            outward = -d if end == 0 else d
            touching = any(
                j != i
                and lines[j].distance(Point(tuple(p))) <= DANGLE_TOUCH_TOL_M
                for j in range(len(lines))
            )
            if touching:
                continue
            ray = LineString(
                [
                    tuple(p + outward * 0.01),
                    tuple(p + outward * DANGLE_MAX_EXTEND_M),
                ]
            )
            best: float | None = None
            for j in range(len(lines)):
                if j == i:
                    continue
                inter = ray.intersection(lines[j])
                if inter.is_empty:
                    continue
                for g in getattr(inter, "geoms", [inter]):
                    for q in getattr(g, "coords", []):
                        t = float(np.hypot(q[0] - p[0], q[1] - p[1]))
                        if best is None or t < best:
                            best = t
            if best is not None:
                if end == 0:
                    p0 = p0 + outward * (best + 0.02)
                else:
                    p1 = p1 + outward * (best + 0.02)
        out.append(
            WallSegment(
                p0=p0,
                p1=p1,
                thickness=s.thickness,
                plane_ids=s.plane_ids,
                weight=s.weight,
            )
        )
    return out


def polygonize_rooms(segments: list[WallSegment]) -> list[Polygon]:
    """Extend segments to close corner gaps, node the network, polygonize."""
    lines = []
    for s in segments:
        d = _seg_dir(s)
        a = s.p0 - d * SNAP_EXTEND_M
        b = s.p1 + d * SNAP_EXTEND_M
        lines.append(LineString([tuple(a), tuple(b)]))
    if not lines:
        return []
    network = unary_union(MultiLineString(lines))
    return list(polygonize(network))


def detect_storeys(
    planes: list[dict[str, Any]], sweeps: list[tuple[int, np.ndarray]]
) -> list[Storey]:
    """Floor elevation bins + tripod evidence -> storeys (G2 v3 rule family)."""
    floors = [
        p
        for p in planes
        if p["class"] == "floor" and p.get("elevation_y") is not None
    ]
    if not floors:
        return []
    ordered = sorted(floors, key=lambda p: p["elevation_y"])
    bins: list[list[dict[str, Any]]] = [[ordered[0]]]
    for f in ordered[1:]:
        if f["elevation_y"] - bins[-1][-1]["elevation_y"] <= STOREY_BIN_GAP_M:
            bins[-1].append(f)
        else:
            bins.append([f])

    storeys: list[Storey] = []
    for b in bins:
        biggest = max(b, key=lambda p: p["area_m2"])
        if biggest["area_m2"] < MIN_STOREY_PATCH_M2:
            continue
        standing = [
            idx
            for idx, pos in sweeps
            if any(
                TRIPOD_MIN_M <= float(pos[1]) - f["elevation_y"] <= TRIPOD_MAX_M
                for f in b
            )
        ]
        if len(standing) < MIN_STOREY_SWEEPS:
            continue
        areas = np.array([p["area_m2"] for p in b])
        elevs = np.array([p["elevation_y"] for p in b])
        floor_y = float((areas * elevs).sum() / areas.sum())
        span = (float(elevs.min()), float(elevs.max()))

        cands = [
            c
            for c in planes
            if c["class"] == "ceiling"
            and c.get("elevation_y") is not None
            and floor_y + CEILING_MIN_ABOVE_M
            < c["elevation_y"]
            < floor_y + CEILING_MAX_ABOVE_M
        ]
        ceiling_y: float | None = None
        if cands:
            cands.sort(key=lambda c: c["elevation_y"])
            cbins: list[list[dict[str, Any]]] = [[cands[0]]]
            for c in cands[1:]:
                if c["elevation_y"] - cbins[-1][-1]["elevation_y"] <= CEILING_BIN_GAP_M:
                    cbins[-1].append(c)
                else:
                    cbins.append([c])
            best = max(cbins, key=lambda cb: sum(c["area_m2"] for c in cb))
            ca = np.array([c["area_m2"] for c in best])
            ce = np.array([c["elevation_y"] for c in best])
            ceiling_y = float((ca * ce).sum() / ca.sum())
        storeys.append(
            Storey(
                storey_id=len(storeys),
                floor_y=floor_y,
                floor_span=span,
                ceiling_y=ceiling_y,
                sweep_indices=sorted(standing),
            )
        )
    return storeys


def storey_walls(
    planes: list[dict[str, Any]], storey: Storey
) -> list[dict[str, Any]]:
    top = (
        storey.ceiling_y
        if storey.ceiling_y is not None
        else storey.floor_y + DEFAULT_STOREY_HEIGHT_M
    )
    out = []
    for p in planes:
        if p["class"] != "wall":
            continue
        y0, y1 = float(p["bbox_min"][1]), float(p["bbox_max"][1])
        overlap = min(y1, top) - max(y0, storey.floor_y)
        if overlap >= min(WALL_MIN_OVERLAP_M, 0.5 * (top - storey.floor_y)):
            out.append(p)
    return out


def _poly_from_plane(plane: dict[str, Any]) -> Polygon:
    pts = _plan_points(plane)
    if pts.shape[0] >= 3:
        poly = Polygon([tuple(q) for q in pts])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            return poly
    bmin = plane["bbox_min"]
    bmax = plane["bbox_max"]
    return Polygon(
        [
            (bmin[0], bmin[2]),
            (bmax[0], bmin[2]),
            (bmax[0], bmax[2]),
            (bmin[0], bmax[2]),
        ]
    )


def _elect(
    face: Polygon, patches: list[tuple[dict[str, Any], Polygon]]
) -> tuple[dict[str, Any] | None, float]:
    """Patch with maximal overlap with the face; returns (patch, coverage)."""
    best: dict[str, Any] | None = None
    best_area = 0.0
    for plane, poly in patches:
        if poly.is_empty:
            continue
        inter = face.intersection(poly).area
        if inter > best_area:
            best, best_area = plane, inter
    return best, (best_area / face.area if face.area > 0 else 0.0)


def build_rooms(
    planes: list[dict[str, Any]], sweeps: list[tuple[int, np.ndarray]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Core pass: storeys -> wall graph -> rooms -> elections -> exteriors.

    Returns (rooms.json payload, per-storey review data for rendering).
    """
    storeys = detect_storeys(planes, sweeps)
    rooms_payload: list[dict[str, Any]] = []
    exterior_payload: list[dict[str, Any]] = []
    storey_payload: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []

    for st in storeys:
        segments = close_dangling_ends(
            merge_segments(wall_segments(storey_walls(planes, st)))
        )
        faces = polygonize_rooms(segments)

        lo, hi = st.floor_span[0] - 0.15, st.floor_span[1] + 0.15
        floor_patches = [
            (p, _poly_from_plane(p))
            for p in planes
            if p["class"] == "floor"
            and p.get("elevation_y") is not None
            and lo <= p["elevation_y"] <= hi
        ]
        ceiling_patches = [
            (p, _poly_from_plane(p))
            for p in planes
            if p["class"] == "ceiling"
            and p.get("elevation_y") is not None
            and st.floor_y + CEILING_MIN_ABOVE_M
            < p["elevation_y"]
            < st.floor_y + CEILING_MAX_ABOVE_M
        ]
        floor_union = unary_union([poly for _, poly in floor_patches])
        sweep_pts = [
            (idx, Point(float(pos[0]), float(pos[2])))
            for idx, pos in sweeps
            if idx in set(st.sweep_indices)
        ]

        accepted: list[tuple[str, Polygon]] = []
        for face in sorted(faces, key=lambda f: -f.area):
            if face.area < MIN_ROOM_AREA_M2:
                continue
            if face.buffer(-MIN_ROOM_WIDTH_M / 2).is_empty:
                continue  # sliver between unpaired opposing wall faces
            cover = (
                face.intersection(floor_union).area / face.area
                if not floor_union.is_empty
                else 0.0
            )
            inside_sweeps = [i for i, pt in sweep_pts if face.contains(pt)]
            if not (
                cover >= MIN_FLOOR_COVER
                or (inside_sweeps and cover >= MIN_FLOOR_COVER_SWEPT)
            ):
                continue
            floor_plane, floor_cov = _elect(face, floor_patches)
            ceiling_plane, ceiling_cov = _elect(face, ceiling_patches)
            ring = face.exterior
            ring_buf = ring.buffer(WALL_ELECT_BUFFER_M)
            wall_ids: list[str] = []
            for s in segments:
                line = LineString([tuple(s.p0), tuple(s.p1)])
                if line.intersection(ring_buf).length >= WALL_ELECT_MIN_LENGTH_M:
                    wall_ids.extend(s.plane_ids)
            outline = face.simplify(POLYGON_SIMPLIFY_M)
            if outline.is_empty or outline.geom_type != "Polygon":
                outline = face
            coords = list(outline.exterior.coords)
            edges: list[dict[str, Any]] = []
            for a, b in zip(coords[:-1], coords[1:]):
                mid = Point((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                best: WallSegment | None = None
                best_d = EDGE_MATCH_MAX_DIST_M
                for s in segments:
                    dist = LineString([tuple(s.p0), tuple(s.p1)]).distance(mid)
                    if dist < best_d:
                        best_d, best = dist, s
                edges.append(
                    {
                        "p0": [round(float(a[0]), 4), round(float(a[1]), 4)],
                        "p1": [round(float(b[0]), 4), round(float(b[1]), 4)],
                        "thickness_m": best.thickness if best else None,
                        "wall_planes": best.plane_ids if best else [],
                    }
                )
            room_id = f"s{st.storey_id}_room_{len(accepted):02d}"
            rooms_payload.append(
                {
                    "room_id": room_id,
                    "storey_id": st.storey_id,
                    "area_m2": round(face.area, 3),
                    "polygon_plan": [
                        [round(float(x), 4), round(float(z), 4)] for x, z in coords
                    ],
                    "edges": edges,
                    "floor_plane": floor_plane["plane_id"] if floor_plane else None,
                    "floor_elevation_y": (
                        floor_plane["elevation_y"] if floor_plane else None
                    ),
                    "floor_coverage": round(floor_cov, 3),
                    "ceiling_plane": (
                        ceiling_plane["plane_id"] if ceiling_plane else None
                    ),
                    "ceiling_elevation_y": (
                        ceiling_plane["elevation_y"] if ceiling_plane else None
                    ),
                    "ceiling_coverage": round(ceiling_cov, 3),
                    "wall_planes": sorted(set(wall_ids)),
                    "sweeps": inside_sweeps,
                }
            )
            accepted.append((room_id, face))

        room_union = (
            unary_union([f for _, f in accepted]) if accepted else None
        )
        exterior_polys: list[Polygon] = []
        for p, poly in floor_patches:
            if poly.is_empty or poly.area <= 0:
                continue
            inside = (
                poly.intersection(room_union).area / poly.area
                if room_union is not None
                else 0.0
            )
            if inside < EXTERIOR_INSIDE_MAX:
                exterior_payload.append(
                    {
                        "plane_id": p["plane_id"],
                        "storey_id": st.storey_id,
                        "area_m2": p["area_m2"],
                        "elevation_y": p["elevation_y"],
                        "inside_fraction": round(inside, 3),
                    }
                )
                exterior_polys.append(poly)

        storey_payload.append(
            {
                "storey_id": st.storey_id,
                "floor_y": round(st.floor_y, 4),
                "ceiling_y": (
                    round(st.ceiling_y, 4) if st.ceiling_y is not None else None
                ),
                "sweep_count": len(st.sweep_indices),
                "room_count": len(accepted),
            }
        )
        review.append(
            {
                "storey": st,
                "segments": segments,
                "rooms": accepted,
                "exterior_polys": exterior_polys,
                "sweep_pts": sweep_pts,
            }
        )

    payload = {
        "schema_version": ROOMS_SCHEMA_VERSION,
        "coordinate_frame": "runtime_yup",
        "storeys": storey_payload,
        "rooms": rooms_payload,
        "exterior_floors": exterior_payload,
    }
    return payload, review


def _palette(index: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(index * 7919 + 13)
    c = (rng.uniform(0.35, 1.0, size=3) * 255).astype(int)
    return int(c[0]), int(c[1]), int(c[2])


def _exterior_rings(geom: Any) -> list[Any]:
    """Exterior rings of a Polygon or MultiPolygon (render helper).

    _poly_from_plane repairs invalid boundaries with buffer(0), which can
    legally return a MultiPolygon (pinched/self-crossing terrain patches).
    """
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom.exterior]
    if geom.geom_type == "MultiPolygon":
        return [g.exterior for g in geom.geoms]
    return []


def render_review(review: list[dict[str, Any]], out_dir: Path, size_px: int = 1600) -> None:
    """Per-storey 2D plan render: walls, room polygons, exteriors, sweeps."""
    for item in review:
        st: Storey = item["storey"]
        xs: list[float] = []
        zs: list[float] = []
        for s in item["segments"]:
            xs.extend([float(s.p0[0]), float(s.p1[0])])
            zs.extend([float(s.p0[1]), float(s.p1[1])])
        for poly in item["exterior_polys"]:
            bx0, bz0, bx1, bz1 = poly.bounds
            xs.extend([bx0, bx1])
            zs.extend([bz0, bz1])
        for _, pt in item["sweep_pts"]:
            xs.append(pt.x)
            zs.append(pt.y)
        if not xs:
            continue
        xmin, zmin = min(xs), min(zs)
        extent = max(max(xs) - xmin, max(zs) - zmin, 1e-6)
        scale = (size_px - 40) / extent
        width = int((max(xs) - xmin) * scale) + 40
        height = int((max(zs) - zmin) * scale) + 40

        def px(x: float, z: float) -> tuple[int, int]:
            return int((x - xmin) * scale) + 20, int((z - zmin) * scale) + 20

        image = np.zeros((height, width, 3), dtype=np.uint8)
        overlay = image.copy()
        for k, (room_id, face) in enumerate(item["rooms"]):
            for ring in _exterior_rings(face):
                pts = np.array(
                    [px(x, z) for x, z in ring.coords], dtype=np.int32
                )
                cv2.fillPoly(overlay, [pts], _palette(k))
        image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)
        for poly in item["exterior_polys"]:
            for ring in _exterior_rings(poly):
                pts = np.array(
                    [px(x, z) for x, z in ring.coords], dtype=np.int32
                )
                cv2.polylines(image, [pts], True, (110, 110, 110), 1)
        for s in item["segments"]:
            th = max(2, int(round((s.thickness or 0.06) * scale)))
            cv2.line(
                image,
                px(float(s.p0[0]), float(s.p0[1])),
                px(float(s.p1[0]), float(s.p1[1])),
                (255, 255, 255),
                th,
            )
        for idx, pt in item["sweep_pts"]:
            u, v = px(pt.x, pt.y)
            cv2.drawMarker(
                image, (u, v), (0, 220, 255), cv2.MARKER_CROSS, 12, 2
            )
            cv2.putText(
                image,
                str(idx),
                (u + 8, v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        for k, (room_id, face) in enumerate(item["rooms"]):
            c = face.representative_point()
            cv2.putText(
                image,
                room_id.split("_", 1)[1],
                px(c.x, c.y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(out_dir / f"rooms_review_storey{st.storey_id}.png"), image)


def build_property(planes_path: Path, poses_path: Path, out_dir: Path) -> dict[str, Any]:
    from vs_pipeline.stages.pose import read_poses_json

    data = json.loads(planes_path.read_text(encoding="utf-8"))
    planes = data["planes"]
    sweeps = [(s.index, s.pose_rt[:3, 3]) for s in read_poses_json(poses_path)]
    payload, review = build_rooms(planes, sweeps)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rooms.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    render_review(review, out_dir)

    for st in payload["storeys"]:
        print(
            f"storey {st['storey_id']}: floor y = {st['floor_y']:+.3f}, "
            f"ceiling y = {st['ceiling_y']}, sweeps {st['sweep_count']}, "
            f"rooms {st['room_count']}"
        )
    for r in payload["rooms"]:
        print(
            f"  {r['room_id']:<14} {r['area_m2']:7.2f} m^2  "
            f"floor {r['floor_plane']} ({r['floor_coverage']:.0%})  "
            f"ceiling {r['ceiling_plane']} ({r['ceiling_coverage']:.0%})  "
            f"walls {len(r['wall_planes'])}  sweeps {len(r['sweeps'])}"
        )
    ext_area = sum(e["area_m2"] for e in payload["exterior_floors"])
    print(
        f"exterior floors: {len(payload['exterior_floors'])} "
        f"({ext_area:.1f} m^2) -- excluded from shell synthesis"
    )
    print(f"outputs -> {out_dir}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 6 (G3) slice 1: wall-line graph -> rooms -> election."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="storeys -> wall graph -> rooms")
    p_build.add_argument(
        "--planes", type=Path, required=True, help="structural_planes.json"
    )
    p_build.add_argument(
        "--poses", type=Path, required=True, help="poses.json (tripod evidence)"
    )
    p_build.add_argument("--output", "-o", type=Path, default=Path("rooms_out"))
    args = parser.parse_args(argv)

    if not args.planes.is_file():
        parser.error(f"planes file not found: {args.planes}")
    if not args.poses.is_file():
        parser.error(f"poses file not found: {args.poses}")
    build_property(args.planes, args.poses, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
