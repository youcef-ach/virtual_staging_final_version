"""Tests for Stage 6 (G3) slice 1: wall graph, polygonization, election."""

from __future__ import annotations

import numpy as np

from vs_pipeline.stages.rooms import (
    build_rooms,
    merge_segments,
    wall_segments,
)


def _wall(pid, p0, p1, y0=0.0, y1=2.7, paired=None, thickness=None):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    corners = [
        [p0[0], y0, p0[1]],
        [p1[0], y0, p1[1]],
        [p1[0], y1, p1[1]],
        [p0[0], y1, p0[1]],
    ]
    xs = [p0[0], p1[0]]
    zs = [p0[1], p1[1]]
    length = float(np.linalg.norm(p1 - p0))
    return {
        "plane_id": pid,
        "class": "wall",
        "coefficients": [
            float(n[0]),
            0.0,
            float(n[1]),
            float(-(n[0] * p0[0] + n[1] * p0[1])),
        ],
        "inlier_count": 1000,
        "area_m2": length * (y1 - y0),
        "centroid": [
            float((p0[0] + p1[0]) / 2),
            (y0 + y1) / 2,
            float((p0[1] + p1[1]) / 2),
        ],
        "bbox_min": [min(xs), y0, min(zs)],
        "bbox_max": [max(xs), y1, max(zs)],
        "elevation_y": None,
        "thickness_m": thickness,
        "paired_with": paired,
        "member_sweeps": [],
        "boundary_2d": [],
        "boundary_3d": corners,
    }


def _horizontal(pid, cls, x0, z0, x1, z1, y):
    ny = 1.0 if cls == "floor" else -1.0
    corners = [[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]]
    return {
        "plane_id": pid,
        "class": cls,
        "coefficients": [0.0, ny, 0.0, float(-ny * y)],
        "inlier_count": 1000,
        "area_m2": abs(x1 - x0) * abs(z1 - z0),
        "centroid": [(x0 + x1) / 2, y, (z0 + z1) / 2],
        "bbox_min": [min(x0, x1), y, min(z0, z1)],
        "bbox_max": [max(x0, x1), y, max(z0, z1)],
        "elevation_y": y,
        "thickness_m": None,
        "paired_with": None,
        "member_sweeps": [],
        "boundary_2d": [],
        "boundary_3d": corners,
    }


def _two_room_planes():
    """8 x 4 outline with a divider at x = 4 -> two 4 x 4 rooms."""
    return [
        _wall("wall_00", (0, 0), (8, 0)),
        _wall("wall_01", (8, 0), (8, 4)),
        _wall("wall_02", (8, 4), (0, 4)),
        _wall("wall_03", (0, 4), (0, 0)),
        _wall("wall_04", (4, 0), (4, 4)),
        _horizontal("floor_00", "floor", 0, 0, 4, 4, 0.0),
        _horizontal("floor_01", "floor", 4, 0, 8, 4, 0.0),
        _horizontal("ceiling_00", "ceiling", 0, 0, 4, 4, 2.7),
        _horizontal("ceiling_01", "ceiling", 4, 0, 8, 4, 2.7),
    ]


def _sweeps(positions):
    return [(i, np.asarray(p, dtype=float)) for i, p in enumerate(positions)]


def test_two_rooms_polygonize_and_elect():
    sweeps = _sweeps(
        [[2, 1.4, 2], [2.5, 1.4, 1.5], [6, 1.4, 2], [6.5, 1.4, 2.5]]
    )
    payload, _ = build_rooms(_two_room_planes(), sweeps)

    assert len(payload["storeys"]) == 1
    assert payload["storeys"][0]["ceiling_y"] is not None
    rooms = payload["rooms"]
    assert len(rooms) == 2
    for room in rooms:
        assert 12.0 < room["area_m2"] < 20.0
        assert room["floor_coverage"] > 0.8
        assert room["ceiling_plane"] is not None
        assert len(room["wall_planes"]) >= 4
    by_floor = {r["floor_plane"] for r in rooms}
    assert by_floor == {"floor_00", "floor_01"}


def test_garden_floor_reported_exterior():
    planes = _two_room_planes() + [
        _horizontal("floor_02", "floor", 12, 0, 16, 4, 0.0)
    ]
    sweeps = _sweeps(
        [
            [2, 1.4, 2],
            [2.5, 1.4, 1.5],
            [6, 1.4, 2],
            [14, 1.4, 2],  # standing on the garden patch
        ]
    )
    payload, _ = build_rooms(planes, sweeps)

    assert len(payload["rooms"]) == 2  # garden creates no room (no walls)
    exterior = {e["plane_id"] for e in payload["exterior_floors"]}
    assert "floor_02" in exterior
    assert "floor_00" not in exterior
    assert "floor_01" not in exterior


def test_duplicate_walls_merge_to_one_canonical_segment():
    a = _wall("wall_10", (0, 0), (6, 0))
    b = _wall("wall_11", (0.1, 0.03), (5.9, 0.03))  # near-collinear duplicate
    merged = merge_segments(wall_segments([a, b]))
    assert len(merged) == 1
    assert set(merged[0].plane_ids) == {"wall_10", "wall_11"}


def test_paired_walls_collapse_to_centerline_with_thickness():
    a = _wall("wall_20", (0, 0), (6, 0), paired="wall_21", thickness=0.2)
    b = _wall("wall_21", (0, 0.2), (6, 0.2), paired="wall_20", thickness=0.2)
    segs = wall_segments([a, b])
    assert len(segs) == 1
    assert segs[0].thickness == 0.2
    # centerline sits midway between the faces
    assert abs(float(segs[0].p0[1]) - 0.1) < 0.02
    assert abs(float(segs[0].p1[1]) - 0.1) < 0.02


def test_doorway_gap_closed_by_dangling_extension():
    """A 0.8 m doorway gap in one wall must not break the room."""
    planes = [
        _wall("wall_30", (0, 0), (1.6, 0)),
        _wall("wall_31", (2.4, 0), (4, 0)),  # 0.8 m doorway between them
        _wall("wall_32", (4, 0), (4, 4)),
        _wall("wall_33", (4, 4), (0, 4)),
        _wall("wall_34", (0, 4), (0, 0)),
        _horizontal("floor_00", "floor", 0, 0, 4, 4, 0.0),
        _horizontal("ceiling_00", "ceiling", 0, 0, 4, 4, 2.7),
    ]
    sweeps = _sweeps([[2, 1.4, 2], [1, 1.4, 1], [3, 1.4, 3]])
    payload, _ = build_rooms(planes, sweeps)
    assert len(payload["rooms"]) == 1
    assert 12.0 < payload["rooms"][0]["area_m2"] < 20.0


def test_sliver_between_opposing_faces_rejected():
    """A 0.5 m wedge between unpaired opposing wall faces is not a room."""
    planes = _two_room_planes() + [
        _wall("wall_40", (4.5, 0), (4.5, 4))  # opposing face of the divider
    ]
    sweeps = _sweeps(
        [[2, 1.4, 2], [2.5, 1.4, 1.5], [6, 1.4, 2], [6.5, 1.4, 2.5]]
    )
    payload, _ = build_rooms(planes, sweeps)
    assert len(payload["rooms"]) == 2  # sliver face (0.5 x 4 m) dropped


def test_ceiling_closure_recovers_glass_wall_room():
    """A room whose 4th side is glass (no wall plane) closes via its ceiling."""
    planes = [
        _wall("wall_50", (0, 0), (4, 0)),
        _wall("wall_51", (4, 0), (4, 4)),
        _wall("wall_52", (0, 4), (0, 0)),
        # north side (0,4)-(4,4): glass, no LiDAR wall returns
        _horizontal("floor_00", "floor", 0, 0, 4, 4, 0.0),
        _horizontal("ceiling_00", "ceiling", 0, 0, 4, 4, 2.7),
    ]
    sweeps = _sweeps([[2, 1.4, 2], [1, 1.4, 1], [3, 1.4, 3]])
    payload, _ = build_rooms(planes, sweeps)
    assert len(payload["rooms"]) == 1
    assert 12.0 < payload["rooms"][0]["area_m2"] < 20.0
