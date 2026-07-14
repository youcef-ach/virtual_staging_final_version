"""Tests for Stage 6 (G3) slice 3: door opening detection."""

from __future__ import annotations

from vs_pipeline.stages.openings import detect_wall_openings


def _wall_with_door():
    """4 x 2.7 m wall with a 0.9 m door notch (u 1.5-2.4, up to v 2.0)."""
    boundary = [
        [0.0, 0.0],
        [1.5, 0.0],
        [1.5, 2.0],
        [2.4, 2.0],
        [2.4, 0.0],
        [4.0, 0.0],
        [4.0, 2.7],
        [0.0, 2.7],
    ]
    return {
        "plane_id": "wall_00",
        "class": "wall",
        "frame": {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0]},
        "boundary_2d": boundary,
    }


def test_detect_door_notch():
    openings = detect_wall_openings(_wall_with_door())
    assert len(openings) == 1
    o = openings[0]
    assert o["kind"] == "door"
    assert 0.7 < o["width_m"] < 1.1
    assert 1.8 < o["height_m"] < 2.2
    ys = [c[1] for c in o["corners_3d"]]
    assert min(ys) < 0.1
    assert 1.8 < max(ys) < 2.2


def test_solid_wall_has_no_openings():
    wall = _wall_with_door()
    wall["boundary_2d"] = [[0.0, 0.0], [4.0, 0.0], [4.0, 2.7], [0.0, 2.7]]
    assert detect_wall_openings(wall) == []


def test_wall_opening_cut_reduces_area():
    from shapely.geometry import Polygon

    from vs_pipeline.stages.proxy import room_shell_mesh

    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    rects = [[(1.0, 2.0, 2.0)], [], [], []]
    parts = room_shell_mesh(poly, 0.0, 2.7, rects)
    full = 4.0 * 2.7
    cut = float(parts["wall_00"].area)
    assert abs(cut - (full - 1.0 * 2.0)) < 0.1
