"""Tests for Stage 6 (G3) slice 2: inset sealing + shell meshes."""

from __future__ import annotations

from shapely.geometry import Polygon

from vs_pipeline.stages.proxy import inset_room_polygon, room_shell_mesh


def test_inset_square_seals_corners():
    ring = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    poly = inset_room_polygon(ring, [0.1, 0.1, 0.1, 0.1])
    assert poly is not None
    assert abs(poly.area - 3.8 * 3.8) < 1e-9
    xs = sorted({round(x, 6) for x, _ in poly.exterior.coords})
    assert xs[0] == 0.1 and xs[-1] == 3.9


def test_inset_respects_per_edge_thickness():
    ring = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    # one thick wall (0.3 inset) on the first edge, thin elsewhere
    poly = inset_room_polygon(ring, [0.3, 0.05, 0.05, 0.05])
    assert poly is not None
    assert abs(poly.area - 3.9 * 3.65) < 1e-9


def test_room_shell_mesh_watertight_and_budget():
    import trimesh

    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    parts = room_shell_mesh(poly, 0.0, 2.7)
    assert set(parts) == {"floor", "ceiling", "wall_00", "wall_01", "wall_02", "wall_03"}
    total = sum(len(m.faces) for m in parts.values())
    assert total == 12  # 2 caps x 2 tris + 4 wall quads x 2 tris
    merged = trimesh.util.concatenate(list(parts.values()))
    merged.merge_vertices()
    assert merged.is_watertight
    assert abs(float(merged.bounds[0][1]) - 0.0) < 1e-9
    assert abs(float(merged.bounds[1][1]) - 2.7) < 1e-9


def test_window_hole_in_wall():
    """A 4-tuple rect cuts an INTERIOR hole: wall area drops by the hole,
    and the wall face still spans floor..ceiling (the hole never touches
    the boundary -- unlike a door notch)."""
    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    rects = [[(1.0, 2.2, 0.9, 2.0)], [], [], []]  # 1.2 x 1.1 m window
    parts = room_shell_mesh(poly, 0.0, 2.7, rects)
    wall = parts["wall_00"]
    full = 4.0 * 2.7
    hole = 1.2 * 1.1
    assert abs(float(wall.area) - (full - hole)) < 0.05
    # Interior hole: the wall still reaches both floor and ceiling.
    assert abs(float(wall.bounds[0][1]) - 0.0) < 1e-6
    assert abs(float(wall.bounds[1][1]) - 2.7) < 1e-6


def test_merge_rects_pair_face_dedup_but_not_across_y():
    from vs_pipeline.stages.proxy import _merge_rects

    # Pair-face duplicates of the same door -> one rect.
    merged = _merge_rects([(1.0, 1.9, -0.01, 2.0), (1.02, 1.92, -0.01, 2.05)])
    assert len(merged) == 1
    s0, s1, y0, y1 = merged[0]
    assert s0 == 1.0 and abs(s1 - 1.92) < 1e-9 and abs(y1 - 2.05) < 1e-9
    # A door and a HIGH window at the same s must stay separate.
    kept = _merge_rects([(1.0, 1.9, -0.01, 1.0), (1.0, 1.9, 1.8, 2.4)])
    assert len(kept) == 2


def test_window_assigned_geometrically_without_plane_id():
    """Fusion windows carry plane ids the room's edge list may never
    reference (duplicate walls) -- assignment must be geometric."""
    from vs_pipeline.stages.proxy import _room_edge_openings

    poly = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    edges = [{"wall_planes": []} for _ in range(4)]  # no ids known
    win = {
        "plane_id": "wall_duplicate_999",
        "kind": "window",
        "corners_3d": [
            [1.0, 0.9, 0.05],
            [2.2, 0.9, 0.05],
            [2.2, 2.0, 0.05],
            [1.0, 2.0, 0.05],
        ],
    }
    far = {
        "plane_id": "wall_far",
        "kind": "window",
        "corners_3d": [
            [1.0, 0.9, 9.0],
            [2.2, 0.9, 9.0],
            [2.2, 2.0, 9.0],
            [1.0, 2.0, 9.0],
        ],
    }
    rects, count = _room_edge_openings(poly, edges, {}, [win, far], 0.0, 2.7)
    assert count == 1  # near window assigned, far one rejected by proximity
    flat = [r for edge in rects for r in edge]
    (s0, s1, y0, y1) = flat[0]
    assert abs((s1 - s0) - 1.2) < 0.05
    assert abs(y0 - 0.9) < 0.05 and abs(y1 - 2.0) < 0.05
