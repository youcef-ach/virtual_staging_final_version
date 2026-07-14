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
