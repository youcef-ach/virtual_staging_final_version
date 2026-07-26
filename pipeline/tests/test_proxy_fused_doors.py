"""Fused-door consumption gates (proxy --doors-fused, #23/#12)."""

from __future__ import annotations

from shapely.geometry import Polygon

from vs_pipeline.stages.proxy import _room_edge_openings


def _fused_door(x0: float, x1: float, z: float, y_top: float) -> dict:
    return {
        "kind": "door",
        "source": "fusion",
        "corners_3d": [
            [x0, 0.05, z],
            [x1, 0.05, z],
            [x1, y_top, z],
            [x0, y_top, z],
        ],
    }


def test_fused_wide_passage_cuts_floor_reaching_hole() -> None:
    # 2 m passage -- wider than G3's 1.4 m jamb-detection ceiling (#12).
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    rects, count = _room_edge_openings(
        poly, [], {}, [_fused_door(1.0, 3.0, 0.0, 2.05)], 0.0, 2.7
    )
    assert count == 1
    hit = [r for edge in rects for r in edge]
    s0, s1, y0, y1 = hit[0]
    assert s1 - s0 > 1.4  # the class of opening G3 cannot produce
    assert y0 < 0.0  # floor-reaching (door semantics)
    assert abs(y1 - 2.05) < 0.15


def test_fused_low_rect_rejected() -> None:
    # Partial/low rect (#21 evidence splitting): a half-height cutout would
    # leave a phantom LINTEL -- must be rejected, not consumed.
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    _rects, count = _room_edge_openings(
        poly, [], {}, [_fused_door(1.0, 3.0, 0.0, 1.0)], 0.0, 2.7
    )
    assert count == 0


def test_fused_oversize_width_rejected() -> None:
    # A wall-sized false rect must not delete the wall.
    poly = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])
    _rects, count = _room_edge_openings(
        poly, [], {}, [_fused_door(1.0, 6.5, 0.0, 2.05)], 0.0, 2.7
    )
    assert count == 0
