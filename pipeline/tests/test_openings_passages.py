"""Wide-gap passage mode (openings.py, #12/#23/#26)."""

from __future__ import annotations

from vs_pipeline.stages.openings import detect_wall_openings


def _plane(boundary: list[tuple[float, float]]) -> dict:
    return {
        "plane_id": "wall_test",
        "class": "wall",
        "frame": {"origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 1, 0]},
        "boundary_2d": [list(p) for p in boundary],
    }


def test_wide_passage_detected_with_jambs() -> None:
    # 6 m wall, 2.7 m tall; 2 m floor-reaching gap (u 2..4) with lintel at 2.1.
    plane = _plane(
        [(0, 0), (2, 0), (2, 2.1), (4, 2.1), (4, 0), (6, 0), (6, 2.7), (0, 2.7)]
    )
    openings = detect_wall_openings(plane)
    assert len(openings) == 1
    o = openings[0]
    assert o.get("passage") is True
    assert o["kind"] == "door"  # downstream consumption unchanged
    assert 1.4 < o["width_m"] <= 2.2
    assert 1.9 < o["height_m"] <= 2.3  # floor-reaching up to the lintel


def test_end_touching_wide_gap_rejected() -> None:
    # The gap touches the wall's left end: no left jamb -> ambiguous -> reject.
    plane = _plane([(0, 2.1), (2, 2.1), (2, 0), (6, 0), (6, 2.7), (0, 2.7)])
    assert detect_wall_openings(plane) == []


def test_oversize_gap_rejected() -> None:
    # 4 m gap in an 8 m wall exceeds PASSAGE_MAX_WIDTH_M.
    plane = _plane(
        [(0, 0), (2, 0), (2, 2.1), (6, 2.1), (6, 0), (8, 0), (8, 2.7), (0, 2.7)]
    )
    assert detect_wall_openings(plane) == []
