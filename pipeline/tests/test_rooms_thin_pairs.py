"""G3 fix (2026-07-18): paper-thin wall pairs leave the wall-line graph.

Verified defect: pair wall_389+wall_629 (thickness 3.3 mm) split the real
living room with a phantom interior wall (s1_room_01/wall_04, G7.1 triage).
A real wall is never thinner than MIN_PAIR_THICKNESS_M; such pairs are
freestanding thin objects (curtains, glass panels) or flipped-normal
duplicate fits, and BOTH planes must be excluded (not merely unpaired --
unpaired planes re-enter as segments).
"""

from vs_pipeline.stages.rooms import MIN_PAIR_THICKNESS_M, wall_segments


def _wall(pid: str, mate: str, thickness: float, x: float) -> dict:
    return {
        "plane_id": pid,
        "paired_with": mate,
        "thickness_m": thickness,
        "area_m2": 10.0,
        "bbox_min": [x, 0.0, 0.0],
        "bbox_max": [x, 2.5, 5.0],
    }


def test_paper_thin_pair_dropped_entirely():
    thin = [
        _wall("wall_a", "wall_b", 0.0033, 0.0),
        _wall("wall_b", "wall_a", 0.0033, 0.0033),
    ]
    real = [
        _wall("wall_c", "wall_d", 0.15, 3.0),
        _wall("wall_d", "wall_c", 0.15, 3.15),
    ]
    segments = wall_segments(thin + real)
    ids = {pid for s in segments for pid in s.plane_ids}
    assert "wall_a" not in ids and "wall_b" not in ids, "thin pair must vanish"
    assert {"wall_c", "wall_d"} <= ids, "real pair must survive"


def test_threshold_boundary_pair_survives():
    at_threshold = [
        _wall("wall_e", "wall_f", MIN_PAIR_THICKNESS_M, 0.0),
        _wall("wall_f", "wall_e", MIN_PAIR_THICKNESS_M, MIN_PAIR_THICKNESS_M),
    ]
    segments = wall_segments(at_threshold)
    ids = {pid for s in segments for pid in s.plane_ids}
    assert {"wall_e", "wall_f"} <= ids


def test_unpaired_walls_unaffected():
    solo = [
        {
            "plane_id": "wall_solo",
            "paired_with": None,
            "thickness_m": None,
            "area_m2": 8.0,
            "bbox_min": [1.0, 0.0, 0.0],
            "bbox_max": [1.0, 2.5, 4.0],
        }
    ]
    segments = wall_segments(solo)
    assert any("wall_solo" in s.plane_ids for s in segments)
