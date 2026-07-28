"""Tests for vs_pipeline.stages.navigation (T1 navigation graph)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vs_pipeline.stages.navigation import build_navigation_graph


def _make_pose(uid: str, index: int, x: float, y: float, z: float) -> dict:
    """Minimal pose entry with position at (x, y, z)."""
    return {
        "sweep_uuid": uid,
        "index": index,
        "pose_rt": [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
    }


def _make_room(
    rid: str, sweeps: list[int], wall_planes: list[str],
    floor_y: float = 0.0, ceil_y: float = 2.5,
) -> dict:
    """Minimal room with sweep indices and wall plane ids."""
    return {
        "room_id": rid,
        "storey_id": int(rid.split("_")[0][1:]),
        "area_m2": 10.0,
        "polygon_plan": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
        "edges": [{"wall_planes": wall_planes}],
        "floor_plane": None,
        "floor_elevation_y": floor_y,
        "floor_coverage": 1.0,
        "ceiling_plane": None,
        "ceiling_elevation_y": ceil_y,
        "ceiling_coverage": 1.0,
        "wall_planes": wall_planes,
        "sweeps": sweeps,
    }


def _make_opening(plane_id: str, kind: str = "door", cx: float = 0.0, cy: float = 1.0, cz: float = 0.0) -> dict:
    """Minimal opening with 4 corners around a centroid."""
    hw = 0.4
    hh = 1.0
    return {
        "plane_id": plane_id,
        "kind": kind,
        "u_range": [-hw, hw],
        "v_range": [0, 2 * hh],
        "width_m": 2 * hw,
        "height_m": 2 * hh,
        "corners_3d": [
            [cx - hw, cy - hh, cz],
            [cx + hw, cy - hh, cz],
            [cx + hw, cy + hh, cz],
            [cx - hw, cy + hh, cz],
        ],
    }


def _write_fixtures(
    tmp_path: Path,
    poses: list[dict],
    rooms: list[dict],
    openings: list[dict],
) -> tuple[Path, Path, Path, Path]:
    """Write poses/rooms/openings JSON fixtures and return paths."""
    poses_path = tmp_path / "poses.json"
    rooms_path = tmp_path / "rooms.json"
    openings_path = tmp_path / "openings.json"
    output_path = tmp_path / "nav_out" / "navigation_graph.json"

    poses_path.write_text(json.dumps({
        "schema_version": 1,
        "coordinate_frame": "runtime_yup",
        "conversion": "z_up_to_y_up",
        "sweep_count": len(poses),
        "sweeps": poses,
    }))
    rooms_path.write_text(json.dumps({
        "schema_version": 2,
        "rooms": rooms,
        "build_params": {},
    }))
    openings_path.write_text(json.dumps({
        "schema_version": 1,
        "coordinate_frame": "runtime_yup",
        "opening_count": len(openings),
        "openings": openings,
    }))
    return poses_path, rooms_path, openings_path, output_path


def test_two_rooms_with_door_creates_portal_edge(tmp_path: Path) -> None:
    """A door between two rooms should produce a portal edge."""
    poses = [
        _make_pose("aaa", 0, -2.0, 1.0, 0.0),
        _make_pose("bbb", 1, 2.0, 1.0, 0.0),
    ]
    rooms = [
        _make_room("s0_room_00", [0], ["wall_shared"]),
        _make_room("s0_room_01", [1], ["wall_shared"]),
    ]
    openings = [_make_opening("wall_shared", "door", cx=0.0, cy=1.0, cz=0.0)]
    pp, rp, op, out = _write_fixtures(tmp_path, poses, rooms, openings)

    result = build_navigation_graph(pp, rp, op, None, out)

    assert result["node_count"] == 2
    portal_edges = [e for e in result["edges"] if e["portal"]]
    assert len(portal_edges) == 1
    assert portal_edges[0]["opening_kind"] == "door"
    assert portal_edges[0]["opening_plane"] == "wall_shared"
    assert portal_edges[0]["distance_m"] == pytest.approx(4.0, abs=0.01)


def test_intra_room_all_pairs_connected(tmp_path: Path) -> None:
    """All sweeps in one room should be pairwise connected."""
    poses = [
        _make_pose("s1", 0, 0.0, 1.0, 0.0),
        _make_pose("s2", 1, 1.0, 1.0, 0.0),
        _make_pose("s3", 2, 0.5, 1.0, 1.0),
    ]
    rooms = [_make_room("s0_room_00", [0, 1, 2], ["w1"])]
    pp, rp, op, out = _write_fixtures(tmp_path, poses, rooms, [])

    result = build_navigation_graph(pp, rp, op, None, out)

    # 3 sweeps in one room -> 3 choose 2 = 3 edges
    assert result["edge_count"] == 3
    assert all(not e["portal"] for e in result["edges"])


def test_portal_edge_distance_cap(tmp_path: Path) -> None:
    """Portal edges exceeding MAX_PORTAL_DISTANCE_M should be dropped."""
    poses = [
        _make_pose("near", 0, 0.0, 1.0, 0.0),
        _make_pose("far", 1, 15.0, 1.0, 0.0),  # 15 m away > 10 m cap
    ]
    rooms = [
        _make_room("s0_room_00", [0], ["wall_x"]),
        _make_room("s0_room_01", [1], ["wall_x"]),
    ]
    openings = [_make_opening("wall_x", "door", cx=7.5, cy=1.0, cz=0.0)]
    pp, rp, op, out = _write_fixtures(tmp_path, poses, rooms, openings)

    result = build_navigation_graph(pp, rp, op, None, out)

    portal_edges = [e for e in result["edges"] if e["portal"]]
    assert len(portal_edges) == 0, "portal edge > 10 m should be dropped"


def test_unassigned_sweeps_get_exterior_group(tmp_path: Path) -> None:
    """Sweeps not in any room should be grouped as 'exterior'."""
    poses = [
        _make_pose("inside", 0, 0.0, 1.0, 0.0),
        _make_pose("outside1", 1, -3.0, 1.0, 0.0),
        _make_pose("outside2", 2, -4.0, 1.0, 0.0),
    ]
    rooms = [_make_room("s0_room_00", [0], ["w1"])]
    pp, rp, op, out = _write_fixtures(tmp_path, poses, rooms, [])

    result = build_navigation_graph(pp, rp, op, None, out)

    exterior_nodes = [n for n in result["nodes"] if n.get("room") == "exterior"]
    assert len(exterior_nodes) == 2


def test_output_schema_valid(tmp_path: Path) -> None:
    """Output JSON contains all required top-level fields."""
    poses = [_make_pose("a", 0, 0.0, 1.0, 0.0)]
    rooms = [_make_room("s0_room_00", [0], ["w1"])]
    pp, rp, op, out = _write_fixtures(tmp_path, poses, rooms, [])

    result = build_navigation_graph(pp, rp, op, None, out)

    for key in ("schema_version", "coordinate_frame", "node_count", "edge_count",
                "portal_count", "connected_components", "nodes", "edges"):
        assert key in result, f"missing key: {key}"
    assert result["coordinate_frame"] == "runtime_yup"
    assert out.is_file()
    roundtrip = json.loads(out.read_text())
    assert roundtrip["node_count"] == result["node_count"]
