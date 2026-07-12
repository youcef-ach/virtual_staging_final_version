"""Tests for Stage 5 (G2) plane fitting & boundary extraction.

Math/clustering tests are pure numpy; Open3D-dependent tests skip cleanly
when the 'pointcloud' extra is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vs_pipeline.stages.planes import (
    GlobalPlane,
    PlaneProposal,
    classify_normal,
    cluster_proposals,
    fit_plane_lsq,
    pair_walls,
    parse_sweep_spec,
)


def _proposal(normal, d, bbox_min, bbox_max, sweep=0, inliers=1000):
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    return PlaneProposal(
        sweep_index=sweep,
        cls=classify_normal(normal),
        normal=normal,
        d=float(d),
        inlier_count=inliers,
        centroid=(bbox_min + bbox_max) / 2,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )


def test_fit_plane_lsq_recovers_plane():
    rng = np.random.default_rng(0)
    # y = 1.5 plane
    pts = np.column_stack(
        [rng.uniform(-2, 2, 500), np.full(500, 1.5), rng.uniform(-2, 2, 500)]
    )
    normal, d = fit_plane_lsq(pts)
    assert abs(abs(float(normal[1])) - 1.0) < 1e-9
    assert abs(abs(d) - 1.5) < 1e-9


def test_classify_normal():
    assert classify_normal(np.array([0.0, 1.0, 0.0])) == "floor"
    assert classify_normal(np.array([0.0, -1.0, 0.0])) == "ceiling"
    assert classify_normal(np.array([1.0, 0.0, 0.0])) == "wall"
    slanted = np.array([0.0, 0.7, 0.714])
    assert classify_normal(slanted / np.linalg.norm(slanted)) == "other"


def test_cluster_merges_overlapping_coplanar():
    a = _proposal([1, 0, 0], -2.0, [2.0, 0.0, 0.0], [2.02, 2.5, 3.0], sweep=0)
    b = _proposal([1, 0, 0], -2.01, [2.0, 0.0, 2.8], [2.02, 2.5, 5.5], sweep=1)
    clusters = cluster_proposals([a, b])
    assert len(clusters) == 1


def test_cluster_keeps_distant_coplanar_apart():
    """The plane-bleeding guard: same infinite plane, far apart -> two planes."""
    a = _proposal([1, 0, 0], -2.0, [2.0, 0.0, 0.0], [2.02, 2.5, 3.0], sweep=0)
    b = _proposal([1, 0, 0], -2.0, [2.0, 0.0, 10.0], [2.02, 2.5, 13.0], sweep=5)
    clusters = cluster_proposals([a, b])
    assert len(clusters) == 2


def test_cluster_keeps_opposing_normals_apart():
    """Two sides of the same wall must remain distinct entities."""
    a = _proposal([1, 0, 0], -2.0, [2.0, 0.0, 0.0], [2.02, 2.5, 3.0])
    b = _proposal([-1, 0, 0], 2.1, [2.08, 0.0, 0.0], [2.1, 2.5, 3.0])
    clusters = cluster_proposals([a, b])
    assert len(clusters) == 2


def _global_wall(pid, normal, d, bbox_min, bbox_max):
    normal = np.asarray(normal, dtype=np.float64)
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    return GlobalPlane(
        plane_id=pid,
        cls="wall",
        normal=normal / np.linalg.norm(normal),
        d=float(d),
        inlier_count=1000,
        area_m2=5.0,
        centroid=(bbox_min + bbox_max) / 2,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        member_sweeps=[0],
    )


def test_pair_walls_thickness():
    a = _global_wall("wall_00", [1, 0, 0], -2.0, [2.0, 0.0, 0.0], [2.0, 2.5, 3.0])
    b = _global_wall("wall_01", [-1, 0, 0], 2.1, [2.1, 0.0, 0.0], [2.1, 2.5, 3.0])
    far = _global_wall("wall_02", [1, 0, 0], -8.0, [8.0, 0.0, 0.0], [8.0, 2.5, 3.0])
    pair_walls([a, b, far])
    assert a.paired_with == "wall_01" and b.paired_with == "wall_00"
    assert a.thickness_m == pytest.approx(0.1, abs=0.005)
    assert far.paired_with is None


def test_parse_sweep_spec():
    available = list(range(10))
    assert parse_sweep_spec(None, available) == available
    assert parse_sweep_spec("3", available) == [3]
    assert parse_sweep_spec("2-4,8", available) == [2, 3, 4, 8]
    assert parse_sweep_spec("8-20", available) == [8, 9]


def test_synthetic_room_end_to_end(tmp_path):
    o3d = pytest.importorskip("open3d")
    from vs_pipeline.stages.planes import fit_property

    rng = np.random.default_rng(4)

    def surface(u_axis, v_axis, origin, nu, nv, normal):
        us = rng.uniform(0, 1, (nu * nv, 1))
        vs = rng.uniform(0, 1, (nu * nv, 1))
        pts = origin + us * u_axis + vs * v_axis
        pts += rng.normal(scale=0.002, size=pts.shape)
        nrm = np.tile(normal, (pts.shape[0], 1))
        return pts, nrm

    n = 3500  # per surface -> ~ dozens of coarse inliers per m^2
    sx, sz = np.array([4.0, 0, 0]), np.array([0, 0, 4.0])
    sy = np.array([0, 2.5, 0])
    parts = [
        surface(sx, sz, np.zeros(3), n, 1, np.array([0, 1.0, 0])),  # floor
        surface(sx, sz, np.array([0, 2.5, 0]), n, 1, np.array([0, -1.0, 0])),  # ceiling
        surface(sy, sz, np.zeros(3), n, 1, np.array([1.0, 0, 0])),  # wall x=0
        surface(sy, sz, np.array([4.0, 0, 0]), n, 1, np.array([-1.0, 0, 0])),  # wall x=4
    ]
    # A furniture blob in the middle (residual candidate).
    blob = rng.uniform(0, 0.6, size=(800, 3)) + np.array([2.0, 0.1, 2.0])
    blob_nrm = np.tile(np.array([0, 1.0, 0]), (blob.shape[0], 1))

    pts = np.vstack([p for p, _ in parts] + [blob])
    nrm = np.vstack([m for _, m in parts] + [blob_nrm])

    cache = tmp_path / "sweep_cache"
    cache.mkdir()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cloud.normals = o3d.utility.Vector3dVector(nrm)
    o3d.io.write_point_cloud(str(cache / "scan_000.ply"), cloud, write_ascii=False)

    out = tmp_path / "planes_out"
    planes = fit_property(cache, out, sweep_spec=None)

    classes = sorted(p.cls for p in planes)
    assert "floor" in classes and "ceiling" in classes
    assert classes.count("wall") >= 2

    report = json.loads((out / "structural_planes.json").read_text(encoding="utf-8"))
    assert report["plane_count"] == len(planes)
    floor = next(p for p in report["planes"] if p["class"] == "floor")
    assert abs(floor["elevation_y"]) < 0.02
    assert len(floor["boundary_3d"]) >= 4

    residuals = json.loads(
        (out / "residual_clusters.json").read_text(encoding="utf-8")
    )
    assert residuals["cluster_count"] >= 1
    assert (out / "planes_review.ply").exists()
    assert (out / "planes_review_topdown.png").exists()
    assert (out / "planes_review_front.png").exists()


def _wall_with_points(pid, z0, z1, d_val=-2.0, seed=0):
    rng = np.random.default_rng(seed)
    count = 800
    ys = rng.uniform(0.0, 2.5, count)
    zs = rng.uniform(z0, z1, count)
    xs = np.full(count, -d_val) + rng.normal(scale=0.002, size=count)
    pts = np.column_stack([xs, ys, zs])
    return GlobalPlane(
        plane_id=pid,
        cls="wall",
        normal=np.array([1.0, 0.0, 0.0]),
        d=float(d_val),
        inlier_count=count,
        area_m2=(z1 - z0) * 2.5,
        centroid=pts.mean(axis=0),
        bbox_min=pts.min(axis=0),
        bbox_max=pts.max(axis=0),
        member_sweeps=[0],
        points=pts,
    )


def test_merge_refined_combines_duplicates_keeps_distant():
    from vs_pipeline.stages.planes import merge_refined_planes

    a = _wall_with_points("wall_00", 0.0, 2.0, seed=1)
    b = _wall_with_points("wall_01", 1.8, 4.0, seed=2)  # same wall, overlapping
    far = _wall_with_points("wall_02", 10.0, 12.0, seed=3)  # bleeding guard
    merged = merge_refined_planes([a, b, far])
    assert len(merged) == 2
    big = max(merged, key=lambda p: p.inlier_count)
    assert big.inlier_count == 1600


def _horizontal(pid, area, y):
    return GlobalPlane(
        plane_id=pid,
        cls="floor",
        normal=np.array([0.0, 1.0, 0.0]),
        d=float(-y),
        inlier_count=1000,
        area_m2=float(area),
        centroid=np.array([2.0, y, 2.0]),
        bbox_min=np.array([0.0, y, 0.0]),
        bbox_max=np.array([4.0, y, 4.0]),
        member_sweeps=[0],
        elevation_y=float(y),
    )


def test_reclassify_tabletop_as_surface():
    from vs_pipeline.stages.planes import reclassify_and_rename

    floor = _horizontal("floor_00", 20.0, 0.0)
    table = _horizontal("floor_01", 1.5, 0.75)
    reclassify_and_rename([floor, table])
    assert floor.cls == "floor"
    assert table.cls == "surface"
    assert table.plane_id.startswith("surface_")
