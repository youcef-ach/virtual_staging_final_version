"""Tests for Stage 3 (G1) point-cloud preprocessing.

Frame-conversion tests are pure numpy; Open3D-dependent tests skip cleanly
when the 'pointcloud' extra is not installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vs_pipeline.coordinates import ZUP_TO_YUP
from vs_pipeline.stages.pointcloud import convert_points_zup_to_yup


def test_points_conversion_matches_coordinate_lock():
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(100, 3))
    expected = (ZUP_TO_YUP[:3, :3] @ pts.T).T
    np.testing.assert_allclose(convert_points_zup_to_yup(pts), expected)


def test_points_conversion_axis_mapping():
    # X_rt = X_mp, Y_rt = Z_mp, Z_rt = -Y_mp
    pts = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(convert_points_zup_to_yup(pts), np.array([[1.0, 3.0, -2.0]]))


def test_points_conversion_preserves_origin_and_lengths():
    np.testing.assert_allclose(
        convert_points_zup_to_yup(np.zeros((1, 3))), np.zeros((1, 3))
    )
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(50, 3))
    np.testing.assert_allclose(
        np.linalg.norm(convert_points_zup_to_yup(pts), axis=1),
        np.linalg.norm(pts, axis=1),
    )


def test_points_conversion_rejects_bad_shape():
    with pytest.raises(ValueError):
        convert_points_zup_to_yup(np.zeros(3))


def test_sor_removes_far_outlier():
    pytest.importorskip("open3d")
    from vs_pipeline.stages.pointcloud import preprocess_scan

    xs, zs = np.meshgrid(np.linspace(0.0, 1.0, 40), np.linspace(0.0, 1.0, 40))
    plane = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()])
    outlier = np.array([[25.0, 25.0, 25.0]])
    cloud, stats = preprocess_scan(
        np.vstack([plane, outlier]),
        sensor_origin_rt=np.array([0.5, 2.0, 0.5]),
        voxel_m=0.001,
        sor_neighbors=10,
        sor_std=2.0,
        normal_radius_m=0.2,
    )
    assert stats["raw"] == plane.shape[0] + 1
    assert np.asarray(cloud.points).max() < 5.0


def test_voxel_downsample_reduces_density():
    pytest.importorskip("open3d")
    from vs_pipeline.stages.pointcloud import preprocess_scan

    rng = np.random.default_rng(1)
    pts = rng.uniform(0.0, 0.2, size=(5000, 3))
    cloud, stats = preprocess_scan(
        pts,
        sensor_origin_rt=np.zeros(3),
        voxel_m=0.05,
        sor_neighbors=10,
        sor_std=4.0,
        normal_radius_m=0.2,
    )
    assert stats["after_voxel"] < stats["after_sor"]
    assert len(cloud.points) <= 6 * 6 * 6


def test_normals_oriented_toward_sensor():
    pytest.importorskip("open3d")
    from vs_pipeline.stages.pointcloud import preprocess_scan

    xs, zs = np.meshgrid(np.linspace(0.0, 1.0, 30), np.linspace(0.0, 1.0, 30))
    floor = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()])
    cloud, _ = preprocess_scan(
        floor,
        sensor_origin_rt=np.array([0.5, 1.5, 0.5]),
        voxel_m=0.001,
        sor_neighbors=10,
        sor_std=5.0,
        normal_radius_m=0.2,
    )
    normals = np.asarray(cloud.normals)
    assert normals.size > 0
    # Sensor sits above the floor plane -> normals must point up (+Y).
    assert (normals[:, 1] > 0).mean() > 0.95


def test_merge_deduplicates_and_writes_meta(tmp_path):
    o3d = pytest.importorskip("open3d")
    from vs_pipeline.stages.pointcloud import merge_cached

    cache = tmp_path / "sweep_cache"
    cache.mkdir()
    rng = np.random.default_rng(2)
    pts = rng.uniform(0.0, 1.0, size=(2000, 3))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    cloud.estimate_normals()
    for name in ("scan_000.ply", "scan_001.ply"):
        o3d.io.write_point_cloud(str(cache / name), cloud, write_ascii=False)

    out_path, merged = merge_cached(tmp_path, voxel_m=0.05)

    # Two identical sweeps must collapse to ~the same cloud as one sweep.
    single = cloud.voxel_down_sample(0.05)
    assert out_path.exists()
    assert abs(len(merged.points) - len(single.points)) <= max(
        3, int(0.05 * len(single.points))
    )

    meta = json.loads((tmp_path / "preprocess_meta.json").read_text(encoding="utf-8"))
    assert meta["final_points"] == len(merged.points)
    assert meta["working_cloud"] == "working_cloud.ply"
