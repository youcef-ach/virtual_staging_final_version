"""Tests for Stage 7 (G4): ray geometry + world-position derivation."""

from __future__ import annotations

import numpy as np

from vs_pipeline.stages.gbuffer import depth_to_worldpos, face_ray_directions


def test_face_ray_directions_unit_and_90deg_fov():
    dirs = face_ray_directions(64)
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0)
    centre = dirs[32, 32]
    assert centre[2] < -0.99  # looks down -Z
    edge = dirs[32, 0]  # left edge of a 90-deg face: ~45 deg off-axis
    assert abs(abs(edge[0]) - abs(edge[2])) < 0.05


def test_depth_to_worldpos_recovers_plane():
    size = 32
    cam = np.eye(4)
    cam[:3, 3] = [1.0, 2.0, 3.0]  # camera looking down -Z from z = 3
    dirs = face_ray_directions(size)
    depth = 3.0 / -dirs[..., 2]  # ray length to the world plane z = 0
    pts = depth_to_worldpos(depth, cam)
    assert np.allclose(pts[..., 2], 0.0, atol=1e-9)
    # x/y spread matches a 90-deg frustum at distance 3
    assert 3.8 < float(pts[..., 0].max() - pts[..., 0].min()) < 6.0
