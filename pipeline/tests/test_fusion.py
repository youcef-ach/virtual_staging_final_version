"""Tests for Stage 4 (G6) slice 3: 2D->3D fusion (torch-free, synthetic)."""

from __future__ import annotations

import numpy as np

from vs_pipeline.stages.fusion import (
    GRID_M,
    PlaneAccumulator,
    WallPlane,
    aligned_worldpos_yup,
    assign_to_walls,
    detect_mirror_mask,
    extract_rects,
    invert_orientation,
    opening_dict,
    project_box_footprint,
    ray_plane_lift,
    vote_box,
    yup_to_zup_points,
    zup_to_yup_points,
)
from vs_pipeline.stages.pose_qa import ORIENTATION_OPS, apply_orientation
from vs_pipeline.stages.segmentation import CLASS_IDS, CLASSES


def _wall_x0() -> WallPlane:
    """Vertical wall at x = 0, u along +Z, v along +Y (runtime Y-up)."""
    return WallPlane(
        plane_id="wall_test",
        normal=np.array([1.0, 0.0, 0.0]),
        d=0.0,
        origin=np.zeros(3),
        u_axis=np.array([0.0, 0.0, 1.0]),
        v_axis=np.array([0.0, 1.0, 0.0]),
        u_range=(0.0, 4.0),
        v_range=(0.0, 3.0),
    )


def test_invert_orientation_round_trip_all_ops():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, size=(8, 8), dtype=np.uint8)
    vol = rng.random((8, 8, 3))
    for op in ORIENTATION_OPS:
        assert np.array_equal(invert_orientation(apply_orientation(img, op), op), img)
        assert np.allclose(invert_orientation(apply_orientation(vol, op), op), vol)


def test_frame_conversion_round_trip():
    rng = np.random.default_rng(5)
    pts = rng.random((10, 3)) * 10 - 5
    assert np.allclose(yup_to_zup_points(zup_to_yup_points(pts)), pts)
    assert np.allclose(zup_to_yup_points(yup_to_zup_points(pts)), pts)
    # Coordinate Lock rule: (x, y, z)_mp -> (x, z, -y)_rt
    assert np.allclose(zup_to_yup_points(np.array([1.0, 2.0, 3.0])), [1.0, 3.0, -2.0])


def test_aligned_worldpos_matches_oriented_reference():
    rng = np.random.default_rng(7)
    depth = rng.random((8, 8)) * 3 + 0.5
    cam = np.eye(4)
    cam[:3, 3] = [1.0, 2.0, 3.0]
    from vs_pipeline.stages.gbuffer import depth_to_worldpos

    ref = zup_to_yup_points(depth_to_worldpos(depth, cam))
    for op in ("rot0", "rot90_flip", "rot180"):
        world, valid = aligned_worldpos_yup(depth, cam, op)
        assert np.allclose(world, apply_orientation(ref, op))
        assert bool(valid.all())


def test_assign_to_walls_distance_and_bbox_gates():
    wall = _wall_x0()
    pts = np.array(
        [
            [0.05, 1.0, 2.0],  # on-plane, inside bbox -> assigned
            [1.0, 1.0, 2.0],  # 1 m off the plane -> rejected
            [0.05, 1.0, 9.0],  # on-plane but far outside the boundary -> rejected
        ]
    )
    best, dist = assign_to_walls(pts, [wall], max_dist=0.15)
    assert best.tolist() == [0, -1, -1]
    assert abs(dist[0] - 0.05) < 1e-9


def test_accumulate_and_extract_window_rect():
    wall = _wall_x0()
    acc = PlaneAccumulator(wall)
    # Dense window patch: u in [1.0, 2.0], v in [0.8, 2.0] on the plane.
    us = np.arange(1.0, 2.0, 0.01)
    vs = np.arange(0.8, 2.0, 0.01)
    uu, vv = np.meshgrid(us, vs)
    pts = (
        wall.origin
        + uu.reshape(-1, 1) * wall.u_axis
        + vv.reshape(-1, 1) * wall.v_axis
    )
    added = acc.add("window", pts, "sweep-a")
    assert added == pts.shape[0]
    rects = extract_rects(acc.grids["window"])
    assert len(rects) == 1
    ua, ub, vb, vt, support = rects[0]
    assert abs((acc.u0 + ua) - 1.0) < 3 * GRID_M
    assert abs((acc.u0 + ub) - 2.0) < 3 * GRID_M
    assert abs((acc.v0 + vb) - 0.8) < 3 * GRID_M
    assert abs((acc.v0 + vt) - 2.0) < 3 * GRID_M
    assert support > 0
    o = opening_dict(wall, "window", acc.u0 + ua, acc.u0 + ub, acc.v0 + vb, acc.v0 + vt, support, 1)
    assert o["kind"] == "window" and o["plane_id"] == "wall_test"
    assert 0.9 < o["width_m"] < 1.1 and 1.1 < o["height_m"] < 1.3
    assert len(o["corners_3d"]) == 4


def test_extract_rects_rejects_noise():
    grid = np.zeros((40, 40), dtype=np.int32)
    grid[5, 5] = 100  # single hot cell: too small
    grid[20:22, 20:22] = 100  # 10 cm blob: below min dim
    assert extract_rects(grid) == []


def _identity_cam_box():
    """Camera at origin (Matterport frame, -Z view) + a Y-up box in front.

    Y-up point (x, y, z) -> Matterport (x, -z, y); putting the box at
    y_yup in [-3.5, -2.5] places it at z_mp in [2.5, 3.5]... in FRONT means
    z_mp negative, so use y_yup in [2.5, 3.5] -> z_mp in [-3.5, -2.5]? No:
    yup_to_zup: (x, y, z)_rt -> (x, -z, y)_mp, so z_mp = y_yup. In front of
    the -Z-looking camera requires z_mp < 0 -> y_yup < 0.
    """
    cam = np.eye(4)
    bmin = np.array([-0.5, -3.5, -0.5])
    bmax = np.array([0.5, -2.5, 0.5])
    return cam, bmin, bmax


def test_project_box_footprint_in_front_and_behind():
    cam, bmin, bmax = _identity_cam_box()
    foot = project_box_footprint(cam, bmin, bmax, size=64)
    assert foot is not None and bool(foot.any())
    # Center of the image is inside the footprint (box is centered on axis).
    assert bool(foot[32, 32])
    # Same box behind the camera -> no footprint.
    behind = project_box_footprint(cam, bmin + [0, 6, 0], bmax + [0, 6, 0], size=64)
    assert behind is None


def test_vote_box_counts_and_occlusion_guard():
    cam, bmin, bmax = _identity_cam_box()
    size = 256
    cls_render = np.full((size, size), CLASS_IDS["furniture"], dtype=np.uint8)
    depth_far = np.full((size, size), 10.0)
    votes = vote_box(cam, depth_far, cls_render, bmin, bmax)
    assert votes is not None
    assert votes[CLASS_IDS["furniture"]] == votes.sum() > 0
    assert len(votes) == len(CLASSES)
    # A proxy wall 0.5 m in front of the camera occludes the 3 m distant box.
    depth_near = np.full((size, size), 0.5)
    assert vote_box(cam, depth_near, cls_render, bmin, bmax) is None


def test_detect_mirror_mask_on_wall_component():
    wall = _wall_x0()
    size = 32
    world = np.zeros((size, size, 3))
    world[..., 0] = 5.0  # default: far from the wall plane
    world[..., 1] = 1.5
    world[..., 2] = 2.0
    # A coherent block of pixels sitting ON the wall (x ~ 0).
    world[8:24, 8:24, 0] = 0.02
    cand = np.zeros((size, size), dtype=bool)
    cand[4:28, 4:28] = True  # candidate classes cover more than the on-wall block
    mask = detect_mirror_mask(cand, world, [wall], min_area_px=50)
    assert bool(mask[10:22, 10:22].all())
    assert not bool(mask[0:4].any())
    # Without walls, nothing fires.
    assert not detect_mirror_mask(cand, world, [], min_area_px=50).any()


def test_detect_mirror_normal_gate():
    """v2 normal gate: only VERTICAL surfaces ALIGNED with the assigned wall
    survive -- ceiling strips (normal down), horizontal rims (normal up) and
    perpendicular vertical surfaces are rejected even when the position sits
    exactly on the wall plane."""
    wall = _wall_x0()  # normal +X
    size = 32
    world = np.zeros((size, size, 3))
    world[..., 0] = 0.02  # every pixel ON the wall plane
    world[..., 1] = 1.5
    world[..., 2] = 2.0
    cand = np.ones((size, size), dtype=bool)
    normals = np.zeros((size, size, 3))
    normals[:8] = [1.0, 0.0, 0.0]  # wall-aligned vertical -> mirror candidate
    normals[8:16] = [0.0, 0.0, 1.0]  # vertical but PERPENDICULAR -> rejected
    normals[16:24] = [0.0, -1.0, 0.0]  # ceiling strip (points down) -> rejected
    normals[24:] = [0.0, 1.0, 0.0]  # horizontal rim (points up) -> rejected
    mask = detect_mirror_mask(cand, world, [wall], min_area_px=10, normals_yup=normals)
    assert bool(mask[:8].all())
    assert not bool(mask[8:].any())
    # Without normals the gate is skipped -> everything on-wall fires (v1).
    v1 = detect_mirror_mask(cand, world, [wall], min_area_px=10)
    assert bool(v1.all())


def test_ray_plane_lift_door_hole_occluder_nearest():
    """Door case: proxy depth BEYOND the plane (ray through the G3 door
    hole) must still land the lift ON the plane; a proxy hit clearly in
    FRONT blocks; the nearest passing plane wins."""
    def _wall(depth_y: float) -> WallPlane:
        # Horizontal-normal test plane at y_yup = -depth_y, facing +Y.
        return WallPlane(
            plane_id=f"w{depth_y}",
            normal=np.array([0.0, 1.0, 0.0]),
            d=depth_y,
            origin=np.array([0.0, -depth_y, 0.0]),
            u_axis=np.array([1.0, 0.0, 0.0]),
            v_axis=np.array([0.0, 0.0, 1.0]),
            u_range=(-8.0, 8.0),
            v_range=(-8.0, 8.0),
        )

    wall = _wall(3.0)
    cam = np.eye(4)  # Matterport frame, looking down -Z_mp = -Y_yup
    size = 16
    cls = np.full((size, size), CLASS_IDS["door"], dtype=np.uint8)
    # Proxy hit at 5 m: BEYOND the 3 m plane (door hole) -> passes.
    depth_beyond = np.full((size, size), 5.0)
    pts, widx = ray_plane_lift(cam, cls, depth_beyond, [wall], CLASS_IDS["door"])
    assert pts.shape[0] == size * size
    assert np.allclose(pts[:, 1], -3.0, atol=1e-9)
    assert bool((widx == 0).all())
    # Proxy hit at 1 m: an occluding wall in FRONT of the plane -> blocked.
    depth_front = np.full((size, size), 1.0)
    pts2, _ = ray_plane_lift(cam, cls, depth_front, [wall], CLASS_IDS["door"])
    assert pts2.shape[0] == 0
    # Nearest passing plane wins over a farther coplanar-family plane.
    far_wall = _wall(4.5)
    pts3, widx3 = ray_plane_lift(
        cam, cls, depth_beyond, [far_wall, wall], CLASS_IDS["door"]
    )
    assert bool((widx3 == 1).all())
    assert np.allclose(pts3[:, 1], -3.0, atol=1e-9)
