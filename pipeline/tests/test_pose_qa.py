import json
from pathlib import Path

import numpy as np

from vs_pipeline.coordinates import convert_pose_zup_to_yup
from vs_pipeline.stages.pose import SweepPose, write_poses_json
from vs_pipeline.stages.pose_qa import (
    FACE_NAMES,
    ORIENTATION_OPS,
    apply_orientation,
    edge_overlap_score,
    face_camera_matrix,
    is_spot_check,
    prepare_render_jobs,
)


def _random_pose(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    pose = np.eye(4)
    pose[:3, :3] = q
    pose[:3, 3] = rng.normal(scale=5.0, size=3)
    return pose


def test_face_camera_matrices_are_rigid() -> None:
    for seed in range(5):
        pose = _random_pose(seed)
        for face in FACE_NAMES:
            cam = face_camera_matrix(pose, face)
            rotation = cam[:3, :3]
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
            assert np.linalg.det(rotation) > 0.999
            np.testing.assert_allclose(cam[:3, 3], pose[:3, 3], atol=0)


def test_face_camera_looks_along_face_direction() -> None:
    pose = np.eye(4)
    cam = face_camera_matrix(pose, "px")
    view_dir = -cam[:3, 2]  # Blender camera looks along local -Z
    np.testing.assert_allclose(view_dir, [1.0, 0.0, 0.0], atol=1e-12)


def test_orientation_ops_are_distinct() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    variants = {apply_orientation(image, op).tobytes() for op in ORIENTATION_OPS}
    assert len(variants) == len(ORIENTATION_OPS)


def test_edge_overlap_score_bounds() -> None:
    pano = np.zeros((128, 128), dtype=np.uint8)
    cv2_rect = (32, 32, 96, 96)
    pano[cv2_rect[1] : cv2_rect[3], cv2_rect[0] : cv2_rect[2]] = 255
    render_same = pano.copy()
    assert edge_overlap_score(pano, render_same) > 0.9

    render_disjoint = np.zeros_like(pano)
    render_disjoint[100:120, 4:20] = 255
    assert edge_overlap_score(pano, render_disjoint) < 0.3

    render_empty = np.zeros_like(pano)
    assert edge_overlap_score(pano, render_empty) == 0.0


def test_spot_check_is_deterministic_and_roughly_ten_percent() -> None:
    uuids = [f"sweep-{i:05d}" for i in range(2000)]
    first = [is_spot_check(u) for u in uuids]
    second = [is_spot_check(u) for u in uuids]
    assert first == second  # deterministic
    rate = sum(first) / len(first)
    assert 0.05 < rate < 0.15


def test_prepare_render_jobs_round_trip(tmp_path: Path) -> None:
    pose_a = convert_pose_zup_to_yup(np.eye(4))
    pose_b_mp = np.eye(4)
    pose_b_mp[:3, 3] = [1.0, 2.0, 3.0]
    pose_b = convert_pose_zup_to_yup(pose_b_mp)
    poses_path = tmp_path / "poses.json"
    write_poses_json(
        [
            SweepPose(sweep_uuid="aaaa-0001", index=0, pose_rt=pose_a),
            SweepPose(sweep_uuid="aaaa-0002", index=1, pose_rt=pose_b),
        ],
        poses_path,
        source_file="scan.e57",
        source_sha256="ab" * 32,
    )

    jobs_path = tmp_path / "render_jobs.json"
    count = prepare_render_jobs(poses_path, jobs_path, render_size=256)
    assert count == 2

    payload = json.loads(jobs_path.read_text())
    assert payload["render_size"] == 256
    assert len(payload["sweeps"]) == 2
    faces = payload["sweeps"][1]["faces"]
    assert set(faces) == set(FACE_NAMES)
    # Camera position must equal the sweep position in the MATTERPORT frame.
    matrix = np.asarray(faces["px"])
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0], atol=1e-9)
