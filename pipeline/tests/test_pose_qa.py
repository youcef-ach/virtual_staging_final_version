import json
from pathlib import Path

import cv2
import numpy as np

from vs_pipeline.coordinates import convert_pose_zup_to_yup
from vs_pipeline.stages.pose import SweepPose, write_poses_json
from vs_pipeline.stages.pose_qa import (
    ADE_MIRROR_LABEL_ID,
    FACE_NAMES,
    ORIENTATION_OPS,
    SEG_WINDOW_CLASS_ID,
    _bijective_assignment,
    _normalized_overlap,
    apply_orientation,
    edge_overlap_score,
    face_camera_matrix,
    is_spot_check,
    load_exclusion_mask,
    prepare_render_jobs,
)


def test_bijective_assignment_prefers_best_scores() -> None:
    combo_scores = {
        ("px", "nx", "rot180"): 0.95,
        ("px", "px", "rot0"): 0.40,
        ("nx", "px", "rot180"): 0.90,
        ("nx", "nx", "rot0"): 0.42,
        ("py", "py", "rot0"): 0.88,
        ("ny", "ny", "rot0_flip"): 0.87,
        ("pz", "nz", "rot90"): 0.86,
        ("nz", "pz", "rot270"): 0.85,
    }
    mapping = _bijective_assignment(combo_scores)
    assert mapping["px"]["render_face"] == "nx"
    assert mapping["nx"]["render_face"] == "px"
    assert mapping["pz"]["render_face"] == "nz"
    # Bijection: each rendered face used at most once.
    used = [entry["render_face"] for entry in mapping.values()]
    assert len(used) == len(set(used))


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


# --------------------------------------------- G6 slice 4.2: glass/mirror exclusion


def test_load_exclusion_mask_prefers_refined_and_unions_mirror(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "seg" / "uuid-1"
    sweep_dir.mkdir(parents=True)
    cls = np.zeros((8, 8), dtype=np.uint8)
    cls[0:2, :] = SEG_WINDOW_CLASS_ID  # coarse-only window rows (must be superseded)
    refined = np.zeros((8, 8), dtype=np.uint8)
    refined[6:8, :] = SEG_WINDOW_CLASS_ID  # SAM-refined window rows
    ade = np.zeros((8, 8), dtype=np.uint8)
    ade[3, 3] = ADE_MIRROR_LABEL_ID  # mirror only visible in the RAW map
    cv2.imwrite(str(sweep_dir / "px_cls.png"), cls)
    cv2.imwrite(str(sweep_dir / "px_cls_refined.png"), refined)
    cv2.imwrite(str(sweep_dir / "px_ade.png"), ade)

    mask = load_exclusion_mask(tmp_path / "seg", "uuid-1", "px", (16, 16))
    assert mask is not None
    assert mask.dtype == bool and mask.shape == (16, 16)
    assert mask[15, 0]  # refined window region (NEAREST-upscaled)
    assert not mask[0, 0]  # coarse-only region ignored: refined map preferred
    assert mask[6, 6]  # mirror pixel from the RAW ADE map, unioned in

    # No segmentation for this face -> None (graceful legacy fallback).
    assert load_exclusion_mask(tmp_path / "seg", "uuid-1", "py", (16, 16)) is None
    assert load_exclusion_mask(tmp_path / "seg", "uuid-2", "px", (16, 16)) is None


def test_exclusion_valid_px_restores_signal_outside_glass() -> None:
    # Glass half (cols < 32): dense pano "content" edges at even columns, render
    # junk at odd columns (all misses). Outside: one clean structural edge where
    # the render lands exactly.
    pano_dilated = np.zeros((64, 64), dtype=np.uint8)
    pano_dilated[:, 0:32:2] = 255  # glass content edges
    pano_dilated[:, 48] = 255  # real structural edge
    render_edges = np.zeros((64, 64), dtype=np.uint8)
    render_edges[:, 1:32:2] = 255  # junk inside the glass (misses the pano edges)
    render_edges[:, 48] = 255  # render hits the structure

    unmasked = _normalized_overlap(pano_dilated, render_edges)

    exclusion = np.zeros((64, 64), dtype=bool)
    exclusion[:, :32] = True
    pano_masked = pano_dilated.copy()
    pano_masked[exclusion] = 0
    render_masked = render_edges.copy()
    render_masked[exclusion] = 0
    valid_px = int(exclusion.size - np.count_nonzero(exclusion))
    masked = _normalized_overlap(pano_masked, render_masked, valid_px)

    # Removing the glass noise turns a chance-level face into a near-perfect one.
    assert masked > unmasked
    assert masked > 0.9
    assert unmasked < 0.5


def test_exclusion_window_id_matches_segmentation_contract() -> None:
    # pose_qa deliberately does not import the segmentation module at runtime;
    # this test pins the id contract so drift is impossible to miss.
    from vs_pipeline.stages.segmentation import CLASS_IDS

    assert SEG_WINDOW_CLASS_ID == CLASS_IDS["window"]
