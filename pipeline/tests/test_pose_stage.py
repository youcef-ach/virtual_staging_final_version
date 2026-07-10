import json
from pathlib import Path

import numpy as np

from vs_pipeline.coordinates import convert_pose_zup_to_yup
from vs_pipeline.stages.pose import (
    POSES_SCHEMA_VERSION,
    SweepPose,
    fallback_sweep_uuid,
    read_poses_json,
    write_poses_json,
)


def _sample_sweeps() -> list[SweepPose]:
    pose_a = convert_pose_zup_to_yup(np.eye(4))
    pose_b_mp = np.eye(4)
    pose_b_mp[:3, 3] = [2.5, -1.0, 1.4]
    pose_b = convert_pose_zup_to_yup(pose_b_mp)
    return [
        SweepPose(sweep_uuid="aaaa-0001", index=0, pose_rt=pose_a),
        SweepPose(sweep_uuid="aaaa-0002", index=1, pose_rt=pose_b),
    ]


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "poses.json"
    sweeps = _sample_sweeps()
    write_poses_json(sweeps, output, source_file="scan.e57", source_sha256="ab" * 32)

    loaded = read_poses_json(output)
    assert len(loaded) == 2
    for original, restored in zip(sweeps, loaded, strict=True):
        assert restored.sweep_uuid == original.sweep_uuid
        assert restored.index == original.index
        np.testing.assert_allclose(restored.pose_rt, original.pose_rt, atol=0)


def test_poses_json_schema_fields(tmp_path: Path) -> None:
    output = tmp_path / "poses.json"
    write_poses_json(_sample_sweeps(), output, source_file="scan.e57", source_sha256="ab" * 32)

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == POSES_SCHEMA_VERSION
    assert payload["coordinate_frame"] == "runtime_yup"
    assert payload["conversion"] == "matterport_zup_to_runtime_yup"
    assert payload["source_file"] == "scan.e57"
    assert payload["sweep_count"] == 2
    assert len(payload["sweeps"][0]["pose_rt"]) == 4  # 4x4 nested lists


def test_translation_is_in_runtime_frame(tmp_path: Path) -> None:
    # A sweep at (2.5, -1.0, 1.4) in Matterport Z-up must land at
    # (2.5, 1.4, +1.0) in runtime Y-up: X_rt=X, Y_rt=Z, Z_rt=-Y.
    output = tmp_path / "poses.json"
    write_poses_json(_sample_sweeps(), output, source_file="scan.e57", source_sha256="ab" * 32)
    loaded = read_poses_json(output)
    np.testing.assert_allclose(loaded[1].pose_rt[:3, 3], [2.5, 1.4, 1.0], atol=1e-12)


def test_fallback_uuid_is_deterministic() -> None:
    a = fallback_sweep_uuid("scan.e57", 3)
    b = fallback_sweep_uuid("scan.e57", 3)
    c = fallback_sweep_uuid("scan.e57", 4)
    assert a == b  # idempotent across re-runs
    assert a != c  # distinct per sweep
