import json
from pathlib import Path

import numpy as np
import pytest

from vs_pipeline.stages.panorama import (
    FACE_NAMES,
    PANORAMA_SCHEMA_VERSION,
    SweepFaces,
    apply_face_transform,
    dominant_axis,
    quat_conjugate,
    quat_rotate,
    write_manifest,
)


def test_quat_rotate_identity() -> None:
    np.testing.assert_allclose(
        quat_rotate([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12
    )


def test_quat_rotate_90_about_x() -> None:
    # 90 deg about X: z -> -y
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    np.testing.assert_allclose(
        quat_rotate([s, 0.0, 0.0, c], [0.0, 0.0, 1.0]), [0.0, -1.0, 0.0], atol=1e-12
    )


def test_quat_conjugate_inverts_rotation() -> None:
    rng = np.random.default_rng(7)
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    v = rng.normal(size=3)
    np.testing.assert_allclose(quat_rotate(quat_conjugate(q), quat_rotate(q, v)), v, atol=1e-12)


def test_dominant_axis() -> None:
    assert dominant_axis([0.9, 0.1, -0.2]) == "+X"
    assert dominant_axis([0.0, -2.0, 0.5]) == "-Y"
    assert dominant_axis([0.1, 0.2, 0.9]) == "+Z"


def test_apply_face_transform_ops() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert apply_face_transform(image, "none") is image
    assert apply_face_transform(image, "rot90_cw").shape == (4, 2, 3)
    assert apply_face_transform(image, "rot90_ccw").shape == (4, 2, 3)
    np.testing.assert_array_equal(apply_face_transform(image, "rot180"), image[::-1, ::-1])
    np.testing.assert_array_equal(apply_face_transform(image, "flip_h"), image[:, ::-1])
    np.testing.assert_array_equal(apply_face_transform(image, "flip_v"), image[::-1])
    with pytest.raises(ValueError):
        apply_face_transform(image, "transpose")


def test_manifest_reports_incomplete_sweeps(tmp_path: Path) -> None:
    complete = SweepFaces(
        sweep_uuid="aaaa-0001",
        index=0,
        faces={name: f"aaaa-0001/{name}.jpg" for name in FACE_NAMES},
    )
    partial = SweepFaces(sweep_uuid="aaaa-0002", index=1, faces={"px": "aaaa-0002/px.jpg"})

    manifest_path = write_manifest(
        [complete, partial], tmp_path, source_file="scan.e57", source_sha256="ab" * 32
    )
    payload = json.loads(manifest_path.read_text())

    assert payload["schema_version"] == PANORAMA_SCHEMA_VERSION
    assert payload["sweep_count"] == 2
    assert payload["incomplete_sweeps"] == ["aaaa-0002"]
    assert set(payload["face_transforms"]) == set(FACE_NAMES)
    assert payload["sweeps"][0]["faces"]["pz"] == "aaaa-0001/pz.jpg"
