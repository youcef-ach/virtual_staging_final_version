import numpy as np
import pytest

from vs_pipeline.coordinates import (
    YUP_TO_ZUP,
    ZUP_TO_YUP,
    convert_point_zup_to_yup,
    convert_pose_yup_to_zup,
    convert_pose_zup_to_yup,
)


def _random_rigid_pose(seed: int) -> np.ndarray:
    """Random valid rigid transform (orthonormal rotation, det=+1) + translation."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    pose = np.eye(4)
    pose[:3, :3] = q
    pose[:3, 3] = rng.normal(scale=5.0, size=3)
    return pose


def test_axis_mapping() -> None:
    # X_rt = X_mp, Y_rt = Z_mp, Z_rt = -Y_mp
    np.testing.assert_allclose(convert_point_zup_to_yup([1.0, 2.0, 3.0]), [1.0, 3.0, -2.0])


def test_origin_is_preserved_exactly() -> None:
    # Hard invariant: the E57 origin never moves.
    result = convert_point_zup_to_yup([0.0, 0.0, 0.0])
    assert np.array_equal(result, np.zeros(3))  # exact, not approximate


def test_conversion_matrix_is_pure_rotation() -> None:
    rotation = ZUP_TO_YUP[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=0)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.array_equal(ZUP_TO_YUP[:3, 3], np.zeros(3))  # zero translation


def test_inverse_is_exact() -> None:
    np.testing.assert_array_equal(ZUP_TO_YUP @ YUP_TO_ZUP, np.eye(4))


@pytest.mark.parametrize("seed", range(10))
def test_pose_round_trip(seed: int) -> None:
    pose_mp = _random_rigid_pose(seed)
    round_tripped = convert_pose_yup_to_zup(convert_pose_zup_to_yup(pose_mp))
    np.testing.assert_allclose(round_tripped, pose_mp, atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_pose_conversion_preserves_rigidity(seed: int) -> None:
    pose_rt = convert_pose_zup_to_yup(_random_rigid_pose(seed))
    rotation = pose_rt[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_array_equal(pose_rt[3], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("seed", range(10))
def test_pose_translation_converts_like_a_point(seed: int) -> None:
    pose_mp = _random_rigid_pose(seed)
    pose_rt = convert_pose_zup_to_yup(pose_mp)
    np.testing.assert_allclose(
        pose_rt[:3, 3], convert_point_zup_to_yup(pose_mp[:3, 3]), atol=1e-12
    )


def test_shape_validation() -> None:
    with pytest.raises(ValueError):
        convert_point_zup_to_yup([1.0, 2.0])
    with pytest.raises(ValueError):
        convert_pose_zup_to_yup(np.eye(3))
