import numpy as np
import pytest

from vs_pipeline.cubemap_transforms import (
    AXES,
    apply_canonical_transforms,
    dir_to_equirect_uv,
    flip_lossless,
    rotate_cw_lossless,
    transform_face,
)


def _reference_canonical(faces: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Verbatim port of the original proven script — the regression oracle."""
    face_rotations = {"+X": 90, "-X": 270, "+Y": 0, "-Y": 180, "+Z": 270, "-Z": 270}

    def rot(a: np.ndarray, ang: int) -> np.ndarray:
        return np.rot90(a, k=(4 - ang // 90) % 4)

    rotated = {k: rot(v, face_rotations[k]) for k, v in faces.items()}
    out: dict[str, np.ndarray] = {}
    for f in ["+X", "-X"]:
        if f in rotated:
            out[f] = np.fliplr(rotated[f])
    for f in ["+Y", "-Y"]:
        if f in rotated:
            out[f] = np.flipud(rotated[f])
    if "+Z" in rotated and "-Z" in rotated:
        out["+Z"] = np.fliplr(rotated["-Z"])
        out["-Z"] = np.fliplr(rotated["+Z"])
    elif "+Z" in rotated:
        out["-Z"] = np.fliplr(rotated["+Z"])
    elif "-Z" in rotated:
        out["+Z"] = np.fliplr(rotated["-Z"])
    return out


def test_parity_with_reference_implementation() -> None:
    """Byte-identical to the certified original across all faces."""
    rng = np.random.default_rng(11)
    faces = {axis: rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8) for axis in AXES}
    ours = apply_canonical_transforms(faces)
    ref = _reference_canonical(faces)
    assert set(ours) == set(ref)
    for key in ref:
        np.testing.assert_array_equal(ours[key], ref[key])


def test_parity_with_partial_face_sets() -> None:
    rng = np.random.default_rng(12)
    for subset in (["+Z"], ["-Z"], ["+X", "+Y", "-Z"]):
        faces = {a: rng.integers(0, 255, size=(4, 4), dtype=np.uint8) for a in subset}
        ours = apply_canonical_transforms(faces)
        ref = _reference_canonical(faces)
        assert set(ours) == set(ref)
        for key in ref:
            np.testing.assert_array_equal(ours[key], ref[key])


def test_rotation_direction_is_clockwise() -> None:
    arr = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(rotate_cw_lossless(arr, 90), [[3, 1], [4, 2]])
    np.testing.assert_array_equal(rotate_cw_lossless(arr, 0), arr)
    with pytest.raises(ValueError):
        rotate_cw_lossless(arr, 45)


def test_z_faces_swap_slots() -> None:
    pz = np.full((2, 2), 1, dtype=np.uint8)
    nz = np.full((2, 2), 2, dtype=np.uint8)
    _, out_axis_pz = transform_face(pz, "+Z")
    _, out_axis_nz = transform_face(nz, "-Z")
    assert out_axis_pz == "-Z"
    assert out_axis_nz == "+Z"
    result = apply_canonical_transforms({"+Z": pz, "-Z": nz})
    assert result["+Z"][0, 0] == 2  # content came from source -Z
    assert result["-Z"][0, 0] == 1  # content came from source +Z


def test_transforms_are_lossless_for_float32() -> None:
    """Pure permutations: the multiset of float32 values is preserved exactly."""
    rng = np.random.default_rng(13)
    depth = rng.normal(size=(16, 16)).astype(np.float32)
    for axis in AXES:
        transformed, _ = transform_face(depth, axis)
        np.testing.assert_array_equal(np.sort(transformed, axis=None), np.sort(depth, axis=None))


def test_flip_ops() -> None:
    arr = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(flip_lossless(arr, "flip_h"), [[2, 1], [4, 3]])
    np.testing.assert_array_equal(flip_lossless(arr, "flip_v"), [[3, 4], [1, 2]])
    assert flip_lossless(arr, "none") is arr
    with pytest.raises(ValueError):
        flip_lossless(arr, "diagonal")


def test_dir_to_equirect_uv() -> None:
    u, v = dir_to_equirect_uv(0.0, 1.0, 0.0)
    assert u == pytest.approx(0.5)
    assert v == pytest.approx(0.5)
    _, v_up = dir_to_equirect_uv(0.0, 0.0, 1.0)
    assert v_up == pytest.approx(0.0)
    _, v_down = dir_to_equirect_uv(0.0, 0.0, -1.0)
    assert v_down == pytest.approx(1.0)
