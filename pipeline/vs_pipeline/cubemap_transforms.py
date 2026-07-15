"""Shared, lossless cubemap pixel transforms (Set A canonical pipeline).

CANONICAL ORDER OF OPERATIONS (load-bearing):
1. Set A rotation — per-face clockwise 90-degree multiples
2. Z-inversion — per-face flips + explicit +Z/-Z content SWAP

All operations are strict numpy permutations (rot90 / flip): no image-library
interpolation. Byte-exact for color, and safe for float32 metric depth and
confidence arrays (G-buffer stage reuses this module).

The values were empirically derived for Matterport raw pinhole faces in the
previous production pipeline and are re-validated visually against the
Babylon.js viewer in F4.4 (front-end conventions may add one global
adjustment on top; it must be applied at runtime, never baked here).
"""

from __future__ import annotations

from typing import Any

import numpy as np

AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

#: Set A rotations (clockwise degrees), empirically derived.
FACE_ROTATIONS_CW: dict[str, int] = {
    "+X": 90,
    "-X": 270,
    "+Y": 0,
    "-Y": 180,
    "+Z": 270,
    "-Z": 270,
}

#: Z-inversion step: per-face flip applied AFTER the Set A rotation.
FACE_FLIPS: dict[str, str] = {
    "+X": "flip_h",
    "-X": "flip_h",
    "+Y": "flip_v",
    "-Y": "flip_v",
    "+Z": "flip_h",
    "-Z": "flip_h",
}

#: Z-inversion step: the +Z/-Z content swap, expressed as source axis ->
#: output face slot (X and Y faces stay in place).
FACE_OUTPUT_AXIS: dict[str, str] = {
    "+X": "+X",
    "-X": "-X",
    "+Y": "+Y",
    "-Y": "-Y",
    "+Z": "-Z",
    "-Z": "+Z",
}


def rotate_cw_lossless(array: np.ndarray, angle_deg: int) -> np.ndarray:
    """Lossless clockwise rotation by a 90-degree multiple (np.rot90 is CCW)."""
    if angle_deg not in (0, 90, 180, 270):
        raise ValueError(f"invalid rotation angle: {angle_deg} (must be multiple of 90)")
    return np.rot90(array, k=(4 - angle_deg // 90) % 4)


def flip_lossless(array: np.ndarray, op: str) -> np.ndarray:
    if op == "none":
        return array
    if op == "flip_h":
        return np.fliplr(array)
    if op == "flip_v":
        return np.flipud(array)
    raise ValueError(f"unknown flip op: {op}")


def transform_face(array: np.ndarray, source_axis: str) -> tuple[np.ndarray, str]:
    """Full canonical transform for one face.

    Returns ``(transformed_array, output_axis)`` — the output axis differs from
    the source axis for +Z/-Z (content swap).
    """
    if source_axis not in AXES:
        raise ValueError(f"unknown face axis: {source_axis}")
    rotated = rotate_cw_lossless(array, FACE_ROTATIONS_CW[source_axis])
    flipped = flip_lossless(rotated, FACE_FLIPS[source_axis])
    return flipped, FACE_OUTPUT_AXIS[source_axis]


def apply_canonical_transforms(faces_by_axis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Transform a dict of faces keyed by source axis; returns output-axis keyed dict."""
    unknown = set(faces_by_axis) - set(AXES)
    if unknown:
        raise ValueError(f"unexpected face keys: {unknown}")
    out: dict[str, np.ndarray] = {}
    for axis, array in faces_by_axis.items():
        transformed, output_axis = transform_face(array, axis)
        out[output_axis] = transformed
    return out


FACE_OUTPUT_AXIS_INV: dict[str, str] = {v: k for k, v in FACE_OUTPUT_AXIS.items()}


def inverse_transform_face(array: np.ndarray, output_axis: str) -> tuple[np.ndarray, str]:
    """Exact inverse of :func:`transform_face`.

    Canonical stored face -> the raw (upright pinhole) orientation.
    Returns ``(array, source_axis)``. Pure permutations: lossless.
    """
    if output_axis not in AXES:
        raise ValueError(f"unknown face axis: {output_axis}")
    source_axis = FACE_OUTPUT_AXIS_INV[output_axis]
    unflipped = flip_lossless(array, FACE_FLIPS[source_axis])
    angle = (360 - FACE_ROTATIONS_CW[source_axis]) % 360
    return rotate_cw_lossless(unflipped, angle), source_axis


def dir_to_equirect_uv(dx: Any, dy: Any, dz: Any) -> tuple[Any, Any]:
    """3D direction -> equirectangular UV in [0, 1] (standard spherical mapping)."""
    phi = np.arcsin(np.clip(dz, -1.0, 1.0))
    theta = np.arctan2(dx, dy)
    u = (theta / (2.0 * np.pi)) + 0.5
    v = 0.5 - (phi / np.pi)
    return u, v
