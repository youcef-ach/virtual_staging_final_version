"""Coordinate Lock Rule (hard invariant) — the single place where frames convert.

The E57 origin (0, 0, 0) is preserved unchanged through every stage. One fixed
conversion is applied exactly once, at ingestion:

    Matterport Z-up  ->  runtime Y-up
    X_rt = X_mp,   Y_rt = Z_mp,   Z_rt = -Y_mp

No other module may define or apply a frame conversion. Downstream code always
works in the runtime Y-up frame.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Matrix4 = npt.NDArray[np.float64]
Vector3 = npt.NDArray[np.float64]

#: Homogeneous change-of-basis matrix C: p_rt = C @ p_mp.
#: Pure rotation (det = +1), zero translation -> origin is preserved exactly.
ZUP_TO_YUP: Matrix4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

#: Exact inverse (C is orthonormal with zero translation, so C^-1 = C^T).
YUP_TO_ZUP: Matrix4 = ZUP_TO_YUP.T.copy()


def convert_point_zup_to_yup(point_mp: npt.ArrayLike) -> Vector3:
    """Convert a 3D point from the Matterport Z-up frame to the runtime Y-up frame."""
    p = np.asarray(point_mp, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"expected shape (3,), got {p.shape}")
    result: Vector3 = ZUP_TO_YUP[:3, :3] @ p
    return result


def convert_pose_zup_to_yup(pose_mp: npt.ArrayLike) -> Matrix4:
    """Re-express a rigid 4x4 transform in the runtime Y-up frame.

    Similarity transform: M_rt = C @ M_mp @ C^T. Both the world frame and the
    sweep-local frame are re-expressed consistently; rigid-body structure
    (orthonormal rotation, det = +1) is preserved, and the translation part
    converts exactly like a point.
    """
    m = np.asarray(pose_mp, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"expected shape (4, 4), got {m.shape}")
    result: Matrix4 = ZUP_TO_YUP @ m @ YUP_TO_ZUP
    return result


def convert_pose_yup_to_zup(pose_rt: npt.ArrayLike) -> Matrix4:
    """Exact inverse of :func:`convert_pose_zup_to_yup` (used only in tests/QA)."""
    m = np.asarray(pose_rt, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"expected shape (4, 4), got {m.shape}")
    result: Matrix4 = YUP_TO_ZUP @ m @ ZUP_TO_YUP
    return result
