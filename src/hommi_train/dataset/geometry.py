from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


def quaternion_wxyz_to_matrix(quaternion: npt.ArrayLike) -> FloatArray:
    """Convert ``[..., qw, qx, qy, qz]`` quaternions to rotation matrices.

    HoMMI dataset files store quaternions in WXYZ order. Inputs are normalized
    defensively so small interpolation / serialization errors do not leak into
    the relative-pose transform.
    """
    q = np.asarray(quaternion, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"expected quaternion (..., 4), got {q.shape}")

    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("zero-length quaternion found")
    q = q / norm

    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    out[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    out[..., 0, 1] = 2.0 * (xy - zw)
    out[..., 0, 2] = 2.0 * (xz + yw)
    out[..., 1, 0] = 2.0 * (xy + zw)
    out[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    out[..., 1, 2] = 2.0 * (yz - xw)
    out[..., 2, 0] = 2.0 * (xz - yw)
    out[..., 2, 1] = 2.0 * (yz + xw)
    out[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


def pose7_wxyz_to_matrix(pose: npt.ArrayLike) -> FloatArray:
    """Convert ``[..., x, y, z, qw, qx, qy, qz]`` to homogeneous matrices."""
    value = np.asarray(pose, dtype=np.float32)
    if value.shape[-1] != 7:
        raise ValueError(f"expected pose (..., 7), got {value.shape}")

    out = np.zeros(value.shape[:-1] + (4, 4), dtype=np.float32)
    out[..., :3, :3] = quaternion_wxyz_to_matrix(value[..., 3:7])
    out[..., :3, 3] = value[..., :3]
    out[..., 3, 3] = 1.0
    return out


def matrix_to_rotation_6d_umi(matrix: npt.ArrayLike) -> FloatArray:
    """Return the UMI / HoMMI 6D rotation representation.

    HoMMI follows UMI and flattens the first two *rows* of the rotation matrix:
    ``[r00, r01, r02, r10, r11, r12]``.
    """
    value = np.asarray(matrix, dtype=np.float32)
    if value.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {value.shape}")
    return value[..., :2, :].copy().reshape(value.shape[:-2] + (6,))


def rotation_6d_to_matrix_umi(
    rotation_6d: npt.ArrayLike,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Inverse of :func:`matrix_to_rotation_6d_umi` using row Gram-Schmidt."""
    d6 = np.asarray(rotation_6d, dtype=np.float32)
    if d6.shape[-1] != 6:
        raise ValueError(f"expected (..., 6), got {d6.shape}")

    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), eps)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), eps)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2).astype(np.float32, copy=False)


def relative_pose9(base_matrix: npt.ArrayLike, matrices: npt.ArrayLike) -> FloatArray:
    """Express transforms in ``base_matrix`` and return ``[pos3, rotation6d]``."""
    base = np.asarray(base_matrix, dtype=np.float32)
    values = np.asarray(matrices, dtype=np.float32)
    if base.shape != (4, 4):
        raise ValueError(f"base_matrix must be (4, 4), got {base.shape}")
    if values.shape[-2:] != (4, 4):
        raise ValueError(f"matrices must end in (4, 4), got {values.shape}")

    base_inv = np.linalg.inv(base).astype(np.float32)
    rel = np.einsum("ij,...jk->...ik", base_inv, values).astype(np.float32)
    return np.concatenate(
        (rel[..., :3, 3], matrix_to_rotation_6d_umi(rel[..., :3, :3])),
        axis=-1,
    ).astype(np.float32, copy=False)
