from __future__ import annotations

import numpy as np

from hommi_train.dataset.geometry import (
    matrix_to_rotation_6d_umi,
    pose7_wxyz_to_matrix,
    quaternion_wxyz_to_matrix,
    relative_pose9,
    rotation_6d_to_matrix_umi,
)


def test_identity_wxyz_quaternion() -> None:
    matrix = quaternion_wxyz_to_matrix(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-6)


def test_rotation_6d_round_trip() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(256, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    r = quaternion_wxyz_to_matrix(q)
    d6 = matrix_to_rotation_6d_umi(r)
    r2 = rotation_6d_to_matrix_umi(d6)
    np.testing.assert_allclose(r2, r, atol=2e-5)


def test_relative_pose_uses_current_pose_as_reference() -> None:
    pose = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    matrix = pose7_wxyz_to_matrix(pose)
    relative = relative_pose9(matrix[1], matrix)
    np.testing.assert_allclose(relative[:, 0], [-1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(relative[:, 3:], [[1, 0, 0, 0, 1, 0]] * 3, atol=1e-6)
