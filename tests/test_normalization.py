from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from hommi_train.dataset import HommiHDF5Dataset
from hommi_train.normalization import build_hommi_normalizer


def _write_single_arm(path: Path) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="single-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left"], dtype=string_dtype)
        group = f.create_group("episode_001")
        action = np.zeros((6, 8), dtype=np.float32)
        action[:, 0] = np.arange(6, dtype=np.float32) * 0.1
        action[:, 3] = 1.0
        action[:, 7] = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        group.create_dataset("action", data=action)
        group.create_dataset("video_index", data=np.arange(6, dtype=np.int32)[:, None])
        video_group = group.create_group("video")
        video = video_group.create_dataset("left", data=np.frombuffer(b"fake", dtype=np.uint8))
        video.attrs["container"] = "mp4"
        video.attrs["codec"] = "h264"


def _write_dual_arm(path: Path) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="dual-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left", "right"], dtype=string_dtype)
        group = f.create_group("episode_001")
        action = np.zeros((6, 16), dtype=np.float32)
        action[:, 0] = np.arange(6, dtype=np.float32) * 0.1
        action[:, 3] = 1.0
        action[:, 7] = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        action[:, 8] = np.arange(6, dtype=np.float32) * 2.0
        action[:, 11] = 1.0
        action[:, 15] = np.linspace(0.2, 0.8, 6, dtype=np.float32)
        group.create_dataset("action", data=action)
        group.create_dataset("video_index", data=np.tile(np.arange(6, dtype=np.int32)[:, None], (1, 2)))
        video_group = group.create_group("video")
        for side in ("left", "right"):
            video = video_group.create_dataset(side, data=np.frombuffer(b"fake", dtype=np.uint8))
            video.attrs["container"] = "mp4"
            video.attrs["codec"] = "h264"


def test_build_hommi_normalizer_single_arm(tmp_path: Path) -> None:
    path = tmp_path / "single.hdf5"
    _write_single_arm(path)
    dataset = HommiHDF5Dataset(
        path,
        obs_horizon=2,
        action_horizon=3,
        decode_images=False,
    )
    normalizer = build_hommi_normalizer(dataset)

    assert set(normalizer.params_dict.keys()) == {
        "camera0_main_rgb",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "action",
    }
    assert normalizer["action"].params_dict["scale"].shape == (10,)
    torch.testing.assert_close(
        normalizer["robot0_eef_rot_axis_angle"].params_dict["scale"],
        torch.ones(6),
    )
    torch.testing.assert_close(
        normalizer["action"].params_dict["scale"][3:9],
        torch.ones(6),
    )


def test_dual_arm_action_normalization_is_per_arm(tmp_path: Path) -> None:
    path = tmp_path / "dual.hdf5"
    _write_dual_arm(path)
    dataset = HommiHDF5Dataset(
        path,
        obs_horizon=2,
        action_horizon=3,
        decode_images=False,
    )
    normalizer = build_hommi_normalizer(dataset)
    scale = normalizer["action"].params_dict["scale"]

    assert scale.shape == (20,)
    # The right arm moves 20x farther per step than the left arm. Per-arm
    # fitting must therefore produce different position scales.
    assert float(scale[0]) > float(scale[10])
    torch.testing.assert_close(scale[3:9], torch.ones(6))
    torch.testing.assert_close(scale[13:19], torch.ones(6))
    assert "robot1_eef_pos" in normalizer.params_dict
