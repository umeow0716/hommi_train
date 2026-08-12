from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from hommi_train.dataset import HommiHDF5Dataset, inspect_hommi_hdf5


def _write_fixture(path: Path, video_bytes: bytes | None = None) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="single-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left"], dtype=string_dtype)
        g = f.create_group("episode_001")
        action = np.zeros((6, 8), dtype=np.float32)
        action[:, 0] = np.arange(6, dtype=np.float32) * 0.1
        action[:, 3] = 1.0  # qw
        action[:, 7] = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        g.create_dataset("action", data=action)
        g.create_dataset("video_index", data=np.array([[0], [1], [2], [3], [4], [5]], np.int32))
        vg = g.create_group("video")
        payload = np.frombuffer(video_bytes or b"not-a-real-mp4", dtype=np.uint8)
        video = vg.create_dataset("left", data=payload)
        video.attrs["container"] = "mp4"
        video.attrs["codec"] = "h264"


def test_schema_and_lowdim_sample(tmp_path: Path) -> None:
    path = tmp_path / "dataset.hdf5"
    _write_fixture(path)

    info = inspect_hommi_hdf5(path)
    assert info.hz == 20.0
    assert info.arm_order == ("left",)
    assert info.num_episodes == 1
    assert info.num_samples == 6

    ds = HommiHDF5Dataset(
        path,
        obs_horizon=2,
        action_horizon=3,
        image_size=32,
        decode_images=False,
    )
    assert len(ds) == 6
    assert ds.action_dim == 10
    assert ds.shape_meta["action"]["shape"] == [10]

    sample = ds[2]
    assert sample["action"].shape == (3, 10)
    assert sample["obs"]["robot0_eef_pos"].shape == (2, 3)
    assert sample["obs"]["robot0_eef_rot_axis_angle"].shape == (2, 6)
    assert sample["obs"]["robot0_gripper_width"].shape == (2, 1)
    assert "camera0_main_rgb" not in sample["obs"]

    # t=2 is the reference: obs positions t=1,t=2 -> -0.1,0;
    # actions t=2,t=3,t=4 -> 0,0.1,0.2.
    torch.testing.assert_close(
        sample["obs"]["robot0_eef_pos"][:, 0],
        torch.tensor([-0.1, 0.0]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        sample["action"][:, 0],
        torch.tensor([0.0, 0.1, 0.2]),
        atol=1e-6,
        rtol=0,
    )


def test_action_padding(tmp_path: Path) -> None:
    path = tmp_path / "dataset.hdf5"
    _write_fixture(path)
    ds = HommiHDF5Dataset(path, action_horizon=4, decode_images=False)
    sample = ds[len(ds) - 1]
    assert sample["action"].shape == (4, 10)
    torch.testing.assert_close(sample["action"][:, :9], torch.tensor([[0, 0, 0, 1, 0, 0, 0, 1, 0]] * 4, dtype=torch.float32))


def test_no_padding_reduces_sample_count(tmp_path: Path) -> None:
    path = tmp_path / "dataset.hdf5"
    _write_fixture(path)
    ds = HommiHDF5Dataset(path, action_horizon=4, action_padding=False, decode_images=False)
    assert len(ds) == 3


def test_rejects_wrong_action_width(tmp_path: Path) -> None:
    path = tmp_path / "dataset.hdf5"
    _write_fixture(path)
    with h5py.File(path, "r+") as f:
        del f["episode_001/action"]
        f["episode_001"].create_dataset("action", data=np.zeros((6, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="action must be"):
        inspect_hommi_hdf5(path)


def test_dual_arm_layout(tmp_path: Path) -> None:
    path = tmp_path / "dual.hdf5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="dual-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left", "right"], dtype=string_dtype)
        g = f.create_group("episode_001")
        action = np.zeros((4, 16), dtype=np.float32)
        action[:, 3] = 1.0
        action[:, 11] = 1.0
        action[:, 8] = np.arange(4, dtype=np.float32) * 0.2
        g.create_dataset("action", data=action)
        g.create_dataset("video_index", data=np.arange(8, dtype=np.int32).reshape(4, 2))
        vg = g.create_group("video")
        for side in ("left", "right"):
            video = vg.create_dataset(side, data=np.frombuffer(b"fake", dtype=np.uint8))
            video.attrs["container"] = "mp4"
            video.attrs["codec"] = "h264"

    ds = HommiHDF5Dataset(path, obs_horizon=2, action_horizon=2, decode_images=False)
    sample = ds[2]
    assert ds.arm_order == ("left", "right")
    assert ds.action_dim == 20
    assert sample["action"].shape == (2, 20)
    assert "robot1_eef_pos" in sample["obs"]
    assert ds.shape_meta["obs"]["camera1_main_rgb"]["shape"] == [3, 224, 224]
