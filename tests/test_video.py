from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from hommi_train.dataset.hommi_hdf5 import HommiHDF5Dataset


def test_embedded_mp4_decodes_with_torchcodec(tmp_path: Path) -> None:
    # The test runner writes this fixture path before pytest starts.
    mp4_path = Path("/mnt/data/tiny.mp4")
    if not mp4_path.exists():
        return

    path = tmp_path / "dataset.hdf5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="single-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left"], dtype=string_dtype)
        g = f.create_group("episode_001")
        action = np.zeros((6, 8), dtype=np.float32)
        action[:, 3] = 1.0
        g.create_dataset("action", data=action)
        g.create_dataset("video_index", data=np.array([[0], [1], [2], [3], [4], [5]], np.int32))
        vg = g.create_group("video")
        payload = np.frombuffer(mp4_path.read_bytes(), dtype=np.uint8)
        video = vg.create_dataset("left", data=payload)
        video.attrs["container"] = "mp4"
        video.attrs["codec"] = "h264"

    ds = HommiHDF5Dataset(path, obs_horizon=2, image_size=32, video_device="cpu")
    sample = ds[3]
    image = sample["obs"]["camera0_main_rgb"]
    assert image.shape == (2, 3, 32, 32)
    assert image.dtype == torch.float32
    assert image.device.type == "cpu"
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert ds.frame_cache_mode == "ram"
    assert ds.num_cached_frames == 6
    assert ds.frame_cache_bytes == 6 * 3 * 32 * 32
    ds.close()


def test_ram_cache_works_with_dataloader_workers(tmp_path: Path) -> None:
    mp4_path = Path("/mnt/data/tiny.mp4")
    if not mp4_path.exists():
        return

    path = tmp_path / "dataset_workers.hdf5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="single-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left"], dtype=string_dtype)
        g = f.create_group("episode_001")
        action = np.zeros((6, 8), dtype=np.float32)
        action[:, 3] = 1.0
        g.create_dataset("action", data=action)
        g.create_dataset(
            "video_index",
            data=np.arange(6, dtype=np.int32)[:, None],
        )
        vg = g.create_group("video")
        video = vg.create_dataset(
            "left", data=np.frombuffer(mp4_path.read_bytes(), dtype=np.uint8)
        )
        video.attrs["container"] = "mp4"
        video.attrs["codec"] = "h264"

    ds = HommiHDF5Dataset(path, obs_horizon=2, image_size=32, frame_cache="ram")
    loader = DataLoader(ds, batch_size=2, num_workers=2, shuffle=False)
    batch = next(iter(loader))
    image = batch["obs"]["camera0_main_rgb"]
    assert image.shape == (2, 2, 3, 32, 32)
    assert image.dtype == torch.float32
    assert ds.num_cached_frames == 6
    ds.close()
