from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from hommi_train.dataset import inspect_hommi_hdf5, split_episode_keys


def _write_multi_episode_fixture(path: Path, count: int = 10) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="single-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left"], dtype=string_dtype)
        for episode_idx in range(count):
            group = f.create_group(f"episode_{episode_idx + 1:03d}")
            action = np.zeros((4, 8), dtype=np.float32)
            action[:, 3] = 1.0
            group.create_dataset("action", data=action)
            group.create_dataset("video_index", data=np.arange(4, dtype=np.int32)[:, None])
            video_group = group.create_group("video")
            video = video_group.create_dataset("left", data=np.frombuffer(b"fake", dtype=np.uint8))
            video.attrs["container"] = "mp4"
            video.attrs["codec"] = "h264"


def test_episode_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    path = tmp_path / "dataset.hdf5"
    _write_multi_episode_fixture(path)
    info = inspect_hommi_hdf5(path)

    first = split_episode_keys(info, val_ratio=0.2, seed=42)
    second = split_episode_keys(info, val_ratio=0.2, seed=42)

    assert first == second
    assert len(first.train_keys) == 8
    assert len(first.val_keys) == 2
    assert not set(first.train_keys) & set(first.val_keys)
    assert set(first.train_keys) | set(first.val_keys) == {
        episode.key for episode in info.episodes
    }


def test_episode_split_requires_two_episodes(tmp_path: Path) -> None:
    path = tmp_path / "dataset.hdf5"
    _write_multi_episode_fixture(path, count=1)
    info = inspect_hommi_hdf5(path)
    with pytest.raises(ValueError, match="at least two episodes"):
        split_episode_keys(info)
