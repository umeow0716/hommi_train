from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from hommi_train.configuration import init_task_config, load_task_config


def _write_dataset(path: Path) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("hz", data=np.float32(20.0))
        f.create_dataset("type", data="single-arm", dtype=string_dtype)
        f.create_dataset("arm_order", data=["left"], dtype=string_dtype)
        group = f.create_group("episode_001")
        action = np.zeros((4, 8), dtype=np.float32)
        action[:, 3] = 1.0
        group.create_dataset("action", data=action)
        group.create_dataset("video_index", data=np.arange(4, dtype=np.int32)[:, None])
        video_group = group.create_group("video")
        video = video_group.create_dataset("left", data=np.zeros(8, dtype=np.uint8))
        video.attrs["container"] = "mp4"
        video.attrs["codec"] = "h264"


def test_init_task_config_contains_shape_meta_and_defaults(tmp_path: Path) -> None:
    dataset = tmp_path / "demo.hdf5"
    config_path = tmp_path / "configs" / "pick.yaml"
    _write_dataset(dataset)

    created = init_task_config(dataset, config_path, name="pick")
    loaded = load_task_config(config_path)

    assert created.task.name == "pick"
    assert loaded.task.arm_order == ("left",)
    assert loaded.task.shape_meta["obs"]["camera0_main_rgb"]["shape"] == [3, 224, 224]
    assert loaded.task.shape_meta["action"]["shape"] == [10]
    assert loaded.config.model.encoder.train_crop_ratio == 0.95
    assert loaded.config.evaluation.backend == "auto"
    assert loaded.resolve_dataset_path() == dataset.resolve()


def test_task_validation_rejects_hz_mismatch(tmp_path: Path) -> None:
    from dataclasses import replace

    from hommi_train.configuration import TaskConfigFile, validate_task_against_dataset

    dataset = tmp_path / "demo.hdf5"
    config_path = tmp_path / "task.yaml"
    _write_dataset(dataset)
    task = init_task_config(dataset, config_path)
    mismatched = TaskConfigFile(
        task=replace(task.task, hz=10.0),
        config=task.config,
        format_version=task.format_version,
        source_path=task.source_path,
    )

    try:
        validate_task_against_dataset(mismatched, dataset)
    except ValueError as exc:
        assert "hz" in str(exc)
    else:
        raise AssertionError("expected HDF5/task Hz mismatch rejection")
