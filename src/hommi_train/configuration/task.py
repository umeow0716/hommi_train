from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..config import HommiTrainConfig, hommi_train_config_from_mapping
from ..dataset import inspect_hommi_hdf5, make_shape_meta

TASK_CONFIG_VERSION = 1


@dataclass(frozen=True, slots=True)
class TaskMetadata:
    name: str
    dataset_path: str
    dataset_type: str
    hz: float
    arm_order: tuple[str, ...]
    shape_meta: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TaskConfigFile:
    task: TaskMetadata
    config: HommiTrainConfig
    format_version: int = TASK_CONFIG_VERSION
    source_path: Path | None = None

    def resolve_dataset_path(self) -> Path:
        raw = Path(self.task.dataset_path).expanduser()
        if raw.is_absolute():
            return raw.resolve()
        base = self.source_path.parent if self.source_path is not None else Path.cwd()
        return (base / raw).resolve()


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a YAML mapping")
    return value


def load_task_config(path: str | Path) -> TaskConfigFile:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    root = _require_mapping(payload, name="task config")
    version = int(root.get("format_version", 1))
    if version != TASK_CONFIG_VERSION:
        raise ValueError(
            f"unsupported task config format_version={version}; "
            f"expected {TASK_CONFIG_VERSION}"
        )

    task = _require_mapping(root.get("task"), name="task")
    arm_order_raw = task.get("arm_order", ())
    if not isinstance(arm_order_raw, (list, tuple)):
        raise TypeError("task.arm_order must be a YAML sequence")
    shape_meta = task.get("shape_meta")
    if not isinstance(shape_meta, dict):
        raise TypeError("task.shape_meta must be a mapping")

    metadata = TaskMetadata(
        name=str(task.get("name", resolved.stem)),
        dataset_path=str(task["dataset_path"]),
        dataset_type=str(task["dataset_type"]),
        hz=float(task["hz"]),
        arm_order=tuple(str(x) for x in arm_order_raw),
        shape_meta=shape_meta,
    )
    config_mapping = root.get("config")
    config = hommi_train_config_from_mapping(
        _require_mapping(config_mapping, name="config") if config_mapping is not None else None,
        strict=True,
    )
    return TaskConfigFile(
        task=metadata,
        config=config,
        format_version=version,
        source_path=resolved,
    )


def task_config_to_mapping(value: TaskConfigFile) -> dict[str, Any]:
    task = asdict(value.task)
    task["arm_order"] = list(value.task.arm_order)
    return {
        "format_version": value.format_version,
        "task": task,
        "config": asdict(value.config),
    }


def save_task_config(value: TaskConfigFile, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            task_config_to_mapping(value),
            file,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
    return output


def init_task_config(
    input_path: str | Path,
    output_path: str | Path,
    *,
    name: str | None = None,
    config: HommiTrainConfig | None = None,
    absolute_dataset_path: bool = False,
) -> TaskConfigFile:
    """Inspect one HoMMI HDF5 file and create a complete editable task YAML."""
    input_resolved = Path(input_path).expanduser().resolve()
    output_resolved = Path(output_path).expanduser().resolve()
    cfg = config or HommiTrainConfig()
    info = inspect_hommi_hdf5(input_resolved)
    shape_meta = make_shape_meta(
        info,
        obs_horizon=cfg.dataset.obs_horizon,
        action_horizon=cfg.dataset.action_horizon,
        image_size=cfg.dataset.image_size,
    )
    if absolute_dataset_path:
        dataset_path = str(input_resolved)
    else:
        dataset_path = os.path.relpath(input_resolved, output_resolved.parent)

    result = TaskConfigFile(
        task=TaskMetadata(
            name=name or input_resolved.stem,
            dataset_path=dataset_path,
            dataset_type=info.dataset_type,
            hz=info.hz,
            arm_order=info.arm_order,
            shape_meta=shape_meta,
        ),
        config=cfg,
        source_path=output_resolved,
    )
    save_task_config(result, output_resolved)
    return result


def validate_task_against_dataset(task: TaskConfigFile, input_path: str | Path) -> None:
    info = inspect_hommi_hdf5(input_path)
    expected = make_shape_meta(
        info,
        obs_horizon=task.config.dataset.obs_horizon,
        action_horizon=task.config.dataset.action_horizon,
        image_size=task.config.dataset.image_size,
    )
    if tuple(info.arm_order) != tuple(task.task.arm_order):
        raise ValueError(
            f"task YAML arm_order={task.task.arm_order} differs from HDF5 {info.arm_order}"
        )
    if info.dataset_type != task.task.dataset_type:
        raise ValueError(
            f"task YAML dataset_type={task.task.dataset_type!r} differs from HDF5 "
            f"{info.dataset_type!r}"
        )
    if not math.isclose(
        float(info.hz),
        float(task.task.hz),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"task YAML hz={task.task.hz:g} differs from HDF5 hz={info.hz:g}"
        )
    if task.task.shape_meta != expected:
        raise ValueError(
            "task YAML shape_meta does not match the selected HDF5/config. "
            "Regenerate it with `python -m hommi_train init-config` after changing dataset "
            "horizons, image size, or arm layout."
        )
