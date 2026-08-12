from __future__ import annotations

import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .ema import HommiEMAModel


CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG streams required for epoch-boundary training resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG streams captured by :func:`capture_rng_state`."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for device_idx, cuda_state in enumerate(cuda_states[:device_count]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_idx)


def _serializable_config(config: Any) -> Any:
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("checkpoint config must be a dataclass, mapping, or None")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def save_training_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    ema: HommiEMAModel,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    trainer_state: Mapping[str, Any],
    metrics: Mapping[str, float],
    config: Any = None,
    shape_meta: Mapping[str, Any] | None = None,
    train_episode_keys: tuple[str, ...] | None = None,
    val_episode_keys: tuple[str, ...] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save a resumable training checkpoint."""
    state = dict(trainer_state)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        # Top-level aliases keep checkpoints easy to inspect and compatible
        # with the previous standalone trainer's naming.
        "epoch": int(state.get("epoch", 0)),
        "global_step": int(state.get("global_step", 0)),
        "model": policy.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "trainer_state": state,
        "metrics": {key: float(value) for key, value in metrics.items()},
        "config": _serializable_config(config),
        "shape_meta": dict(shape_meta) if shape_meta is not None else None,
        "train_episode_keys": tuple(train_episode_keys or ()),
        "val_episode_keys": tuple(val_episode_keys or ()),
        "rng_state": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    _atomic_torch_save(payload, Path(path))


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and minimally validate a hommi-train checkpoint."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    version = int(checkpoint.get("format_version", 0))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format_version={version}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    for key in ("model", "ema", "optimizer", "lr_scheduler", "trainer_state"):
        if key not in checkpoint:
            raise KeyError(f"checkpoint is missing required key {key!r}")
    return checkpoint


def restore_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    policy: nn.Module,
    ema: HommiEMAModel,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    restore_rng: bool = True,
    strict_model: bool = True,
) -> dict[str, Any]:
    """Restore all mutable training state and return the serialized trainer state."""
    policy.load_state_dict(checkpoint["model"], strict=strict_model)
    ema.load_state_dict(checkpoint["ema"], strict=strict_model)
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if restore_rng and "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return dict(checkpoint["trainer_state"])
