from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DiTTrainConfig


def build_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: DiTTrainConfig,
) -> tuple[DataLoader, DataLoader]:
    """Construct train/validation DataLoaders from runtime training settings."""
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    persistent_workers = bool(config.persistent_workers and config.num_workers > 0)
    common: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": config.drop_last,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)

    if len(train_loader) == 0:
        raise ValueError("train DataLoader is empty; reduce batch_size or disable drop_last")
    if len(val_loader) == 0:
        raise ValueError(
            "validation DataLoader is empty; reduce batch_size, disable drop_last, "
            "or increase the validation split"
        )
    return train_loader, val_loader


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move model inputs to the training device without copying metadata."""
    return {
        "obs": {
            key: value.to(device, non_blocking=True)
            for key, value in batch["obs"].items()
        },
        "action": batch["action"].to(device, non_blocking=True),
        **(
            {
                "valid_action_mask": batch["valid_action_mask"].to(
                    device, non_blocking=True
                )
            }
            if "valid_action_mask" in batch
            else {}
        ),
    }
