from __future__ import annotations

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..config import DiTTrainConfig


def build_optimizer(policy: nn.Module, config: DiTTrainConfig) -> Optimizer:
    """Build the policy-defined AdamW groups with HoMMI learning-rate defaults."""
    get_optimizer = getattr(policy, "get_optimizer", None)
    if get_optimizer is None:
        raise TypeError("policy must implement get_optimizer()")
    return get_optimizer(
        lr=config.lr,
        weight_decay=config.weight_decay,
        obs_encoder_lr=config.obs_encoder_lr,
        obs_encoder_weight_decay=config.obs_encoder_weight_decay,
        betas=config.betas,
    )


def build_lr_scheduler(
    optimizer: Optimizer,
    config: DiTTrainConfig,
    *,
    num_training_steps: int,
) -> LRScheduler:
    """Build HoMMI's cosine scheduler through diffusers.optimization."""
    if num_training_steps < 1:
        raise ValueError("num_training_steps must be >= 1")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")

    from diffusers.optimization import get_scheduler

    return get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=int(num_training_steps),
    )
