from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """HoMMI / UMI effective 20 Hz dataset defaults."""

    obs_horizon: int = 2
    action_horizon: int = 16
    image_size: int = 224
    val_ratio: float = 0.05
    action_padding: bool = True
    frame_cache: Literal["none", "lru", "ram"] = "ram"
    frame_cache_size: int = 2048
    frame_preload_batch_size: int = 8


@dataclass(frozen=True, slots=True)
class DiTTrainConfig:
    """Defaults mirrored from HoMMI ``umi_policy_dit.yaml``."""

    batch_size: int = 16
    num_workers: int = 8
    epochs: int = 1000
    lr: float = 7.5e-5
    obs_encoder_lr: float = 7.5e-6
    weight_decay: float = 1.0e-6
    obs_encoder_weight_decay: float = 1.0e-6
    warmup_steps: int = 50
    clip_grad_norm: float = 5.0
    sample_every: int = 10
    log_grad_norm_every: int = 50
    seed: int = 42


@dataclass(frozen=True, slots=True)
class DiTModelConfig:
    """Defaults mirrored from HoMMI ``diffusion_dit.yaml``."""

    model_name: str = "vit_base_patch16_clip_224.openai"
    pretrained: bool = True
    n_action_steps: int = 8
    num_inference_steps: int = 16
    attention_embed_dim: int = 768
    diffusion_timestep_embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    train_diffusion_n_samples: int = 8
