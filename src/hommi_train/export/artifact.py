from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..config import HommiTrainConfig, hommi_train_config_from_mapping
from ..policy import build_dit_policy
from ..training import load_training_checkpoint

PORTABLE_MODEL_FORMAT = "hommi-train.portable-diffusion-policy"
PORTABLE_MODEL_FORMAT_VERSION = 1


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.unlink(missing_ok=True)
    try:
        torch.save(dict(payload), temp)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def portable_payload_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    source_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Strip a resumable training checkpoint down to EMA inference state."""
    ema = checkpoint.get("ema")
    if not isinstance(ema, Mapping) or "averaged_model" not in ema:
        raise KeyError("training checkpoint is missing ema.averaged_model")
    shape_meta = checkpoint.get("shape_meta")
    if not isinstance(shape_meta, Mapping):
        raise KeyError("training checkpoint is missing shape_meta")

    config = hommi_train_config_from_mapping(checkpoint.get("config"))
    trainer_state = checkpoint.get("trainer_state", {})
    return {
        "format": PORTABLE_MODEL_FORMAT,
        "format_version": PORTABLE_MODEL_FORMAT_VERSION,
        "policy_type": "dit",
        "state_source": "ema",
        "policy_state": ema["averaged_model"],
        "shape_meta": dict(shape_meta),
        "config": asdict(config),
        "metrics": {
            key: float(value) for key, value in checkpoint.get("metrics", {}).items()
        },
        "train_episode_keys": tuple(checkpoint.get("train_episode_keys", ())),
        "val_episode_keys": tuple(checkpoint.get("val_episode_keys", ())),
        "provenance": {
            "source_checkpoint": source_checkpoint,
            "epoch": int(checkpoint.get("epoch", trainer_state.get("epoch", 0))),
            "global_step": int(
                checkpoint.get("global_step", trainer_state.get("global_step", 0))
            ),
        },
    }


def save_portable_checkpoint_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    checkpoint = load_training_checkpoint(checkpoint_path, map_location="cpu")
    payload = portable_payload_from_checkpoint(
        checkpoint,
        source_checkpoint=checkpoint_path.name,
    )
    _atomic_torch_save(payload, output_path)
    return output_path


def load_portable_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load the portable artifact with PyTorch's restricted weights-only loader."""
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("portable model payload must be a dictionary")
    if payload.get("format") != PORTABLE_MODEL_FORMAT:
        raise ValueError(f"not a {PORTABLE_MODEL_FORMAT!r} artifact")
    version = int(payload.get("format_version", 0))
    if version != PORTABLE_MODEL_FORMAT_VERSION:
        raise ValueError(
            f"unsupported portable model format_version={version}; "
            f"expected {PORTABLE_MODEL_FORMAT_VERSION}"
        )
    for key in ("policy_state", "shape_meta", "config"):
        if key not in payload:
            raise KeyError(f"portable model is missing required key {key!r}")
    return payload


def load_portable_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruct an EMA policy without touching training-only state.

    Pretrained backbone initialization is forcibly disabled because the complete
    backbone parameters are already included in ``policy_state``.
    """
    payload = load_portable_payload(path, map_location="cpu")
    config: HommiTrainConfig = hommi_train_config_from_mapping(payload["config"])
    policy = build_dit_policy(
        payload["shape_meta"],
        model_config=config.model,
        ddim_config=config.ddim,
        pretrained_override=False,
    )
    policy.load_state_dict(payload["policy_state"], strict=strict)
    policy.to(torch.device(device))
    policy.eval()
    policy.requires_grad_(False)
    return policy, payload
