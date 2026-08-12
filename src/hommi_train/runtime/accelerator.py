from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Precision = Literal["fp32", "bf16"]


@dataclass(frozen=True, slots=True)
class ResolvedAccelerator:
    device: torch.device
    precision: Precision
    pin_memory: bool


def resolve_device(value: str | torch.device = "auto") -> torch.device:
    if isinstance(value, torch.device):
        device = value
    elif value == "auto":
        accelerator = None
        if hasattr(torch, "accelerator"):
            accelerator = torch.accelerator.current_accelerator(check_available=True)
        device = accelerator if accelerator is not None else torch.device("cpu")
    else:
        device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    return device


def resolve_precision(
    value: Literal["auto", "fp32", "bf16"],
    device: torch.device,
) -> Precision:
    if value == "fp32":
        return "fp32"
    if value == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but the selected CUDA device does not support it")
        if device.type not in {"cuda", "cpu"}:
            raise RuntimeError(f"BF16 autocast is not configured for {device.type!r}")
        return "bf16"

    # Automatic mode is intentionally conservative outside CUDA. BF16 is a
    # strong default on modern NVIDIA hardware, while CPU/MPS support and speed
    # vary more by platform.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def resolve_pin_memory(value: bool | Literal["auto"], device: torch.device) -> bool:
    if value == "auto":
        return device.type == "cuda"
    return bool(value)


def resolve_accelerator(
    *,
    device: str | torch.device = "auto",
    precision: Literal["auto", "fp32", "bf16"] = "auto",
    pin_memory: bool | Literal["auto"] = "auto",
) -> ResolvedAccelerator:
    resolved_device = resolve_device(device)
    return ResolvedAccelerator(
        device=resolved_device,
        precision=resolve_precision(precision, resolved_device),
        pin_memory=resolve_pin_memory(pin_memory, resolved_device),
    )
