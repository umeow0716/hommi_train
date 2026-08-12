from __future__ import annotations

import importlib
from typing import Literal

import torch
from torch import nn

from ..config import TensorRTConfig

EvaluationBackend = Literal["auto", "eager", "inductor", "tensorrt"]


def tensorrt_available() -> bool:
    """Return whether Torch-TensorRT can actually be imported on this CUDA host.

    ``find_spec()`` alone is not sufficient: a mismatched TensorRT/CUDA install
    can expose the package while failing during module import.  Treat those
    runtime import failures as an unavailable backend so ``backend=auto`` can
    safely fall back to eager evaluation.
    """
    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module("torch_tensorrt")
    except Exception:
        return False
    return True


def resolve_evaluation_backend(value: EvaluationBackend, device: torch.device) -> str:
    if value != "auto":
        if value == "tensorrt" and device.type != "cuda":
            raise RuntimeError("TensorRT evaluation requires a CUDA device")
        return value
    if device.type == "cuda" and tensorrt_available():
        return "tensorrt"
    return "eager"


def _compile_tensorrt_module(
    module: nn.Module,
    config: TensorRTConfig,
    *,
    precision: Literal["fp32", "bf16"],
) -> nn.Module:
    try:
        import torch_tensorrt  # noqa: F401 - registers torch.compile backend
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT backend requested but the required torch-tensorrt runtime could not be imported. "
            "Reinstall hommi-train with a PyTorch/CUDA/TensorRT-compatible environment."
        ) from exc
    if config.min_block_size < 1:
        raise ValueError("TensorRT min_block_size must be >= 1")
    if not 0 <= config.optimization_level <= 5:
        raise ValueError("TensorRT optimization_level must be between 0 and 5")

    options: dict[str, object] = {
        "min_block_size": config.min_block_size,
        "optimization_level": config.optimization_level,
    }
    if precision == "bf16":
        # Torch-TensorRT's graph-aware autocast keeps numerically sensitive
        # nodes in FP32 while lowering eligible TensorRT segments to BF16.
        options["enable_autocast"] = True
        options["autocast_low_precision_type"] = torch.bfloat16

    return torch.compile(
        module,
        backend="torch_tensorrt",
        dynamic=config.dynamic,
        options=options,
    )


def configure_evaluation_backend(
    policy: nn.Module,
    *,
    backend: EvaluationBackend,
    device: torch.device,
    compile_mode: str,
    tensorrt: TensorRTConfig,
    precision: Literal["fp32", "bf16"] = "fp32",
) -> str:
    """Configure inference acceleration and return the resolved backend name.

    TensorRT is intentionally applied to the two compute-heavy pure tensor
    submodules (vision backbone(s) and ActionDiT denoiser). DDIM scheduler
    orchestration remains in eager PyTorch, avoiding export/graph-break issues
    in the Python diffusion loop while accelerating repeated denoiser calls.
    """
    resolved = resolve_evaluation_backend(backend, device)
    if resolved == "eager":
        return resolved

    if resolved == "inductor":
        if hasattr(policy, "model"):
            policy.model = torch.compile(policy.model, backend="inductor", mode=compile_mode)
        encoder = getattr(policy, "obs_encoder", None)
        key_model_map = getattr(encoder, "key_model_map", None)
        if key_model_map is not None:
            for key in list(key_model_map.keys()):
                key_model_map[key] = torch.compile(
                    key_model_map[key], backend="inductor", mode=compile_mode
                )
        return resolved

    if resolved == "tensorrt":
        if not tensorrt_available():
            raise RuntimeError(
                "TensorRT evaluation requested but CUDA + torch-tensorrt are not available"
            )
        if tensorrt.compile_denoiser:
            model = getattr(policy, "model", None)
            if not isinstance(model, nn.Module):
                raise TypeError("policy does not expose an nn.Module 'model' denoiser")
            policy.model = _compile_tensorrt_module(model, tensorrt, precision=precision)
        if tensorrt.compile_backbone:
            encoder = getattr(policy, "obs_encoder", None)
            key_model_map = getattr(encoder, "key_model_map", None)
            if key_model_map is None:
                raise TypeError("policy observation encoder has no key_model_map")
            for key in list(key_model_map.keys()):
                key_model_map[key] = _compile_tensorrt_module(
                    key_model_map[key], tensorrt, precision=precision
                )
        return resolved

    raise ValueError(f"unsupported evaluation backend: {resolved!r}")
