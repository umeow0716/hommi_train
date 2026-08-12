from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    mode: Literal["sampled", "full"]
    action_mse: float
    num_batches: int
    num_samples: int
    num_action_values: int
    device: str
    precision: Literal["fp32", "bf16"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError(f"unsupported precision: {precision!r}")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 evaluation requested but this CUDA device does not support BF16")
    if device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"BF16 autocast is not configured for {device.type!r}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _move_obs(obs: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in obs.items()}


def evaluate_policy(
    policy: nn.Module,
    data_loader: DataLoader,
    *,
    device: str | torch.device,
    mode: Literal["sampled", "full"] = "sampled",
    precision: Literal["fp32", "bf16"] = "bf16",
    seed: int = 42,
) -> EvaluationResult:
    """Evaluate EMA inference using full-horizon action MSE.

    ``sampled`` intentionally evaluates one batch to mirror the historical
    HoMMI workspace metric. ``full`` consumes the entire loader and aggregates
    squared error by action element so incomplete final batches are weighted
    correctly.
    """
    if mode not in {"sampled", "full"}:
        raise ValueError("mode must be 'sampled' or 'full'")
    resolved = torch.device(device)
    policy = policy.to(resolved).eval()

    # predict_action uses global torch RNG for diffusion noise. fork_rng keeps a
    # standalone evaluation reproducible without perturbing the caller's RNG.
    cuda_devices: list[int] = []
    if resolved.type == "cuda":
        cuda_devices = [resolved.index if resolved.index is not None else torch.cuda.current_device()]

    total_sq_error = 0.0
    total_values = 0
    total_samples = 0
    num_batches = 0

    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if resolved.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            for batch in data_loader:
                obs = _move_obs(batch["obs"], resolved)
                target = batch["action"].to(resolved, non_blocking=True)
                with _autocast(resolved, precision):
                    prediction = policy.predict_action(obs)["action_pred"]
                diff = prediction.float() - target.float()
                total_sq_error += float(diff.square().sum().cpu())
                total_values += int(diff.numel())
                total_samples += int(target.shape[0])
                num_batches += 1
                if mode == "sampled":
                    break

    if total_values == 0:
        raise ValueError("evaluation DataLoader produced no action values")
    return EvaluationResult(
        mode=mode,
        action_mse=total_sq_error / total_values,
        num_batches=num_batches,
        num_samples=total_samples,
        num_action_values=total_values,
        device=str(resolved),
        precision=precision,
    )


def save_evaluation_result(result: EvaluationResult, path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    temp.replace(path)
    return path
