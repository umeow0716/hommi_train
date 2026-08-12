from __future__ import annotations

import torch
from torch import nn


def gradient_norm(model: nn.Module) -> float:
    """Return the global L2 gradient norm before gradient clipping."""
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    total = torch.zeros((), device=device, dtype=torch.float32)
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.detach().float().norm(2).square()
    return float(total.sqrt().cpu())


def action_mse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Full-horizon action mean-squared error as a Python float."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"action shapes differ: prediction={tuple(prediction.shape)} "
            f"target={tuple(target.shape)}"
        )
    return float(torch.nn.functional.mse_loss(prediction.float(), target.float()).cpu())
