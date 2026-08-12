from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


class HommiEMAModel:
    """HoMMI / diffusion-policy EMA warmup implementation.

    Parameters are kept in a full averaged model so inference can run directly
    from ``averaged_model`` without swapping weights in and out of the training
    policy. BatchNorm parameters are copied instead of exponentially averaged,
    matching the previous standalone trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, int(optimization_step) - self.update_after_step - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: nn.Module) -> None:
        self.decay = self.get_decay(self.optimization_step)
        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules(), strict=True
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False),
                ema_module.parameters(recurse=False),
                strict=True,
            ):
                source = param.detach().to(
                    device=ema_param.device,
                    dtype=ema_param.dtype,
                )
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    ema_param.copy_(source)
                elif not param.requires_grad:
                    ema_param.copy_(source)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(source, alpha=1.0 - self.decay)
        self.optimization_step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "averaged_model": self.averaged_model.state_dict(),
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "decay": self.decay,
            "optimization_step": self.optimization_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any], *, strict: bool = True) -> None:
        self.averaged_model.load_state_dict(
            state_dict["averaged_model"],
            strict=strict,
        )
        self.update_after_step = int(
            state_dict.get("update_after_step", self.update_after_step)
        )
        self.inv_gamma = float(state_dict.get("inv_gamma", self.inv_gamma))
        self.power = float(state_dict.get("power", self.power))
        self.min_value = float(state_dict.get("min_value", self.min_value))
        self.max_value = float(state_dict.get("max_value", self.max_value))
        self.decay = float(state_dict.get("decay", 0.0))
        self.optimization_step = int(state_dict.get("optimization_step", 0))
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
