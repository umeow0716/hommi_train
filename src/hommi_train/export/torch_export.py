from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .artifact import load_portable_policy


class PolicyInferenceModule(nn.Module):
    """Tensor-only inference facade around the policy's dictionary API.

    Inputs are passed as one tuple in ``obs_keys`` order. The output is the
    executable ``action`` chunk rather than the full diagnostic dictionary.
    This facade is useful for ``torch.compile`` and experimental ``torch.export``.
    """

    def __init__(self, policy: nn.Module, obs_keys: Sequence[str]) -> None:
        super().__init__()
        self.policy = policy
        self.obs_keys = tuple(obs_keys)

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(inputs) != len(self.obs_keys):
            raise ValueError(
                f"expected {len(self.obs_keys)} observation tensors, got {len(inputs)}"
            )
        obs = {key: value for key, value in zip(self.obs_keys, inputs, strict=True)}
        return self.policy.predict_action(obs)["action"]


def active_observation_keys(shape_meta: Mapping[str, Any]) -> tuple[str, ...]:
    obs = shape_meta.get("obs")
    if not isinstance(obs, Mapping):
        raise ValueError("shape_meta must contain an obs mapping")
    return tuple(
        sorted(
            key
            for key, attr in obs.items()
            if not attr.get("ignore_by_policy", False)
            and attr.get("type", "low_dim") in {"rgb", "low_dim"}
        )
    )


def example_observation_inputs(
    shape_meta: Mapping[str, Any],
    *,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    obs = shape_meta["obs"]
    keys = active_observation_keys(shape_meta)
    tensors: list[torch.Tensor] = []
    resolved = torch.device(device)
    for key in keys:
        attr = obs[key]
        horizon = int(attr.get("horizon", 1))
        shape = tuple(int(value) for value in attr["shape"])
        tensors.append(
            torch.zeros(
                (batch_size, horizon, *shape),
                dtype=torch.float32,
                device=resolved,
            )
        )
    return keys, tuple(tensors)


def build_inference_module(
    policy: nn.Module,
    shape_meta: Mapping[str, Any],
    *,
    compile: bool = False,
    compile_mode: str = "reduce-overhead",
) -> nn.Module:
    module: nn.Module = PolicyInferenceModule(policy, active_observation_keys(shape_meta))
    module.eval()
    if compile:
        module = torch.compile(module, mode=compile_mode)
    return module


def export_policy_pt2(
    policy: nn.Module,
    shape_meta: Mapping[str, Any],
    output_path: str | Path,
    *,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> Path:
    """Attempt a static-batch ``torch.export`` inference artifact.

    This is intentionally separate from the portable ``model.pt`` path. Full
    diffusion inference must be exportable as one graph; otherwise this function
    raises the original exporter error instead of silently producing a partial
    or eager fallback artifact.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = torch.device(device)
    policy = policy.to(resolved).eval()
    keys, example_inputs = example_observation_inputs(
        shape_meta,
        batch_size=batch_size,
        device=resolved,
    )
    wrapper = PolicyInferenceModule(policy, keys).eval()
    exported = torch.export.export(
        wrapper,
        args=(example_inputs,),
        strict=strict,
    )
    metadata = {
        "format": "hommi-train.torch-export-inference",
        "format_version": 1,
        "obs_keys": list(keys),
        "batch_size": batch_size,
        "shape_meta": shape_meta,
    }
    torch.export.save(
        exported,
        output_path,
        extra_files={"hommi_metadata.json": json.dumps(metadata, sort_keys=True)},
    )
    return output_path


def export_portable_model_pt2(
    model_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> Path:
    policy, payload = load_portable_policy(model_path, device=device)
    return export_policy_pt2(
        policy,
        payload["shape_meta"],
        output_path,
        batch_size=batch_size,
        device=device,
        strict=strict,
    )
