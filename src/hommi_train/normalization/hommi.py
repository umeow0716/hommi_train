from __future__ import annotations

import numpy as np
import torch

from hommi_diffusion_policy import LinearNormalizer, SingleFieldLinearNormalizer

from ..dataset.hommi_hdf5 import HommiHDF5Dataset


def _stats(data: np.ndarray) -> dict[str, torch.Tensor]:
    tensor = torch.as_tensor(data, dtype=torch.float32).reshape(-1, data.shape[-1])
    return {
        "min": tensor.amin(dim=0),
        "max": tensor.amax(dim=0),
        "mean": tensor.mean(dim=0),
        "std": tensor.std(dim=0),
    }


def _identity_normalizer(data: np.ndarray) -> SingleFieldLinearNormalizer:
    dim = int(data.shape[-1])
    return SingleFieldLinearNormalizer.create_manual(
        scale=torch.ones(dim, dtype=torch.float32),
        offset=torch.zeros(dim, dtype=torch.float32),
        input_stats=_stats(data),
    )


def _range_normalizer(data: np.ndarray) -> SingleFieldLinearNormalizer:
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, last_n_dims=1, mode="limits")
    return normalizer


def _concat_normalizers(
    parts: list[SingleFieldLinearNormalizer],
) -> SingleFieldLinearNormalizer:
    if not parts:
        raise ValueError("cannot concatenate zero normalizers")

    scale = torch.cat([part.params_dict["scale"].detach() for part in parts], dim=0)
    offset = torch.cat([part.params_dict["offset"].detach() for part in parts], dim=0)
    stats = {
        name: torch.cat(
            [part.params_dict["input_stats"][name].detach() for part in parts],
            dim=0,
        )
        for name in ("min", "max", "mean", "std")
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats=stats,
    )


def build_hommi_normalizer(dataset: HommiHDF5Dataset) -> LinearNormalizer:
    """Fit HoMMI normalization from a *training* dataset only.

    Per arm, the model representation is ``pos(3) + rot6(6) + gripper(1)``:

    - position: range/limits normalization
    - rotation-6D: identity
    - gripper: range/limits normalization
    - RGB: identity here; pretrained vision normalization remains in the encoder

    The resulting normalizer is injected into the policy with
    ``policy.set_normalizer(normalizer)``. Validation data must never be used to
    fit these statistics.
    """
    obs, action = dataset.collect_lowdim_training_arrays()
    expected_action_dim = dataset.num_arms * 10
    if action.ndim != 2 or action.shape[1] != expected_action_dim:
        raise ValueError(
            f"expected action arrays [N,{expected_action_dim}], got {action.shape}"
        )

    normalizer = LinearNormalizer()
    action_parts: list[SingleFieldLinearNormalizer] = []

    for arm_idx in range(dataset.num_arms):
        camera_key = f"camera{arm_idx}_main_rgb"
        pos_key = f"robot{arm_idx}_eef_pos"
        rot_key = f"robot{arm_idx}_eef_rot_axis_angle"
        gripper_key = f"robot{arm_idx}_gripper_width"

        normalizer[camera_key] = SingleFieldLinearNormalizer.create_identity()
        normalizer[pos_key] = _range_normalizer(obs[pos_key])
        normalizer[rot_key] = _identity_normalizer(obs[rot_key])
        normalizer[gripper_key] = _range_normalizer(obs[gripper_key])

        start = arm_idx * 10
        action_parts.extend(
            (
                _range_normalizer(action[:, start : start + 3]),
                _identity_normalizer(action[:, start + 3 : start + 9]),
                _range_normalizer(action[:, start + 9 : start + 10]),
            )
        )

    normalizer["action"] = _concat_normalizers(action_parts)
    return normalizer
