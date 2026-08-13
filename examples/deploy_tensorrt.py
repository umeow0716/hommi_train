"""Minimal real-robot deployment example for a precompiled HoMMI TensorRT bundle.

Build the bundle first::

    hommi-train tensorrt -i runs/pick_place

Then run this example::

    python examples/deploy_tensorrt.py -m runs/pick_place/model.trt.ep

The synthetic observation is only a stand-in for your camera and robot state.
Replace ``make_dummy_observation`` with synchronized real observations while
preserving the same [B, T, ...] tensor shapes.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from hommi_train import hommi_train_config_from_mapping, load_tensorrt_policy, resolve_device


IDENTITY_ROTATION_6D = np.asarray(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one inference iteration from a precompiled HoMMI TensorRT bundle."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="model.trt.ep produced by 'hommi-train tensorrt'",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def make_dummy_observation(
    shape_meta: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create one B=1 observation batch matching ``shape_meta``."""
    obs: dict[str, torch.Tensor] = {}

    for key, attr in shape_meta["obs"].items():
        if attr.get("ignore_by_policy", False):
            continue

        horizon = int(attr.get("horizon", 1))
        shape = tuple(int(value) for value in attr["shape"])

        if attr.get("type", "low_dim") == "rgb":
            channels, height, width = shape
            # Real deployment: replace this with your synchronized RGB frames.
            rgb = np.zeros((horizon, height, width, channels), dtype=np.uint8)
            tensor = (
                torch.from_numpy(rgb)
                .permute(0, 3, 1, 2)
                .to(dtype=torch.float32)
                .div_(255.0)
            )
        elif "eef_rot" in key and shape == (6,):
            value = np.broadcast_to(IDENTITY_ROTATION_6D, (horizon, 6)).copy()
            tensor = torch.from_numpy(value)
        else:
            tensor = torch.zeros((horizon, *shape), dtype=torch.float32)

        obs[key] = tensor.unsqueeze(0).to(device=device, non_blocking=True)

    return obs


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    policy, artifact = load_tensorrt_policy(args.model, device=device)
    config = hommi_train_config_from_mapping(artifact["config"])
    precision = artifact["tensorrt_bundle"]["precision"]

    print(f"model={args.model}")
    print(f"device={device}")
    print(f"precision={precision}")
    print(f"n_action_steps={config.policy.n_action_steps}")

    obs = make_dummy_observation(artifact["shape_meta"], device=device)

    # The TensorRT submodules in the bundle handle their own input/output dtype
    # boundary.  This autocast keeps the surrounding eager policy consistent
    # with the precision used when the bundle was built.
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )

    with torch.inference_mode(), autocast:
        prediction = policy.predict_action(obs)

    action = prediction["action"]
    print("action shape:", tuple(action.shape))
    print("first action:", action[0, 0].float().cpu().numpy())

    # Typical real-robot loop:
    #
    # while True:
    #     obs = read_synchronized_observation()
    #     with torch.inference_mode(), autocast:
    #         action_chunk = policy.predict_action(obs)["action"][0]
    #     send_action(action_chunk[0])
    #
    # Re-acquire a fresh observation before replanning.  HoMMI returns an
    # action chunk; whether you execute one or several rows before replanning is
    # a controller-rate / latency trade-off for your robot.


if __name__ == "__main__":
    main()
