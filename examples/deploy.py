"""Minimal real-robot inference loop for a HoMMI portable ``model.pt``.

This example deliberately uses synthetic observations so it can be read without
robot/camera SDK dependencies:

- RGB observations are black NumPy images.
- EEF positions are all zeros.
- EEF rotations use a valid identity rotation-6D representation.
- Gripper observations are zero (closed in the default 0=closed, 1=open convention).

Replace ``make_dummy_observation()`` with your camera + robot state reader on the
real system.  The model output is already unnormalized by the policy.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from hommi_train import (
    configure_evaluation_backend,
    hommi_train_config_from_mapping,
    load_portable_policy,
    resolve_device,
    resolve_precision,
)
from hommi_train.dataset import rotation_6d_to_matrix_umi


IDENTITY_ROTATION_6D = np.asarray(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one TensorRT-accelerated HoMMI inference iteration."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="portable model.pt produced by hommi-train",
    )
    return parser.parse_args()


def make_dummy_observation(
    shape_meta: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build one synthetic observation batch matching the trained model.

    A real deployment should keep exactly these tensor shapes and replace the
    NumPy values with synchronized camera frames / robot states.  EEF history in
    the trained dataset is expressed relative to the latest EEF frame: keep the
    last ``horizon`` world poses and convert them with
    ``relative_pose9(current_world_eef, world_pose_history)`` instead of feeding
    raw world XYZ / quaternion values directly.
    """
    obs: dict[str, torch.Tensor] = {}

    for key, attr in shape_meta["obs"].items():
        if attr.get("ignore_by_policy", False):
            continue

        horizon = int(attr.get("horizon", 1))
        shape = tuple(int(value) for value in attr["shape"])
        obs_type = attr.get("type", "low_dim")

        if obs_type == "rgb":
            # shape_meta stores RGB as [C, H, W].  Pretend the camera returned
            # black uint8 HWC frames, then convert to the model's [0, 1] CHW input.
            channels, height, width = shape
            black_rgb = np.zeros(
                (horizon, height, width, channels),
                dtype=np.uint8,
            )
            tensor = (
                torch.from_numpy(black_rgb)
                .permute(0, 3, 1, 2)
                .to(dtype=torch.float32)
                .div_(255.0)
            )

        elif "eef_rot" in key and shape == (6,):
            # Rotation-6D must represent a valid rotation.  Six zeros are not a
            # valid rotation, so use identity R = I instead.
            value = np.broadcast_to(
                IDENTITY_ROTATION_6D,
                (horizon, 6),
            ).copy()
            tensor = torch.from_numpy(value)

        else:
            # EEF position = [0, 0, 0], gripper = 0, and any other low-dim
            # observation is zero in this standalone example.
            value = np.zeros((horizon, *shape), dtype=np.float32)
            tensor = torch.from_numpy(value)

        # Policy input convention is [B, T, ...].  This example has B=1.
        obs[key] = tensor.unsqueeze(0).to(device=device, non_blocking=True)

    return obs


def decode_single_arm_action(
    action10: np.ndarray,
    *,
    current_world_eef: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convert one 10-D network action into a target 4x4 EEF pose + gripper.

    HoMMI actions are future poses expressed relative to the last observation
    EEF frame, not a raw XYZ Euler-angle command:

        [rel_x, rel_y, rel_z,
         r00, r01, r02, r10, r11, r12,
         gripper]

    ``rotation_6d_to_matrix_umi`` performs the same row-based Gram-Schmidt
    reconstruction used by the training representation.
    """
    if action10.shape != (10,):
        raise ValueError(f"expected one single-arm 10-D action, got {action10.shape}")

    relative = np.eye(4, dtype=np.float32)
    relative[:3, 3] = action10[:3]
    relative[:3, :3] = rotation_6d_to_matrix_umi(action10[3:9])

    target_world_eef = current_world_eef @ relative

    # The dataset convention is 0.0 = closed, 1.0 = fully open.  Clipping before
    # sending a physical command is a useful final safety boundary.
    gripper = float(np.clip(action10[9], 0.0, 1.0))
    return target_world_eef, gripper


def main() -> None:
    args = parse_args()

    # TensorRT deployment targets CUDA.  ``precision=auto`` selects BF16 when
    # the CUDA device supports it, otherwise FP32.
    device = resolve_device("cuda")
    policy, artifact = load_portable_policy(args.model, device=device)
    config = hommi_train_config_from_mapping(artifact["config"])
    precision = resolve_precision("auto", device)

    backend = configure_evaluation_backend(
        policy,
        backend="tensorrt",
        device=device,
        compile_mode=config.evaluation.compile_mode,
        tensorrt=config.evaluation.tensorrt,
        precision=precision,
    )
    print(f"device={device}, precision={precision}, backend={backend}")

    # ---------------------------------------------------------------------
    # One deployment iteration.
    # Replace this function with your synchronized camera + robot state read.
    # ---------------------------------------------------------------------
    obs = make_dummy_observation(artifact["shape_meta"], device=device)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        prediction = policy.predict_action(obs)

    # ``action`` is the executable receding-horizon chunk selected by the policy.
    # With the default HoMMI config this is typically:
    #
    #   single arm: [B=1, n_action_steps=8, action_dim=10]
    #   dual arm:   [B=1, n_action_steps=8, action_dim=20]
    #
    # ``action_pred`` is the full diffusion horizon, normally [1, 16, D].
    action_chunk = prediction["action"][0].float().cpu().numpy()
    print("action chunk shape:", action_chunk.shape)
    print("first action:", action_chunk[0])

    # Example single-arm value (illustrative only; your trained model will differ):
    # [ 0.012, -0.004,  0.020,
    #   0.999,  0.010, -0.030,  -0.009,  1.000,  0.004,
    #   0.72 ]
    #   |-------- rel XYZ --------| |-------- rotation 6D ---------|  grip
    #
    # For dual arm, action_dim=20 and the vector is two consecutive 10-D blocks:
    #   [robot0 action(10), robot1 action(10)]
    # following the arm order used by the training dataset/task config.

    if action_chunk.shape[-1] == 10:
        # The dummy observation has position zero + identity orientation, so its
        # world EEF transform is identity.  On a real robot, replace this with the
        # FK / state-estimator transform captured at the SAME observation time.
        current_world_eef = np.eye(4, dtype=np.float32)

        target_world_eef, gripper = decode_single_arm_action(
            action_chunk[0],
            current_world_eef=current_world_eef,
        )
        print("target world EEF pose:\n", target_world_eef)
        print("gripper:", gripper)

        # Typical real-robot integration:
        #
        # robot.command_eef_pose(target_world_eef)
        # robot.command_gripper(gripper)
        #
        # Then acquire a NEW synchronized observation and call predict_action()
        # again (receding-horizon control).  You may also execute more than one
        # row from action_chunk before replanning, depending on your controller
        # rate / latency trade-off.
        #
        # Important: every row in this action_chunk is relative to the SAME
        # current_world_eef captured for this prediction.  Do not chain row 1 on
        # top of row 0 as if each row were an incremental delta transform.


if __name__ == "__main__":
    main()
