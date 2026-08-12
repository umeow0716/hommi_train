from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_example_module():
    path = Path(__file__).parents[1] / "examples" / "deploy.py"
    spec = importlib.util.spec_from_file_location("hommi_train_deploy_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deploy_example_dummy_observation_and_action_decode() -> None:
    module = _load_example_module()
    shape_meta = {
        "obs": {
            "camera0_main_rgb": {
                "shape": [3, 32, 32],
                "horizon": 2,
                "type": "rgb",
                "ignore_by_policy": False,
            },
            "robot0_eef_pos": {
                "shape": [3],
                "horizon": 2,
                "type": "low_dim",
                "ignore_by_policy": False,
            },
            "robot0_eef_rot_axis_angle": {
                "shape": [6],
                "horizon": 2,
                "type": "low_dim",
                "ignore_by_policy": False,
            },
            "robot0_gripper_width": {
                "shape": [1],
                "horizon": 2,
                "type": "low_dim",
                "ignore_by_policy": False,
            },
        }
    }

    obs = module.make_dummy_observation(shape_meta, device=torch.device("cpu"))
    assert obs["camera0_main_rgb"].shape == (1, 2, 3, 32, 32)
    assert torch.count_nonzero(obs["camera0_main_rgb"]) == 0
    assert torch.count_nonzero(obs["robot0_eef_pos"]) == 0
    assert torch.allclose(
        obs["robot0_eef_rot_axis_angle"][0, 0],
        torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32),
    )

    action = np.asarray(
        [0.1, -0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.75],
        dtype=np.float32,
    )
    target, gripper = module.decode_single_arm_action(
        action,
        current_world_eef=np.eye(4, dtype=np.float32),
    )
    np.testing.assert_allclose(target[:3, 3], action[:3])
    np.testing.assert_allclose(target[:3, :3], np.eye(3), atol=1e-6)
    assert gripper == 0.75
