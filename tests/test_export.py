from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from hommi_train.config import HommiTrainConfig
from hommi_train.export import (
    PolicyInferenceModule,
    export_policy_pt2,
    load_portable_payload,
    portable_payload_from_checkpoint,
)


class _TinyInferencePolicy(nn.Module):
    def predict_action(self, obs):
        x = obs["x"]
        return {"action": x[:, -1:] * 2.0, "action_pred": x * 2.0}


def _checkpoint() -> dict:
    model = nn.Linear(2, 2)
    return {
        "ema": {"averaged_model": model.state_dict()},
        "shape_meta": {
            "obs": {"x": {"shape": [2], "horizon": 2, "type": "low_dim"}},
            "action": {"shape": [2], "horizon": 2},
        },
        "config": asdict(HommiTrainConfig()),
        "metrics": {"val_action_mse_error": 0.25},
        "train_episode_keys": ("episode_001",),
        "val_episode_keys": ("episode_002",),
        "epoch": 3,
        "global_step": 99,
    }


def test_portable_payload_keeps_only_inference_and_provenance() -> None:
    payload = portable_payload_from_checkpoint(_checkpoint(), source_checkpoint="best.pt")
    assert payload["state_source"] == "ema"
    assert payload["val_episode_keys"] == ("episode_002",)
    assert payload["provenance"]["global_step"] == 99
    assert "optimizer" not in payload
    assert "lr_scheduler" not in payload
    assert "rng_state" not in payload


def test_portable_payload_round_trip_uses_weights_only_loader(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(portable_payload_from_checkpoint(_checkpoint()), path)
    loaded = load_portable_payload(path)
    assert loaded["format_version"] == 1
    assert loaded["shape_meta"]["action"]["shape"] == [2]
    assert loaded["metrics"]["val_action_mse_error"] == 0.25


def test_policy_inference_module_adapts_ordered_tensor_tuple() -> None:
    module = PolicyInferenceModule(_TinyInferencePolicy(), ("x",))
    x = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    output = module(x)
    torch.testing.assert_close(output, x[:, -1:] * 2.0)


def test_torch_export_pt2_round_trip_for_exportable_policy(tmp_path: Path) -> None:
    shape_meta = {
        "obs": {"x": {"shape": [2], "horizon": 2, "type": "low_dim"}},
        "action": {"shape": [2], "horizon": 2},
    }
    path = tmp_path / "tiny.pt2"
    export_policy_pt2(_TinyInferencePolicy(), shape_meta, path, batch_size=1)
    assert path.exists()

    extra = {"hommi_metadata.json": ""}
    exported = torch.export.load(path, extra_files=extra)
    x = torch.ones(1, 2, 2)
    output = exported.module()(x)
    torch.testing.assert_close(output, x[:, -1:] * 2.0)
    assert '"obs_keys": ["x"]' in extra["hommi_metadata.json"]


def test_tensorrt_directory_resolution(tmp_path: Path) -> None:
    from hommi_train.export import default_tensorrt_path, resolve_model_path

    model = tmp_path / "model.pt"
    model.touch()
    assert resolve_model_path(tmp_path) == model.resolve()
    assert default_tensorrt_path(tmp_path) == (tmp_path / "model.trt.ep").resolve()
