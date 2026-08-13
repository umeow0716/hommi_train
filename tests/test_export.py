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


def test_explicit_bf16_prep_casts_timestep_embedding_before_linear() -> None:
    from hommi_train.export.tensorrt import _prepare_explicit_bf16_module

    class FloatEmbedding(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 4)

    class TinyDenoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.timestep_embedding = nn.Sequential(
                FloatEmbedding(),
                nn.Linear(4, 4),
            )

    module = _prepare_explicit_bf16_module(TinyDenoiser())

    assert module.timestep_embedding[1].weight.dtype == torch.bfloat16
    embedded = module.timestep_embedding[0](torch.tensor([1, 2], dtype=torch.long))
    assert embedded.dtype == torch.bfloat16

    output = module.timestep_embedding(torch.tensor([1, 2], dtype=torch.long))
    assert output.dtype == torch.bfloat16


def test_load_tensorrt_policy_does_not_eval_exported_program_modules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import json
    import sys
    import types
    import zipfile

    import hommi_train.export.tensorrt as trt_export

    class InferenceOnlyModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def train(self, mode: bool = True):
            raise NotImplementedError("Calling train() is not supported yet.")

    class FakeExportedProgram:
        def module(self) -> nn.Module:
            return InferenceOnlyModule()

    fake_torch_tensorrt = types.SimpleNamespace(
        load=lambda _path: FakeExportedProgram(),
    )
    monkeypatch.setitem(sys.modules, "torch_tensorrt", fake_torch_tensorrt)

    class DummyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.key_model_map = nn.ModuleDict({"camera": nn.Identity()})

    class DummyPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Identity()
            self.obs_encoder = DummyEncoder()

    eager_policy = DummyPolicy().eval()
    monkeypatch.setattr(
        trt_export,
        "load_portable_policy",
        lambda *_args, **_kwargs: (eager_policy, {"shape_meta": {}}),
    )

    bundle = tmp_path / "model.trt.ep"
    manifest = {
        "format": trt_export._BUNDLE_FORMAT,
        "format_version": trt_export._BUNDLE_VERSION,
        "portable_model": "model.pt",
        "precision": "fp32",
        "modules": {
            "denoiser": "denoiser.ep",
            "backbone_0": "backbone_0.ep",
        },
        "backbone_aliases": {"backbone_0": ["camera"]},
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("model.pt", b"unused")
        archive.writestr("denoiser.ep", b"unused")
        archive.writestr("backbone_0.ep", b"unused")

    policy, payload = trt_export.load_tensorrt_policy(bundle, device="cuda:0")

    assert policy is eager_policy
    assert not policy.training
    assert isinstance(policy.model, InferenceOnlyModule)
    assert isinstance(policy.obs_encoder.key_model_map["camera"], InferenceOnlyModule)
    assert payload["tensorrt_bundle"]["precision"] == "fp32"
