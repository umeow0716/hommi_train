from __future__ import annotations

import sys
import types

import hommi_diffusion_policy

from hommi_train.config import DDIMConfig, DiTModelConfig
from hommi_train.policy import build_ddim_scheduler, build_dit_policy


class _FakeScheduler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = types.SimpleNamespace(num_train_timesteps=kwargs["num_train_timesteps"])


class _FakeEncoder:
    def __init__(self, shape_meta, *, model_name: str, pretrained: bool):
        self.shape_meta = shape_meta
        self.model_name = model_name
        self.pretrained = pretrained


class _FakePolicy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _shape_meta() -> dict:
    return {
        "obs": {
            "camera0_main_rgb": {
                "shape": [3, 224, 224],
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
        },
        "action": {"shape": [10], "horizon": 16, "rotation_rep": "rotation_6d"},
    }


def test_ddim_factory_uses_hommi_defaults(monkeypatch) -> None:
    module = types.ModuleType("diffusers")
    module.DDIMScheduler = _FakeScheduler
    monkeypatch.setitem(sys.modules, "diffusers", module)

    scheduler = build_ddim_scheduler(DDIMConfig())
    assert scheduler.kwargs == {
        "num_train_timesteps": 50,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "set_alpha_to_one": True,
        "steps_offset": 0,
        "prediction_type": "epsilon",
    }


def test_dit_policy_factory_wires_shape_meta_and_defaults(monkeypatch) -> None:
    module = types.ModuleType("diffusers")
    module.DDIMScheduler = _FakeScheduler
    monkeypatch.setitem(sys.modules, "diffusers", module)
    monkeypatch.setattr(hommi_diffusion_policy, "DiTObsEncoderLite", _FakeEncoder)
    monkeypatch.setattr(hommi_diffusion_policy, "DiffusionDiTImagePolicy", _FakePolicy)

    policy = build_dit_policy(_shape_meta(), model_config=DiTModelConfig())
    kwargs = policy.kwargs
    assert kwargs["horizon"] == 16
    assert kwargs["n_obs_steps"] == 2
    assert kwargs["n_action_steps"] == 8
    assert kwargs["num_inference_steps"] == 16
    assert kwargs["attention_embed_dim"] == 768
    assert kwargs["depth"] == 8
    assert kwargs["num_heads"] == 8
    assert kwargs["use_rms_norm"] is True
    assert isinstance(kwargs["noise_scheduler"], _FakeScheduler)
    assert kwargs["obs_encoder"].model_name == "vit_base_patch16_clip_224.openai"


def test_dit_policy_factory_can_disable_pretrained_initialization_for_restore(monkeypatch) -> None:
    module = types.ModuleType("diffusers")
    module.DDIMScheduler = _FakeScheduler
    monkeypatch.setitem(sys.modules, "diffusers", module)
    monkeypatch.setattr(hommi_diffusion_policy, "DiTObsEncoderLite", _FakeEncoder)
    monkeypatch.setattr(hommi_diffusion_policy, "DiffusionDiTImagePolicy", _FakePolicy)

    policy = build_dit_policy(
        _shape_meta(),
        model_config=DiTModelConfig(pretrained=True),
        pretrained_override=False,
    )
    assert policy.kwargs["obs_encoder"].pretrained is False
