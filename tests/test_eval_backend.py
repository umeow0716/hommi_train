from __future__ import annotations

import torch

from hommi_train.evaluation.backend import resolve_evaluation_backend


def test_evaluation_backend_auto_is_eager_on_cpu() -> None:
    assert resolve_evaluation_backend("auto", torch.device("cpu")) == "eager"


def test_tensorrt_requires_cuda() -> None:
    try:
        resolve_evaluation_backend("tensorrt", torch.device("cpu"))
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("expected TensorRT CPU rejection")


def test_tensorrt_backend_compiles_denoiser_and_backbone(monkeypatch) -> None:
    from torch import nn

    from hommi_train.config import TensorRTConfig
    from hommi_train.evaluation import backend as backend_module

    class FakeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.key_model_map = nn.ModuleDict({"rgb": nn.Linear(4, 4)})

    class FakePolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Linear(4, 4)
            self.obs_encoder = FakeEncoder()

    policy = FakePolicy()
    original_model = policy.model
    original_backbone = policy.obs_encoder.key_model_map["rgb"]
    compiled: list[nn.Module] = []

    def fake_compile(
        module: nn.Module, config: TensorRTConfig, *, precision: str
    ) -> nn.Module:
        del config
        assert precision == "bf16"
        compiled.append(module)
        return nn.Sequential(module)

    monkeypatch.setattr(backend_module, "tensorrt_available", lambda: True)
    monkeypatch.setattr(backend_module, "_compile_tensorrt_module", fake_compile)

    resolved = backend_module.configure_evaluation_backend(
        policy,
        backend="tensorrt",
        device=torch.device("cuda"),
        compile_mode="reduce-overhead",
        tensorrt=TensorRTConfig(),
        precision="bf16",
    )

    assert resolved == "tensorrt"
    assert compiled == [original_model, original_backbone]
    assert isinstance(policy.model, nn.Sequential)
    assert isinstance(policy.obs_encoder.key_model_map["rgb"], nn.Sequential)


def test_tensorrt_available_rejects_broken_runtime_import(monkeypatch) -> None:
    from hommi_train.evaluation import backend as backend_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def broken_import(name: str):
        assert name == "torch_tensorrt"
        raise OSError("missing TensorRT shared library")

    monkeypatch.setattr(backend_module.importlib, "import_module", broken_import)
    assert backend_module.tensorrt_available() is False
