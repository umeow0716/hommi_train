from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from ..config import hommi_train_config_from_mapping
from ..runtime import resolve_device, resolve_precision
from .artifact import load_portable_policy
from .torch_export import example_observation_inputs


TensorRTPrecision = Literal["auto", "fp32", "bf16"]
_BUNDLE_FORMAT = "hommi-train.torch-tensorrt-bundle"
_BUNDLE_VERSION = 1


def resolve_model_path(input_dir: str | Path) -> Path:
    """Resolve ``<input_dir>/model.pt`` and fail with a useful error."""
    directory = Path(input_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"TensorRT input must be a run directory: {directory}")
    model_path = directory / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"portable model not found: {model_path}")
    return model_path


def default_tensorrt_path(input_dir: str | Path) -> Path:
    """Default single-file TensorRT bundle path next to ``model.pt``."""
    return Path(input_dir).expanduser().resolve() / "model.trt.ep"


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def _clone_capture_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_clone_capture_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_capture_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_capture_value(item) for key, item in value.items()}
    return value


def _cast_floating_tensors(value: Any, dtype: torch.dtype) -> Any:
    """Recursively cast only floating-point tensors, preserving integer inputs."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(_cast_floating_tensors(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_floating_tensors(item, dtype) for item in value]
    if isinstance(value, dict):
        return {
            key: _cast_floating_tensors(item, dtype)
            for key, item in value.items()
        }
    return value


class _TensorRTBF16Adapter(nn.Module):
    """Keep the policy's public FP32 boundary around an explicitly-BF16 TRT module.

    Torch-TensorRT's Dynamo path uses strong typing.  We therefore compile BF16
    artifacts with BF16 weights/inputs instead of its rule-based autocast pass
    (which currently mishandles list-valued FX metadata from ops such as
    ``unbind``/``chunk``).  At runtime this tiny adapter casts floating inputs to
    BF16 and floating outputs back to FP32 so the surrounding HoMMI policy and
    DDIM scheduler retain their existing eager-PyTorch dtype behavior.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        bf16_args = _cast_floating_tensors(args, torch.bfloat16)
        bf16_kwargs = _cast_floating_tensors(kwargs, torch.bfloat16)
        output = self.module(*bf16_args, **bf16_kwargs)
        return _cast_floating_tensors(output, torch.float32)


class _CastFloatingOutput(nn.Module):
    """Cast floating outputs from a wrapped module while preserving integers."""

    def __init__(self, module: nn.Module, dtype: torch.dtype) -> None:
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return _cast_floating_tensors(self.module(*args, **kwargs), self.dtype)


def _prepare_explicit_bf16_module(module: nn.Module) -> nn.Module:
    """Make HoMMI's explicit-BF16 graph strongly typed for TensorRT.

    ``SinusoidalPosEmb`` intentionally creates FP32 frequencies.  Under normal
    eager training, autocast converts the following Linear as needed.  The
    TensorRT Dynamo path is strongly typed, so after converting the denoiser
    weights to BF16 we must make the FP32 -> BF16 edge explicit *before* the
    first timestep Linear.  Casting only after the whole timestep MLP is too
    late and produces a Float x BFloat16 matrix multiply during conversion.
    """
    module = module.to(dtype=torch.bfloat16)

    timestep_embedding = getattr(module, "timestep_embedding", None)
    if isinstance(timestep_embedding, nn.Sequential) and len(timestep_embedding) >= 2:
        first = timestep_embedding[0]
        if not isinstance(first, _CastFloatingOutput):
            timestep_embedding[0] = _CastFloatingOutput(first, torch.bfloat16)

    return module


def _capture_module_inputs(
    policy: nn.Module,
    shape_meta: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    compile_backbone: bool,
    compile_denoiser: bool,
) -> tuple[
    dict[str, tuple[tuple[Any, ...], dict[str, Any]]],
    dict[str, list[str]],
]:
    """Run one eager diffusion inference and capture pure-tensor submodule calls.

    Full ``predict_action`` cannot be exported reliably because the diffusers
    DDIM scheduler contains Python/data-dependent control flow.  The expensive
    neural-network pieces are pure tensor graphs, though, so capture the exact
    concrete calls that normal inference makes and compile those independently.
    """
    targets: dict[str, nn.Module] = {}
    aliases: dict[str, list[str]] = {}

    if compile_denoiser:
        model = getattr(policy, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError("policy does not expose an nn.Module 'model' denoiser")
        targets["denoiser"] = model

    if compile_backbone:
        encoder = getattr(policy, "obs_encoder", None)
        key_model_map = getattr(encoder, "key_model_map", None)
        if key_model_map is None:
            raise TypeError("policy observation encoder has no key_model_map")

        # HoMMI can share one ViT between multiple RGB keys. Compile each unique
        # module exactly once and remember which observation keys use it.
        unique: dict[int, str] = {}
        for key in list(key_model_map.keys()):
            module = key_model_map[key]
            if not isinstance(module, nn.Module):
                raise TypeError(f"backbone for {key!r} is not an nn.Module")
            ident = id(module)
            if ident in unique:
                aliases[unique[ident]].append(str(key))
                continue
            name = f"backbone_{len(unique)}"
            unique[ident] = name
            targets[name] = module
            aliases[name] = [str(key)]

    captures: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    hooks = []

    for name, target in targets.items():
        def hook(
            _module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            *,
            _name: str = name,
        ) -> None:
            if _name not in captures:
                captures[_name] = (
                    tuple(_clone_capture_value(item) for item in args),
                    {key: _clone_capture_value(item) for key, item in kwargs.items()},
                )

        hooks.append(target.register_forward_pre_hook(hook, with_kwargs=True))

    try:
        keys, example_inputs = example_observation_inputs(
            shape_meta,
            batch_size=batch_size,
            device=device,
        )
        obs = dict(zip(keys, example_inputs, strict=True))
        with torch.inference_mode():
            policy.predict_action(obs)
    finally:
        for handle in hooks:
            handle.remove()

    missing = sorted(set(targets) - set(captures))
    if missing:
        raise RuntimeError(
            "failed to capture inference inputs for TensorRT module(s): "
            + ", ".join(missing)
        )
    return captures, aliases


def _compile_one_module(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output_path: Path,
    *,
    torch_tensorrt: Any,
    min_block_size: int,
    optimization_level: int,
    precision: Literal["fp32", "bf16"],
) -> None:
    """AOT compile one pure-tensor module and serialize embedded TRT engines."""
    compile_module = module.eval()
    compile_args = args
    compile_kwargs_inputs = kwargs

    if precision == "bf16":
        # Dynamo/TensorRT uses strong typing. Compile an explicitly-BF16 graph
        # instead of enabling Torch-TensorRT's graph autocast pass. HoMMI's
        # sinusoidal timestep embedding emits FP32 by design, so make that
        # FP32 -> BF16 edge explicit before its first Linear as well.
        compile_module = _prepare_explicit_bf16_module(compile_module)
        compile_args = _cast_floating_tensors(args, torch.bfloat16)
        compile_kwargs_inputs = _cast_floating_tensors(kwargs, torch.bfloat16)

    exported = torch.export.export(
        compile_module,
        args=compile_args,
        kwargs=compile_kwargs_inputs,
        strict=False,
    )

    # The deployment target can be a memory-constrained Jetson. Offloading the
    # source module after conversion frees one GPU-side model copy while the TRT
    # builder works.
    compile_options: dict[str, Any] = {
        "arg_inputs": compile_args,
        "kwarg_inputs": compile_kwargs_inputs,
        "min_block_size": min_block_size,
        "optimization_level": optimization_level,
        "offload_module_to_cpu": True,
    }

    compiled = torch_tensorrt.dynamo.compile(exported, **compile_options)
    torch_tensorrt.save(
        compiled,
        output_path,
        arg_inputs=compile_args,
        kwarg_inputs=compile_kwargs_inputs,
        output_format="exported_program",
        retrace=False,
    )


def compile_portable_model_tensorrt(
    input_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "cuda",
    precision: TensorRTPrecision = "auto",
    batch_size: int = 1,
) -> dict[str, Path]:
    """Build a single-file TensorRT deployment bundle from ``<run>/model.pt``.

    Only the compute-heavy pure-tensor networks are AOT compiled: the ActionDiT
    denoiser and unique vision backbone(s). The DDIM scheduler and diffusion
    orchestration intentionally remain eager PyTorch because diffusers' scheduler
    uses data-dependent Python branches that ``torch.export`` cannot represent as
    one static graph.

    The ``model.trt.ep`` output is a ZIP-based HoMMI bundle containing the
    portable model plus serialized Torch-TensorRT ExportedPrograms. Load it with
    :func:`load_tensorrt_policy`; the returned policy keeps the normal
    ``predict_action(obs)`` API.

    TensorRT engine bytes are hardware/runtime-specific. Build this artifact on
    the deployment machine when portability matters.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    model_path = resolve_model_path(input_dir)
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_tensorrt_path(input_dir)
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError("TensorRT compilation requires a CUDA device")

    # Torch-TensorRT reads this when creating TensorRT builders. Enable its
    # malloc trimming before importing/using the runtime so transient host-side
    # model copies are returned more aggressively on memory-constrained systems.
    os.environ.setdefault("TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM", "1")

    try:
        import torch_tensorrt
    except Exception as exc:  # pragma: no cover - depends on CUDA/TensorRT host
        raise RuntimeError(
            "Torch-TensorRT could not be imported. Install a PyTorch/CUDA/"
            "TensorRT-compatible torch-tensorrt build on the compilation host."
        ) from exc

    policy, payload = load_portable_policy(model_path, device=resolved_device)
    config = hommi_train_config_from_mapping(payload["config"])
    resolved_precision = resolve_precision(precision, resolved_device)
    trt_cfg = config.evaluation.tensorrt

    captures, aliases = _capture_module_inputs(
        policy,
        payload["shape_meta"],
        batch_size=batch_size,
        device=resolved_device,
        compile_backbone=trt_cfg.compile_backbone,
        compile_denoiser=trt_cfg.compile_denoiser,
    )

    targets: dict[str, nn.Module] = {}
    if trt_cfg.compile_denoiser:
        targets["denoiser"] = policy.model
    if trt_cfg.compile_backbone:
        key_model_map = policy.obs_encoder.key_model_map
        for name, keys in aliases.items():
            targets[name] = key_model_map[keys[0]]

    with tempfile.TemporaryDirectory(prefix="hommi-trt-build-") as temp_name:
        temp = Path(temp_name)
        shutil.copy2(model_path, temp / "model.pt")

        module_files: dict[str, str] = {}
        with torch.inference_mode():
            for name, module in targets.items():
                args, kwargs = captures[name]
                filename = f"{name}.ep"
                _compile_one_module(
                    module,
                    args,
                    kwargs,
                    temp / filename,
                    torch_tensorrt=torch_tensorrt,
                    min_block_size=trt_cfg.min_block_size,
                    optimization_level=trt_cfg.optimization_level,
                    precision=resolved_precision,
                )
                module_files[name] = filename

        manifest = {
            "format": _BUNDLE_FORMAT,
            "format_version": _BUNDLE_VERSION,
            "source_model": model_path.name,
            "portable_model": "model.pt",
            "batch_size": batch_size,
            "precision": resolved_precision,
            "device": str(resolved_device),
            "modules": module_files,
            "backbone_aliases": aliases,
            "tensorrt": {
                "min_block_size": trt_cfg.min_block_size,
                "optimization_level": trt_cfg.optimization_level,
                "compile_backbone": trt_cfg.compile_backbone,
                "compile_denoiser": trt_cfg.compile_denoiser,
            },
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        tmp_output = output.with_name(output.name + ".tmp")
        if tmp_output.exists():
            tmp_output.unlink()
        with zipfile.ZipFile(
            tmp_output,
            mode="w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            for child in sorted(temp.iterdir()):
                archive.write(child, arcname=child.name)
        tmp_output.replace(output)

    metadata_path = _metadata_path(output)
    metadata_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"tensorrt": output, "metadata": metadata_path}


def load_tensorrt_policy(
    path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a HoMMI ``model.trt.ep`` bundle without rebuilding TRT engines."""
    bundle = Path(path).expanduser().resolve()
    if not bundle.is_file():
        raise FileNotFoundError(bundle)

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise RuntimeError("TensorRT bundle execution requires a CUDA device")

    try:
        import torch_tensorrt
    except Exception as exc:  # pragma: no cover - depends on deployment runtime
        raise RuntimeError(
            "Torch-TensorRT runtime is required to load a TensorRT bundle."
        ) from exc

    tempdir = tempfile.TemporaryDirectory(prefix="hommi-trt-load-")
    temp = Path(tempdir.name)
    try:
        with zipfile.ZipFile(bundle, mode="r") as archive:
            archive.extractall(temp)

        manifest = json.loads((temp / "manifest.json").read_text())
        if manifest.get("format") != _BUNDLE_FORMAT:
            raise ValueError(f"not a {_BUNDLE_FORMAT!r} artifact")
        if int(manifest.get("format_version", 0)) != _BUNDLE_VERSION:
            raise ValueError(
                "unsupported TensorRT bundle format_version="
                f"{manifest.get('format_version')}"
            )

        policy, payload = load_portable_policy(
            temp / manifest["portable_model"],
            device=resolved_device,
        )

        modules = manifest["modules"]
        explicit_bf16 = manifest.get("precision") == "bf16"

        def load_module(filename: str) -> nn.Module:
            loaded = torch_tensorrt.load(temp / filename).module()
            if explicit_bf16:
                return _TensorRTBF16Adapter(loaded)
            return loaded

        if "denoiser" in modules:
            policy.model = load_module(modules["denoiser"])

        key_model_map = getattr(policy.obs_encoder, "key_model_map", None)
        for name, keys in manifest.get("backbone_aliases", {}).items():
            if name not in modules:
                continue
            loaded = load_module(modules[name])
            for key in keys:
                key_model_map[key] = loaded

        # Keep extracted files alive for runtimes that lazily access serialized
        # resources after load. nn.Module permits arbitrary Python attributes.
        policy._hommi_tensorrt_tempdir = tempdir  # type: ignore[attr-defined]
        payload["tensorrt_bundle"] = manifest
        return policy, payload
    except Exception:
        tempdir.cleanup()
        raise
