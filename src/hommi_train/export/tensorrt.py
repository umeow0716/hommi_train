from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch

from ..config import hommi_train_config_from_mapping
from ..runtime import resolve_device, resolve_precision
from .artifact import load_portable_policy
from .torch_export import PolicyInferenceModule, example_observation_inputs


TensorRTPrecision = Literal["auto", "fp32", "bf16"]


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
    return Path(input_dir).expanduser().resolve() / "model.trt.ep"


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def compile_portable_model_tensorrt(
    input_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "cuda",
    precision: TensorRTPrecision = "auto",
    batch_size: int = 1,
) -> dict[str, Path]:
    """Compile ``<run>/model.pt`` into a serialized Torch-TensorRT program.

    The artifact uses Torch-TensorRT's Dynamo AOT frontend. The policy facade has
    a tensor-only positional input contract while preserving the fixed HoMMI
    diffusion inference path inside the exported program. Unsupported graph
    regions may remain as PyTorch regions; TensorRT-compatible regions are
    lowered into embedded TensorRT engines by Torch-TensorRT.

    The resulting engine code is hardware/runtime-specific. Build this artifact
    on the deployment machine (for example, the Jetson Orin Nano) rather than on
    an unrelated desktop GPU when portability matters.
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

    keys, example_inputs = example_observation_inputs(
        payload["shape_meta"],
        batch_size=batch_size,
        device=resolved_device,
    )
    module = PolicyInferenceModule(policy, keys).eval()

    trt_cfg = config.evaluation.tensorrt
    compile_kwargs: dict[str, Any] = {
        "ir": "dynamo",
        # PolicyInferenceModule receives the observation tensor tuple as one
        # positional argument, hence this deliberately nested arg_inputs tuple.
        "arg_inputs": (example_inputs,),
        "min_block_size": trt_cfg.min_block_size,
        "optimization_level": trt_cfg.optimization_level,
    }
    if resolved_precision == "bf16":
        compile_kwargs["enable_autocast"] = True
        compile_kwargs["autocast_low_precision_type"] = torch.bfloat16

    with torch.inference_mode():
        compiled = torch_tensorrt.compile(module, **compile_kwargs)
        torch_tensorrt.save(
            compiled,
            output,
            arg_inputs=(example_inputs,),
            output_format="exported_program",
        )

    metadata = {
        "format": "hommi-train.torch-tensorrt",
        "format_version": 1,
        "source_model": model_path.name,
        "artifact": output.name,
        "obs_keys": list(keys),
        "batch_size": batch_size,
        "precision": resolved_precision,
        "device": str(resolved_device),
        "shape_meta": payload["shape_meta"],
        "tensorrt": {
            "min_block_size": trt_cfg.min_block_size,
            "optimization_level": trt_cfg.optimization_level,
        },
    }
    metadata_path = _metadata_path(output)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"tensorrt": output, "metadata": metadata_path}
