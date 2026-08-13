"""Benchmark end-to-end HoMMI TensorRT policy inference throughput.

This measures ``policy.predict_action(obs)`` including the eager DDIM sampling
loop plus the TensorRT ViT/DiT submodules.  It intentionally excludes model load
and TensorRT engine build time, so pass an already compiled ``model.trt.ep``.

Example::

    python examples/benchmark_tensorrt.py \
        -m runs/pick_place/model.trt.ep \
        --warmup 10 --iterations 100
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch

from hommi_train import load_tensorrt_policy, resolve_device


IDENTITY_ROTATION_6D = np.asarray(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end HoMMI TensorRT inference latency and Hz."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="precompiled model.trt.ep",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def make_dummy_observation(
    shape_meta: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    obs: dict[str, torch.Tensor] = {}

    for key, attr in shape_meta["obs"].items():
        if attr.get("ignore_by_policy", False):
            continue

        horizon = int(attr.get("horizon", 1))
        shape = tuple(int(value) for value in attr["shape"])

        if attr.get("type", "low_dim") == "rgb":
            channels, height, width = shape
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


def percentile(values_ms: list[float], q: float) -> float:
    """Linear-interpolated percentile without adding a NumPy dependency here."""
    ordered = sorted(values_ms)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sequence")
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")

    device = resolve_device(args.device)
    policy, artifact = load_tensorrt_policy(args.model, device=device)
    precision = artifact["tensorrt_bundle"]["precision"]
    obs = make_dummy_observation(artifact["shape_meta"], device=device)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )

    print(f"model={args.model}")
    print(f"device={device}")
    print(f"precision={precision}")
    print(f"warmup={args.warmup}, iterations={args.iterations}")

    with torch.inference_mode(), autocast:
        for _ in range(args.warmup):
            policy.predict_action(obs)
        synchronize(device)

        latencies_ms: list[float] = []
        for _ in range(args.iterations):
            synchronize(device)
            start = perf_counter()
            prediction = policy.predict_action(obs)
            synchronize(device)
            latencies_ms.append((perf_counter() - start) * 1000.0)

    avg_ms = mean(latencies_ms)
    p50_ms = percentile(latencies_ms, 0.50)
    p95_ms = percentile(latencies_ms, 0.95)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    hz = 1000.0 / avg_ms

    action_shape = tuple(prediction["action"].shape)
    finite = bool(torch.isfinite(prediction["action"]).all().item())

    print()
    print("HoMMI TensorRT end-to-end benchmark")
    print(f"action shape : {action_shape}")
    print(f"finite output: {finite}")
    print(f"mean latency : {avg_ms:.3f} ms")
    print(f"p50 latency  : {p50_ms:.3f} ms")
    print(f"p95 latency  : {p95_ms:.3f} ms")
    print(f"min latency  : {min_ms:.3f} ms")
    print(f"max latency  : {max_ms:.3f} ms")
    print(f"throughput   : {hz:.2f} Hz")
    print()
    print(
        "Note: this is model inference throughput on synthetic, already-GPU-resident "
        "observations. Camera capture, decoding, preprocessing outside the policy, "
        "robot I/O, and control-loop work are not included."
    )


if __name__ == "__main__":
    main()
