from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from ..config import hommi_train_config_from_mapping
from ..dataset import HommiHDF5Dataset
from ..export import load_portable_policy
from ..runtime import resolve_device, resolve_pin_memory, resolve_precision
from .backend import EvaluationBackend, configure_evaluation_backend
from .evaluator import EvaluationResult, evaluate_policy, save_evaluation_result


def run_evaluation(
    input_path: str | Path,
    model_path: str | Path,
    *,
    mode: Literal["sampled", "full"] | None = None,
    device: str = "auto",
    precision: Literal["auto", "fp32", "bf16"] | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    frame_cache: Literal["none", "lru", "ram"] | None = None,
    seed: int | None = None,
    backend: EvaluationBackend | None = None,
    compile: bool | None = None,
    compile_mode: str | None = None,
    output_path: str | Path | None = None,
) -> EvaluationResult:
    """Load a portable model and evaluate its saved validation split."""
    resolved_device = resolve_device(device)
    policy, payload = load_portable_policy(model_path, device=resolved_device)
    config = hommi_train_config_from_mapping(payload["config"])
    eval_cfg = config.evaluation
    resolved_precision = resolve_precision(
        precision or eval_cfg.precision,
        resolved_device,
    )

    selected_keys = tuple(str(key) for key in payload.get("val_episode_keys", ()))
    if not selected_keys:
        raise ValueError(
            "portable model does not contain validation episode keys; "
            "cannot reproduce the training validation split"
        )

    dataset_cfg = config.dataset
    runtime_cfg = config.runtime
    training_cfg = config.training
    dataset = HommiHDF5Dataset(
        input_path,
        episode_keys=selected_keys,
        obs_horizon=dataset_cfg.obs_horizon,
        action_horizon=dataset_cfg.action_horizon,
        image_size=dataset_cfg.image_size,
        action_padding=dataset_cfg.action_padding,
        video_device=runtime_cfg.video_device,
        decoder_cache_size=runtime_cfg.decoder_cache_size,
        video_seek_mode=runtime_cfg.video_seek_mode,
        video_num_threads=runtime_cfg.video_num_threads,
        frame_cache=frame_cache or dataset_cfg.frame_cache,
        frame_cache_size=dataset_cfg.frame_cache_size,
        frame_preload_batch_size=dataset_cfg.frame_preload_batch_size,
    )
    try:
        if dataset.shape_meta != payload["shape_meta"]:
            raise ValueError("evaluation dataset shape_meta differs from the portable model")
        workers = training_cfg.num_workers if num_workers is None else int(num_workers)
        if workers < 0:
            raise ValueError("num_workers must be >= 0")
        loader = DataLoader(
            dataset,
            batch_size=training_cfg.batch_size if batch_size is None else int(batch_size),
            shuffle=False,
            num_workers=workers,
            pin_memory=resolve_pin_memory(training_cfg.pin_memory, resolved_device),
            persistent_workers=bool(training_cfg.persistent_workers and workers > 0),
            drop_last=False,
        )
        if len(loader) == 0:
            raise ValueError("evaluation DataLoader is empty")

        selected_backend = backend or ("inductor" if compile else eval_cfg.backend)
        resolved_backend = configure_evaluation_backend(
            policy,
            backend=selected_backend,
            device=resolved_device,
            compile_mode=compile_mode or eval_cfg.compile_mode,
            tensorrt=eval_cfg.tensorrt,
            precision=resolved_precision,
        )
        result = evaluate_policy(
            policy,
            loader,
            device=resolved_device,
            mode=mode or eval_cfg.mode,
            precision=resolved_precision,
            seed=eval_cfg.seed if seed is None else int(seed),
            backend=resolved_backend,
        )
        if output_path is not None:
            save_evaluation_result(result, output_path)
        return result
    finally:
        dataset.close()
