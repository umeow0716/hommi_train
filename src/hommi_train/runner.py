from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import HommiTrainConfig
from .dataset import EpisodeSplit, HommiHDF5Dataset, inspect_hommi_hdf5, split_episode_keys
from .normalization import build_hommi_normalizer
from .export import save_portable_checkpoint_model
from .policy import build_dit_policy
from .training import (
    Trainer,
    TrainerState,
    build_dataloaders,
    load_training_checkpoint,
    seed_everything,
)


def _resume_split(
    checkpoint: Mapping[str, Any],
    *,
    config: HommiTrainConfig,
) -> EpisodeSplit:
    train_keys = tuple(str(key) for key in checkpoint.get("train_episode_keys", ()))
    val_keys = tuple(str(key) for key in checkpoint.get("val_episode_keys", ()))
    if not train_keys or not val_keys:
        raise ValueError(
            "resume checkpoint does not contain train/validation episode keys; "
            "cannot guarantee the original split"
        )
    return EpisodeSplit(
        train_keys=train_keys,
        val_keys=val_keys,
        seed=config.training.seed,
        val_ratio=config.dataset.val_ratio,
    )


def _dataset_kwargs(config: HommiTrainConfig) -> dict[str, Any]:
    dataset = config.dataset
    runtime = config.runtime
    return {
        "obs_horizon": dataset.obs_horizon,
        "action_horizon": dataset.action_horizon,
        "image_size": dataset.image_size,
        "action_padding": dataset.action_padding,
        "video_device": runtime.video_device,
        "decoder_cache_size": runtime.decoder_cache_size,
        "video_seek_mode": runtime.video_seek_mode,
        "video_num_threads": runtime.video_num_threads,
        "frame_cache": dataset.frame_cache,
        "frame_cache_size": dataset.frame_cache_size,
        "frame_preload_batch_size": dataset.frame_preload_batch_size,
    }


def _validate_resume_shape(
    checkpoint: Mapping[str, Any],
    shape_meta: Mapping[str, Any],
) -> None:
    saved = checkpoint.get("shape_meta")
    if saved is not None and saved != shape_meta:
        raise ValueError(
            "resume checkpoint shape_meta differs from the dataset/config selected for "
            "this run. Keep obs/action horizons, image size, and arm layout identical."
        )


def run_training(
    input_path: str | Path,
    output_dir: str | Path,
    config: HommiTrainConfig,
    *,
    resume_from: str | Path | None = None,
    resume_checkpoint: Mapping[str, Any] | None = None,
) -> TrainerState:
    """Compose dataset -> normalizer -> policy -> Trainer and execute training.

    This function is intentionally independent of argparse so notebooks, tests,
    and future config frontends can use the exact same training composition.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if (
        config.runtime.video_device == "cuda"
        and config.dataset.frame_cache != "ram"
        and config.training.num_workers > 0
    ):
        raise ValueError(
            "video_device='cuda' with DataLoader workers requires frame_cache='ram'; "
            "otherwise use --num-workers 0 or --video-device cpu"
        )
    resume_path = (
        Path(resume_from).expanduser().resolve() if resume_from is not None else None
    )

    if resume_checkpoint is not None and resume_path is None:
        raise ValueError("resume_checkpoint requires resume_from")
    checkpoint: Mapping[str, Any] | None = resume_checkpoint
    if resume_path is not None and checkpoint is None:
        checkpoint = load_training_checkpoint(resume_path, map_location="cpu")

    last_path = output_dir / "checkpoints" / "last.pt"
    if resume_path is None and last_path.exists():
        raise FileExistsError(
            f"{last_path} already exists; use --resume {last_path} or choose another -o"
        )

    # Seed before policy construction so non-pretrained/random model parameters
    # are reproducible. A resumed checkpoint restores its saved RNG state later.
    seed_everything(config.training.seed)

    info = inspect_hommi_hdf5(input_path)
    split = (
        _resume_split(checkpoint, config=config)
        if checkpoint is not None
        else split_episode_keys(
            info,
            val_ratio=config.dataset.val_ratio,
            seed=config.training.seed,
        )
    )

    common = _dataset_kwargs(config)
    train_dataset: HommiHDF5Dataset | None = None
    val_dataset: HommiHDF5Dataset | None = None
    try:
        # Split before dataset construction so RAM mode only preloads frames from
        # the episodes that belong to each split.
        train_dataset = HommiHDF5Dataset(
            input_path,
            episode_keys=split.train_keys,
            **common,
        )
        val_dataset = HommiHDF5Dataset(
            input_path,
            episode_keys=split.val_keys,
            **common,
        )
        if train_dataset.shape_meta != val_dataset.shape_meta:
            raise RuntimeError("train and validation shape_meta differ")
        if checkpoint is not None:
            _validate_resume_shape(checkpoint, train_dataset.shape_meta)

        normalizer = build_hommi_normalizer(train_dataset)
        policy = build_dit_policy(
            train_dataset.shape_meta,
            model_config=config.model,
            ddim_config=config.ddim,
            # A resume checkpoint already contains every vision-backbone weight.
            # Avoid a redundant timm pretrained-weight initialization/download.
            pretrained_override=False if checkpoint is not None else None,
        )
        policy.set_normalizer(normalizer)

        train_loader, val_loader = build_dataloaders(
            train_dataset,
            val_dataset,
            config.training,
        )
        trainer = Trainer(
            policy=policy,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.training,
            run_config=config,
            output_dir=output_dir,
            device=None if config.runtime.device == "auto" else config.runtime.device,
            progress=config.runtime.progress,
        )
        if checkpoint is not None:
            trainer.restore_checkpoint(checkpoint)
        state = trainer.fit()

        if config.export.auto_export:
            checkpoint_dir = output_dir / "checkpoints"
            preferred = (
                checkpoint_dir / "best.pt"
                if config.export.source == "best"
                else checkpoint_dir / "last.pt"
            )
            source = preferred if preferred.exists() else checkpoint_dir / "last.pt"
            if not source.exists():
                raise FileNotFoundError(
                    f"cannot create portable model because {source} does not exist"
                )
            save_portable_checkpoint_model(
                source,
                output_dir / config.export.artifact_name,
            )
        return state
    finally:
        if train_dataset is not None:
            train_dataset.close()
        if val_dataset is not None:
            val_dataset.close()
