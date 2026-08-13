from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from hommi_train.cli import build_parser, config_from_args
from hommi_train.config import HommiTrainConfig, hommi_train_config_from_mapping
from hommi_train.runner import _resume_split


def test_cli_defaults_come_from_config_dataclasses() -> None:
    args = build_parser().parse_args(["-i", "dataset.hdf5"])
    assert config_from_args(args) == HommiTrainConfig()


def test_cli_applies_only_explicit_overrides() -> None:
    base = HommiTrainConfig(
        training=replace(HommiTrainConfig().training, batch_size=32, epochs=777),
        model=replace(
            HommiTrainConfig().model,
            depth=12,
            encoder=replace(HommiTrainConfig().model.encoder, pretrained=True),
        ),
        runtime=replace(HommiTrainConfig().runtime, device="cuda:1"),
    )
    args = build_parser().parse_args(
        [
            "-i",
            "dataset.hdf5",
            "--epochs",
            "1200",
            "--no-pretrained",
            "--device",
            "cuda",
        ]
    )
    cfg = config_from_args(args, base=base)
    assert cfg.training.batch_size == 32
    assert cfg.training.epochs == 1200
    assert cfg.model.depth == 12
    assert cfg.model.encoder.pretrained is False
    assert cfg.runtime.device == "cuda"


def test_checkpoint_config_mapping_accepts_03_without_runtime() -> None:
    original = HommiTrainConfig(
        training=replace(HommiTrainConfig().training, betas=(0.9, 0.99)),
        model=replace(HommiTrainConfig().model, depth=10),
    )
    raw = asdict(original)
    raw.pop("runtime")
    raw["future_section"] = {"ignored": True}
    restored = hommi_train_config_from_mapping(raw)
    assert restored.training.betas == (0.9, 0.99)
    assert restored.model.depth == 10
    assert restored.runtime == HommiTrainConfig().runtime


def test_resume_split_uses_checkpoint_episode_keys() -> None:
    cfg = HommiTrainConfig()
    split = _resume_split(
        {
            "train_episode_keys": ("episode_001", "episode_003"),
            "val_episode_keys": ("episode_002",),
        },
        config=cfg,
    )
    assert split.train_keys == ("episode_001", "episode_003")
    assert split.val_keys == ("episode_002",)


def test_cli_boolean_optional_flags() -> None:
    args = build_parser().parse_args(
        [
            "-i",
            "dataset.hdf5",
            "--no-pin-memory",
            "--no-persistent-workers",
            "--no-drop-last",
            "--no-progress",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.training.pin_memory is False
    assert cfg.training.persistent_workers is False
    assert cfg.training.drop_last is False
    assert cfg.runtime.progress is False


def test_checkpoint_config_mapping_accepts_04_without_export() -> None:
    raw = asdict(HommiTrainConfig())
    raw.pop("export")
    restored = hommi_train_config_from_mapping(raw)
    assert restored.export == HommiTrainConfig().export


def test_cli_export_overrides() -> None:
    args = build_parser().parse_args(
        ["-i", "dataset.hdf5", "--no-auto-export", "--export-source", "last", "--artifact-name", "ema.pt"]
    )
    cfg = config_from_args(args)
    assert cfg.export.auto_export is False
    assert cfg.export.source == "last"
    assert cfg.export.artifact_name == "ema.pt"


def test_tensorrt_cli_accepts_run_directory() -> None:
    from hommi_train.cli import build_tensorrt_parser

    args = build_tensorrt_parser().parse_args(
        ["-i", "runs/pick_place", "--precision", "bf16", "--batch-size", "1"]
    )
    assert args.input == Path("runs/pick_place")
    assert args.precision == "bf16"
    assert args.batch_size == 1
