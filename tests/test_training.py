from __future__ import annotations

import random
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hommi_train.config import DiTTrainConfig, HommiTrainConfig
from hommi_train.training import (
    HommiEMAModel,
    Trainer,
    build_dataloaders,
    build_lr_scheduler,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
)


class _TinyDataset(Dataset):
    def __init__(self, count: int = 8, *, episode_key: str = "episode_001") -> None:
        self.count = count
        self.episode_keys = (episode_key,)
        self.shape_meta = {
            "obs": {"x": {"shape": [2], "horizon": 2, "type": "low_dim"}},
            "action": {"shape": [2], "horizon": 2},
        }

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        x = torch.tensor(
            [[index / 10.0, 1.0], [(index + 1) / 10.0, 1.0]],
            dtype=torch.float32,
        )
        return {
            "obs": {"x": x},
            "action": 0.5 * x,
            "metadata": {"index": index},
        }


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def get_optimizer(self, lr: float, **kwargs):
        kwargs.pop("obs_encoder_lr", None)
        kwargs.pop("obs_encoder_weight_decay", None)
        weight_decay = kwargs.pop("weight_decay", 0.0)
        betas = kwargs.pop("betas", (0.9, 0.999))
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            **kwargs,
        )

    def compute_loss(self, batch):
        prediction = self.linear(batch["obs"]["x"])
        return torch.nn.functional.mse_loss(prediction, batch["action"])

    def predict_action(self, obs):
        return {"action_pred": self.linear(obs["x"])}


def _trainer(
    tmp_path: Path,
    *,
    epochs: int,
    policy: _TinyPolicy | None = None,
) -> Trainer:
    train_ds = _TinyDataset(8, episode_key="episode_train")
    val_ds = _TinyDataset(8, episode_key="episode_val")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    config = DiTTrainConfig(
        batch_size=4,
        num_workers=0,
        epochs=epochs,
        lr=1e-2,
        obs_encoder_lr=1e-2,
        warmup_steps=0,
        sample_every=1,
        log_grad_norm_every=-1,
        precision="fp32",
        persistent_workers=False,
        keep_best_k=1,
    )
    model = policy or _TinyPolicy()
    optimizer = model.get_optimizer(
        lr=config.lr,
        weight_decay=config.weight_decay,
        obs_encoder_lr=config.obs_encoder_lr,
        obs_encoder_weight_decay=config.obs_encoder_weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return Trainer(
        policy=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        run_config=HommiTrainConfig(training=config),
        output_dir=tmp_path,
        device="cpu",
        optimizer=optimizer,
        lr_scheduler=scheduler,
        progress=False,
    )


def test_ema_state_round_trip() -> None:
    model = nn.Linear(2, 2)
    ema = HommiEMAModel(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.step(model)
    state = ema.state_dict()

    restored = HommiEMAModel(nn.Linear(2, 2))
    restored.load_state_dict(state)
    assert restored.optimization_step == ema.optimization_step
    assert restored.decay == ema.decay
    for left, right in zip(
        ema.averaged_model.parameters(),
        restored.averaged_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)


def test_rng_state_round_trip() -> None:
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(()).item())
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(()).item())
    assert actual == expected


def test_dataloader_builder_uses_runtime_config() -> None:
    cfg = DiTTrainConfig(
        batch_size=4,
        num_workers=0,
        precision="fp32",
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
    )
    train, val = build_dataloaders(_TinyDataset(), _TinyDataset(), cfg)
    assert train.batch_size == 4
    assert train.drop_last is True
    assert train.pin_memory is True
    # Must be forced false when num_workers == 0.
    assert train.persistent_workers is False
    assert val.drop_last is True


def test_diffusers_cosine_scheduler_factory(monkeypatch) -> None:
    calls = {}

    def fake_get_scheduler(name, **kwargs):
        calls["name"] = name
        calls.update(kwargs)
        return torch.optim.lr_scheduler.LambdaLR(kwargs["optimizer"], lambda _: 1.0)

    optimization = types.ModuleType("diffusers.optimization")
    optimization.get_scheduler = fake_get_scheduler
    package = types.ModuleType("diffusers")
    package.optimization = optimization
    monkeypatch.setitem(sys.modules, "diffusers", package)
    monkeypatch.setitem(sys.modules, "diffusers.optimization", optimization)

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = DiTTrainConfig(warmup_steps=50)
    scheduler = build_lr_scheduler(optimizer, config, num_training_steps=1000)

    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    assert calls["name"] == "cosine"
    assert calls["num_warmup_steps"] == 50
    assert calls["num_training_steps"] == 1000


def test_trainer_writes_resumable_last_and_best_checkpoints(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, epochs=2)
    state = trainer.fit()

    assert state.epoch == 2
    assert state.global_step == 4
    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "last.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    metric_files = list(checkpoint_dir.glob("epoch=*-val_action_mse_error=*.pt"))
    assert len(metric_files) == 1

    checkpoint = load_training_checkpoint(checkpoint_dir / "last.pt")
    assert checkpoint["epoch"] == 2
    assert checkpoint["global_step"] == 4
    assert checkpoint["train_episode_keys"] == ("episode_train",)
    assert checkpoint["val_episode_keys"] == ("episode_val",)
    assert checkpoint["shape_meta"]["action"]["shape"] == [2]
    assert checkpoint["config"]["training"]["precision"] == "fp32"
    assert "rng_state" in checkpoint
    assert "optimizer" in checkpoint
    assert "lr_scheduler" in checkpoint
    assert "ema" in checkpoint


def test_resume_continues_from_next_epoch_and_restores_optimizer(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first = _trainer(first_dir, epochs=1)
    first_state = first.fit()
    assert first_state.epoch == 1
    assert first_state.global_step == 2

    resume_path = first_dir / "checkpoints" / "last.pt"
    second_dir = tmp_path / "second"
    second = _trainer(second_dir, epochs=2)
    final_state = second.fit(resume_from=resume_path)

    # Exactly one additional epoch (2 batches) is executed.
    assert final_state.epoch == 2
    assert final_state.global_step == 4
    assert second.ema.optimization_step == 4
    assert (second_dir / "checkpoints" / "last.pt").exists()
