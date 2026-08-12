from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import DiTTrainConfig
from .checkpoint import (
    load_training_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from .data import move_batch
from .ema import HommiEMAModel
from .metrics import action_mse, gradient_norm
from .optimizer import build_lr_scheduler, build_optimizer


@dataclass(slots=True)
class TrainerState:
    """Mutable progress state stored in every resumable checkpoint."""

    # Next epoch index to run. A value of 7 means epochs [0, 7) are complete.
    epoch: int = 0
    global_step: int = 0
    best_val_action_mse: float = math.inf
    best_checkpoints: list[tuple[float, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainerState":
        raw_best = value.get("best_checkpoints", [])
        return cls(
            epoch=int(value.get("epoch", 0)),
            global_step=int(value.get("global_step", 0)),
            best_val_action_mse=float(value.get("best_val_action_mse", math.inf)),
            best_checkpoints=[
                (float(score), str(filename)) for score, filename in raw_best
            ],
        )


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch before constructing a policy for reproducibility."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but torch.cuda.is_available() is false")
    return resolved


class Trainer:
    """HoMMI-aligned DiT training loop without CLI or dataset-construction logic.

    The trainer owns mutable optimization state only. Dataset splitting,
    normalizer fitting, and policy construction remain separate composition
    layers so future CLI/config frontends do not become coupled to this loop.
    """

    def __init__(
        self,
        *,
        policy: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DiTTrainConfig,
        output_dir: str | Path,
        device: str | torch.device | None = None,
        run_config: Any = None,
        optimizer: Optimizer | None = None,
        lr_scheduler: LRScheduler | None = None,
        ema: HommiEMAModel | None = None,
        progress: bool = True,
    ) -> None:
        self.policy = policy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.device = resolve_training_device(device)
        self.run_config = run_config if run_config is not None else config
        self.progress = bool(progress)
        self.state = TrainerState()
        self.last_metrics: dict[str, float] = {}

        self._validate_config()
        if len(self.train_loader) < 1:
            raise ValueError("train_loader must contain at least one batch")
        if len(self.val_loader) < 1:
            raise ValueError("val_loader must contain at least one batch")

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        self.policy.to(self.device)
        self.optimizer = optimizer or build_optimizer(self.policy, config)
        total_steps = len(self.train_loader) * config.epochs
        self.lr_scheduler = lr_scheduler or build_lr_scheduler(
            self.optimizer,
            config,
            num_training_steps=total_steps,
        )
        self.ema = ema or HommiEMAModel(
            self.policy,
            inv_gamma=1.0,
            max_value=0.9999,
            min_value=0.0,
            power=0.75,
            update_after_step=0,
        )
        self.ema.averaged_model.to(self.device)
        self.ema.averaged_model.eval()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if cfg.clip_grad_norm < 0:
            raise ValueError("clip_grad_norm must be >= 0")
        if cfg.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if cfg.log_grad_norm_every == 0 or cfg.log_grad_norm_every < -1:
            raise ValueError("log_grad_norm_every must be -1 or >= 1")
        if cfg.keep_best_k < 0:
            raise ValueError("keep_best_k must be >= 0")
        if cfg.precision not in {"fp32", "bf16"}:
            raise ValueError(f"unsupported precision: {cfg.precision!r}")
        if (
            cfg.precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                "precision='bf16' requested but this CUDA device does not support BF16"
            )

    def _autocast(self):
        if self.config.precision == "fp32":
            return nullcontext()
        if self.device.type not in {"cpu", "cuda"}:
            raise RuntimeError(
                f"BF16 autocast is not configured for device type {self.device.type!r}"
            )
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
        )

    @property
    def train_episode_keys(self) -> tuple[str, ...]:
        return tuple(getattr(self.train_loader.dataset, "episode_keys", ()))

    @property
    def val_episode_keys(self) -> tuple[str, ...]:
        return tuple(getattr(self.val_loader.dataset, "episode_keys", ()))

    @property
    def shape_meta(self) -> Mapping[str, Any] | None:
        value = getattr(self.train_loader.dataset, "shape_meta", None)
        if value is not None:
            return value
        return getattr(self.policy, "shape_meta", None)

    def resume(self, path: str | Path, *, restore_rng: bool = True) -> None:
        checkpoint = load_training_checkpoint(path, map_location="cpu")
        raw_state = restore_training_checkpoint(
            checkpoint,
            policy=self.policy,
            ema=self.ema,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            restore_rng=restore_rng,
        )
        self.state = TrainerState.from_dict(raw_state)
        self.last_metrics = {
            key: float(value) for key, value in checkpoint.get("metrics", {}).items()
        }
        if self.state.epoch > self.config.epochs:
            raise ValueError(
                f"checkpoint next epoch {self.state.epoch} exceeds configured "
                f"epochs={self.config.epochs}"
            )

    def _checkpoint_payload_extra(self) -> dict[str, Any]:
        return {
            "trainer": "hommi_train.training.Trainer",
            "precision": self.config.precision,
        }

    def _save_checkpoint(self, path: Path, metrics: Mapping[str, float]) -> None:
        save_training_checkpoint(
            path,
            policy=self.policy,
            ema=self.ema,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            trainer_state=asdict(self.state),
            metrics=metrics,
            config=self.run_config,
            shape_meta=self.shape_meta,
            train_episode_keys=self.train_episode_keys,
            val_episode_keys=self.val_episode_keys,
            extra=self._checkpoint_payload_extra(),
        )

    def _evaluate_sampled(
        self,
        train_batch: dict[str, Any],
    ) -> dict[str, float]:
        """Match HoMMI workspace behavior: one train batch and one val batch."""
        eval_policy = self.ema.averaged_model
        eval_policy.eval()
        with torch.inference_mode():
            with self._autocast():
                train_prediction = eval_policy.predict_action(train_batch["obs"])[
                    "action_pred"
                ]
            train_error = action_mse(train_prediction, train_batch["action"])

            val_batch = move_batch(next(iter(self.val_loader)), self.device)
            with self._autocast():
                val_prediction = eval_policy.predict_action(val_batch["obs"])[
                    "action_pred"
                ]
            val_error = action_mse(val_prediction, val_batch["action"])

        return {
            "train_action_mse_error": train_error,
            "val_action_mse_error": val_error,
        }

    def _update_best_checkpoints(
        self,
        *,
        completed_epoch: int,
        metrics: Mapping[str, float],
    ) -> None:
        if "val_action_mse_error" not in metrics:
            return

        score = float(metrics["val_action_mse_error"])
        metric_path = self.checkpoint_dir / (
            f"epoch={completed_epoch:04d}-val_action_mse_error={score:.6f}.pt"
        )
        is_new_best = score < self.state.best_val_action_mse
        if is_new_best:
            self.state.best_val_action_mse = score

        if self.config.keep_best_k > 0:
            self.state.best_checkpoints.append((score, metric_path.name))
            self.state.best_checkpoints.sort(key=lambda item: item[0])
            while len(self.state.best_checkpoints) > self.config.keep_best_k:
                _, filename = self.state.best_checkpoints.pop()
                (self.checkpoint_dir / filename).unlink(missing_ok=True)

            retained_names = {filename for _, filename in self.state.best_checkpoints}
            if metric_path.name in retained_names:
                # Save after updating all score bookkeeping so this snapshot can
                # itself be used as a consistent resume point.
                self._save_checkpoint(metric_path, metrics)

        if is_new_best:
            self._save_checkpoint(self.checkpoint_dir / "best.pt", metrics)

    def fit(self, *, resume_from: str | Path | None = None) -> TrainerState:
        """Train until ``config.epochs`` and return the final mutable state."""
        if resume_from is not None:
            self.resume(resume_from)
        elif self.state.epoch == 0 and self.state.global_step == 0:
            # This seeds the training RNG streams. For deterministic model
            # initialization, callers should also call seed_everything() before
            # build_dit_policy(); the 0.4 CLI will do that automatically.
            seed_everything(self.config.seed)

        for epoch in range(self.state.epoch, self.config.epochs):
            self.policy.train()
            train_losses: list[float] = []
            train_sampling_batch: dict[str, Any] | None = None

            iterator = tqdm(
                self.train_loader,
                desc=f"train {epoch + 1:04d}/{self.config.epochs:04d}",
                mininterval=1.0,
                disable=not self.progress,
            )
            for batch in iterator:
                batch = move_batch(batch, self.device)
                train_sampling_batch = batch
                self.optimizer.zero_grad(set_to_none=True)

                with self._autocast():
                    loss = self.policy.compute_loss(batch)
                if loss.ndim != 0:
                    raise ValueError(
                        f"policy.compute_loss() must return a scalar, got {tuple(loss.shape)}"
                    )
                if not torch.isfinite(loss.detach()):
                    raise FloatingPointError(
                        f"non-finite training loss at global_step={self.state.global_step}: "
                        f"{float(loss.detach().float().cpu())}"
                    )
                loss.backward()

                grad = None
                if (
                    self.config.log_grad_norm_every != -1
                    and self.state.global_step % self.config.log_grad_norm_every == 0
                ):
                    grad = gradient_norm(self.policy)

                if self.config.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.clip_grad_norm,
                    )

                self.optimizer.step()
                self.lr_scheduler.step()
                self.ema.step(self.policy)
                self.state.global_step += 1

                value = float(loss.detach().float().cpu())
                train_losses.append(value)
                if self.progress:
                    postfix: dict[str, str] = {
                        "loss": f"{value:.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        "ema": f"{self.ema.decay:.6f}",
                    }
                    if grad is not None:
                        postfix["grad"] = f"{grad:.2f}"
                    iterator.set_postfix(**postfix)

            if not train_losses or train_sampling_batch is None:
                raise RuntimeError("training epoch produced no batches")

            metrics: dict[str, float] = {
                "train_loss": float(np.mean(train_losses)),
            }
            if epoch % self.config.sample_every == 0:
                metrics.update(self._evaluate_sampled(train_sampling_batch))

            # State epoch always means the next epoch to execute on resume.
            self.state.epoch = epoch + 1
            self.last_metrics = metrics
            self._update_best_checkpoints(
                completed_epoch=self.state.epoch,
                metrics=metrics,
            )
            self._save_checkpoint(self.checkpoint_dir / "last.pt", metrics)

            if self.progress:
                summary = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                print(f"epoch={self.state.epoch:04d} {summary}")

        return self.state
