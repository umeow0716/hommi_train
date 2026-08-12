from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import HommiHDF5Info


@dataclass(frozen=True, slots=True)
class EpisodeSplit:
    """Deterministic episode-level train/validation split.

    Episodes, rather than individual windows, are split so validation never
    shares trajectories or decoded-frame cache entries with training.
    """

    train_keys: tuple[str, ...]
    val_keys: tuple[str, ...]
    seed: int
    val_ratio: float

    def __post_init__(self) -> None:
        if not self.train_keys:
            raise ValueError("training split cannot be empty")
        if not self.val_keys:
            raise ValueError("validation split cannot be empty")
        overlap = set(self.train_keys).intersection(self.val_keys)
        if overlap:
            raise ValueError(f"train/validation split overlap: {sorted(overlap)}")


def split_episode_keys(
    info: HommiHDF5Info,
    *,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> EpisodeSplit:
    """Split an HDF5 dataset at episode granularity.

    The rounding/clamping matches the previous standalone trainer: at least one
    episode is kept in each split, and validation size is ``round(N*ratio)``.
    Returned keys retain the canonical HDF5 episode order for reproducibility.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must satisfy 0 < val_ratio < 1")

    keys = tuple(episode.key for episode in info.episodes)
    if len(keys) < 2:
        raise ValueError("at least two episodes are required for train/validation split")

    n_val = min(max(1, round(len(keys) * float(val_ratio))), len(keys) - 1)
    rng = np.random.default_rng(int(seed))
    val_indices = set(
        int(index)
        for index in rng.choice(len(keys), size=n_val, replace=False).tolist()
    )
    train_keys = tuple(key for index, key in enumerate(keys) if index not in val_indices)
    val_keys = tuple(key for index, key in enumerate(keys) if index in val_indices)
    return EpisodeSplit(
        train_keys=train_keys,
        val_keys=val_keys,
        seed=int(seed),
        val_ratio=float(val_ratio),
    )
