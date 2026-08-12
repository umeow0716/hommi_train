from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
import torch

from .video import HDF5VideoDecoderCache, preprocess_rgb_uint8


FrameCacheMode = Literal["none", "lru", "ram"]


@dataclass(slots=True)
class _RamVideoFrames:
    indices: np.ndarray  # sorted int64 [K]
    frames: torch.Tensor  # CPU uint8 [K,3,H,W]


class HDF5VideoFrameCache:
    """Decoded/preprocessed frame cache layered on ``HDF5VideoDecoderCache``.

    Modes:
      ``none``: decode and preprocess every request.
      ``lru``:  retain up to ``capacity`` individual CPU uint8 frames per process.
      ``ram``:  preload the referenced working set once as CPU uint8 tensors.

    ``ram`` is intended for long training runs. Source H.264 remains the compact
    on-disk format; only dataset-referenced frames are decoded, black-padded to square,
    resized, and retained in host RAM.
    """

    def __init__(
        self,
        decoder_cache: HDF5VideoDecoderCache,
        *,
        image_size: int,
        mode: FrameCacheMode = "ram",
        capacity: int = 2048,
        preload_batch_size: int = 8,
    ) -> None:
        if mode not in {"none", "lru", "ram"}:
            raise ValueError("frame cache mode must be 'none', 'lru', or 'ram'")
        if capacity < 1:
            raise ValueError("frame cache capacity must be >= 1")
        if preload_batch_size < 1:
            raise ValueError("preload_batch_size must be >= 1")

        self.decoder_cache = decoder_cache
        self.image_size = int(image_size)
        self.mode: FrameCacheMode = mode
        self.capacity = int(capacity)
        self.preload_batch_size = int(preload_batch_size)
        self._pid = os.getpid()
        self._lru: OrderedDict[tuple[str, str, int], torch.Tensor] = OrderedDict()
        self._ram: dict[tuple[str, str], _RamVideoFrames] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # LRU state is intentionally per worker/process. The full RAM working
        # set is immutable after preload and can be inherited by DataLoader workers.
        state["_lru"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state

    def _reset_lru_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        self._lru = OrderedDict()
        self._pid = current_pid

    @property
    def num_cached_frames(self) -> int:
        if self.mode == "ram":
            return sum(int(entry.frames.shape[0]) for entry in self._ram.values())
        if self.mode == "lru":
            return len(self._lru)
        return 0

    @property
    def cache_bytes(self) -> int:
        if self.mode == "ram":
            return sum(entry.frames.numel() * entry.frames.element_size() for entry in self._ram.values())
        if self.mode == "lru":
            return sum(frame.numel() * frame.element_size() for frame in self._lru.values())
        return 0

    def _decode_preprocessed(
        self,
        episode_key: str,
        side: str,
        indices: np.ndarray,
    ) -> torch.Tensor:
        decoded = self.decoder_cache.get_frames(episode_key, side, indices)
        prepared = preprocess_rgb_uint8(decoded, image_size=self.image_size)
        # RAM/LRU caches are explicitly host-memory caches even if the decoder
        # itself is running on CUDA for single-process profiling/eval.
        return prepared.to(device="cpu", non_blocking=False).contiguous()

    def preload(
        self,
        references: Mapping[tuple[str, str], np.ndarray],
        *,
        share_memory: bool = False,
    ) -> None:
        """Populate ``ram`` mode with only the frame indices the dataset uses."""
        if self.mode != "ram":
            raise RuntimeError("preload() is only valid with frame cache mode='ram'")
        if self._ram:
            raise RuntimeError("RAM frame cache has already been preloaded")

        for key, raw_indices in references.items():
            episode_key, side = key
            indices = np.unique(np.asarray(raw_indices, dtype=np.int64).reshape(-1))
            if indices.size == 0:
                continue
            if np.any(indices < 0):
                raise ValueError(f"negative video frame index in {episode_key}/{side}")

            chunks: list[torch.Tensor] = []
            for start in range(0, indices.size, self.preload_batch_size):
                chunk_indices = indices[start : start + self.preload_batch_size]
                chunks.append(self._decode_preprocessed(episode_key, side, chunk_indices))
            frames = torch.cat(chunks, dim=0)
            if share_memory:
                frames.share_memory_()
            self._ram[key] = _RamVideoFrames(indices=indices, frames=frames)

    def _get_ram(
        self,
        episode_key: str,
        side: str,
        indices: np.ndarray,
    ) -> torch.Tensor:
        key = (episode_key, side)
        entry = self._ram.get(key)
        if entry is None:
            raise KeyError(f"RAM frame cache has no preloaded entry for {episode_key}/{side}")

        positions = np.searchsorted(entry.indices, indices)
        valid = positions < entry.indices.size
        if not np.all(valid) or not np.array_equal(entry.indices[positions], indices):
            missing = indices[~valid] if not np.all(valid) else indices[entry.indices[positions] != indices]
            raise KeyError(
                f"RAM frame cache missing referenced frame(s) for {episode_key}/{side}: "
                f"{missing[:8].tolist()}"
            )
        gather = torch.as_tensor(positions, dtype=torch.long)
        return entry.frames.index_select(0, gather)

    def _get_lru(
        self,
        episode_key: str,
        side: str,
        indices: np.ndarray,
    ) -> torch.Tensor:
        self._reset_lru_after_fork()
        unique, inverse = np.unique(indices, return_inverse=True)
        resolved: dict[int, torch.Tensor] = {}
        missing: list[int] = []

        for frame_idx in unique.tolist():
            key = (episode_key, side, int(frame_idx))
            frame = self._lru.pop(key, None)
            if frame is None:
                missing.append(int(frame_idx))
            else:
                self._lru[key] = frame
                resolved[int(frame_idx)] = frame

        if missing:
            batch = self._decode_preprocessed(
                episode_key,
                side,
                np.asarray(missing, dtype=np.int64),
            )
            for frame_idx, frame in zip(missing, batch, strict=True):
                frame = frame.contiguous()
                key = (episode_key, side, frame_idx)
                self._lru[key] = frame
                resolved[frame_idx] = frame

            while len(self._lru) > self.capacity:
                self._lru.popitem(last=False)

        unique_batch = torch.stack([resolved[int(frame_idx)] for frame_idx in unique], dim=0)
        return unique_batch.index_select(0, torch.as_tensor(inverse, dtype=torch.long))

    def get_frames(
        self,
        episode_key: str,
        side: str,
        frame_indices: np.ndarray,
    ) -> torch.Tensor:
        indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("frame_indices cannot be empty")
        if np.any(indices < 0):
            raise ValueError("frame_indices cannot be negative")

        if self.mode == "none":
            return self._decode_preprocessed(episode_key, side, indices)
        if self.mode == "lru":
            return self._get_lru(episode_key, side, indices)
        return self._get_ram(episode_key, side, indices)

    def close(self) -> None:
        self._lru.clear()
        self.decoder_cache.close()
