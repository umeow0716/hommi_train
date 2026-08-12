from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(slots=True)
class _DecoderEntry:
    h5file: h5py.File
    source: Any
    decoder: Any

    def close(self) -> None:
        try:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()
        finally:
            self.h5file.close()


class HDF5VideoDecoderCache:
    """Per-process LRU cache of TorchCodec decoders backed by HDF5 MP4 bytes.

    This cache owns codec/HDF5 handles only. Decoded image caching is handled by
    :class:`HDF5VideoFrameCache` so decoder lifetime and training working-set
    lifetime remain separate concerns.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        capacity: int = 4,
        seek_mode: Literal["exact", "approximate"] = "exact",
        num_ffmpeg_threads: int = 1,
    ) -> None:
        if capacity < 1:
            raise ValueError("decoder cache capacity must be >= 1")
        if num_ffmpeg_threads < 1:
            raise ValueError("num_ffmpeg_threads must be >= 1")

        self.path = Path(path).expanduser().resolve()
        self.device = torch.device(device)
        self.capacity = int(capacity)
        self.seek_mode = seek_mode
        self.num_ffmpeg_threads = int(num_ffmpeg_threads)
        self._pid = os.getpid()
        self._entries: OrderedDict[tuple[str, str], _DecoderEntry] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_entries"] = OrderedDict()
        state["_pid"] = os.getpid()
        return state

    def _reset_after_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid == self._pid:
            return
        # Never reuse h5py / codec handles inherited from another process.
        self._entries = OrderedDict()
        self._pid = current_pid

    def _create_entry(self, episode_key: str, side: str) -> _DecoderEntry:
        # Lazy imports keep schema / low-dimensional dataset inspection usable
        # without initializing the video stack.
        from hommi_dataset.reader import HDF5VideoFile
        from torchcodec.decoders import VideoDecoder

        h5file = h5py.File(self.path, "r")
        try:
            video_dataset = h5file[f"{episode_key}/video/{side}"]
            source = HDF5VideoFile(video_dataset)
            decoder = VideoDecoder(
                source,
                device=self.device,
                seek_mode=self.seek_mode,
                num_ffmpeg_threads=self.num_ffmpeg_threads,
                dimension_order="NCHW",
            )
        except BaseException:
            h5file.close()
            raise
        return _DecoderEntry(h5file=h5file, source=source, decoder=decoder)

    def _entry(self, episode_key: str, side: str) -> _DecoderEntry:
        self._reset_after_fork()
        key = (episode_key, side)
        entry = self._entries.pop(key, None)
        if entry is None:
            entry = self._create_entry(episode_key, side)
        self._entries[key] = entry

        while len(self._entries) > self.capacity:
            _, stale = self._entries.popitem(last=False)
            stale.close()
        return entry

    def get_frames(
        self,
        episode_key: str,
        side: str,
        frame_indices: np.ndarray,
    ) -> torch.Tensor:
        """Decode requested original-video frame indices as ``uint8 [N,C,H,W]``."""
        indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("frame_indices cannot be empty")
        if np.any(indices < 0):
            raise ValueError("frame_indices cannot be negative")

        # Observation padding and timestamp alignment can repeat a frame. Decode
        # each unique index once, then restore the requested order.
        unique, inverse = np.unique(indices, return_inverse=True)
        entry = self._entry(episode_key, side)
        batch = entry.decoder.get_frames_at(indices=unique.tolist()).data
        inverse_tensor = torch.as_tensor(inverse, dtype=torch.long, device=batch.device)
        return batch.index_select(0, inverse_tensor)

    def close(self) -> None:
        for entry in self._entries.values():
            entry.close()
        self._entries.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def preprocess_rgb_uint8(frames: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """Deterministically prepare decoded RGB for the host-RAM working set.

    The original video is center-square cropped and resized to ``image_size``.
    The returned representation stays ``uint8 [N,3,H,W]`` so a full training
    cache uses one quarter of the RAM of a float32 cache.

    This is intentionally *not* model augmentation. HoMMI's stochastic 95%
    RandomCrop/Resize/ColorJitter (and eval CenterCrop/Resize) remain in
    ``DiTObsEncoderLite`` and are applied after policy-side normalization.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(f"expected [N,3,H,W] video frames, got {tuple(frames.shape)}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    x = frames[..., top : top + side, left : left + side]

    if side != image_size:
        # Interpolation needs floating-point math, but the temporary float tensor
        # is discarded immediately. Only the compact uint8 result is cached.
        x = F.interpolate(
            x.to(dtype=torch.float32),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = x.round_().clamp_(0, 255).to(dtype=torch.uint8)
    elif x.dtype != torch.uint8:
        x = x.round().clamp(0, 255).to(dtype=torch.uint8)

    return x.contiguous()
