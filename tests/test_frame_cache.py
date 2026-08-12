from __future__ import annotations

import numpy as np
import torch

from hommi_train.dataset.frame_cache import HDF5VideoFrameCache
from hommi_train.dataset.video import preprocess_rgb_uint8


class _FakeDecoderCache:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[int, ...]]] = []

    def get_frames(self, episode_key: str, side: str, frame_indices: np.ndarray) -> torch.Tensor:
        indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        self.calls.append((episode_key, side, tuple(int(x) for x in indices)))
        frames = torch.empty((indices.size, 3, 10, 12), dtype=torch.uint8)
        for row, frame_idx in enumerate(indices):
            frames[row].fill_(int(frame_idx) % 256)
        return frames

    def close(self) -> None:
        pass


def test_preprocess_rgb_uint8_keeps_compact_dtype() -> None:
    frames = torch.arange(2 * 3 * 10 * 12, dtype=torch.int32).reshape(2, 3, 10, 12)
    frames = (frames % 256).to(torch.uint8)
    output = preprocess_rgb_uint8(frames, image_size=8)
    assert output.shape == (2, 3, 8, 8)
    assert output.dtype == torch.uint8
    assert output.is_contiguous()


def test_ram_cache_preloads_only_referenced_unique_frames() -> None:
    decoder = _FakeDecoderCache()
    cache = HDF5VideoFrameCache(
        decoder, image_size=8, mode="ram", preload_batch_size=2
    )
    cache.preload({("episode_001", "left"): np.array([3, 1, 3, 5], dtype=np.int64)})

    # Unique referenced frames are 1, 3, 5. With batch size 2 this is two decode calls.
    assert decoder.calls == [
        ("episode_001", "left", (1, 3)),
        ("episode_001", "left", (5,)),
    ]
    assert cache.num_cached_frames == 3
    assert cache.cache_bytes == 3 * 3 * 8 * 8

    calls_before = len(decoder.calls)
    output = cache.get_frames(
        "episode_001", "left", np.array([5, 1, 5], dtype=np.int64)
    )
    assert len(decoder.calls) == calls_before
    assert output.shape == (3, 3, 8, 8)
    assert output.dtype == torch.uint8
    assert int(output[0, 0, 0, 0]) == 5
    assert int(output[1, 0, 0, 0]) == 1


def test_lru_cache_reuses_hits() -> None:
    decoder = _FakeDecoderCache()
    cache = HDF5VideoFrameCache(decoder, image_size=8, mode="lru", capacity=2)

    cache.get_frames("episode_001", "left", np.array([1, 2], dtype=np.int64))
    assert len(decoder.calls) == 1
    cache.get_frames("episode_001", "left", np.array([2], dtype=np.int64))
    assert len(decoder.calls) == 1
    cache.get_frames("episode_001", "left", np.array([3], dtype=np.int64))
    assert len(decoder.calls) == 2
    assert cache.num_cached_frames == 2
