from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .geometry import pose7_wxyz_to_matrix, relative_pose9
from .schema import HommiHDF5Info, inspect_hommi_hdf5
from .frame_cache import FrameCacheMode, HDF5VideoFrameCache
from .video import HDF5VideoDecoderCache


@dataclass(frozen=True, slots=True)
class _EpisodeData:
    key: str
    action: np.ndarray  # [T, A, 8] = xyz + qwxyz + gripper
    pose_matrix: np.ndarray  # [T, A, 4, 4]
    video_index: np.ndarray  # [T, A]

    @property
    def length(self) -> int:
        return int(self.action.shape[0])


class HommiHDF5Dataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for files produced by ``hommi_dataset>=0.2``.

    On-disk action convention per arm::

        [x, y, z, qw, qx, qy, qz, gripper]

    Model convention per arm::

        [relative_x, relative_y, relative_z, rotation6d(6), gripper]

    For a sample anchored at timestep ``t``, every arm uses its own pose at
    ``t`` as the reference frame. Observation history and the future action
    chunk are both expressed in that frame, matching the UMI / HoMMI training
    convention used by the previous standalone trainer.

    Encoded H.264 stays in HDF5 as the storage format. For long training runs,
    ``frame_cache="ram"`` decodes only dataset-referenced frames once, performs
    deterministic center-square crop + resize, and retains compact uint8 frames
    in host RAM. Train/eval augmentation remains the encoder's job.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        episode_keys: Sequence[str] | None = None,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        image_size: int = 224,
        action_padding: bool = True,
        decode_images: bool = True,
        video_device: str | torch.device = "cpu",
        decoder_cache_size: int = 4,
        video_seek_mode: Literal["exact", "approximate"] = "exact",
        video_num_threads: int = 1,
        frame_cache: FrameCacheMode = "ram",
        frame_cache_size: int = 2048,
        frame_preload_batch_size: int = 8,
        strict_video_attrs: bool = True,
    ) -> None:
        super().__init__()
        if obs_horizon < 1:
            raise ValueError("obs_horizon must be >= 1")
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        if image_size < 1:
            raise ValueError("image_size must be >= 1")

        self.info = inspect_hommi_hdf5(path, strict_video_attrs=strict_video_attrs)
        self.path = self.info.path
        self.obs_horizon = int(obs_horizon)
        self.action_horizon = int(action_horizon)
        self.image_size = int(image_size)
        self.action_padding = bool(action_padding)
        self.decode_images = bool(decode_images)
        self.video_device = torch.device(video_device)

        if self.video_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("video_device='cuda' requested but torch.cuda.is_available() is false")

        available = {episode.key for episode in self.info.episodes}
        if episode_keys is None:
            selected = [episode.key for episode in self.info.episodes]
        else:
            selected = list(episode_keys)
            if not selected:
                raise ValueError("episode_keys cannot be empty")
            unknown = [key for key in selected if key not in available]
            if unknown:
                raise KeyError(f"unknown episode keys: {unknown}")
            if len(set(selected)) != len(selected):
                raise ValueError("episode_keys contains duplicates")

        self.episode_keys = tuple(selected)
        self._episodes = self._load_lowdim(self.episode_keys)
        self._indices: list[tuple[int, int]] = []
        for episode_idx, episode in enumerate(self._episodes):
            if self.action_padding:
                valid_t = range(episode.length)
            else:
                last_t = episode.length - self.action_horizon
                valid_t = range(max(0, last_t + 1))
            self._indices.extend((episode_idx, t) for t in valid_t)

        if not self._indices:
            raise ValueError("dataset contains no valid training samples")

        self._decoder_cache = HDF5VideoDecoderCache(
            self.path,
            device=self.video_device,
            capacity=decoder_cache_size,
            seek_mode=video_seek_mode,
            num_ffmpeg_threads=video_num_threads,
        )
        self._frame_cache = HDF5VideoFrameCache(
            self._decoder_cache,
            image_size=self.image_size,
            mode=frame_cache,
            capacity=frame_cache_size,
            preload_batch_size=frame_preload_batch_size,
        )
        if self.decode_images and frame_cache == "ram":
            self._frame_cache.preload(self._referenced_video_frames())
            # RAM mode is self-contained after preloading; release codec/HDF5
            # handles before DataLoader workers are created.
            self._decoder_cache.close()

    @property
    def hz(self) -> float:
        return self.info.hz

    @property
    def arm_order(self) -> tuple[str, ...]:
        return self.info.arm_order

    @property
    def num_arms(self) -> int:
        return self.info.num_arms

    @property
    def action_dim(self) -> int:
        # relative xyz(3) + rotation6d(6) + gripper(1), per arm
        return self.num_arms * 10

    @property
    def shape_meta(self) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        for arm_idx, _side in enumerate(self.arm_order):
            obs[f"camera{arm_idx}_main_rgb"] = {
                "shape": [3, self.image_size, self.image_size],
                "horizon": self.obs_horizon,
                "type": "rgb",
                "ignore_by_policy": False,
            }
            obs[f"robot{arm_idx}_eef_pos"] = {
                "shape": [3],
                "horizon": self.obs_horizon,
                "type": "low_dim",
                "ignore_by_policy": False,
            }
            obs[f"robot{arm_idx}_eef_rot_axis_angle"] = {
                "raw_shape": [3],
                "shape": [6],
                "horizon": self.obs_horizon,
                "type": "low_dim",
                "rotation_rep": "rotation_6d",
                "ignore_by_policy": False,
            }
            obs[f"robot{arm_idx}_gripper_width"] = {
                "shape": [1],
                "horizon": self.obs_horizon,
                "type": "low_dim",
                "ignore_by_policy": False,
            }
        return {
            "image_resolution": self.image_size,
            "obs": obs,
            "action": {
                "shape": [self.action_dim],
                "horizon": self.action_horizon,
                "rotation_rep": "rotation_6d",
            },
        }

    def _load_lowdim(self, episode_keys: Sequence[str]) -> tuple[_EpisodeData, ...]:
        episodes: list[_EpisodeData] = []
        with h5py.File(self.path, "r") as h5file:
            for episode_key in episode_keys:
                group = h5file[episode_key]
                action_flat = np.asarray(group["action"][:], dtype=np.float32)
                action = action_flat.reshape(action_flat.shape[0], self.num_arms, 8)
                video_index = np.asarray(group["video_index"][:], dtype=np.int64)
                pose_matrix = pose7_wxyz_to_matrix(action[..., :7])
                episodes.append(
                    _EpisodeData(
                        key=episode_key,
                        action=action,
                        pose_matrix=pose_matrix,
                        video_index=video_index,
                    )
                )
        return tuple(episodes)

    def _obs_indices(self, t: int) -> np.ndarray:
        indices = np.arange(t - self.obs_horizon + 1, t + 1, dtype=np.int64)
        return np.maximum(indices, 0)

    def _action_indices(self, episode: _EpisodeData, t: int) -> np.ndarray:
        indices = np.arange(t, t + self.action_horizon, dtype=np.int64)
        if self.action_padding:
            return np.minimum(indices, episode.length - 1)
        if indices[-1] >= episode.length:
            raise IndexError("action window exceeds episode and padding is disabled")
        return indices

    def lowdim_sample(
        self,
        episode_idx: int,
        t: int,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Build one deterministic low-dimensional HoMMI sample without video decode."""
        episode = self._episodes[episode_idx]
        if not 0 <= t < episode.length:
            raise IndexError(t)

        obs_indices = self._obs_indices(t)
        action_indices = self._action_indices(episode, t)
        obs: dict[str, np.ndarray] = {}
        action_parts: list[np.ndarray] = []

        for arm_idx in range(self.num_arms):
            base = episode.pose_matrix[t, arm_idx]
            obs_pose9 = relative_pose9(base, episode.pose_matrix[obs_indices, arm_idx])
            action_pose9 = relative_pose9(base, episode.pose_matrix[action_indices, arm_idx])

            obs[f"robot{arm_idx}_eef_pos"] = obs_pose9[:, :3]
            # Name retained for HoMMI config compatibility. Values are 6D
            # rotation rows after preprocessing, not 3D axis-angle values.
            obs[f"robot{arm_idx}_eef_rot_axis_angle"] = obs_pose9[:, 3:9]
            obs[f"robot{arm_idx}_gripper_width"] = episode.action[
                obs_indices, arm_idx, 7:8
            ].astype(np.float32, copy=False)

            action_gripper = episode.action[action_indices, arm_idx, 7:8]
            action_parts.append(
                np.concatenate((action_pose9, action_gripper), axis=-1).astype(
                    np.float32, copy=False
                )
            )

        return obs, np.concatenate(action_parts, axis=-1).astype(np.float32, copy=False)

    def iter_lowdim_samples(self) -> Iterator[tuple[dict[str, np.ndarray], np.ndarray]]:
        for episode_idx, t in self._indices:
            yield self.lowdim_sample(episode_idx, t)

    def collect_lowdim_training_arrays(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Materialize low-dimensional samples for training-set normalizer fitting.

        This preserves the exact sample/padding distribution seen by the policy.
        It intentionally excludes RGB because the image field normalizer is
        identity and pretrained-backbone normalization lives in the encoder.
        """
        obs_cache: dict[str, list[np.ndarray]] = {}
        action_cache: list[np.ndarray] = []
        for obs, action in self.iter_lowdim_samples():
            for key, value in obs.items():
                obs_cache.setdefault(key, []).append(value)
            action_cache.append(action)

        return (
            {
                key: np.concatenate(values, axis=0).astype(np.float32, copy=False)
                for key, values in obs_cache.items()
            },
            np.concatenate(action_cache, axis=0).astype(np.float32, copy=False),
        )

    @property
    def frame_cache_mode(self) -> FrameCacheMode:
        return self._frame_cache.mode

    @property
    def frame_cache_bytes(self) -> int:
        return self._frame_cache.cache_bytes

    @property
    def num_cached_frames(self) -> int:
        return self._frame_cache.num_cached_frames

    def _referenced_video_frames(self) -> dict[tuple[str, str], np.ndarray]:
        """Return only original-video frames reachable by dataset observations."""
        by_episode: dict[int, list[int]] = {}
        for episode_idx, t in self._indices:
            by_episode.setdefault(episode_idx, []).extend(self._obs_indices(t).tolist())

        references: dict[tuple[str, str], np.ndarray] = {}
        for episode_idx, raw_obs_indices in by_episode.items():
            episode = self._episodes[episode_idx]
            obs_indices = np.unique(np.asarray(raw_obs_indices, dtype=np.int64))
            for arm_idx, side in enumerate(self.arm_order):
                references[(episode.key, side)] = np.unique(
                    episode.video_index[obs_indices, arm_idx]
                ).astype(np.int64, copy=False)
        return references

    def _load_images(
        self,
        episode: _EpisodeData,
        obs_indices: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        if not self.decode_images:
            return {}
        worker = get_worker_info()
        if (
            self.video_device.type == "cuda"
            and worker is not None
            and self.frame_cache_mode != "ram"
        ):
            raise RuntimeError(
                "CUDA video decoding inside DataLoader worker processes is intentionally "
                "disabled. Use num_workers=0 with video_device='cuda', or decode on CPU "
                "workers and move the batch to CUDA in the trainer."
            )

        images: dict[str, torch.Tensor] = {}
        for arm_idx, side in enumerate(self.arm_order):
            frame_indices = episode.video_index[obs_indices, arm_idx]
            cached = self._frame_cache.get_frames(
                episode.key,
                side,
                frame_indices,
            )
            # Keep the public dataset contract unchanged: policy/encoder inputs
            # are float32 RGB in [0, 1]. Only the persistent RAM working set is
            # uint8; this per-sample float tensor is short-lived.
            images[f"camera{arm_idx}_main_rgb"] = cached.to(
                dtype=torch.float32
            ).div_(255.0)
        return images

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_idx, t = self._indices[index]
        episode = self._episodes[episode_idx]
        obs_indices = self._obs_indices(t)
        lowdim, action = self.lowdim_sample(episode_idx, t)

        obs: dict[str, torch.Tensor] = self._load_images(episode, obs_indices)
        obs.update({key: torch.from_numpy(value) for key, value in lowdim.items()})
        return {
            "obs": obs,
            "action": torch.from_numpy(action),
            "metadata": {
                "episode_idx": torch.tensor(episode_idx, dtype=torch.int64),
                "episode_key": episode.key,
                "t": torch.tensor(t, dtype=torch.int64),
            },
        }

    def close(self) -> None:
        self._frame_cache.close()
