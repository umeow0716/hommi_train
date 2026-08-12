from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

_EPISODE_RE = re.compile(r"^episode_(\d+)$")


@dataclass(frozen=True, slots=True)
class EpisodeInfo:
    key: str
    sample_count: int


@dataclass(frozen=True, slots=True)
class HommiHDF5Info:
    path: Path
    hz: float
    dataset_type: str
    arm_order: tuple[str, ...]
    episodes: tuple[EpisodeInfo, ...]

    @property
    def num_arms(self) -> int:
        return len(self.arm_order)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_samples(self) -> int:
        return sum(episode.sample_count for episode in self.episodes)


def natural_episode_key(key: str) -> tuple[int, str]:
    match = _EPISODE_RE.fullmatch(key)
    return (int(match.group(1)) if match else 10**12, key)


def _decode_text(value: object, *, field: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_text(value.item(), field=field)
    raise ValueError(f"{field} must contain UTF-8 text, got {type(value).__name__}")


def _decode_text_array(values: Iterable[object], *, field: str) -> tuple[str, ...]:
    result = tuple(_decode_text(value, field=field) for value in values)
    if not result:
        raise ValueError(f"{field} must contain at least one arm")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} contains duplicate arm names: {result}")
    return result



def make_shape_meta(
    info: HommiHDF5Info,
    *,
    obs_horizon: int = 2,
    action_horizon: int = 16,
    image_size: int = 224,
) -> dict[str, object]:
    """Build the canonical model observation/action contract from HDF5 metadata.

    No video decode or episode payload read is required, which makes this safe
    for task-YAML generation and configuration validation.
    """
    if obs_horizon < 1 or action_horizon < 1 or image_size < 1:
        raise ValueError("obs_horizon, action_horizon, and image_size must be >= 1")

    obs: dict[str, object] = {}
    for arm_idx, _side in enumerate(info.arm_order):
        obs[f"camera{arm_idx}_main_rgb"] = {
            "shape": [3, image_size, image_size],
            "horizon": obs_horizon,
            "type": "rgb",
            "ignore_by_policy": False,
        }
        obs[f"robot{arm_idx}_eef_pos"] = {
            "shape": [3],
            "horizon": obs_horizon,
            "type": "low_dim",
            "ignore_by_policy": False,
        }
        obs[f"robot{arm_idx}_eef_rot_axis_angle"] = {
            "raw_shape": [3],
            "shape": [6],
            "horizon": obs_horizon,
            "type": "low_dim",
            "rotation_rep": "rotation_6d",
            "ignore_by_policy": False,
        }
        obs[f"robot{arm_idx}_gripper_width"] = {
            "shape": [1],
            "horizon": obs_horizon,
            "type": "low_dim",
            "ignore_by_policy": False,
        }
    return {
        "image_resolution": image_size,
        "obs": obs,
        "action": {
            "shape": [info.num_arms * 10],
            "horizon": action_horizon,
            "rotation_rep": "rotation_6d",
        },
    }

def inspect_hommi_hdf5(path: str | Path, *, strict_video_attrs: bool = True) -> HommiHDF5Info:
    """Validate the on-disk ``hommi_dataset>=0.2`` HDF5 schema.

    The function only reads metadata / shapes; it does not decode or preload any
    embedded MP4 payload.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    with h5py.File(resolved, "r") as h5file:
        for key in ("hz", "type", "arm_order"):
            if key not in h5file:
                raise KeyError(f"{resolved}: missing root dataset {key!r}")

        hz = float(np.asarray(h5file["hz"][()]).item())
        if not np.isfinite(hz) or hz <= 0.0:
            raise ValueError(f"{resolved}: hz must be positive and finite, got {hz}")

        dataset_type = _decode_text(h5file["type"][()], field="type")
        arm_order = _decode_text_array(h5file["arm_order"][:], field="arm_order")
        num_arms = len(arm_order)

        episode_keys = sorted(
            (key for key in h5file.keys() if _EPISODE_RE.fullmatch(key)),
            key=natural_episode_key,
        )
        if not episode_keys:
            raise ValueError(f"{resolved}: no episode_### groups found")

        episodes: list[EpisodeInfo] = []
        for episode_key in episode_keys:
            group = h5file[episode_key]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"{resolved}:{episode_key} must be an HDF5 group")
            for key in ("action", "video_index", "video"):
                if key not in group:
                    raise KeyError(f"{resolved}:{episode_key} missing {key!r}")

            action = group["action"]
            video_index = group["video_index"]
            if action.ndim != 2 or action.shape[1] != num_arms * 8:
                raise ValueError(
                    f"{resolved}:{episode_key}/action must be [T,{num_arms * 8}], "
                    f"got {action.shape}"
                )
            if video_index.ndim != 2 or video_index.shape != (action.shape[0], num_arms):
                raise ValueError(
                    f"{resolved}:{episode_key}/video_index must be "
                    f"[{action.shape[0]},{num_arms}], got {video_index.shape}"
                )
            if action.shape[0] == 0:
                raise ValueError(f"{resolved}:{episode_key} contains zero samples")
            if np.any(video_index[:] < 0):
                raise ValueError(f"{resolved}:{episode_key}/video_index contains negative indices")

            video_group = group["video"]
            if not isinstance(video_group, h5py.Group):
                raise TypeError(f"{resolved}:{episode_key}/video must be a group")
            for side in arm_order:
                if side not in video_group:
                    raise KeyError(f"{resolved}:{episode_key}/video missing arm {side!r}")
                video = video_group[side]
                if video.ndim != 1 or video.dtype != np.dtype(np.uint8):
                    raise ValueError(
                        f"{resolved}:{episode_key}/video/{side} must be uint8[bytes], "
                        f"got dtype={video.dtype}, shape={video.shape}"
                    )
                if strict_video_attrs:
                    container = video.attrs.get("container")
                    codec = video.attrs.get("codec")
                    container = container.decode() if isinstance(container, bytes) else container
                    codec = codec.decode() if isinstance(codec, bytes) else codec
                    if container != "mp4":
                        raise ValueError(
                            f"{resolved}:{episode_key}/video/{side} container must be 'mp4', "
                            f"got {container!r}"
                        )
                    if codec != "h264":
                        raise ValueError(
                            f"{resolved}:{episode_key}/video/{side} codec must be 'h264', "
                            f"got {codec!r}"
                        )

            episodes.append(EpisodeInfo(episode_key, int(action.shape[0])))

    return HommiHDF5Info(
        path=resolved,
        hz=hz,
        dataset_type=dataset_type,
        arm_order=arm_order,
        episodes=tuple(episodes),
    )
