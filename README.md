# hommi-train

Training-side package for datasets produced by `hommi-dataset` and policies from
`hommi-diffusion-policy`.

Version **0.2.0** completes the composition layer between the HDF5 dataset and
the policy library: episode split, training-only normalizer fitting, and a
HoMMI-aligned DiT policy factory.

## Package boundary

`hommi-train` owns dataset/training-recipe knowledge. `hommi-diffusion-policy`
remains the reusable model/runtime package and does not know how HDF5 episodes
are stored or how training statistics are fitted.

```text
hommi_train/
├── dataset/
│   ├── geometry.py       # WXYZ pose -> matrix -> UMI rotation-6D
│   ├── schema.py         # HDF5 schema validation / metadata
│   ├── split.py          # deterministic episode-level train/val split
│   ├── video.py          # HDF5/TorchCodec decoder + deterministic resize
│   ├── frame_cache.py    # none/LRU/full-RAM uint8 working-set cache
│   └── hommi_hdf5.py     # torch.utils.data.Dataset
├── normalization/
│   └── hommi.py          # fit policy normalizer from training split only
├── policy/
│   └── dit.py            # DDIM + DiT construction recipe
├── config.py             # one source of HoMMI-aligned defaults
├── cli.py
└── __main__.py

# later milestones
hommi_train/training/      # optimizer, EMA, BF16, train/val loop, resume
hommi_train/export/        # portable .pt + optimized eval artifact
```

## Input HDF5

The reader targets `hommi-dataset >= 0.2.0`:

```text
hz                  float32
type                str
arm_order           str[A]
episode_001/
  action             float32[N, A*8]
  video_index        int32[N, A]
  video/
    left             uint8[original H.264 MP4 bytes]
    right            uint8[...]                 # dual arm
```

Each stored arm action is:

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

`HommiHDF5Dataset` converts this to HoMMI's model representation per arm:

```text
[pos(3), rotation6d(6), gripper(1)]
```

The pose at the last observation timestep is the reference frame. Observation
history and future action chunks are expressed relative to that pose.

## Video path and RAM working set

H.264 remains the compact on-disk representation. The default training working
set is compact decoded host RAM:

```text
HDF5 embedded H.264 (original 1920x1440)
    -> only unique frames referenced by selected episodes/observations
    -> TorchCodec decode
    -> deterministic center-square crop
    -> resize to 224x224
    -> uint8 CHW RAM cache
```

The persistent cache is `uint8`, not float32. `__getitem__` converts only the
requested observation frames to short-lived float32 `[0,1]` tensors.

Three cache modes are available:

- `frame_cache="ram"` (default): preload referenced frames once.
- `frame_cache="lru"`: bounded lazy decoded-frame cache.
- `frame_cache="none"`: decode every request; useful for low-RAM CUDA/NVDEC eval.

The dataset center-square crop is deterministic storage preprocessing. The DiT
encoder's 0.95 RandomCrop/CenterCrop remains model augmentation and is not
removed.

## 0.2.0 composition flow

Split **episodes first**, then construct independent datasets. This ensures the
RAM cache and normalization statistics never include validation trajectories.

```python
from hommi_train import (
    DatasetConfig,
    HommiHDF5Dataset,
    build_dit_policy,
    build_hommi_normalizer,
    inspect_hommi_hdf5,
    split_episode_keys,
)

path = "dataset.hdf5"
cfg = DatasetConfig()
info = inspect_hommi_hdf5(path)
split = split_episode_keys(info, val_ratio=cfg.val_ratio, seed=42)

train_ds = HommiHDF5Dataset(
    path,
    episode_keys=split.train_keys,
    obs_horizon=cfg.obs_horizon,
    action_horizon=cfg.action_horizon,
    image_size=cfg.image_size,
    action_padding=cfg.action_padding,
    frame_cache=cfg.frame_cache,
    frame_cache_size=cfg.frame_cache_size,
    frame_preload_batch_size=cfg.frame_preload_batch_size,
)
val_ds = HommiHDF5Dataset(
    path,
    episode_keys=split.val_keys,
    obs_horizon=cfg.obs_horizon,
    action_horizon=cfg.action_horizon,
    image_size=cfg.image_size,
    action_padding=cfg.action_padding,
    frame_cache=cfg.frame_cache,
    frame_cache_size=cfg.frame_cache_size,
    frame_preload_batch_size=cfg.frame_preload_batch_size,
)

normalizer = build_hommi_normalizer(train_ds)  # training split only
policy = build_dit_policy(train_ds.shape_meta)
policy.set_normalizer(normalizer)
```

### Normalization ownership

`hommi_train.normalization` fits statistics; the policy applies them during both
training and inference.

Per arm:

```text
RGB       identity in LinearNormalizer (vision mean/std lives in encoder)
position  limits/range -> [-1, 1]
rotation6 identity
gripper   limits/range -> [-1, 1]
```

For dual-arm data, action normalization is fitted independently per 10-D arm
block and then concatenated in HDF5 `arm_order`.

Do not fit a separate normalizer on validation data.

## Policy factory

`build_dit_policy(shape_meta)` owns the HoMMI training recipe while the actual
model classes stay in `hommi-diffusion-policy`.

The factory builds:

```text
DiTObsEncoderLite
+ DDIMScheduler
+ DiffusionDiTImagePolicy
```

`DDIMConfig` exposes the previously hard-coded scheduler defaults:

- train timesteps: 50
- beta start/end: `0.0001` / `0.02`
- schedule: `squaredcos_cap_v2`
- clip sample: true
- set alpha to one: true
- steps offset: 0
- prediction type: `epsilon`

`DiTModelConfig` keeps the HoMMI-aligned model defaults, including ViT CLIP
backbone, 768 hidden size, 8 blocks, 8 heads, RMSNorm, 8 action steps, 16 DDIM
inference steps, and 8 diffusion samples per training batch.

## Dataset validation CLI

The final target remains:

```bash
python -m hommi_train -i file.hdf5 -o output/
```

The trainer loop is deliberately not wired yet. Dataset validation works now:

```bash
python -m hommi_train -i file.hdf5 -o output/ --inspect-dataset
```

## Next milestone

The next package layer is `training/`:

```text
training/
├── trainer.py
├── ema.py
├── optimizer.py
├── checkpoint.py
└── metrics.py
```

It will add BF16 CUDA training, DataLoader construction, HoMMI optimizer/LR
scheduler defaults, EMA, validation, checkpoint/resume, and finally connect the
main CLI.

## Lock file

Regenerate the lock file in a networked checkout after unpacking:

```bash
uv lock
uv sync --extra dev
```
