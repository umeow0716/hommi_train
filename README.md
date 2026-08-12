# hommi-train

Training-side package for datasets produced by `hommi-dataset` and policies from
`hommi-diffusion-policy`.

Version **0.3.0** completes the reusable training layer: DataLoaders, optimizer
and cosine LR scheduler construction, HoMMI EMA, BF16 autocast, sampled
validation, atomic checkpoints, and full resume state restoration.

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
├── training/
│   ├── data.py           # DataLoaders + non-blocking device transfer
│   ├── optimizer.py      # policy optimizer groups + cosine scheduler
│   ├── ema.py            # HoMMI EMA warmup
│   ├── metrics.py        # gradient norm + action MSE
│   ├── checkpoint.py     # atomic save/load + RNG state
│   └── trainer.py        # BF16 train/validation loop + resume
├── config.py             # one source of HoMMI-aligned defaults
├── cli.py
└── __main__.py

# later milestones
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

The dataset center-square crop is deterministic storage preprocessing. The DiT
encoder's 0.95 RandomCrop/CenterCrop remains model augmentation.

## Composition flow

Split **episodes first**, then build independent datasets and fit normalization
from training only.

```python
from pathlib import Path

from hommi_train import (
    HommiTrainConfig,
    HommiHDF5Dataset,
    Trainer,
    build_dataloaders,
    build_dit_policy,
    build_hommi_normalizer,
    inspect_hommi_hdf5,
    seed_everything,
    split_episode_keys,
)

path = Path("dataset.hdf5")
output = Path("outputs/run-001")
cfg = HommiTrainConfig()

# Seed before policy construction so randomly initialized DiT weights are also
# reproducible. Trainer checkpoints additionally preserve all RNG streams.
seed_everything(cfg.training.seed)

info = inspect_hommi_hdf5(path)
split = split_episode_keys(
    info,
    val_ratio=cfg.dataset.val_ratio,
    seed=cfg.training.seed,
)

common_dataset = dict(
    obs_horizon=cfg.dataset.obs_horizon,
    action_horizon=cfg.dataset.action_horizon,
    image_size=cfg.dataset.image_size,
    action_padding=cfg.dataset.action_padding,
    frame_cache=cfg.dataset.frame_cache,
    frame_cache_size=cfg.dataset.frame_cache_size,
    frame_preload_batch_size=cfg.dataset.frame_preload_batch_size,
)
train_ds = HommiHDF5Dataset(path, episode_keys=split.train_keys, **common_dataset)
val_ds = HommiHDF5Dataset(path, episode_keys=split.val_keys, **common_dataset)

normalizer = build_hommi_normalizer(train_ds)
policy = build_dit_policy(
    train_ds.shape_meta,
    model_config=cfg.model,
    ddim_config=cfg.ddim,
)
policy.set_normalizer(normalizer)

train_loader, val_loader = build_dataloaders(
    train_ds,
    val_ds,
    cfg.training,
)
trainer = Trainer(
    policy=policy,
    train_loader=train_loader,
    val_loader=val_loader,
    config=cfg.training,
    run_config=cfg,
    output_dir=output,
)
trainer.fit()
```

## Training defaults

The training/model defaults remain aligned with the previous HoMMI DiT recipe:

```text
batch size                16
epochs                  1000
lr                     7.5e-5
vision-backbone lr      7.5e-6
weight decay             1e-6
warmup steps                50
gradient clip               5.0
sample/eval every           10 epochs
EMA inv_gamma                1.0
EMA power                    0.75
EMA max                      0.9999
```

Runtime defaults are intentionally separate from HoMMI architecture/training
hyperparameters:

```text
precision                 bf16
pin_memory                true
persistent_workers        true
frame cache               ram / uint8
```

On CUDA, `precision="bf16"` uses `torch.autocast` without FP16 GradScaler.
Unsupported CUDA devices fail early with a clear error. `precision="fp32"`
remains available for debugging and CPU tests.

## Validation behavior

To preserve the previous HoMMI workspace behavior, every `sample_every` epochs
the EMA policy evaluates:

- the last train batch from the epoch;
- one validation batch;
- full-horizon action MSE.

A later evaluation milestone will add explicit `sampled` versus `full`
validation modes without coupling them into the core trainer.

## Checkpoints and resume

Training writes:

```text
output/
└── checkpoints/
    ├── last.pt
    ├── best.pt
    └── epoch=XXXX-val_action_mse_error=....pt   # best K, default 3
```

Each checkpoint includes:

```text
model state
EMA model + EMA warmup state
optimizer state
LR scheduler state
next epoch index
global step
best validation state
full hierarchical run config
shape_meta
train/validation episode keys
latest metrics
Python / NumPy / Torch CPU / Torch CUDA RNG states
```

Resume is therefore a real training resume rather than a model-only load:

```python
trainer.fit(resume_from="outputs/run-001/checkpoints/last.pt")
```

`TrainerState.epoch` is the **next** epoch to execute, so an epoch-complete
checkpoint does not repeat the completed epoch after resume.

## Dataset validation CLI

The final target remains:

```bash
python -m hommi_train -i file.hdf5 -o output/
```

The reusable trainer is implemented in 0.3.0, but the complete argparse runner
is deliberately the **0.4.0** milestone. Dataset inspection remains available:

```bash
python -m hommi_train -i file.hdf5 -o output/ --inspect-dataset
```

## Next milestone

**0.4.0** connects the existing composition and trainer layers into the main
CLI. Arguments will be grouped into dataset, training, optimizer, DiT, runtime,
and checkpoint sections while all defaults continue to come from the config
dataclasses rather than being duplicated in argparse.

## Lock file

Regenerate the lock file in a networked checkout after unpacking:

```bash
uv lock
uv sync --extra dev
```
