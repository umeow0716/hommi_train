# hommi-train

Training-side package for datasets produced by `hommi-dataset` and policies from
`hommi-diffusion-policy`.

## Package boundary

`hommi-train` owns training/data-pipeline knowledge. The policy package stays
usable for inference without knowing how HDF5 episodes are stored.

```text
hommi_train/
├── dataset/              # implemented; RAM frame cache added in 0.1.1
│   ├── geometry.py       # WXYZ pose -> matrix -> UMI rotation-6D
│   ├── schema.py         # HDF5 schema validation / metadata
│   ├── video.py          # HDF5/TorchCodec decoder + deterministic resize
│   ├── frame_cache.py    # none/LRU/full-RAM uint8 working-set cache
│   └── hommi_hdf5.py     # torch.utils.data.Dataset
├── config.py             # HoMMI default values, shared by future CLI/trainer
├── cli.py                # final training CLI entry point; inspection works now
└── __main__.py

# next milestones (kept separate rather than mixed into dataset code)
hommi_train/normalization/ # fit LinearNormalizer from training split only
hommi_train/training/      # DiT construction, optimizer, EMA, train/val loop
hommi_train/export/        # .pt checkpoint + optimized eval artifact
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

H.264 remains the compact **on-disk** representation. For long training runs the
default dataset mode is now a compact decoded **host-RAM** working set:

```text
HDF5 embedded H.264 (original 1920x1440)
    -> only unique frames referenced by dataset observations
    -> TorchCodec decode
    -> deterministic center-square crop
    -> resize to 224x224
    -> uint8 CHW RAM cache
```

The persistent cache is `uint8`, not float32. A 224x224 RGB frame therefore
uses 150,528 bytes. `HommiHDF5Dataset.__getitem__` converts only the requested
observation frames to short-lived float32 `[0,1]` tensors, preserving the
existing policy/encoder contract.

Training default:

```python
train_ds = HommiHDF5Dataset(
    "dataset.hdf5",
    obs_horizon=2,
    action_horizon=16,
    image_size=224,
    video_device="cpu",
    frame_cache="ram",
)

print(train_ds.num_cached_frames)
print(train_ds.frame_cache_bytes)
```

Three cache modes are available:

- `frame_cache="ram"` (default): preload the referenced working set once. This is
  recommended for long multi-epoch training.
- `frame_cache="lru"`: decode on first use and retain a bounded number of recent
  uint8 frames per process. Tune with `frame_cache_size`.
- `frame_cache="none"`: no decoded-frame cache; every request goes through
  TorchCodec. Useful for low-RAM evaluation/profiling.

The RAM preloader decodes in small batches (`frame_preload_batch_size=8` by
default) so original-resolution 1920x1440 frames do not create an excessive
temporary float32 batch during resize.

For direct NVDEC profiling/evaluation you can still use:

```python
eval_ds = HommiHDF5Dataset(
    "dataset.hdf5",
    video_device="cuda",
    frame_cache="none",
)
```

Use `num_workers=0` when CUDA decoding is still required at sample time. In RAM
mode decoding is finished during dataset construction and decoder/HDF5 handles
are closed before DataLoader workers are created.

### Why the DiT 0.95 crop stays

The dataset's center-square crop and the encoder's 0.95 crop have different
responsibilities. The first is deterministic aspect-ratio preprocessing for the
RAM working set. `DiTObsEncoderLite`'s `RandomCrop(0.95) -> Resize -> ColorJitter`
is HoMMI training augmentation, while eval uses `CenterCrop(0.95) -> Resize`.
Removing that encoder crop would change the HoMMI-aligned training recipe, so
0.1.1 intentionally leaves it unchanged.

## Dataset sample

Single arm:

```python
sample = train_ds[10]

sample["obs"]["camera0_main_rgb"]             # [2, 3, 224, 224], float32 [0,1]
sample["obs"]["robot0_eef_pos"]              # [2, 3]
sample["obs"]["robot0_eef_rot_axis_angle"]   # [2, 6] (name kept for HoMMI compatibility)
sample["obs"]["robot0_gripper_width"]        # [2, 1]
sample["action"]                              # [16, 10]
```

For dual arm, the same keys are added with `camera1_*` / `robot1_*` and action
shape becomes `[16, 20]`.

`dataset.shape_meta` is ready to pass to `hommi-diffusion-policy` encoders and
policies.

## Dataset validation CLI

The final target remains:

```bash
python -m hommi_train -i file.hdf5 -o output/
```

The training runner is the next milestone. In this dataset milestone the CLI can
already validate the input file:

```bash
python -m hommi_train -i file.hdf5 -o output/ --inspect-dataset
```

## HoMMI defaults reserved in `config.py`

Dataset defaults:

- observation horizon: 2
- action horizon: 16
- image size: 224
- validation ratio: 0.05
- frame cache: `ram`
- LRU capacity when selected: 2048 frames per process
- RAM preload decode batch: 8 frames

Training defaults from the previous HoMMI-aligned trainer:

- batch size: 16
- epochs: 1000
- learning rate: `7.5e-5`
- vision encoder learning rate: `7.5e-6`
- weight decay: `1e-6`
- warmup: 50 steps
- gradient clipping: 5.0
- sample/eval interval: 10 epochs

DiT defaults:

- `vit_base_patch16_clip_224.openai`
- action steps: 8
- DDIM inference steps: 16
- hidden size: 768
- timestep embedding: 256
- depth: 8
- heads: 8
- MLP ratio: 4
- diffusion samples per training batch: 8

These values are defined now so later CLI flags can override one canonical
configuration instead of duplicating defaults across scripts.

## Lock file

The dependency graph changed in this milestone (`torchcodec` is now a direct
runtime dependency), so regenerate and commit `uv.lock` in a networked checkout:

```bash
uv lock
uv sync --extra dev
```
