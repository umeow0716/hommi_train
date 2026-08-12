# hommi-train

Training-side package for datasets produced by `hommi-dataset` and policies from
`hommi-diffusion-policy`.

Version **0.4.0** completes the command-line training runner. The reusable
composition and training layers from 0.2/0.3 are now connected end-to-end, so a
HoMMI HDF5 dataset can be trained directly with:

```bash
python -m hommi_train -i dataset.hdf5 -o outputs/run-001
```

The installed console entry point is equivalent:

```bash
hommi-train -i dataset.hdf5 -o outputs/run-001
```

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
├── runner.py             # dataset -> split -> normalizer -> policy -> Trainer
├── config.py             # one source of HoMMI-aligned defaults
├── cli.py                # argparse only; no training business logic
└── __main__.py

# next milestone
hommi_train/evaluation/
hommi_train/export/
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

## Training from the CLI

The zero-override command uses the HoMMI-aligned model/training defaults and the
runtime optimizations documented below:

```bash
python -m hommi_train \
    -i pick_and_place.hdf5 \
    -o outputs/pick-and-place
```

Typical overrides stay compact:

```bash
python -m hommi_train \
    -i pick_and_place.hdf5 \
    -o outputs/pick-and-place \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 1200
```

Use `--help` to see grouped Dataset, Training, Optimizer, DiT/diffusion,
Runtime, and Checkpoint options.

### Dataset inspection

Schema/metadata inspection does not initialize TorchCodec or the model:

```bash
python -m hommi_train -i pick_and_place.hdf5 --inspect-dataset
```

### Resume

Resume restores model, EMA, optimizer, LR scheduler, trainer progress, split,
and RNG states:

```bash
python -m hommi_train \
    -i pick_and_place.hdf5 \
    -o outputs/pick-and-place \
    --resume outputs/pick-and-place/checkpoints/last.pt
```

The checkpoint's saved hierarchical config is the base config for resume. Only
options explicitly supplied on the new command line override it. This makes it
possible to extend a run without repeating custom architecture settings:

```bash
python -m hommi_train \
    -i pick_and_place.hdf5 \
    -o outputs/pick-and-place \
    --resume outputs/pick-and-place/checkpoints/last.pt \
    --epochs 1500
```

The original train/validation episode keys are always reused on resume. The CLI
also checks the saved `shape_meta` against the selected dataset/config before
restoring weights.

Starting a new run in an output directory that already contains
`checkpoints/last.pt` is refused unless `--resume` is supplied.

## CLI groups and defaults

### Dataset

```text
obs horizon                   2
action horizon               16
image size                  224
validation ratio           0.05
action padding              true
frame cache                  ram
LRU capacity                2048
RAM preload decode batch       8
```

### Training / optimizer

```text
batch size                    16
epochs                      1000
lr                         7.5e-5
vision-backbone lr          7.5e-6
weight decay                 1e-6
vision weight decay          1e-6
AdamW betas           (0.95, 0.999)
warmup steps                    50
gradient clip                  5.0
sample/eval every               10 epochs
grad norm log every             50 steps
seed                            42
```

### DiT / DDIM

```text
vision backbone    vit_base_patch16_clip_224.openai
pretrained                                      true
n_action_steps                                     8
DDIM inference steps                              16
DiT hidden width                                 768
timestep embed                                   256
depth                                               8
heads                                               8
MLP ratio                                         4.0
train diffusion samples                            8
DDIM train timesteps                              50
beta schedule                    squaredcos_cap_v2
prediction type                              epsilon
```

### Runtime

```text
training device                 auto  # CUDA if available
video decode device             cpu
precision                       bf16
DataLoader workers                 8
pin_memory                      true
persistent_workers              true
drop_last                       true
frame cache                      ram / uint8
```

Runtime choices are intentionally separate from HoMMI architecture/training
hyperparameters. BF16 is the default CUDA training optimization; use
`--precision fp32` when required.

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

The train/validation split is decided **before** either dataset is constructed,
so each RAM cache only preloads its own episode set. Validation data is never
used to fit normalizer statistics.

The dataset center-square crop is deterministic preprocessing. The DiT
encoder's 0.95 RandomCrop/CenterCrop and train ColorJitter remain model-side
augmentation.

## Programmatic composition

The CLI is only a frontend. The same complete composition is available without
argparse:

```python
from hommi_train import HommiTrainConfig, run_training

state = run_training(
    "dataset.hdf5",
    "outputs/run-001",
    HommiTrainConfig(),
)
```

Lower layers remain separately reusable when custom composition is needed.

## Validation behavior

To preserve the previous HoMMI workspace behavior, every `sample_every` epochs
the EMA policy evaluates the last train batch and one validation batch using
full-horizon action MSE.

The next milestone will separate this into explicit sampled/full evaluation and
produce portable/optimized inference artifacts.

## Checkpoints

Training writes:

```text
output/
└── checkpoints/
    ├── last.pt
    ├── best.pt
    └── epoch=XXXX-val_action_mse_error=....pt   # best K, default 3
```

Each checkpoint includes model/EMA/optimizer/scheduler state, next epoch,
global step, full config, `shape_meta`, train/validation episode keys, metrics,
and Python/NumPy/Torch CPU/CUDA RNG states.

## Next milestone

**0.5.0** adds explicit evaluation/export functionality:

- sampled versus full validation/evaluation;
- portable EMA `model.pt` without optimizer/training state;
- investigation of `torch.export` for the full diffusion policy;
- an optimized CUDA edge-inference artifact when technically appropriate.

## Lock file

Regenerate the lock file in a networked checkout after unpacking:

```bash
uv lock
uv sync --extra dev
```
