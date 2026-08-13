# hommi-train

Training, evaluation, configuration, and deployment tooling for HoMMI-style diffusion policies.

`hommi-train` consumes HDF5 datasets created by [`hommi-dataset`](https://github.com/umeow0716/hommi_dataset), builds policies from [`hommi-diffusion-policy`](https://github.com/umeow0716/hommi_diffusion_policy), and provides a task-YAML workflow from dataset inspection through training, TensorRT-accelerated evaluation, checkpointing, and portable model export.

## Requirements

- Python 3.12+
- NVIDIA CUDA environment for TensorRT evaluation
- PyTorch, Torch-TensorRT, and TensorRT versions that are mutually compatible

TensorRT support is a **required package dependency**, not an optional extra. This intentionally makes the published package target Torch-TensorRT-supported hosts (official prebuilt Torch-TensorRT packages are provided for Linux x86 and Windows). The package still keeps eager PyTorch fallback paths for environments where CUDA is unavailable, so dataset inspection and configuration tooling remain usable after installation.

## Installation

### pip

After the packages are published:

```bash
python -m pip install git+https://github.com/umeow0716/hommi-train
```

From a local checkout:

```bash
python -m pip install .
```

### uv

Install into the active environment:

```bash
uv pip install git+https://github.com/umeow0716/hommi-train
```

Or add it to an existing uv project:

```bash
uv add git+https://github.com/umeow0716/hommi-train
```

From a local checkout:

```bash
uv pip install .
```

`hommi-train` also installs a console-script entry point, but this README uses the module form consistently:

```bash
python -m hommi_train --help
```

Inside a uv-managed project, use:

```bash
uv run python -m hommi_train --help
```

> `hommi-dataset` and `hommi-diffusion-policy` must also be published/installable before a registry install of `hommi-train` can resolve them. For local repository development, this project keeps uv Git sources for those dependencies.

## Quick start

### 1. Generate a task YAML from HDF5

```bash
python -m hommi_train init-config \
  -i data/pick_place.hdf5 \
  -o configs/tasks/pick_place.yaml \
  --name pick_place
```

With uv:

```bash
uv run python -m hommi_train init-config \
  -i data/pick_place.hdf5 \
  -o configs/tasks/pick_place.yaml \
  --name pick_place
```

The generated YAML contains:

- dataset path and metadata
- arm order and dataset rate
- complete `shape_meta`
- HoMMI-aligned dataset/training/DiT/DDIM defaults
- encoder augmentation settings
- runtime accelerator settings
- evaluation/TensorRT settings
- export settings

### 2. Train from YAML

```bash
python -m hommi_train \
  -c configs/tasks/pick_place.yaml \
  -o runs/pick_place
```

The HDF5 path can still be overridden explicitly:

```bash
python -m hommi_train \
  -c configs/tasks/pick_place.yaml \
  -i /datasets/pick_place_v2.hdf5 \
  -o runs/pick_place_v2
```

CLI values override YAML values for ordinary hyperparameters:

```bash
python -m hommi_train \
  -c configs/tasks/pick_place.yaml \
  -o runs/pick_place_bs32 \
  --batch-size 32 \
  --lr 1e-4
```

For uv-managed environments, prefix the same commands with `uv run`:

```bash
uv run python -m hommi_train \
  -c configs/tasks/pick_place.yaml \
  -o runs/pick_place
```

Shape-changing settings such as image size or horizons are validated against the YAML `shape_meta`. Regenerate the task YAML after changing them.

### 3. Evaluate with TensorRT

On CUDA, `backend=auto` selects TensorRT when the required Torch-TensorRT runtime imports successfully. You can request it explicitly:

```bash
python -m hommi_train eval \
  -i data/pick_place.hdf5 \
  -m runs/pick_place/model.pt \
  --mode full \
  --device cuda \
  --backend tensorrt
```

With uv:

```bash
uv run python -m hommi_train eval \
  -i data/pick_place.hdf5 \
  -m runs/pick_place/model.pt \
  --mode full \
  --device cuda \
  --backend tensorrt
```

TensorRT compilation targets the compute-heavy vision backbone and ActionDiT denoiser. DDIM scheduler orchestration stays in PyTorch.

### 4. Real-time inference / deployment from Python

For a robot process you normally load `model.pt` once, configure TensorRT once,
and call `policy.predict_action()` for every synchronized observation.  A minimal
single-iteration example is included at
[`examples/deploy.py`](https://github.com/umeow0716/hommi_train/blob/main/examples/deploy.py).
It deliberately uses black NumPy images and zero EEF positions so the input
shapes and action semantics are visible without requiring a camera or robot SDK.
The real dataset represents the EEF observation history relative to its newest
EEF pose, so a stationary dummy history naturally has zero position and identity
rotation-6D:

```python
from contextlib import nullcontext

import numpy as np
import torch

from hommi_train import (
    configure_evaluation_backend,
    hommi_train_config_from_mapping,
    load_portable_policy,
    resolve_device,
    resolve_precision,
)

model_path = "runs/pick_place/model.pt"
device = resolve_device("cuda")
policy, artifact = load_portable_policy(model_path, device=device)
config = hommi_train_config_from_mapping(artifact["config"])
precision = resolve_precision("auto", device)

configure_evaluation_backend(
    policy,
    backend="tensorrt",
    device=device,
    compile_mode=config.evaluation.compile_mode,
    tensorrt=config.evaluation.tensorrt,
    precision=precision,
)

# Default single-arm shape_meta: image [3,224,224], obs horizon 2.
black_rgb = np.zeros((2, 224, 224, 3), dtype=np.uint8)
rgb = torch.from_numpy(black_rgb).permute(0, 3, 1, 2).float() / 255.0

# Position is zero. Rotation-6D uses identity R; six zeros are not a valid rotation.
position = torch.zeros((2, 3), dtype=torch.float32)
rotation6d = torch.tensor(
    [[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, 1, 0]],
    dtype=torch.float32,
)
gripper = torch.zeros((2, 1), dtype=torch.float32)

obs = {
    "camera0_main_rgb": rgb.unsqueeze(0).to(device),        # [1,2,3,224,224]
    "robot0_eef_pos": position.unsqueeze(0).to(device),    # [1,2,3]
    "robot0_eef_rot_axis_angle": rotation6d.unsqueeze(0).to(device),  # [1,2,6]
    "robot0_gripper_width": gripper.unsqueeze(0).to(device),          # [1,2,1]
}

autocast = (
    torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "bf16"
    else nullcontext()
)
with torch.inference_mode(), autocast:
    prediction = policy.predict_action(obs)

# Default single-arm result: [B=1, n_action_steps=8, action_dim=10].
action_chunk = prediction["action"]
first_action = action_chunk[0, 0].float().cpu().numpy()

# One 10-D step is:
# [rel_x, rel_y, rel_z,
#  r00, r01, r02, r10, r11, r12,
#  gripper]
#
# Example only:
# [0.012, -0.004, 0.020, 0.999, 0.010, -0.030,
#  -0.009, 1.000, 0.004, 0.72]
print(first_action)
```

The XYZ and rotation-6D fields describe a future EEF pose **relative to the last
observation EEF frame**.  For a single-arm controller, reconstruct `T_rel` from
the first 9 values and apply:

```text
T_world_target = T_world_current @ T_rel
```

Then send `T_world_target` to the robot pose controller and the final scalar to
the gripper controller (`0 = closed`, `1 = open`).  `examples/deploy.py` contains
the complete `rotation6D -> R -> T_rel -> T_world_target` code.

For real observations, keep the last `obs_horizon` EEF world transforms and
express that history relative to the newest transform before inference (the
library's `hommi_train.dataset.relative_pose9()` uses the same convention as the
training dataset).  Likewise, every row in one predicted action chunk is relative
to the **same EEF frame at prediction time**; the rows are future target poses,
not transforms that should be chained together.

In a normal receding-horizon loop, acquire a fresh synchronized camera/robot
observation, run `predict_action()`, execute the first action (or a short prefix),
and repeat.

For offline validation-dataset metrics, keep using `run_evaluation()` or
[`examples/eval.py`](https://github.com/umeow0716/hommi_train/blob/main/examples/eval.py).

## Task YAML

A generated task file has the following top-level structure:

```yaml
format_version: 1

task:
  name: pick_place
  dataset_path: ../../data/pick_place.hdf5
  dataset_type: single-arm
  hz: 20.0
  arm_order: [left]
  shape_meta:
    image_resolution: 224
    obs:
      camera0_main_rgb:
        shape: [3, 224, 224]
        horizon: 2
        type: rgb
        ignore_by_policy: false
      robot0_eef_pos:
        shape: [3]
        horizon: 2
        type: low_dim
        ignore_by_policy: false
    action:
      shape: [10]
      horizon: 16
      rotation_rep: rotation_6d

config:
  dataset: {...}
  training: {...}
  model:
    encoder: {...}
  ddim: {...}
  runtime: {...}
  evaluation: {...}
  export: {...}
```

A complete editable reference lives at [`configs/default.yaml`](https://github.com/umeow0716/hommi_train/blob/main/configs/default.yaml).

Relative `task.dataset_path` values are resolved relative to the YAML file, so task configs can be committed to the repository and moved with the project.

Task YAML loading is strict: unknown or misspelled configuration keys are rejected instead of silently falling back to defaults. Checkpoint loading remains backward-compatible and migrates the older 0.3–0.5 encoder fields automatically.

## Configuration reference

### `DatasetConfig`

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `obs_horizon` | `int` | `2` | Observation history length |
| `action_horizon` | `int` | `16` | Predicted action horizon |
| `image_size` | `int` | `224` | Square RGB model input |
| `val_ratio` | `float` | `0.05` | Episode-level validation ratio |
| `action_padding` | `bool` | `true` | Pad action windows at episode end |
| `frame_cache` | `none\|lru\|ram` | `ram` | Decoded image working-set cache |
| `frame_cache_size` | `int` | `2048` | LRU capacity when using `lru` |
| `frame_preload_batch_size` | `int` | `8` | Batch size used while preloading RAM cache |

The persistent RAM frame cache stores resized `uint8 CHW` frames, not decoded 1920×1440 float images.

### `DiTObsEncoderConfig` (`hommi-diffusion-policy`)

Encoder settings are defined by the reusable policy package and passed through by `hommi-train`.

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `model_name` | `str` | `vit_base_patch16_clip_224.openai` | timm backbone |
| `pretrained` | `bool` | `true` | Initialize pretrained backbone for new runs |
| `frozen` | `bool` | `false` | Freeze backbone parameters |
| `global_pool` | `str` | `""` | timm global pooling argument |
| `feature_aggregation` | `cls\|avg\|max` | `cls` | Token feature aggregation |
| `use_group_norm` | `bool` | `true` | HoMMI compatibility setting |
| `share_rgb_model` | `bool` | `true` | Share one backbone across RGB streams |
| `use_vision_norm` | `bool` | `true` | Apply timm pretrained mean/std |
| `train_crop_ratio` | `float` | `0.95` | RandomCrop ratio during training |
| `eval_crop_ratio` | `float` | `0.95` | CenterCrop ratio during evaluation |
| `color_jitter_brightness` | `float` | `0.3` | Training ColorJitter brightness |
| `color_jitter_contrast` | `float` | `0.4` | Training ColorJitter contrast |
| `color_jitter_saturation` | `float` | `0.5` | Training ColorJitter saturation |
| `color_jitter_hue` | `float` | `0.08` | Training ColorJitter hue |

These defaults mirror the HoMMI single-task image augmentation recipe. The crop ratios are no longer hard-coded inside `hommi-train`.

### `DiTModelConfig`

| Parameter | Type | Default |
|---|---|---:|
| `encoder` | `DiTObsEncoderConfig` | HoMMI encoder defaults |
| `n_action_steps` | `int` | `8` |
| `num_inference_steps` | `int` | `16` |
| `attention_embed_dim` | `int` | `768` |
| `diffusion_timestep_embed_dim` | `int` | `256` |
| `depth` | `int` | `8` |
| `num_heads` | `int` | `8` |
| `mlp_ratio` | `float` | `4.0` |
| `train_diffusion_n_samples` | `int` | `8` |
| `qkv_bias` | `bool` | `true` |
| `use_rms_norm` | `bool` | `true` |
| `input_perturbation` | `float` | `0.0` |
| `obs_as_global_cond` | `bool` | `true` |
| `use_flow_matching` | `bool` | `false` |
| `fm_tsampler` | `uniform\|beta` | `uniform` |

### `DDIMConfig`

| Parameter | Type | Default |
|---|---|---:|
| `num_train_timesteps` | `int` | `50` |
| `beta_start` | `float` | `0.0001` |
| `beta_end` | `float` | `0.02` |
| `beta_schedule` | `str` | `squaredcos_cap_v2` |
| `clip_sample` | `bool` | `true` |
| `set_alpha_to_one` | `bool` | `true` |
| `steps_offset` | `int` | `0` |
| `prediction_type` | `str` | `epsilon` |

### `DiTTrainConfig`

| Parameter | Type | Default |
|---|---|---:|
| `batch_size` | `int` | `16` |
| `num_workers` | `int` | `8` |
| `epochs` | `int` | `1000` |
| `lr` | `float` | `7.5e-5` |
| `obs_encoder_lr` | `float` | `7.5e-6` |
| `weight_decay` | `float` | `1e-6` |
| `obs_encoder_weight_decay` | `float` | `1e-6` |
| `warmup_steps` | `int` | `50` |
| `clip_grad_norm` | `float` | `5.0` |
| `sample_every` | `int` | `10` |
| `log_grad_norm_every` | `int` | `50` |
| `seed` | `int` | `42` |
| `betas` | `[float, float]` | `[0.95, 0.999]` |
| `precision` | `auto\|fp32\|bf16` | `auto` |
| `pin_memory` | `auto\|bool` | `auto` |
| `persistent_workers` | `bool` | `true` |
| `drop_last` | `bool` | `true` |
| `keep_best_k` | `int` | `3` |

### `RuntimeConfig`

| Parameter | Type | Default | Behavior |
|---|---|---:|---|
| `device` | `str` | `auto` | Uses PyTorch's current available accelerator, otherwise CPU |
| `video_device` | `cpu\|cuda` | `cpu` | TorchCodec decoder device |
| `decoder_cache_size` | `int` | `4` | Open video decoder LRU capacity |
| `video_seek_mode` | `exact\|approximate` | `exact` | TorchCodec seek mode |
| `video_num_threads` | `int` | `1` | FFmpeg threads per decoder |
| `progress` | `bool` | `true` | Show dataset preload, normalizer, and training progress |

Hardware-aware defaults intentionally affect runtime only:

```text
device=auto
  -> torch.accelerator current available device
  -> CPU fallback

precision=auto
  -> CUDA + BF16 support: BF16
  -> otherwise: FP32

pin_memory=auto
  -> CUDA: true
  -> otherwise: false
```

`auto` deliberately does **not** guess batch size, learning rate, or worker count. Those depend on model shape, dataset, VRAM, storage, and the training objective; the package keeps the HoMMI recipe defaults unless you override them in YAML or on the CLI.

Batch size, learning rate, and worker count are **not** guessed from hardware because there is no universally optimal value without workload-specific benchmarking.

### `EvaluationConfig`

| Parameter | Type | Default |
|---|---|---:|
| `mode` | `sampled\|full` | `sampled` |
| `seed` | `int` | `42` |
| `precision` | `auto\|fp32\|bf16` | `auto` |
| `backend` | `auto\|eager\|inductor\|tensorrt` | `auto` |
| `compile_mode` | `str` | `reduce-overhead` |
| `tensorrt` | `TensorRTConfig` | see below |

### `TensorRTConfig`

| Parameter | Type | Default | Description |
|---|---|---:|---|
| `min_block_size` | `int` | `5` | Minimum TensorRT-capable operator block |
| `optimization_level` | `int` | `3` | TensorRT compilation optimization level |
| `dynamic` | `bool` | `false` | Compile dynamic shapes |
| `compile_backbone` | `bool` | `true` | TensorRT compile timm vision backbone(s) |
| `compile_denoiser` | `bool` | `true` | TensorRT compile ActionDiT denoiser |

TensorRT compilation deliberately targets the compute-heavy vision backbone and ActionDiT module. The DDIM scheduler loop remains PyTorch, avoiding unnecessary coupling between Python scheduler orchestration and TensorRT graph conversion. With `precision: bf16` (including `precision: auto` on a BF16-capable CUDA device), the TensorRT backend enables graph-aware BF16 autocast for eligible TensorRT segments while retaining sensitive operations in FP32. The `torch.compile(..., backend="torch_tensorrt")` path is JIT: the first matching batch shape pays compilation cost, while later batches in the same process reuse the compiled engine.

### `ExportConfig`

| Parameter | Type | Default |
|---|---|---:|
| `auto_export` | `bool` | `true` |
| `source` | `best\|last` | `best` |
| `artifact_name` | `str` | `model.pt` |

## Training outputs and resume

A normal run produces:

```text
runs/pick_place/
├── checkpoints/
│   ├── last.pt
│   ├── best.pt
│   └── epoch=....pt
└── model.pt
```

`checkpoints/*.pt` contain model, EMA, optimizer, scheduler, trainer progress, RNG state, config, and train/validation split.

Resume:

```bash
python -m hommi_train \
  -c configs/tasks/pick_place.yaml \
  -o runs/pick_place \
  --resume runs/pick_place/checkpoints/last.pt
```

The checkpoint's saved training config and episode split are authoritative during resume; explicit CLI options can still override non-shape hyperparameters.

`model.pt` is the compact EMA inference artifact and excludes optimizer/scheduler/training RNG state.

## Dataset representation

Input HDF5 layout (`hommi-dataset >= 0.2`):

```text
hz                  float32
type                str
arm_order           str[A]
episode_001/
  action             float32[N, A*8]
  video_index        int32[N, A]
  video/
    left             uint8[original H.264 MP4 bytes]
    right            uint8[...]   # dual-arm
```

Stored action per arm:

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

Network action per arm:

```text
[relative position(3), rotation6d(6), gripper(1)]
```

Image pipeline:

```text
HDF5 H.264
  -> referenced source frames only
  -> decode once
  -> deterministic black-pad-to-square
  -> resize to image_size
  -> uint8 RAM cache
  -> float [0,1] sample
  -> encoder RandomCrop/ColorJitter (train)
     or CenterCrop (eval)
```

## Programmatic API

```python
from hommi_train import load_task_config, run_training

spec = load_task_config("configs/tasks/pick_place.yaml")
run_training(
    spec.resolve_dataset_path(),
    "runs/pick_place",
    spec.config,
)
```

Encoder configuration comes directly from the policy module:

```python
from hommi_diffusion_policy import DiTObsEncoderConfig
from hommi_train import DiTModelConfig

encoder = DiTObsEncoderConfig(
    train_crop_ratio=0.9,
    eval_crop_ratio=1.0,
    color_jitter_brightness=0.2,
)
model = DiTModelConfig(encoder=encoder)
```

## Repository layout

```text
configs/
├── default.yaml
└── tasks/

examples/
├── deploy.py              # portable-model / TensorRT real-robot example
├── deploy_tensorrt.py     # precompiled model.trt.ep deployment example
├── benchmark_tensorrt.py  # end-to-end TensorRT inference latency / Hz
└── eval.py                # offline validation-dataset evaluation

src/hommi_train/
├── configuration/   # task YAML generation/loading
├── dataset/         # HDF5, geometry, video, frame cache, split
├── normalization/   # training-split statistics
├── policy/          # policy factory
├── runtime/         # accelerator/device/precision resolution
├── training/        # Trainer, EMA, optimizer, checkpoints
├── evaluation/      # eager/Inductor/TensorRT evaluation
├── export/          # portable model + PT2 utilities
├── config.py
├── runner.py
└── cli.py
```

## Tests

```bash
uv sync --extra dev
uv run python -m pytest
```

### Build a TensorRT deployment artifact

Compile the portable `model.pt` inside a completed run directory on the target CUDA/TensorRT machine:

```bash
uv run python -m hommi_train tensorrt -i runs/pick_place
```

This reads `runs/pick_place/model.pt` and writes:

```text
runs/pick_place/model.trt.ep
runs/pick_place/model.trt.ep.json
```

The default deployment batch size is 1. Optional overrides:

```bash
uv run python -m hommi_train tensorrt \
  -i runs/pick_place \
  -o runs/pick_place/model.trt.ep \
  --device cuda:0 \
  --precision auto \
  --batch-size 1
```

TensorRT artifacts are hardware/runtime specific. Build this artifact on the deployment target (for example, a Jetson Orin Nano with its JetPack-compatible PyTorch and Torch-TensorRT stack) rather than assuming a desktop-built TensorRT program is portable to Jetson.

Deploy the precompiled bundle with the standalone example:

```bash
uv run python examples/deploy_tensorrt.py \
  -m runs/pick_place/model.trt.ep
```

Estimate end-to-end policy inference latency and throughput on the current host:

```bash
uv run python examples/benchmark_tensorrt.py \
  -m runs/pick_place/model.trt.ep \
  --warmup 10 \
  --iterations 100
```

The benchmark measures `policy.predict_action(obs)`, including the eager DDIM loop and TensorRT ViT/DiT submodules. It synchronizes CUDA around every timed iteration. The reported Hz therefore estimates model inference throughput only; camera capture/decoding, robot I/O, and other control-loop work are excluded.

## License

See the repository license and third-party notices where applicable.
