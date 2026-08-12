# hommi-train

Training, evaluation, and deployment tooling for datasets produced by
`hommi-dataset` and policies from `hommi-diffusion-policy`.

Version **0.5.0** completes the original repository roadmap:

```text
HDF5 dataset
  -> episode split
  -> compact uint8 frame cache
  -> training-only normalizer fitting
  -> HoMMI DiT policy
  -> BF16 Trainer / EMA / checkpoints
  -> portable EMA model.pt
  -> sampled/full evaluation
  -> optional torch.compile / torch.export inference paths
```

## Repository layout

```text
src/hommi_train/
├── dataset/
│   ├── geometry.py
│   ├── schema.py
│   ├── split.py
│   ├── video.py
│   ├── frame_cache.py
│   └── hommi_hdf5.py
├── normalization/
│   └── hommi.py
├── policy/
│   └── dit.py
├── training/
│   ├── data.py
│   ├── optimizer.py
│   ├── ema.py
│   ├── metrics.py
│   ├── checkpoint.py
│   └── trainer.py
├── evaluation/
│   ├── evaluator.py
│   └── runner.py
├── export/
│   ├── artifact.py
│   ├── torch_export.py
│   └── runner.py
├── runner.py
├── config.py
├── cli.py
└── __main__.py
```

`hommi-diffusion-policy` remains the reusable model/runtime library.
`hommi-train` owns HDF5 layout, train/val split, normalization statistics,
training recipes, evaluation, and deployment artifact composition.

## Train

The original one-command interface remains unchanged:

```bash
python -m hommi_train \
    -i dataset.hdf5 \
    -o outputs/run-001
```

or:

```bash
hommi-train -i dataset.hdf5 -o outputs/run-001
```

The default configuration follows the HoMMI DiT recipe while runtime choices
such as BF16, pinned memory, persistent workers, and the uint8 RAM frame cache
remain explicitly separate optimizations.

Typical overrides:

```bash
python -m hommi_train \
    -i dataset.hdf5 \
    -o outputs/run-001 \
    --batch-size 32 \
    --lr 1e-4 \
    --epochs 1200
```

Resume restores model, EMA, optimizer, scheduler, RNG streams, progress, and the
original episode split:

```bash
python -m hommi_train \
    -i dataset.hdf5 \
    -o outputs/run-001 \
    --resume outputs/run-001/checkpoints/last.pt
```

When reconstructing a resume checkpoint, pretrained timm initialization is
skipped because the checkpoint already contains the complete vision-backbone
state.

## Output

By default a completed run contains:

```text
outputs/run-001/
├── checkpoints/
│   ├── last.pt
│   ├── best.pt
│   └── epoch=XXXX-val_action_mse_error=....pt
└── model.pt
```

`checkpoints/*.pt` are **training checkpoints** and contain optimizer/scheduler,
RNG, and trainer state.

`model.pt` is the **portable inference artifact**. It contains only the EMA
policy state plus the information required to reconstruct it:

```text
policy_state           EMA state_dict, including normalizer
shape_meta             observation/action contract
config                  model + DDIM + dataset/runtime metadata
val_episode_keys        reproducible validation split
metrics                 checkpoint metrics
provenance              epoch/global step/source checkpoint
```

It intentionally excludes optimizer, LR scheduler, training RNG state, and the
non-EMA training model.

The artifact is loaded through PyTorch's restricted `weights_only=True` loader.

### Post-training export controls

Portable EMA export is enabled by default and uses `best.pt` when available:

```text
--auto-export / --no-auto-export
--export-source best|last
--artifact-name model.pt
```

## Explicit export

A checkpoint can be stripped independently of training:

```bash
python -m hommi_train export \
    -c outputs/run-001/checkpoints/best.pt \
    -o outputs/run-001/model.pt
```

If `-o` is omitted and the checkpoint lives under `checkpoints/`, the default is
`../model.pt`.

### Experimental torch.export PT2

`model.pt` is the primary artifact. A static-batch `torch.export` artifact can be
attempted explicitly:

```bash
python -m hommi_train export \
    -c outputs/run-001/checkpoints/best.pt \
    -o outputs/run-001/model.pt \
    --pt2 outputs/run-001/model.pt2 \
    --device cuda \
    --batch-size 1
```

The PT2 wrapper exposes active observations as a positional tensor tuple in a
stable, metadata-recorded key order and returns the executable action chunk.
The `.pt2` archive embeds `hommi_metadata.json` with `obs_keys`, `shape_meta`,
and the static export batch size.

Full diffusion inference has to be exportable as one graph. If PT2 generation
fails, the command reports the exporter error while keeping the already-created
portable `model.pt`; it never silently creates a partial artifact.

## Evaluation

Evaluate the exact validation split saved in `model.pt`:

```bash
python -m hommi_train eval \
    -i dataset.hdf5 \
    -m outputs/run-001/model.pt
```

The default `sampled` mode consumes one validation batch to preserve the
historical HoMMI workspace metric behavior.

Full validation:

```bash
python -m hommi_train eval \
    -i dataset.hdf5 \
    -m outputs/run-001/model.pt \
    --mode full \
    -o outputs/run-001/evaluation.json
```

`full` evaluation uses `drop_last=False` and aggregates squared error by action
element, so the incomplete final batch is weighted correctly.

The result contains:

```json
{
  "mode": "full",
  "action_mse": 0.0,
  "num_batches": 0,
  "num_samples": 0,
  "num_action_values": 0,
  "device": "cuda",
  "precision": "bf16"
}
```

The numerical values above are only a schema example.

Diffusion evaluation uses a local forked RNG stream and a fixed seed, so running
evaluation does not perturb the caller's global Torch RNG state.

### CUDA runtime compilation

Portable `model.pt` remains ordinary PyTorch and can use runtime compilation:

```bash
python -m hommi_train eval \
    -i dataset.hdf5 \
    -m outputs/run-001/model.pt \
    --device cuda \
    --compile \
    --compile-mode reduce-overhead
```

Programmatic edge inference can use `build_inference_module(..., compile=True)`
to wrap the dictionary policy API behind a tensor-only module.

## Programmatic inference

```python
from hommi_train import (
    build_inference_module,
    load_portable_policy,
)

policy, artifact = load_portable_policy(
    "outputs/run-001/model.pt",
    device="cuda",
)

module = build_inference_module(
    policy,
    artifact["shape_meta"],
    compile=True,
)
```

The wrapper accepts a tuple of tensors in sorted active-observation-key order.
Use `active_observation_keys()` from `hommi_train.export` when building an edge
adapter.

## Dataset and frame cache

Input HDF5 layout (`hommi-dataset >= 0.2.0`):

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

Stored action per arm:

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

Model action per arm:

```text
[relative position(3), rotation6d(6), gripper(1)]
```

Default image path:

```text
HDF5 H.264
  -> unique observation-referenced source frames only
  -> decode once
  -> deterministic center-square crop
  -> resize 224x224
  -> uint8 CHW RAM cache
  -> per-sample float32 [0,1]
  -> DiT train RandomCrop/ColorJitter or eval CenterCrop
```

The train/validation split is decided before RAM preload and only the training
split is used for normalizer fitting.

## Key defaults

```text
obs_horizon                         2
action_horizon                     16
image_size                        224
val_ratio                         0.05
frame_cache                        ram

batch_size                          16
epochs                            1000
lr                               7.5e-5
obs_encoder_lr                    7.5e-6
warmup_steps                         50
clip_grad_norm                       5
precision                          bf16

model_name      vit_base_patch16_clip_224.openai
n_action_steps                       8
num_inference_steps                 16
attention_embed_dim                768
depth                                8
num_heads                            8
DDIM train timesteps                50
```

## Tests

The 0.5.0 suite covers dataset geometry/schema/video/cache, normalization,
splitting, policy composition, training/checkpoint resume, CLI config,
portable artifact stripping/loading, sampled/full evaluation, and a real
`torch.export -> save -> load -> execute` PT2 round-trip for an exportable
inference policy.

## Lock file

Regenerate the lock file in a networked checkout after unpacking:

```bash
uv lock
uv sync --extra dev
```
