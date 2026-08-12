from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Mapping, TypeVar

from hommi_diffusion_policy import DiTObsEncoderConfig


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """HoMMI / UMI effective 20 Hz dataset defaults."""

    obs_horizon: int = 2
    action_horizon: int = 16
    image_size: int = 224
    val_ratio: float = 0.05
    action_padding: bool = True
    frame_cache: Literal["none", "lru", "ram"] = "ram"
    frame_cache_size: int = 2048
    frame_preload_batch_size: int = 8


@dataclass(frozen=True, slots=True)
class DiTTrainConfig:
    """HoMMI-aligned optimization defaults plus runtime-neutral loader settings."""

    batch_size: int = 16
    num_workers: int = 8
    epochs: int = 1000
    lr: float = 7.5e-5
    obs_encoder_lr: float = 7.5e-6
    weight_decay: float = 1.0e-6
    obs_encoder_weight_decay: float = 1.0e-6
    warmup_steps: int = 50
    clip_grad_norm: float = 5.0
    sample_every: int = 10
    log_grad_norm_every: int = 50
    seed: int = 42
    betas: tuple[float, float] = (0.95, 0.999)

    # ``auto`` is resolved against the selected accelerator at runtime.
    precision: Literal["auto", "fp32", "bf16"] = "auto"
    pin_memory: bool | Literal["auto"] = "auto"
    persistent_workers: bool = True
    drop_last: bool = True

    keep_best_k: int = 3


@dataclass(frozen=True, slots=True)
class DDIMConfig:
    """Noise-scheduler defaults used by HoMMI's DiT training recipe."""

    num_train_timesteps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    prediction_type: str = "epsilon"


@dataclass(frozen=True, slots=True)
class DiTModelConfig:
    """Diffusion-policy architecture config.

    Vision-backbone and augmentation settings live in
    :class:`hommi_diffusion_policy.DiTObsEncoderConfig`, so model behavior has a
    single source of truth in the reusable policy package.
    """

    encoder: DiTObsEncoderConfig = field(default_factory=DiTObsEncoderConfig)
    n_action_steps: int = 8
    num_inference_steps: int = 16
    attention_embed_dim: int = 768
    diffusion_timestep_embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    train_diffusion_n_samples: int = 8
    qkv_bias: bool = True
    use_rms_norm: bool = True
    input_perturbation: float = 0.0
    obs_as_global_cond: bool = True
    use_flow_matching: bool = False
    fm_tsampler: Literal["uniform", "beta"] = "uniform"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Machine/runtime choices that are not part of the learned architecture."""

    # ``auto`` uses torch.accelerator.current_accelerator(check_available=True)
    # and falls back to CPU.
    device: str = "auto"
    video_device: Literal["cpu", "cuda"] = "cpu"
    decoder_cache_size: int = 4
    video_seek_mode: Literal["exact", "approximate"] = "exact"
    video_num_threads: int = 1
    progress: bool = True


@dataclass(frozen=True, slots=True)
class TensorRTConfig:
    """Torch-TensorRT settings for evaluation acceleration."""

    min_block_size: int = 5
    optimization_level: int = 3
    dynamic: bool = False
    compile_backbone: bool = True
    compile_denoiser: bool = True


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Standalone evaluation defaults."""

    mode: Literal["sampled", "full"] = "sampled"
    seed: int = 42
    precision: Literal["auto", "fp32", "bf16"] = "auto"
    backend: Literal["auto", "eager", "inductor", "tensorrt"] = "auto"
    compile_mode: str = "reduce-overhead"
    tensorrt: TensorRTConfig = field(default_factory=TensorRTConfig)


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Post-training portable artifact policy."""

    auto_export: bool = True
    source: Literal["best", "last"] = "best"
    artifact_name: str = "model.pt"


@dataclass(frozen=True, slots=True)
class HommiTrainConfig:
    """Hierarchical configuration carried into YAML files and checkpoints."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: DiTTrainConfig = field(default_factory=DiTTrainConfig)
    model: DiTModelConfig = field(default_factory=DiTModelConfig)
    ddim: DDIMConfig = field(default_factory=DDIMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


_ConfigT = TypeVar("_ConfigT")


def _config_kwargs(cls: type[_ConfigT], value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    known = {item.name for item in fields(cls)}
    return {key: item for key, item in value.items() if key in known}


def _encoder_config_from_mapping(value: Mapping[str, Any] | None) -> DiTObsEncoderConfig:
    if value is None:
        return DiTObsEncoderConfig()
    known = {item.name for item in fields(DiTObsEncoderConfig)}
    return DiTObsEncoderConfig(**{k: v for k, v in value.items() if k in known})




def _validate_mapping_fields(
    name: str,
    cls: type[Any],
    value: object,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")


def _validate_strict_config_mapping(value: Mapping[str, Any]) -> None:
    _validate_mapping_fields("config", HommiTrainConfig, value)
    _validate_mapping_fields("config.dataset", DatasetConfig, value.get("dataset"))
    _validate_mapping_fields("config.training", DiTTrainConfig, value.get("training"))
    _validate_mapping_fields("config.ddim", DDIMConfig, value.get("ddim"))
    _validate_mapping_fields("config.runtime", RuntimeConfig, value.get("runtime"))
    _validate_mapping_fields("config.export", ExportConfig, value.get("export"))

    raw_model = value.get("model")
    _validate_mapping_fields("config.model", DiTModelConfig, raw_model)
    if isinstance(raw_model, Mapping):
        _validate_mapping_fields(
            "config.model.encoder",
            DiTObsEncoderConfig,
            raw_model.get("encoder"),
        )

    raw_eval = value.get("evaluation")
    _validate_mapping_fields("config.evaluation", EvaluationConfig, raw_eval)
    if isinstance(raw_eval, Mapping):
        _validate_mapping_fields(
            "config.evaluation.tensorrt",
            TensorRTConfig,
            raw_eval.get("tensorrt"),
        )


def hommi_train_config_from_mapping(
    value: Mapping[str, Any] | None,
    *,
    strict: bool = False,
) -> HommiTrainConfig:
    """Reconstruct :class:`HommiTrainConfig` from YAML/checkpoint mappings.

    Older 0.3-0.5 checkpoints are migrated automatically: ``model_name`` and
    ``pretrained`` used to live directly in ``model`` and are now moved into
    ``model.encoder``. Missing sections use current defaults. By default unknown
    fields are ignored for checkpoint forward compatibility; ``strict=True`` is
    intended for user-authored YAML and rejects misspelled/unknown keys.
    """
    if value is None:
        return HommiTrainConfig()
    if not isinstance(value, Mapping):
        raise TypeError("training config must be a mapping")
    if strict:
        _validate_strict_config_mapping(value)

    training_kwargs = _config_kwargs(DiTTrainConfig, value.get("training"))
    if "betas" in training_kwargs:
        betas = training_kwargs["betas"]
        training_kwargs["betas"] = (float(betas[0]), float(betas[1]))

    raw_model = value.get("model")
    if not isinstance(raw_model, Mapping):
        raw_model = {}
    raw_encoder = raw_model.get("encoder")
    if not isinstance(raw_encoder, Mapping):
        raw_encoder = {}
    migrated_encoder = dict(raw_encoder)
    for legacy_name in ("model_name", "pretrained"):
        if legacy_name in raw_model and legacy_name not in migrated_encoder:
            migrated_encoder[legacy_name] = raw_model[legacy_name]
    model_kwargs = _config_kwargs(DiTModelConfig, raw_model)
    model_kwargs["encoder"] = _encoder_config_from_mapping(migrated_encoder)

    raw_eval = value.get("evaluation")
    if not isinstance(raw_eval, Mapping):
        raw_eval = {}
    eval_kwargs = _config_kwargs(EvaluationConfig, raw_eval)
    raw_trt = raw_eval.get("tensorrt")
    eval_kwargs["tensorrt"] = TensorRTConfig(
        **_config_kwargs(TensorRTConfig, raw_trt if isinstance(raw_trt, Mapping) else None)
    )

    return HommiTrainConfig(
        dataset=DatasetConfig(**_config_kwargs(DatasetConfig, value.get("dataset"))),
        training=DiTTrainConfig(**training_kwargs),
        model=DiTModelConfig(**model_kwargs),
        ddim=DDIMConfig(**_config_kwargs(DDIMConfig, value.get("ddim"))),
        runtime=RuntimeConfig(**_config_kwargs(RuntimeConfig, value.get("runtime"))),
        evaluation=EvaluationConfig(**eval_kwargs),
        export=ExportConfig(**_config_kwargs(ExportConfig, value.get("export"))),
    )
