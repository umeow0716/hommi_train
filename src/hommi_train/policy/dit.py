from __future__ import annotations

from typing import Any

from ..config import DDIMConfig, DiTModelConfig


def _observation_horizon(shape_meta: dict[str, Any]) -> int:
    obs_meta = shape_meta.get("obs")
    if not isinstance(obs_meta, dict) or not obs_meta:
        raise ValueError("shape_meta must contain non-empty 'obs' metadata")

    horizons = {
        int(value["horizon"])
        for value in obs_meta.values()
        if not value.get("ignore_by_policy", False)
    }
    if len(horizons) != 1:
        raise ValueError(
            "all active observations must share one horizon for DiT; "
            f"got {sorted(horizons)}"
        )
    horizon = horizons.pop()
    if horizon < 1:
        raise ValueError("observation horizon must be >= 1")
    return horizon


def build_ddim_scheduler(config: DDIMConfig | None = None):
    """Build the HoMMI-aligned DDIM scheduler lazily."""
    from diffusers import DDIMScheduler

    cfg = config or DDIMConfig()
    return DDIMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        clip_sample=cfg.clip_sample,
        set_alpha_to_one=cfg.set_alpha_to_one,
        steps_offset=cfg.steps_offset,
        prediction_type=cfg.prediction_type,
    )


def build_dit_policy(
    shape_meta: dict[str, Any],
    *,
    model_config: DiTModelConfig | None = None,
    ddim_config: DDIMConfig | None = None,
    name: str = "diffusion_dit",
    pretrained_override: bool | None = None,
):
    """Construct the HoMMI 2D DiT policy from canonical dataset metadata.

    ``hommi_train`` owns this construction recipe; the actual encoder and policy
    implementations remain in ``hommi_diffusion_policy``. ``pretrained_override``
    is used when reconstructing a fully saved checkpoint/artifact so timm does
    not download or initialize pretrained weights that will immediately be
    overwritten by the saved state dict.
    """
    from hommi_diffusion_policy import DiTObsEncoderConfig, DiTObsEncoderLite, DiffusionDiTImagePolicy

    cfg = model_config or DiTModelConfig()
    action_meta = shape_meta.get("action")
    if not isinstance(action_meta, dict):
        raise ValueError("shape_meta must contain 'action' metadata")
    action_horizon = int(action_meta.get("horizon", 0))
    if action_horizon < 1:
        raise ValueError("shape_meta action horizon must be >= 1")
    if not 1 <= cfg.n_action_steps <= action_horizon:
        raise ValueError(
            f"n_action_steps must satisfy 1 <= n_action_steps <= {action_horizon}"
        )

    obs_horizon = _observation_horizon(shape_meta)
    encoder_cfg = cfg.encoder
    if pretrained_override is not None:
        encoder_cfg = DiTObsEncoderConfig(
            **{
                field: (
                    bool(pretrained_override)
                    if field == "pretrained"
                    else getattr(encoder_cfg, field)
                )
                for field in encoder_cfg.__dataclass_fields__
            }
        )
    obs_encoder = DiTObsEncoderLite(shape_meta, config=encoder_cfg)
    scheduler = build_ddim_scheduler(ddim_config)
    return DiffusionDiTImagePolicy(
        name=name,
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=obs_encoder,
        horizon=action_horizon,
        n_action_steps=cfg.n_action_steps,
        n_obs_steps=obs_horizon,
        num_inference_steps=cfg.num_inference_steps,
        obs_as_global_cond=cfg.obs_as_global_cond,
        train_diffusion_n_samples=cfg.train_diffusion_n_samples,
        attention_embed_dim=cfg.attention_embed_dim,
        diffusion_timestep_embed_dim=cfg.diffusion_timestep_embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        qkv_bias=cfg.qkv_bias,
        use_rms_norm=cfg.use_rms_norm,
        input_perturbation=cfg.input_perturbation,
        use_flow_matching=cfg.use_flow_matching,
        fm_tsampler=cfg.fm_tsampler,
    )
