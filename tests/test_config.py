from __future__ import annotations

from hommi_train.config import hommi_train_config_from_mapping


def test_legacy_model_encoder_fields_migrate_into_encoder_config() -> None:
    cfg = hommi_train_config_from_mapping(
        {
            "model": {
                "model_name": "vit_tiny_patch16_224",
                "pretrained": False,
                "depth": 4,
            }
        }
    )

    assert cfg.model.encoder.model_name == "vit_tiny_patch16_224"
    assert cfg.model.encoder.pretrained is False
    assert cfg.model.depth == 4
    assert cfg.model.encoder.train_crop_ratio == 0.95


def test_strict_config_rejects_unknown_yaml_fields() -> None:
    try:
        hommi_train_config_from_mapping(
            {"training": {"batch_szie": 32}},
            strict=True,
        )
    except ValueError as exc:
        assert "batch_szie" in str(exc)
    else:
        raise AssertionError("expected strict config typo rejection")
