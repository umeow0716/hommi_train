from __future__ import annotations

import torch

from hommi_train.runtime import resolve_device, resolve_pin_memory, resolve_precision


def test_auto_runtime_falls_back_to_cpu_in_cpu_test_environment() -> None:
    device = resolve_device("auto")
    assert device.type == "cpu"
    assert resolve_precision("auto", device) == "fp32"
    assert resolve_pin_memory("auto", device) is False
