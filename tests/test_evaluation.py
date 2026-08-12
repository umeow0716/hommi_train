from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hommi_train.evaluation import evaluate_policy, save_evaluation_result


class _EvalDataset(Dataset):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int):
        x = torch.full((2, 2), float(index), dtype=torch.float32)
        return {"obs": {"x": x}, "action": x + 1.0}


class _EvalPolicy(nn.Module):
    def predict_action(self, obs):
        return {"action_pred": obs["x"] + 1.0}


def test_sampled_evaluation_consumes_one_batch() -> None:
    loader = DataLoader(_EvalDataset(), batch_size=2, drop_last=False)
    result = evaluate_policy(
        _EvalPolicy(), loader, device="cpu", mode="sampled", precision="fp32"
    )
    assert result.action_mse == 0.0
    assert result.num_batches == 1
    assert result.num_samples == 2


def test_full_evaluation_includes_incomplete_final_batch(tmp_path: Path) -> None:
    loader = DataLoader(_EvalDataset(), batch_size=2, drop_last=False)
    result = evaluate_policy(
        _EvalPolicy(), loader, device="cpu", mode="full", precision="fp32"
    )
    assert result.action_mse == 0.0
    assert result.num_batches == 3
    assert result.num_samples == 5
    path = save_evaluation_result(result, tmp_path / "evaluation.json")
    assert path.exists()
    assert '"num_samples": 5' in path.read_text()
