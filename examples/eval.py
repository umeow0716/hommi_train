"""Evaluate a trained HoMMI portable model with TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

from hommi_train import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True, help="HoMMI HDF5 dataset")
    parser.add_argument("-m", "--model", type=Path, required=True, help="portable model.pt")
    parser.add_argument("--mode", choices=("sampled", "full"), default="full")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        args.input,
        args.model,
        mode=args.mode,
        device=args.device,
        precision="auto",
        backend="tensorrt",
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
