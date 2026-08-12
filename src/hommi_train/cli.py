from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import inspect_hommi_hdf5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hommi-train",
        description="HoMMI training pipeline (composition milestone).",
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="HoMMI HDF5 dataset")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/hommi"),
        help="training output directory (used by the upcoming trainer milestone)",
    )
    parser.add_argument(
        "--inspect-dataset",
        action="store_true",
        help="validate the HDF5 schema and print dataset metadata",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.inspect_dataset:
        raise SystemExit(
            "Dataset, split, normalization, and policy composition are implemented; "
            "the training runner is the next milestone. "
            "Use --inspect-dataset for now."
        )

    info = inspect_hommi_hdf5(args.input)
    print(f"path: {info.path}")
    print(f"type: {info.dataset_type}")
    print(f"hz: {info.hz:g}")
    print(f"arms: {', '.join(info.arm_order)}")
    print(f"episodes: {info.num_episodes}")
    print(f"samples: {info.num_samples}")
    return 0
