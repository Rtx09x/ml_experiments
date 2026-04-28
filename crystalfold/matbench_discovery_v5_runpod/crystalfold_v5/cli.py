from __future__ import annotations

import argparse

from .data import run_preprocess
from .train import run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CrystalFold V5 RunPod CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess", help="Download official data and build processed HDF5 artifacts")
    p.add_argument("--root", required=True, help="Shared network-volume root for raw and processed data")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--snapshots", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument("--max-materials", type=int, default=None)
    p.add_argument("--include-wbm", action="store_true")
    p.add_argument("--max-wbm", type=int, default=None)
    p.set_defaults(func=run_preprocess)

    t = sub.add_parser("train", help="Train CrystalFold V5 from a processed HDF5 dataset")
    t.add_argument("--dataset", required=True)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--batch-size", type=int, default=192)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--train-frac", type=float, default=0.9)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    t.add_argument("--compile", action="store_true")
    t.add_argument("--energy-only-epochs", type=int, default=0)
    t.add_argument("--force-every", type=int, default=1)
    t.add_argument("--val-max-batches", type=int, default=None)
    t.add_argument("--local-copy", default=None)
    t.add_argument("--resume", default=None)
    t.set_defaults(func=run_train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
