"""Command line interface for mixupcl."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import TimeSeriesDataset, load_dataset
from .models import FCN
from .trainer import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcl",
        description="Mixup contrastive learning for time series "
        "(Wickstrøm et al., Pattern Recognition Letters, 2022).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "train", help="Train on a UCR/UEA dataset and report 1-NN test accuracy."
    )
    p.add_argument(
        "--dataset",
        default="GunPoint",
        help="Dataset name from the UCR/UEA archives (default: "
        "GunPoint). Downloaded and cached via sktime.",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Beta(alpha, alpha) mixing parameter (default: 1.0).",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--tau", type=float, default=0.5, help="MNT-Xent temperature (default: 0.5)."
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Minibatch size; 0 uses the full training set as a "
        "single batch, as in the paper (default: 0).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Compute device, e.g. 'cpu', 'cuda', 'mps' (default: auto-detect).",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility."
    )
    p.add_argument(
        "--save-history",
        default=None,
        metavar="PATH",
        help="Save per-epoch loss/accuracy history to a CSV file.",
    )
    return parser


def _train_command(args: argparse.Namespace) -> None:
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    x_tr, y_tr = load_dataset(args.dataset, split="train")
    x_te, y_te = load_dataset(args.dataset, split="test")
    train_set = TimeSeriesDataset(x_tr, y_tr)
    test_set = TimeSeriesDataset(x_te, y_te)
    print(
        f"{args.dataset}: {len(train_set)} train / {len(test_set)} test "
        f"series, {train_set.n_channels} channel(s), "
        f"length {train_set.n_timepoints}"
    )

    model = FCN(train_set.n_channels)
    history = train(
        model,
        train_set,
        test_set,
        epochs=args.epochs,
        alpha=args.alpha,
        lr=args.lr,
        tau=args.tau,
        batch_size=args.batch_size or None,
        device=args.device,
    )

    if history["accuracies"]:
        print(f"Final 1-NN test accuracy: {history['accuracies'][-1]:.4f}")
    if args.save_history:
        _save_history(history, args.save_history)


def _save_history(history: dict, path: str) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy"])
        for i, loss in enumerate(history["epoch_losses"]):
            acc = history["accuracies"][i] if i < len(history["accuracies"]) else ""
            writer.writerow([i, loss, acc])
    print(f"History saved to {path}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        _train_command(args)


if __name__ == "__main__":
    main()
