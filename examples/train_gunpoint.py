"""Train mixup contrastive learning on the GunPoint dataset.

Reproduces the example from the original notebook using the package API:
python examples/train_gunpoint.py [--epochs 200] [--alpha 1.0]
"""

import argparse

from mcl.data import TimeSeriesDataset, load_dataset
from mcl.models import FCN
from mcl.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GunPoint")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_tr, y_tr = load_dataset(args.dataset, split="train")
    x_te, y_te = load_dataset(args.dataset, split="test")
    train_set = TimeSeriesDataset(x_tr, y_tr)
    test_set = TimeSeriesDataset(x_te, y_te)
    print(
        f"{args.dataset}: {len(train_set)} train / {len(test_set)} test, "
        f"{train_set.n_channels} channel(s), length {train_set.n_timepoints}"
    )

    model = FCN(train_set.n_channels)
    history = train(model, train_set, test_set, epochs=args.epochs, alpha=args.alpha)

    if history["accuracies"]:
        print(f"Final 1-NN test accuracy: {history['accuracies'][-1]:.4f}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(history["epoch_losses"])
        ax1.set_title("MNT-Xent loss")
        ax1.set_xlabel("Epoch")
        ax2.plot(history["accuracies"])
        ax2.set_title("1-NN test accuracy")
        ax2.set_xlabel("Epoch")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
