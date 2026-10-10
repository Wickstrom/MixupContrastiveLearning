"""Training and evaluation loops for mixup contrastive learning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from .losses import MixupLoss

__all__ = ["embed", "evaluate", "mixup_batch", "resolve_device", "train"]


def resolve_device(device: str | None = None) -> str:
    """Return ``device`` if given, otherwise CUDA when available, else CPU."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def mixup_batch(x: torch.Tensor, alpha: float):
    """Mix a batch with a random permutation of itself.

    Draws ``lam ~ Beta(alpha, alpha)``, forms the mixed batch
    ``x_aug = lam * x + (1 - lam) * x[perm]`` and returns
    ``(x_aug, x_2, lam)`` where ``x_2 = x[perm]``.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    x_2 = x[torch.randperm(x.shape[0], device=x.device)]
    x_aug = lam * x + (1 - lam) * x_2
    return x_aug, x_2, lam


def train(
    model,
    train_set,
    test_set=None,
    epochs: int = 200,
    alpha: float = 1.0,
    lr: float = 1e-3,
    tau: float = 0.5,
    batch_size: int | None = None,
    optimizer=None,
    device: str | None = None,
    verbose: bool = True,
    log_every: int = 10,
) -> dict:
    """Train a model with the MNT-Xent loss.

    By default the full training set is used as a single batch per epoch,
    as in the original paper.

    Parameters
    ----------
    model : nn.Module
        Model mapping a batch to ``(z, h)`` (e.g. :class:`mcl.models.FCN`).
    train_set, test_set : TimeSeriesDataset
        Training data, and optional test data used to report 1-NN accuracy.
    epochs : int
        Number of training epochs.
    alpha : float
        Parameter of the ``Beta(alpha, alpha)`` mixing distribution.
    lr : float
        Learning rate used if ``optimizer`` is not given (Adam).
    tau : float
        Temperature of the MNT-Xent loss.
    batch_size : int, optional
        Minibatch size. ``None`` (default) uses the full training set.
    optimizer : torch.optim.Optimizer, optional
        Defaults to ``Adam(model.parameters(), lr=lr)``.
    device : str, optional
        Compute device; auto-detected when omitted.
    verbose : bool
        Whether to print progress every ``log_every`` epochs.
    log_every : int
        Epochs between progress prints.

    Returns
    -------
    dict
        ``{"losses": per-batch losses, "epoch_losses": per-epoch mean
        losses, "accuracies": per-epoch test accuracies (empty if no
        test set)}``.
    """
    device = resolve_device(device)
    model = model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if batch_size is None:
        batch_size = len(train_set)
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 (BatchNorm requires it).")
    drop_last = batch_size < len(train_set)

    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    criterion = MixupLoss(tau=tau)

    history = {"losses": [], "epoch_losses": [], "accuracies": []}
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for x, _ in loader:
            x = x.to(device)
            x_aug, x_2, lam = mixup_batch(x, alpha)
            z_aug, _ = model(x_aug)
            z_1, _ = model(x)
            z_2, _ = model(x_2)
            loss = criterion(z_aug, z_1, z_2, lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        history["losses"].extend(epoch_losses)
        history["epoch_losses"].append(float(np.mean(epoch_losses)))
        if test_set is not None:
            history["accuracies"].append(
                evaluate(model, train_set, test_set, device=device)
            )
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            acc = (
                f" accuracy={history['accuracies'][-1]:.4f}"
                if history["accuracies"]
                else ""
            )
            print(f"epoch {epoch:4d}  loss={history['epoch_losses'][-1]:.4f}{acc}")
    return history


@torch.no_grad()
def embed(model, dataset, device: str | None = None, batch_size: int = 256):
    """Compute L2-normalized encoder embeddings and labels for a dataset.

    Returns ``(h, y)`` as numpy arrays, where ``h`` has shape
    ``(n_instances, feature_dim)``.
    """
    device = resolve_device(device)
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    hs, ys = [], []
    for x, y in loader:
        _, h = model(x.to(device))
        hs.append(h.cpu())
        ys.append(y.cpu())
    h = F.normalize(torch.cat(hs), dim=1).numpy()
    y = torch.cat(ys).numpy()
    return h, y


def evaluate(
    model, train_set, test_set, device: str | None = None, batch_size: int = 256
) -> float:
    """Evaluate representations by 1-nearest-neighbor accuracy.

    Embeds train and test sets with the encoder, then classifies each test
    sample by the label of its nearest training sample in embedding space.
    """
    h_tr, y_tr = embed(model, train_set, device=device, batch_size=batch_size)
    h_te, y_te = embed(model, test_set, device=device, batch_size=batch_size)
    return float(KNeighborsClassifier(n_neighbors=1).fit(h_tr, y_tr).score(h_te, y_te))
