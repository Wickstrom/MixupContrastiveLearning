import numpy as np
import pytest
import torch

from mcl.data import TimeSeriesDataset
from mcl.models import FCN
from mcl.trainer import embed, evaluate, mixup_batch, resolve_device, train


def test_mixup_batch_convex_combination():
    torch.manual_seed(0)
    x = torch.randn(10, 1, 20)
    x_aug, x_2, lam = mixup_batch(x, alpha=1.0)
    assert 0.0 <= lam <= 1.0
    assert x_aug.shape == x.shape
    assert x_2.shape == x.shape
    reconstructed = lam * x + (1 - lam) * x_2
    assert torch.allclose(x_aug, reconstructed, atol=1e-6)


def test_mixup_batch_uses_permutation():
    torch.manual_seed(0)
    x = torch.arange(20, dtype=torch.float32).reshape(10, 1, 2)
    _, x_2, _ = mixup_batch(x, alpha=1.0)
    assert not torch.equal(x, x_2)
    assert torch.equal(x_2.sort(dim=0).values, x.sort(dim=0).values)


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(None) in ("cpu", "cuda")


def test_train_reduces_loss(synthetic_data):
    torch.manual_seed(0)
    x, y = synthetic_data
    train_set = TimeSeriesDataset(x, y)
    test_set = TimeSeriesDataset(x, y)
    model = FCN(n_channels=1)
    history = train(
        model,
        train_set,
        test_set,
        epochs=30,
        alpha=1.0,
        device="cpu",
        verbose=False,
        log_every=1000,
    )
    assert len(history["epoch_losses"]) == 30
    assert len(history["losses"]) == 30
    assert len(history["accuracies"]) == 30
    assert np.isfinite(history["losses"]).all()
    first = np.mean(history["epoch_losses"][:5])
    last = np.mean(history["epoch_losses"][-5:])
    assert last < 0.9 * first
    assert all(0.0 <= a <= 1.0 for a in history["accuracies"])


def test_train_minibatch(synthetic_data):
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = synthetic_data
    train_set = TimeSeriesDataset(x, y)
    model = FCN(n_channels=1)
    history = train(
        model, train_set, epochs=3, batch_size=6, device="cpu", verbose=False
    )
    assert len(history["epoch_losses"]) == 3
    assert len(history["losses"]) == 9


def test_train_rejects_batch_size_one(synthetic_data):
    x, y = synthetic_data
    train_set = TimeSeriesDataset(x, y)
    with pytest.raises(ValueError):
        train(FCN(1), train_set, epochs=1, batch_size=1, verbose=False)


def test_embed_shapes_and_normalization(synthetic_data):
    torch.manual_seed(0)
    x, y = synthetic_data
    ds = TimeSeriesDataset(x, y)
    model = FCN(n_channels=1)
    h, y_out = embed(model, ds, device="cpu")
    assert h.shape == (20, 128)
    assert np.allclose((h**2).sum(axis=1), 1.0, atol=1e-5)
    assert (y_out == y).all()


def test_evaluate_returns_accuracy(synthetic_data):
    torch.manual_seed(0)
    x, y = synthetic_data
    train_set = TimeSeriesDataset(x, y)
    test_set = TimeSeriesDataset(x, y)
    model = FCN(n_channels=1)
    acc = evaluate(model, train_set, test_set, device="cpu")
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_resolve_device_auto():
    assert resolve_device("auto") in ("cpu", "cuda")


def test_train_verbose_logging(synthetic_data, capsys):
    torch.manual_seed(0)
    np.random.seed(0)
    x, y = synthetic_data
    train_set = TimeSeriesDataset(x, y)
    model = FCN(n_channels=1)
    train(
        model,
        train_set,
        test_set=train_set,
        epochs=2,
        device="cpu",
        verbose=True,
        log_every=1,
    )
    out = capsys.readouterr().out
    assert "epoch" in out
    assert "accuracy=" in out
