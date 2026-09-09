"""Integration tests that download real UCR/UEA datasets via sktime.

Run with: pytest --run-integration
"""

import numpy as np
import pytest
import torch

from mcl.data import TimeSeriesDataset, load_dataset
from mcl.models import FCN
from mcl.trainer import train

pytestmark = pytest.mark.integration


def test_load_gunpoint():
    x_tr, y_tr = load_dataset("GunPoint", split="train")
    x_te, y_te = load_dataset("GunPoint", split="test")
    assert x_tr.shape == (50, 1, 150)
    assert x_te.shape == (150, 1, 150)
    assert x_tr.dtype == np.float32
    assert set(np.unique(y_tr)) == {0, 1}


def test_train_gunpoint_short():
    torch.manual_seed(0)
    np.random.seed(0)
    x_tr, y_tr = load_dataset("GunPoint", split="train")
    x_te, y_te = load_dataset("GunPoint", split="test")
    train_set = TimeSeriesDataset(x_tr, y_tr)
    test_set = TimeSeriesDataset(x_te, y_te)
    model = FCN(train_set.n_channels)
    history = train(model, train_set, test_set, epochs=10, device="cpu", verbose=False)
    assert len(history["epoch_losses"]) == 10
    assert 0.0 <= history["accuracies"][-1] <= 1.0
