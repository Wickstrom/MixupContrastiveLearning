import math

import torch

from mcl.losses import MixupLoss


def test_loss_value_with_identical_embeddings():
    n, d = 8, 16
    z = torch.nn.functional.normalize(torch.ones(n, d), dim=1)
    for lam in (0.0, 0.3, 1.0):
        loss = MixupLoss(tau=0.5)(z, z, z, lam=lam)
        assert torch.isclose(loss, torch.tensor(math.log(2 * n)), atol=1e-5)


def test_loss_near_zero_with_perfect_prediction():
    n, d = 4, 8
    z_1 = torch.eye(n, d)
    z_2 = torch.eye(2 * n, d)[n:]
    z_aug = z_1
    loss = MixupLoss(tau=0.01)(z_aug, z_1, z_2, lam=1.0)
    assert loss.item() < 1e-6


def test_loss_symmetric_in_lam():
    torch.manual_seed(0)
    n, d = 4, 8
    z_1 = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    z_2 = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    z_aug = torch.nn.functional.normalize(torch.randn(n, d), dim=1)
    l_a = MixupLoss()(z_aug, z_1, z_2, lam=0.3)
    l_b = MixupLoss()(z_aug, z_2, z_1, lam=0.7)
    assert torch.isclose(l_a, l_b, atol=1e-6)


def test_loss_gradients_flow():
    torch.manual_seed(0)
    z_1 = torch.randn(4, 8, requires_grad=True)
    z_2 = torch.randn(4, 8, requires_grad=True)
    z_aug = torch.randn(4, 8, requires_grad=True)
    loss = MixupLoss()(z_aug, z_1, z_2, lam=0.5)
    loss.backward()
    assert z_aug.grad is not None
    assert z_1.grad is not None
    assert z_2.grad is not None


def test_loss_handles_any_batch_size():
    torch.manual_seed(0)
    criterion = MixupLoss()
    for n in (3, 7, 16):
        z_1 = torch.randn(n, 8)
        z_2 = torch.randn(n, 8)
        z_aug = torch.randn(n, 8)
        loss = criterion(z_aug, z_1, z_2, lam=0.5)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
