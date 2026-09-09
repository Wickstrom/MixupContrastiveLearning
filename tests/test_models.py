import torch

from mcl.models import FCN


def test_fcn_output_shapes_univariate():
    model = FCN(n_channels=1)
    x = torch.randn(8, 1, 150)
    z, h = model(x)
    assert z.shape == (8, 128)
    assert h.shape == (8, 128)


def test_fcn_output_shapes_multivariate():
    model = FCN(n_channels=3)
    x = torch.randn(4, 3, 60)
    z, h = model(x)
    assert z.shape == (4, 128)
    assert h.shape == (4, 128)


def test_fcn_preserves_temporal_length():
    model = FCN(n_channels=1).eval()
    with torch.no_grad():
        for t in (30, 50, 150):
            x = torch.randn(2, 1, t)
            z, h = model(x)
            assert z.shape == (2, 128)
            assert h.shape == (2, 128)


def test_fcn_custom_feature_dim():
    model = FCN(n_channels=1, feature_dim=64)
    z, _ = model(torch.randn(2, 1, 40))
    assert z.shape == (2, 64)
