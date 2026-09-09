import numpy as np
import pandas as pd
import pytest
import torch

from mcl.data import TimeSeriesDataset, to_numpy3d


def test_to_numpy3d_from_3d_array():
    x = np.random.default_rng(0).standard_normal((4, 2, 10))
    out = to_numpy3d(x)
    assert out.shape == (4, 2, 10)
    assert out.dtype == np.float32


def test_to_numpy3d_from_2d_array():
    x = np.random.default_rng(0).standard_normal((4, 10))
    out = to_numpy3d(x)
    assert out.shape == (4, 1, 10)


def test_to_numpy3d_from_wide_dataframe():
    x = pd.DataFrame(np.random.default_rng(0).standard_normal((4, 10)))
    out = to_numpy3d(x)
    assert out.shape == (4, 1, 10)


def test_to_numpy3d_from_nested_dataframe():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "dim_0": [pd.Series(rng.standard_normal(10)) for _ in range(4)],
            "dim_1": [pd.Series(rng.standard_normal(10)) for _ in range(4)],
        }
    )
    out = to_numpy3d(df)
    assert out.shape == (4, 2, 10)
    assert out.dtype == np.float32


def test_to_numpy3d_from_multiindex_dataframe():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([range(4), range(10)], names=["instance", "time"])
    df = pd.DataFrame(
        {"var_0": rng.standard_normal(40), "var_1": rng.standard_normal(40)},
        index=idx,
    )
    out = to_numpy3d(df)
    assert out.shape == (4, 2, 10)


def test_to_numpy3d_rejects_1d():
    with pytest.raises(ValueError):
        to_numpy3d(np.zeros(5))


def test_dataset_from_numpy(synthetic_data):
    x, y = synthetic_data
    ds = TimeSeriesDataset(x, y)
    assert len(ds) == 20
    xi, yi = ds[0]
    assert xi.shape == (1, 40)
    assert xi.dtype == torch.float32
    assert yi.dtype == torch.int64
    assert ds.n_channels == 1
    assert ds.n_timepoints == 40


def test_dataset_from_tensor(synthetic_data):
    x, y = synthetic_data
    ds = TimeSeriesDataset(torch.from_numpy(x), torch.from_numpy(y))
    assert ds.x.dtype == torch.float32
    assert ds.x.shape == (20, 1, 40)


def test_dataset_device(synthetic_data):
    x, y = synthetic_data
    ds = TimeSeriesDataset(x, y, device="cpu")
    assert ds.x.device.type == "cpu"


def test_dataset_validates_shapes(synthetic_data):
    x, y = synthetic_data
    with pytest.raises(ValueError):
        TimeSeriesDataset(x[0, 0, :], y)
    with pytest.raises(ValueError):
        TimeSeriesDataset(x, y[:-1])
