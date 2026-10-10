import numpy as np
import pandas as pd
import pytest
import torch

from mcl.data import TimeSeriesDataset, load_dataset, to_numpy3d


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


def test_to_numpy3d_from_series():
    series = pd.Series([np.arange(10.0) for _ in range(4)])
    out = to_numpy3d(series)
    assert out.shape == (4, 1, 10)


def test_to_numpy3d_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        to_numpy3d(pd.DataFrame())
    with pytest.raises(ValueError):
        to_numpy3d(pd.DataFrame(columns=["a", "b"]))
    with pytest.raises(ValueError):
        to_numpy3d(pd.DataFrame(index=range(4)))


def test_encode_labels_preserves_numeric_order():
    from mcl.data import _encode_labels

    assert list(_encode_labels(np.array([1, 2, 10]))) == [0, 1, 2]
    assert list(_encode_labels(pd.Series([10, 1, 2]))) == [2, 0, 1]


def test_encode_labels_string_labels():
    from mcl.data import _encode_labels

    assert list(_encode_labels(["b", "a", "b"])) == [1, 0, 1]


def test_encode_labels_accepts_column_vector():
    from mcl.data import _encode_labels

    out = _encode_labels(np.array([[1], [2], [10]]))
    assert out.shape == (3,)
    assert list(out) == [0, 1, 2]


def test_dataset_rejects_2d_tensor():
    with pytest.raises(ValueError):
        TimeSeriesDataset(torch.randn(4, 5), torch.tensor([0, 1, 0, 1]))


def test_to_numpy3d_rejects_unsupported_type():
    with pytest.raises(ValueError):
        to_numpy3d("not a time series")


def test_load_dataset_encodes_and_returns_3d(monkeypatch):
    df = pd.DataFrame(np.arange(40, dtype=np.float32).reshape(4, 10))

    def fake_load(name, split, return_X_y, extract_path):
        return df, pd.Series(["a", "b", "a", "b"])

    monkeypatch.setattr(
        "sktime.datasets.load_UCR_UEA_dataset", fake_load, raising=False
    )
    monkeypatch.setattr("sktime.datasets.load_classification", fake_load, raising=False)

    x, y = load_dataset("Fake", split="train")
    assert x.shape == (4, 1, 10)
    assert x.dtype == np.float32
    assert list(y) == [0, 1, 0, 1]
