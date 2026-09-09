"""Data loading and dataset utilities for time series classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

__all__ = ["TimeSeriesDataset", "load_dataset", "to_numpy3d"]


def _nested_to_numpy3d(X: pd.DataFrame) -> np.ndarray:
    cols = []
    for c in range(X.shape[1]):
        cols.append(np.stack([np.asarray(v, dtype=np.float32) for v in X.iloc[:, c]]))
    return np.stack(cols, axis=1)


def _multiindex_to_numpy3d(X: pd.DataFrame) -> np.ndarray:
    instances = X.index.get_level_values(0).unique()
    cols = []
    for c in X.columns:
        wide = X[c].unstack().loc[instances]
        cols.append(wide.to_numpy(dtype=np.float32))
    return np.stack(cols, axis=1)


def to_numpy3d(X) -> np.ndarray:
    """Convert common time series container formats to a float32 array of
    shape ``(n_instances, n_channels, n_timepoints)``.

    Supports 3D/2D numpy arrays, nested and wide ``pandas.DataFrame``\\s,
    and ``pd-multiindex`` dataframes as returned by sktime.
    """
    if isinstance(X, np.ndarray):
        arr = X.astype(np.float32)
        if arr.ndim == 2:
            return arr[:, None, :]
        if arr.ndim == 3:
            return arr
        raise ValueError(f"Expected a 2D or 3D array, got {arr.ndim}D.")
    if isinstance(X, pd.DataFrame):
        if isinstance(X.index, pd.MultiIndex):
            return _multiindex_to_numpy3d(X)
        if isinstance(X.iloc[0, 0], (pd.Series, np.ndarray, list)):
            return _nested_to_numpy3d(X)
        arr = X.to_numpy(dtype=np.float32)
        return arr[:, None, :]
    if isinstance(X, pd.Series):
        return to_numpy3d(X.to_frame())
    raise ValueError(f"Unsupported input type: {type(X)}.")


def _encode_labels(y) -> np.ndarray:
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    return LabelEncoder().fit_transform(np.asarray(y).astype(str)).astype(np.int64)


def load_dataset(name: str, split: str = "train", extract_path=None):
    """Load a dataset from the UCR/UEA time series classification archives.

    Parameters
    ----------
    name : str
        Dataset name, e.g. ``"GunPoint"`` (UCR) or ``"BasicMotions"`` (UEA).
        Downloads and caches the data via sktime on first use.
    split : str
        Either ``"train"`` or ``"test"``.
    extract_path : str or Path, optional
        Where to store the downloaded archives (sktime's default cache is
        used when omitted).

    Returns
    -------
    x : np.ndarray
        Float32 array of shape ``(n_instances, n_channels, n_timepoints)``.
    y : np.ndarray
        Int64 labels of shape ``(n_instances,)``, encoded as ``0..n_classes-1``.
    """
    try:
        from sktime.datasets import load_UCR_UEA_dataset as _load
    except ImportError:
        from sktime.datasets import load_classification as _load

    X, y = _load(name, split=split, return_X_y=True, extract_path=extract_path)
    return to_numpy3d(X), _encode_labels(y)


class TimeSeriesDataset(Dataset):
    """A simple tensor dataset holding time series of shape
    ``(n_instances, n_channels, n_timepoints)`` and integer labels.

    Parameters
    ----------
    x : array-like
        Time series data, any format accepted by :func:`to_numpy3d`
        (numpy array, nested/wide pandas dataframe, or torch tensor).
    y : array-like
        Labels.
    device : str or torch.device, optional
        If given, tensors are stored directly on this device (as in the
        original implementation). Otherwise they are kept on CPU and moved
        to the compute device by the training/evaluation loops.
    """

    def __init__(self, x, y, device=None) -> None:
        if isinstance(x, torch.Tensor):
            x_t = x.detach().to(dtype=torch.float32)
        else:
            x_t = torch.as_tensor(to_numpy3d(x))
        if isinstance(y, torch.Tensor):
            y_t = y.detach().to(dtype=torch.long)
        else:
            y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
        if x_t.ndim != 3:
            raise ValueError(
                f"x must be 3D (n_instances, n_channels, n_timepoints), "
                f"got {x_t.ndim}D."
            )
        if x_t.shape[0] != y_t.shape[0]:
            raise ValueError(
                f"x and y have mismatched lengths: {x_t.shape[0]} vs {y_t.shape[0]}."
            )
        if device is not None:
            x_t = x_t.to(device)
            y_t = y_t.to(device)
        self.x = x_t
        self.y = y_t

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @property
    def n_channels(self) -> int:
        return self.x.shape[1]

    @property
    def n_timepoints(self) -> int:
        return self.x.shape[2]
