"""Shared fixtures and options for the test suite."""

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that download real datasets.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: downloads real datasets")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="needs --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def synthetic_data():
    """Two-class synthetic dataset: sine waves vs. sawtooth-ish waves."""
    rng = np.random.default_rng(0)
    n, t = 20, 40
    time = np.linspace(0, 4 * np.pi, t)
    x0 = np.sin(time)[None, :] + 0.1 * rng.standard_normal((n // 2, t))
    x1 = np.sign(np.sin(time))[None, :] + 0.1 * rng.standard_normal((n // 2, t))
    x = np.concatenate([x0, x1]).astype(np.float32)[:, None, :]
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    return x, y
