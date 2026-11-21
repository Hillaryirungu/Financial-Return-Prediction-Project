import numpy as np
import pandas as pd

from src.backtest import rank_weights
from src.data import make_windows, purged_chronological_split, scale_splits


def test_windows_and_forward_target():
    idx = pd.bdate_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"SPY": np.arange(100.0) + 100}, index=idx)
    data = make_windows(prices, lookback=10, horizon=5)
    assert data.X.shape[1:] == (10, 1)
    assert len(data.X) == len(data.y) == len(data.dates)
    assert np.isclose(data.y[0, 0], prices.iloc[15, 0] / prices.iloc[10, 0] - 1)


def test_purged_split_has_no_adjacent_boundary_overlap():
    split = purged_chronological_split(500, horizon=25)
    assert split.train[-1] + 25 < split.validation[0]
    assert split.validation[-1] + 25 < split.test[0]


def test_scalers_fit_training_observations_only():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"SPY": 100 * np.cumprod(1 + np.random.default_rng(1).normal(0, .01, 300))}, index=idx)
    data = make_windows(prices, lookback=10, horizon=5)
    split = purged_chronological_split(len(data.X), horizon=5)
    scaled = scale_splits(data, split)
    expected = data.X[split.train].reshape(-1, 1).mean()
    assert np.isclose(scaled.feature_scaler.mean_[0], expected)


def test_long_short_weights_are_market_neutral():
    forecasts = pd.Series({"SPY": .1, "TLT": .05, "SHY": .01, "GLD": -.02, "DBO": -.04})
    weights = rank_weights(forecasts, "long_short_2x2")
    assert np.isclose(weights.sum(), 0)
    assert np.isclose(weights.abs().sum(), 2)
