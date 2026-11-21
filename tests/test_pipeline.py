import numpy as np
import pandas as pd

from src.backtest import equal_weight, long_short_2x2, long_top3
from src.data import HORIZON, LOOKBACK, build_dataset
from src.models import gaf_images


def synthetic_prices():
    dates = pd.bdate_range("2007-01-01", "2023-01-01")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.01, size=(len(dates), 5))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, index=dates, columns=["SPY", "TLT", "SHY", "GLD", "DBO"])


def test_dataset_shapes_targets_and_purged_chronological_periods():
    data = build_dataset(synthetic_prices(), lookback=LOOKBACK, horizon=HORIZON)
    assert data.X.shape[1:] == (60, 5)
    assert data.y.shape == (len(data.X), 5)
    assert data.masks["train"].any()
    assert data.masks["validation"].any()
    assert data.masks["test"].any()
    assert (data.target_end[data.masks["train"]] <= pd.Timestamp("2015-12-31")).all()
    assert (data.dates[data.masks["validation"]] >= pd.Timestamp("2016-01-01")).all()
    assert (data.target_end[data.masks["validation"]] <= pd.Timestamp("2017-12-31")).all()
    assert (data.dates[data.masks["test"]] >= pd.Timestamp("2018-01-01")).all()
    assert (data.target_end[data.masks["test"]] <= pd.Timestamp("2022-12-31")).all()


def test_scaling_uses_training_data_only():
    data = build_dataset(synthetic_prices())
    train_x = data.X[data.masks["train"]]
    expected = train_x.reshape(-1, 5).mean(axis=0)
    assert np.allclose(data.x_mean, expected)
    assert data.Xs("train").shape == train_x.shape
    assert data.ys("test").shape[1] == 5


def test_gaf_has_summation_and_difference_channels():
    images = gaf_images(np.random.default_rng(1).normal(size=(4, 60)).astype("float32"))
    assert images.shape == (4, 60, 60, 2)
    assert np.isfinite(images).all()


def test_portfolio_rules_have_expected_exposures():
    forecast = pd.Series({"SPY": .1, "TLT": .05, "SHY": .01, "GLD": -.02, "DBO": -.04})
    top = long_top3(forecast)
    long_short = long_short_2x2(forecast)
    equal = equal_weight(forecast)
    assert np.isclose(top.sum(), 1.0)
    assert (top > 0).sum() == 3
    assert np.isclose(long_short.sum(), 0.0)
    assert np.isclose(long_short.abs().sum(), 2.0)
    assert np.isclose(equal.sum(), 1.0)
