from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

TICKERS = ["SPY", "TLT", "SHY", "GLD", "DBO"]
LOOKBACK = 60
HORIZON = 25
TRAIN_END = "2015-12-31"
VAL_START = "2016-01-01"
VAL_END = "2017-12-31"
TEST_START = "2018-01-01"
TEST_END = "2022-12-31"


def download_prices(tickers: Sequence[str] = TICKERS, start: str = "2007-01-01",
                    end: str = "2023-01-01") -> pd.DataFrame:
    """Download auto-adjusted daily ETF prices and align the common calendar."""
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no prices; check network and dates.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.reindex(columns=list(tickers)).sort_index().dropna()
    if close.empty or close.isna().any().any():
        raise ValueError("Could not form a complete common price panel for all ETFs.")
    return close.astype(float)


@dataclass
class Dataset:
    prices: pd.DataFrame
    returns: pd.DataFrame
    dates: pd.DatetimeIndex
    target_end: pd.DatetimeIndex
    X: np.ndarray
    y: np.ndarray
    tickers: list[str]
    masks: dict[str, np.ndarray]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def Xs(self, split: str) -> np.ndarray:
        x = self.X[self.masks[split]]
        return ((x - self.x_mean[None, None, :]) / self.x_std[None, None, :]).astype("float32")

    def ys(self, split: str) -> np.ndarray:
        y = self.y[self.masks[split]]
        return ((y - self.y_mean[None, :]) / self.y_std[None, :]).astype("float32")

    def y_raw(self, split: str) -> np.ndarray:
        return self.y[self.masks[split]]

    def dates_of(self, split: str) -> pd.DatetimeIndex:
        return self.dates[self.masks[split]]


def build_dataset(prices: pd.DataFrame, lookback: int = LOOKBACK,
                  horizon: int = HORIZON) -> Dataset:
    """Build 60-day log-return windows and 25-day forward cumulative log returns.

    A sample is assigned to a period only if its full forward target ends within
    that period. Samples whose target crosses a boundary are purged.
    """
    prices = prices.sort_index().dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    tickers = list(prices.columns)
    dates = returns.index[lookback - 1:]
    windows = [np.lib.stride_tricks.sliding_window_view(returns[t].to_numpy(), lookback)
               for t in tickers]
    X = np.stack(windows, axis=-1).astype("float32")

    forward = returns.rolling(horizon).sum().shift(-horizon)
    y_frame = forward.reindex(dates)
    end_dates = pd.Series(returns.index, index=returns.index).shift(-horizon).reindex(dates)
    valid = y_frame.notna().all(axis=1).to_numpy() & end_dates.notna().to_numpy()
    dates = dates[valid]
    X = X[valid]
    y = y_frame.to_numpy(dtype="float32")[valid]
    target_end = pd.DatetimeIndex(end_dates.to_numpy()[valid])

    masks = {
        "train": np.asarray(target_end <= pd.Timestamp(TRAIN_END)),
        "validation": np.asarray((dates >= pd.Timestamp(VAL_START)) & (target_end <= pd.Timestamp(VAL_END))),
        "test": np.asarray((dates >= pd.Timestamp(TEST_START)) & (target_end <= pd.Timestamp(TEST_END))),
    }
    if any(mask.sum() == 0 for mask in masks.values()):
        raise ValueError("A chronological split is empty. Check data coverage and date boundaries.")

    train_X = X[masks["train"]]
    train_y = y[masks["train"]]
    x_mean = train_X.reshape(-1, len(tickers)).mean(axis=0)
    x_std = train_X.reshape(-1, len(tickers)).std(axis=0)
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_std = np.where(y_std == 0, 1.0, y_std)

    return Dataset(prices, returns, dates, target_end, X, y, tickers, masks,
                   x_mean, x_std, y_mean, y_std)
