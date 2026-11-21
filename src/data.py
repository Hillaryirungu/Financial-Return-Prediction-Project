from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_TICKERS = ["SPY", "TLT", "SHY", "GLD", "DBO"]


def download_prices(
    tickers: Sequence[str] = DEFAULT_TICKERS,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily ETF prices through yfinance."""
    import yfinance as yf

    raw = yf.download(
        list(tickers), start=start, end=end, auto_adjust=True,
        progress=False, group_by="column", threads=True,
    )
    if raw.empty:
        raise RuntimeError("No price data returned. Check tickers, dates and network.")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if len(tickers) == 1 and close.shape[1] == 1:
        close.columns = list(tickers)
    close = close.reindex(columns=list(tickers)).sort_index().dropna(how="all")
    close = close.ffill(limit=2).dropna()
    if close.empty or close.isna().any().any():
        raise ValueError("Price data is incomplete after cleaning.")
    return close.astype(float)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    daily_returns: pd.DataFrame
    tickers: list[str]


def make_windows(
    prices: pd.DataFrame,
    lookback: int = 60,
    horizon: int = 25,
) -> Dataset:
    """Create trailing-return windows and forward cumulative-return targets."""
    if lookback < 2 or horizon < 1:
        raise ValueError("lookback must be >=2 and horizon must be >=1")
    prices = prices.sort_index().dropna()
    returns = prices.pct_change().dropna()
    values = returns.to_numpy(dtype=np.float32)
    price_values = prices.loc[returns.index].to_numpy(dtype=np.float64)
    X, y, dates = [], [], []
    for end_idx in range(lookback - 1, len(returns) - horizon):
        X.append(values[end_idx - lookback + 1 : end_idx + 1])
        y.append(price_values[end_idx + horizon] / price_values[end_idx] - 1.0)
        dates.append(returns.index[end_idx])
    if not X:
        raise ValueError("Not enough observations for the requested lookback and horizon.")
    return Dataset(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        dates=pd.DatetimeIndex(dates),
        daily_returns=returns,
        tickers=list(prices.columns),
    )


@dataclass
class TimeSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def purged_chronological_split(
    n_samples: int,
    horizon: int = 25,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TimeSplit:
    """Chronological split with a horizon-sized gap to prevent target overlap."""
    if n_samples <= 3 * horizon:
        raise ValueError("Insufficient samples for the requested purged split.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must be below one.")
    first = int(n_samples * train_fraction)
    second = int(n_samples * (train_fraction + validation_fraction))
    train = np.arange(0, first)
    validation = np.arange(min(first + horizon, second), second)
    test = np.arange(min(second + horizon, n_samples), n_samples)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Purging left an empty split; provide more observations.")
    return TimeSplit(train, validation, test)


@dataclass
class ScaledData:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def scale_splits(dataset: Dataset, split: TimeSplit) -> ScaledData:
    """Fit both scalers on training observations only."""
    Xtr, Xv, Xte = dataset.X[split.train], dataset.X[split.validation], dataset.X[split.test]
    ytr, yv, yte = dataset.y[split.train], dataset.y[split.validation], dataset.y[split.test]
    n_assets = Xtr.shape[-1]
    feature_scaler = StandardScaler().fit(Xtr.reshape(-1, n_assets))
    target_scaler = StandardScaler().fit(ytr)

    def scale_x(x: np.ndarray) -> np.ndarray:
        return feature_scaler.transform(x.reshape(-1, n_assets)).reshape(x.shape).astype(np.float32)

    return ScaledData(
        X_train=scale_x(Xtr), X_validation=scale_x(Xv), X_test=scale_x(Xte),
        y_train=target_scaler.transform(ytr).astype(np.float32),
        y_validation=target_scaler.transform(yv).astype(np.float32),
        y_test=target_scaler.transform(yte).astype(np.float32),
        feature_scaler=feature_scaler, target_scaler=target_scaler,
    )
