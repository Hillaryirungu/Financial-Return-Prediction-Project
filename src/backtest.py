from __future__ import annotations

import numpy as np
import pandas as pd


def rank_weights(
    forecasts: pd.Series,
    strategy: str,
) -> pd.Series:
    """Create weights from forecast rankings; weights sum to one for long-only."""
    forecasts = forecasts.dropna().sort_values(ascending=False)
    weights = pd.Series(0.0, index=forecasts.index)
    if strategy == "equal_weight":
        weights[:] = 1.0 / len(weights)
    elif strategy == "long_only_top3":
        selected = forecasts.index[:3]
        weights.loc[selected] = 1.0 / len(selected)
    elif strategy == "long_short_2x2":
        if len(forecasts) < 4:
            raise ValueError("Long-short strategy requires at least four assets.")
        weights.loc[forecasts.index[:2]] = 0.5
        weights.loc[forecasts.index[-2:]] = -0.5
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return weights


def backtest_daily(
    predictions: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Daily-rebalanced portfolios using each date's 25-day forecast ranking.

    Forecasts are formed at close on date t and applied to daily returns on the
    next available trading date. Returns are gross of transaction costs.
    """
    rows = []
    common = predictions.index.intersection(daily_returns.index)
    for date in common:
        next_positions = daily_returns.index.get_loc(date) + 1
        if next_positions >= len(daily_returns.index):
            continue
        next_date = daily_returns.index[next_positions]
        forecast = predictions.loc[date]
        realised = daily_returns.loc[next_date].reindex(forecast.index)
        row = {"Date": next_date}
        for strategy in ("equal_weight", "long_only_top3", "long_short_2x2"):
            w = rank_weights(forecast, strategy)
            row[strategy] = float((w * realised).sum())
        rows.append(row)
    if not rows:
        raise ValueError("No overlapping forecast and next-day return observations.")
    return pd.DataFrame(rows).set_index("Date").sort_index()


def performance_summary(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Annualised return, volatility, Sharpe and maximum drawdown."""
    summary = {}
    for column in returns:
        r = returns[column].dropna()
        wealth = (1 + r).cumprod()
        drawdown = wealth / wealth.cummax() - 1
        vol = r.std(ddof=1) * np.sqrt(periods)
        summary[column] = {
            "Annualised return": (wealth.iloc[-1] ** (periods / len(r)) - 1) if len(r) else np.nan,
            "Annualised volatility": vol,
            "Sharpe (rf=0)": (r.mean() * periods / vol) if vol > 0 else np.nan,
            "Maximum drawdown": drawdown.min(),
        }
    return pd.DataFrame(summary).T
