from __future__ import annotations

import numpy as np
import pandas as pd


def long_top3(forecast: pd.Series) -> pd.Series:
    weights = pd.Series(0.0, index=forecast.index)
    weights.loc[forecast.nlargest(3).index] = 1.0 / 3.0
    return weights


def long_short_2x2(forecast: pd.Series) -> pd.Series:
    weights = pd.Series(0.0, index=forecast.index)
    weights.loc[forecast.nlargest(2).index] = 0.5
    weights.loc[forecast.nsmallest(2).index] = -0.5
    return weights


def equal_weight(forecast: pd.Series) -> pd.Series:
    return pd.Series(1.0 / len(forecast), index=forecast.index)


def backtest(prices: pd.DataFrame, predictions: pd.DataFrame, rule,
             horizon: int = 25, cost_bps: float = 0.0,
             buy_and_hold: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    """Rebalance every horizon days; holdings drift between rebalance dates.

    Forecasts are observed at the close on each rebalance date. Returns are
    measured from that close onward. Transaction cost is charged as bps times
    one-way L1 turnover at each subsequent rebalance.
    """
    px = prices.loc[predictions.index.min():predictions.index.max()]
    rebalance_dates = px.index[::horizon]
    value = 1.0
    paths, allocations = [], []
    previous_weights = pd.Series(1.0 / len(px.columns), index=px.columns)

    for k, date in enumerate(rebalance_dates):
        forecast = predictions.loc[date]
        weights = previous_weights.copy() if (buy_and_hold and k > 0) else rule(forecast)
        weights = weights.reindex(px.columns).fillna(0.0)
        if k > 0 and cost_bps:
            turnover = float((weights - previous_weights).abs().sum())
            value *= 1.0 - cost_bps / 10000.0 * turnover

        next_date = rebalance_dates[k + 1] if k + 1 < len(rebalance_dates) else px.index[-1]
        relative_prices = px.loc[date:next_date] / px.loc[date] - 1.0
        path = value * (1.0 + relative_prices.mul(weights, axis=1).sum(axis=1))
        paths.append(path if k == 0 else path.iloc[1:])

        block_return = float(relative_prices.iloc[-1] @ weights)
        denominator = 1.0 + block_return
        previous_weights = (weights * (1.0 + relative_prices.iloc[-1]) / denominator
                            if denominator != 0 else weights)
        value = float(path.iloc[-1])
        allocations.append(pd.Series(weights, name=date))

    curve = pd.concat(paths).rename("Portfolio value")
    allocation_table = pd.DataFrame(allocations)
    return curve, allocation_table


def performance_summary(curve: pd.Series) -> dict[str, float]:
    daily = curve.pct_change().dropna()
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1.0 / years) - 1.0
    drawdown = curve / curve.cummax() - 1.0
    volatility = daily.std(ddof=1) * np.sqrt(252)
    return {
        "Total return (%)": (curve.iloc[-1] / curve.iloc[0] - 1.0) * 100,
        "CAGR (%)": cagr * 100,
        "Annualised volatility (%)": volatility * 100,
        "Sharpe (rf=0)": daily.mean() / daily.std(ddof=1) * np.sqrt(252),
        "Maximum drawdown (%)": drawdown.min() * 100,
        "Calmar": cagr / abs(drawdown.min()) if drawdown.min() != 0 else np.nan,
    }
