from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.backtest import backtest, equal_weight, long_short_2x2, long_top3, performance_summary
from src.data import HORIZON, LOOKBACK, TEST_END, TEST_START, TICKERS, build_dataset, download_prices
from src.models import build_cnn_gaf, build_lstm, build_mlp, gaf_images


SEEDS = (0, 1, 2)
MAX_EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 25


def fit_seed_ensemble(builder, X_train, y_train, X_val, y_val,
                      seeds=SEEDS, epochs=MAX_EPOCHS):
    """Fit one network per seed with validation early stopping."""
    models, histories = [], []
    for seed in seeds:
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = builder()
        callback = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=BATCH_SIZE, callbacks=[callback],
            verbose=0, shuffle=True,
        )
        models.append(model)
        histories.append(history.history)
    return models, histories


def ensemble_predict(models, X):
    return np.mean([model.predict(X, verbose=0).squeeze() for model in models], axis=0)


def evaluate(y_true, y_pred, historical_mean):
    from scipy import stats

    error = y_true - y_pred
    pearson = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_pred) > 0 else np.nan
    spearman = stats.spearmanr(y_true, y_pred).statistic if np.std(y_pred) > 0 else np.nan
    denominator = np.sum((y_true - historical_mean) ** 2)
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2 vs train mean": float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else np.nan,
        "IC (Pearson)": float(pearson),
        "IC (Spearman)": float(spearman),
        "Hit rate": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "Share of positive targets": float(np.mean(y_true > 0)),
    }


def run(output_dir="results", epochs=MAX_EPOCHS):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    prices = download_prices(TICKERS)
    data = build_dataset(prices, LOOKBACK, HORIZON)
    predictions = {}
    metric_rows = []
    histories_summary = {}
    single_lstm_models = {}

    # Three single-output architectures, each trained separately for each ETF.
    for family in ("MLP", "CNN-GAF", "LSTM"):
        predictions[family] = {}
        for j, ticker in enumerate(data.tickers):
            X = {split: data.Xs(split)[:, :, j] for split in ("train", "validation", "test")}
            if family == "MLP":
                builder = lambda: build_mlp(LOOKBACK)
                model_X = X
            elif family == "LSTM":
                builder = lambda: build_lstm(LOOKBACK, n_features=1, n_outputs=1, units=32)
                model_X = {k: v[..., None] for k, v in X.items()}
            else:
                builder = lambda: build_cnn_gaf(LOOKBACK)
                model_X = {k: gaf_images(v) for k, v in X.items()}

            y_train = data.ys("train")[:, j]
            y_val = data.ys("validation")[:, j]
            models, histories = fit_seed_ensemble(
                builder, model_X["train"], y_train, model_X["validation"], y_val, epochs=epochs
            )
            if family == "LSTM":
                single_lstm_models[ticker] = models
            histories_summary[f"{family}:{ticker}"] = [len(h["loss"]) for h in histories]

            for split in ("train", "test"):
                scaled_pred = ensemble_predict(models, model_X[split]).reshape(-1)
                pred = scaled_pred * data.y_std[j] + data.y_mean[j]
                predictions[family].setdefault(split, pd.DataFrame(index=data.dates_of(split), columns=data.tickers))
                predictions[family][split][ticker] = pred
                metric_rows.append({
                    "Model": family, "ETF": ticker,
                    "Sample": "in-sample" if split == "train" else "out-of-sample",
                    **evaluate(data.y_raw(split)[:, j], pred, data.y_mean[j]),
                })

    # Joint five-output LSTM: five return histories in, five forecasts out.
    multi_builder = lambda: build_lstm(LOOKBACK, n_features=len(data.tickers),
                                       n_outputs=len(data.tickers), units=64)
    multi_models, multi_histories = fit_seed_ensemble(
        multi_builder, data.Xs("train"), data.ys("train"),
        data.Xs("validation"), data.ys("validation"), epochs=epochs,
    )
    histories_summary["Multi-output LSTM"] = [len(h["loss"]) for h in multi_histories]
    predictions["Multi-output LSTM"] = {}
    for split in ("train", "test"):
        scaled = ensemble_predict(multi_models, data.Xs(split))
        scaled = np.asarray(scaled).reshape(-1, len(data.tickers))
        raw = scaled * data.y_std[None, :] + data.y_mean[None, :]
        frame = pd.DataFrame(raw, index=data.dates_of(split), columns=data.tickers)
        predictions["Multi-output LSTM"][split] = frame
        for j, ticker in enumerate(data.tickers):
            metric_rows.append({
                "Model": "Multi-output LSTM", "ETF": ticker,
                "Sample": "in-sample" if split == "train" else "out-of-sample",
                **evaluate(data.y_raw(split)[:, j], raw[:, j], data.y_mean[j]),
            })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "forecast_metrics.csv", index=False)
    for family, splits in predictions.items():
        for split, frame in splits.items():
            frame.to_csv(output / f"predictions_{family.lower().replace(' ', '_').replace('-', '_')}_{split}.csv")

    # Generate forecasts for every test-period trading date, including the final
    # 25 sessions whose forward labels are unavailable but whose features are known.
    # These dates are required to carry the holdings-based backtest through TEST_END.
    backtest_dates = prices.loc[TEST_START:TEST_END].index
    return_dates = data.returns.index[LOOKBACK - 1:]
    X_bt = {}
    for j, ticker in enumerate(data.tickers):
        windows = np.lib.stride_tricks.sliding_window_view(
            data.returns[ticker].to_numpy(), LOOKBACK
        )
        positions = return_dates.get_indexer(backtest_dates)
        if (positions < 0).any():
            raise ValueError("Backtest dates are missing from the trailing-return window index.")
        raw_windows = windows[positions]
        X_bt[ticker] = ((raw_windows - data.x_mean[j]) / data.x_std[j]).astype("float32")

    X_bt_multi = np.stack([X_bt[t] for t in data.tickers], axis=-1)
    multi_scaled = ensemble_predict(multi_models, X_bt_multi).reshape(-1, len(data.tickers))
    multi_raw = multi_scaled * data.y_std[None, :] + data.y_mean[None, :]
    multi_daily = pd.DataFrame(multi_raw, index=backtest_dates, columns=data.tickers)

    single_daily = pd.DataFrame(index=backtest_dates, columns=data.tickers, dtype=float)
    for j, ticker in enumerate(data.tickers):
        scaled_pred = ensemble_predict(single_lstm_models[ticker], X_bt[ticker][..., None]).reshape(-1)
        single_daily[ticker] = scaled_pred * data.y_std[j] + data.y_mean[j]

    multi_daily.to_csv(output / "predictions_multi_output_lstm_all_test_dates.csv")
    single_daily.to_csv(output / "predictions_single_lstm_all_test_dates.csv")

    # Out-of-sample holdings-based strategy comparison, rebalanced every 25 sessions.
    multi_test = multi_daily
    single_test = single_daily
    strategy_inputs = {
        "Multi-output, long top 3": (multi_test, long_top3, False),
        "Multi-output, long-short 2/2": (multi_test, long_short_2x2, False),
        "Single LSTMs, long top 3": (single_test, long_top3, False),
        "Single LSTMs, long-short 2/2": (single_test, long_short_2x2, False),
        "Equal weight, buy-and-hold": (multi_test, equal_weight, True),
        "Equal weight, rebalanced 25d": (multi_test, equal_weight, False),
    }
    curves, allocations, net_rows = {}, {}, []
    for name, (forecast, rule, buy_hold) in strategy_inputs.items():
        curve, weights = backtest(prices, forecast, rule, HORIZON, cost_bps=0, buy_and_hold=buy_hold)
        curves[name], allocations[name] = curve, weights
        net_curve, _ = backtest(prices, forecast, rule, HORIZON, cost_bps=10, buy_and_hold=buy_hold)
        gross_stats = performance_summary(curve)
        net_stats = performance_summary(net_curve)
        net_rows.append({"Strategy": name, **{f"Gross {k}": v for k, v in gross_stats.items()},
                         **{f"Net of 10bp {k}": v for k, v in net_stats.items()}})
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(output / "backtest_equity_curves.csv")
    pd.concat(allocations, names=["Strategy", "Rebalance date"]).to_csv(output / "allocation_history.csv")
    pd.DataFrame(net_rows).to_csv(output / "backtest_metrics.csv", index=False)

    metadata = {
        "tickers": data.tickers, "lookback_trading_days": LOOKBACK,
        "forecast_horizon_trading_days": HORIZON, "train_end": "2015-12-31",
        "validation_period": ["2016-01-01", "2017-12-31"],
        "test_period": [TEST_START, TEST_END], "seeds": list(SEEDS),
        "epochs_max": epochs, "early_stopping_patience": PATIENCE,
        "batch_size": BATCH_SIZE, "optimizer": "Adam(learning_rate=0.001)",
        "target": "25-day forward cumulative log return",
        "backtest_rebalance_frequency_days": HORIZON,
        "transaction_cost_sensitivity_bps_per_unit_turnover": 10,
        "training_epochs_by_model": histories_summary,
        "note": "Metrics are generated by running this script; no unverified scores are hard-coded.",
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(metrics[metrics["Sample"] == "out-of-sample"].round(4).to_string(index=False))
    print(pd.DataFrame(net_rows).round(3).to_string(index=False))
    print(f"Saved outputs to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MScFE642 multi-ETF forecasting models.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()
    run(output_dir=args.output_dir, epochs=args.epochs)
