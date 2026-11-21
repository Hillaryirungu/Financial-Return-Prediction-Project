from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyts.image import GramianAngularField
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.backtest import backtest_daily, performance_summary
from src.data import DEFAULT_TICKERS, download_prices, make_windows, purged_chronological_split, scale_splits
from src.models import CNNGAFRegressor, LSTMRegressor, MLPRegressor, MultiOutputLSTM


def _fit(model, Xtr, ytr, Xv, yv, epochs=100, patience=12, batch_size=64, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    xv = torch.tensor(Xv, dtype=torch.float32, device=device)
    yv_t = torch.tensor(yv, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv_t).item())
        if val_loss < best_loss - 1e-6:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(Xv, dtype=torch.float32, device=device)).cpu().numpy(), model


def _gaf(X: np.ndarray) -> np.ndarray:
    transformer = GramianAngularField(image_size=X.shape[1], method="summation")
    return transformer.fit_transform(X).astype(np.float32)[:, None, :, :]


def run(
    start="2015-01-01",
    output_dir="results",
    lookback=60,
    horizon=25,
    epochs=100,
    seeds=(42, 43, 44),
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prices = download_prices(DEFAULT_TICKERS, start=start)
    dataset = make_windows(prices, lookback=lookback, horizon=horizon)
    split = purged_chronological_split(len(dataset.X), horizon=horizon)
    scaled = scale_splits(dataset, split)
    tickers = dataset.tickers
    target_scaler = scaled.target_scaler

    # Each single-output architecture is trained independently per ETF.
    forecasts = {name: np.zeros((len(split.test), len(tickers))) for name in ("MLP", "LSTM", "CNN-GAF")}
    Xtr_all, Xv_all, Xte_all = scaled.X_train, scaled.X_validation, scaled.X_test
    for asset_idx, ticker in enumerate(tickers):
        ytr, yv = scaled.y_train[:, asset_idx:asset_idx+1], scaled.y_validation[:, asset_idx:asset_idx+1]
        Xtr = Xtr_all[:, :, asset_idx:asset_idx+1]
        Xv = Xv_all[:, :, asset_idx:asset_idx+1]
        Xte = Xte_all[:, :, asset_idx:asset_idx+1]
        per_model = {"MLP": (Xtr[:, :, 0], Xv[:, :, 0], Xte[:, :, 0], lambda: MLPRegressor(lookback)),
                     "LSTM": (Xtr, Xv, Xte, lambda: LSTMRegressor()),
                     "CNN-GAF": (_gaf(Xtr[:, :, 0]), _gaf(Xv[:, :, 0]), _gaf(Xte[:, :, 0]), lambda: CNNGAFRegressor())}
        for model_name, (a, b, c, factory) in per_model.items():
            seed_preds = []
            for seed in seeds:
                # Validation predictions drive early stopping; the restored best
                # model is then evaluated on the untouched test partition.
                _, model = _fit(factory(), a, ytr, b, yv, epochs=epochs, seed=seed)
                device = next(model.parameters()).device
                model.eval()
                with torch.no_grad():
                    pred = model(torch.tensor(c, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
                seed_preds.append(pred)
            forecasts[model_name][:, asset_idx] = np.mean(seed_preds, axis=0)

    # One multi-output LSTM learns all five ETF targets jointly.
    multi_seed_preds = []
    for seed in seeds:
        _, model = _fit(MultiOutputLSTM(n_assets=len(tickers)), Xtr_all, scaled.y_train,
                        Xv_all, scaled.y_validation, epochs=epochs, seed=seed)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte_all, dtype=torch.float32, device=device)).cpu().numpy()
        multi_seed_preds.append(pred)
    forecasts["Multi-output LSTM"] = np.mean(multi_seed_preds, axis=0)

    test_dates = dataset.dates[split.test]
    pred_frames = {}
    metric_rows = []
    actual = dataset.y[split.test]
    for name, pred_scaled in forecasts.items():
        pred = target_scaler.inverse_transform(pred_scaled)
        frame = pd.DataFrame(pred, index=test_dates, columns=tickers)
        pred_frames[name] = frame
        metric_rows.append({
            "Model": name,
            "MAE": float(np.mean(np.abs(pred - actual))),
            "RMSE": float(np.sqrt(np.mean((pred - actual) ** 2))),
            "Directional accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
        })
        frame.to_csv(out / f"forecasts_{name.lower().replace(' ', '_').replace('-', '_')}.csv")

    metrics = pd.DataFrame(metric_rows).set_index("Model")
    metrics.to_csv(out / "forecast_metrics.csv")
    (out / "run_metadata.json").write_text(json.dumps({
        "tickers": tickers, "lookback_days": lookback, "forecast_horizon_days": horizon,
        "train_samples": len(split.train), "validation_samples": len(split.validation),
        "test_samples": len(split.test), "seeds": list(seeds),
        "note": "Forecast metrics use the purged chronological test set. Strategy results are gross of transaction costs."
    }, indent=2))

    # Backtest each model's forecasts separately against the same daily returns.
    summaries = []
    for name, frame in pred_frames.items():
        daily = backtest_daily(frame, dataset.daily_returns)
        daily.to_csv(out / f"backtest_{name.lower().replace(' ', '_').replace('-', '_')}.csv")
        summary = performance_summary(daily)
        summary.insert(0, "Model", name)
        summaries.append(summary.reset_index(names="Strategy"))
    pd.concat(summaries, ignore_index=True).to_csv(out / "strategy_metrics.csv", index=False)
    print(metrics.round(4))
    print(f"Saved forecasts and backtests to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ETF return forecasting models.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run(start=args.start, output_dir=args.output_dir, epochs=args.epochs)
