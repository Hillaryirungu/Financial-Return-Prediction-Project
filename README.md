# Deep Learning for Financial Return Prediction & Tactical Asset Allocation

A reproducible research project comparing neural-network approaches to **25-trading-day-ahead returns** for five exchange-traded funds (ETFs), then translating the forecasts into transparent portfolio-allocation experiments.

> **Implementation note:** The source code and experiment pipeline are now included in this repository. Model training and the final held-out evaluation must be run in the user's environment to generate empirical results. This README does not publish unverified performance figures.

## Research design

| Component | Specification |
|---|---|
| ETF universe | SPY, TLT, SHY, GLD and DBO |
| Input window | Previous 60 trading days of daily returns |
| Forecast horizon | 25 trading days ahead |
| Prediction target | Forward cumulative simple return for each ETF |
| Models | MLP, CNN with Gramian Angular Fields (CNN-GAF), single-output LSTM, multi-output LSTM |
| Ensemble | Average forecasts across three fixed random seeds |
| Validation | Chronological train / validation / test partitions with a 25-observation purge at each boundary |
| Scaling | Feature and target scalers fitted on training observations only |
| Portfolio experiments | Equal-weight benchmark, long-only top-three, long-short top-two versus bottom-two |

The project uses Yahoo Finance adjusted daily prices via `yfinance`. The default download start date is 2015-01-01; the actual dates available depend on the data provider.

## Why these models?

- **MLP:** learns nonlinear relationships from flattened 60-day return windows.
- **CNN-GAF:** transforms each univariate return window into a Gramian Angular Summation Field image, then learns spatial patterns with a convolutional network.
- **Single-output LSTM:** models sequential dependence separately for each ETF.
- **Multi-output LSTM:** predicts all five ETF returns jointly, allowing the recurrent representation to learn shared cross-asset information.

Each single-output architecture is fitted independently for each ETF. Three seeded runs are averaged to reduce dependence on one initialisation. The multi-output LSTM is trained jointly on the five return targets.

## Validation and leakage controls

Financial time series are not randomly shuffled. The pipeline:

1. Builds each feature window using only trailing observations available at its forecast date.
2. Creates the target from the price 25 trading days after that date.
3. Splits samples chronologically into training, validation and test partitions.
4. Leaves a 25-sample purge between adjacent partitions so training targets do not extend into the next partition.
5. Fits feature and target standardisation on training data only.
6. Uses validation loss for early stopping.
7. Keeps the final test partition out of model fitting and early stopping.

The held-out test period is used once to calculate forecast metrics and strategy results after training.

## Portfolio experiments

At each test date, the strategy ranks ETFs using the model's predicted 25-day return:

- **Equal weight:** holds all five ETFs equally.
- **Long-only top three:** allocates equally to the three highest forecasts.
- **Long-short 2×2:** allocates 50% to each of the two highest forecasts and -50% to each of the two lowest forecasts.

Portfolio weights are formed at a forecast date's close and applied to the next available trading day's realised ETF returns. The strategy is therefore **daily rebalanced using 25-day forecast rankings**; it is not a 25-day buy-and-hold trade. Reported strategy returns are gross of transaction costs, financing, taxes and market impact. These assumptions should be considered when interpreting the results.

## Repository structure

```text
Financial-Return-Prediction-Project/
├── src/
│   ├── data.py          # Price download, windows, purged splits and scaling
│   ├── models.py        # MLP, CNN-GAF, LSTM and multi-output LSTM
│   ├── train.py         # Training, seeded ensembles and evaluation
│   └── backtest.py      # Forecast-ranked portfolio strategies
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── results/             # Generated locally; excluded from Git
└── README.md
```

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Hillaryirungu/Financial-Return-Prediction-Project.git
cd Financial-Return-Prediction-Project
```

### 2. Create a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch will use CUDA when a compatible GPU is available; otherwise, training runs on CPU. The CNN-GAF and recurrent models can take time to train on CPU.

### 4. Run the tests

```bash
python -m pytest tests/ -v
```

### 5. Run the experiment

```bash
python -m src.train --start 2015-01-01 --epochs 100
```

Generated outputs are written to `results/`:

- Per-model ETF return forecasts.
- Held-out MAE, RMSE and directional accuracy.
- Daily returns for each portfolio strategy and model.
- Strategy-level annualised return, volatility, Sharpe ratio and maximum drawdown.
- Run metadata describing the sample and experiment settings.

## Interpretation and limitations

- Forecast accuracy does not by itself establish that a strategy is investable or profitable.
- The strategy is a research backtest and excludes transaction costs, slippage, short-borrow costs, taxes and liquidity constraints.
- Yahoo Finance is a third-party data source; availability, corporate-action adjustments and historical data quality should be checked before relying on results.
- Results may vary with data revisions, package versions, random seeds and training hardware.
- This is an educational quantitative-finance project, not investment advice.

## Author

**Hillary Irungu** · [GitHub](https://github.com/Hillaryirungu) · [Portfolio](https://hillaryirungu.github.io)

## License

Released under the MIT License. See [LICENSE](LICENSE).
