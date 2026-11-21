# Deep Learning for Financial Return Prediction & Tactical Asset Allocation

A reproducible MScFE 642 research project comparing deep-learning models for forecasting **25-trading-day-ahead returns** across five ETFs, then evaluating forecast-ranked portfolio strategies.

## Research question

Do sequence and image-based neural networks provide useful information for ranking diversified ETF returns, and does a joint multi-output LSTM offer an advantage over separately trained single-output LSTMs?

## Data and experimental design

| Item | Specification |
|---|---|
| ETF universe | SPY, TLT, SHY, GLD and DBO |
| Market data | Daily auto-adjusted prices from Yahoo Finance |
| Sample | 2007–2022 |
| Input | Trailing 60 daily log returns per ETF |
| Target | 25-day forward cumulative log return: `ln(P[t+25] / P[t])` |
| Training | Through 31 December 2015 |
| Validation | 2016–2017 |
| Out-of-sample test | 2018–2022 |
| Purging | Samples whose 25-day target crosses a period boundary are excluded |
| Scaling | Each ETF's input and target scalers are estimated from its training observations only |
| Random initialisations | Three-seed prediction ensemble |
| Optimisation | Adam, learning rate 0.001, batch size 64, early stopping (patience 25, restore best weights), up to 300 epochs |

The common price history is downloaded and aligned across all five ETFs. The first and last available dates may vary if the data provider revises or limits its history.

## Models

- **MLP:** one feed-forward network per ETF, using a flattened 60-day return window.
- **CNN-GAF:** one CNN per ETF, using two-channel Gramian Angular Field images: the summation field (GASF) and difference field (GADF).
- **Single-output LSTM:** one recurrent network per ETF, modelling its 60 daily returns in time order.
- **Multi-output LSTM:** one joint recurrent network receiving all five ETF histories and predicting all five forward returns simultaneously.

The three single-output architectures are trained separately for each ETF. Each model family uses three seeded fits, with predictions averaged across the ensemble. The multi-output LSTM uses a 64-unit recurrent layer and a five-value output head.

## Evaluation

The script reports in-sample and out-of-sample results by model and ETF:

- RMSE and MAE in cumulative log-return units.
- Out-of-sample R² relative to the training-sample mean forecast.
- Pearson and Spearman information coefficients.
- Directional hit rate and the share of positive realised targets.

Overlapping 25-day targets are dependent observations. The results should therefore be interpreted with care; the notebook research additionally examines non-overlapping observations and moving-block bootstrap uncertainty.

## Tactical asset-allocation backtest

The test-period strategy rebalances **every 25 trading days**. At each rebalance close, the model ranks the five ETFs by predicted 25-day return and holds the resulting portfolio until the next rebalance. Holdings drift with realised prices between rebalances.

| Strategy | Allocation rule |
|---|---|
| Multi-output LSTM — long-only top 3 | One-third in each of the three highest-ranked ETFs |
| Multi-output LSTM — long-short 2/2 | +50% in each of the two highest-ranked ETFs; -50% in each of the two lowest-ranked |
| Single-output LSTMs — long-only top 3 | Same top-three rule using the five independent LSTM forecasts |
| Single-output LSTMs — long-short 2/2 | Same long-short rule using the five independent LSTM forecasts |
| Equal-weight buy-and-hold | 20% in each ETF, held without rebalancing |
| Equal-weight rebalanced | 20% in each ETF, reset every 25 trading days |

The backtest reports total return, CAGR, annualised volatility, Sharpe ratio, maximum drawdown and Calmar ratio. A separate sensitivity applies **10 basis points per unit of one-way turnover** at each rebalance.

All backtest results are historical and gross/net cost assumptions are stated explicitly. They are not a live trading strategy or investment recommendation.

## Repository structure

```text
Financial-Return-Prediction-Project/
├── src/
│   ├── data.py          # ETF prices, windows, targets, purged splits and scaling
│   ├── models.py        # MLP, two-channel CNN-GAF and LSTM architectures
│   ├── train.py         # Three-seed training, evaluation and backtest orchestration
│   └── backtest.py      # 25-day holdings-based strategy engine
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── results/             # Generated locally; excluded from Git
└── README.md
```

## Run locally

### 1. Clone and install

```bash
git clone https://github.com/Hillaryirungu/Financial-Return-Prediction-Project.git
cd Financial-Return-Prediction-Project
python -m venv .venv
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\\Scripts\\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow uses a compatible GPU when available; otherwise, it runs on CPU. Training all model families for three seeds can take time.

### 2. Run the tests

```bash
python -m pytest tests/ -v
```

### 3. Run the full experiment

```bash
python -m src.train --epochs 300 --output-dir results
```

Use a smaller `--epochs` value for a quick smoke run. Such a run is not directly comparable with the full 300-epoch maximum experiment.

## Generated outputs

The script writes the following to `results/`:

- Per-model, per-ETF train and test predictions.
- Forecast evaluation metrics.
- Portfolio equity curves and rebalance allocation history.
- Gross and 10-basis-point cost-sensitivity performance summaries.
- Run metadata, including dates, seeds and epochs completed.

The repository does not hard-code model scores. The result tables are generated by running the experiment against the downloaded data.

## Limitations and reproducibility

- Yahoo Finance is a third-party data source. Keep a dated local copy of the downloaded adjusted-price panel when exact replication is required.
- Neural-network results depend on data revisions, package versions, hardware and random initialisation.
- A positive information coefficient or historical backtest does not establish future profitability.
- The 10 bp sensitivity is a simplified turnover-cost assumption; it does not fully model market impact, borrow availability, financing, taxes or liquidity.
- The long-short portfolio has both long and short exposure and may not be suitable for every investor.

## Author

**Hillary Irungu** · [GitHub](https://github.com/Hillaryirungu) · [Portfolio](https://hillaryirungu.github.io)

## License

Released under the MIT License. See [LICENSE](LICENSE).
