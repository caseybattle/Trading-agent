---
name: prediction-market-bot
description: >
  Elite multi-agent prediction market trading bot for Polymarket and Kalshi.
  Combines superforecasting calibration, cross-market correlation awareness,
  Thompson sampling strategy selection, walk-forward backtesting, and a live
  Streamlit dashboard. Targets maximum risk-adjusted returns with strict Kelly
  fractional sizing and hard exposure limits.
version: "1.0"
---

# Prediction Market Trading Bot — Master Orchestration

## What This Bot Does

Trades binary prediction markets on Polymarket and Kalshi using a five-stage pipeline:

1. **Data collection** — Incremental scraping of resolved markets (Polymarket Gamma API, Kalshi REST)
2. **Feature engineering + base rates** — Per-category base rates anchored via superforecasting priors
3. **Strategy selection** — Thompson sampling bandit selects among 5 strategies per market
4. **Risk management** — Correlation-adjusted fractional Kelly sizing with hard exposure caps
5. **Monitoring** — Streamlit dashboard with live positions, calibration tracking, and arbitrage signals

---

## Phase Sequence

Run these in order on first setup. Subsequent runs skip to whichever phase has new data.

### Phase 1: Data Collection

```bash
# Polymarket (no auth required, ~5,000–15,000 resolved markets)
python scripts/collect_polymarket.py

# Kalshi (requires KALSHI_API_KEY environment variable)
export KALSHI_API_KEY=<your_key>
python scripts/collect_kalshi.py
```

Both scripts are **incremental**: they cache a cursor in `data/.polymarket_cursor.json` and `data/.kalshi_cursor.json` and resume from the last fetched page on subsequent runs.

Output:
- `data/polymarket_resolved.parquet`
- `data/kalshi_resolved.parquet`

---

### Phase 2: Feature Engineering + Base Rates

```bash
python scripts/build_base_rates.py
```

Reads both parquet files, computes engineered features, enforces the **20% holdout rule** (last 20% of resolved markets by date are never used in training), and writes:
- `data/features/market_features.parquet` — full feature matrix + outcome labels
- `data/base_rates.json` — per-category YES rate, median resolution days, calibration

---

### Phase 3: Correlation Graph

```bash
python scripts/correlation_engine.py \
    --features data/features/market_features.parquet \
    --out data/market_correlations.parquet \
    --tfidf-threshold 0.60 \
    --alpha 0.5
```

Builds a sparse correlation graph from TF-IDF text similarity + binary outcome correlation. Used to penalize Kelly bet sizes for correlated positions and detect cross-market arbitrage.

Output: `data/market_correlations.parquet`

---

### Phase 4: Backtesting

```bash
python scripts/backtest_runner.py \
    --folds 5 \
    --mc-trials 1000 \
    --workers $(python -c "import os; print(os.cpu_count()-1)")
```

Runs walk-forward purged K-fold cross-validation + Monte Carlo stress testing across all 5 strategies. Uses CPU parallelization. Enforces holdout split.

Output:
- `backtest/fold_results.csv` — per-fold Brier, AUC, Kelly return, n_trades
- `backtest/backtest_summary.json` — aggregate metrics + holdout evaluation
- `backtest/monte_carlo_returns.npy` — return distribution from MC trials

---

### Phase 5: Live Dashboard

```bash
streamlit run scripts/dashboard.py
```

Opens at `http://localhost:8501` with 5 tabs:

| Tab | Contents |
|-----|----------|
| Trade Lifecycle | Open/closed positions, cumulative PnL, trade history |
| Strategy Performance | Fold results, MC return distribution, Thompson sampling weights |
| Calibration | Brier score history, calibration curve, isotonic correction map |
| Correlation Map | Market correlation heatmap, top correlated pairs, arbitrage signals |
| Live Positions | Risk gauge metrics, category exposure, position table |

---

## Strategy Descriptions

All strategies are implemented in `scripts/strategy_ensemble.py`. The Thompson sampling bandit (Beta(α, β) posteriors) selects among them based on historical Brier score performance.

| Strategy | Signal Source | Best For |
|----------|---------------|----------|
| Sentiment | News sentiment aggregation | Politics, macro events |
| Momentum | Price change velocity (24h, 7d) | Trending markets |
| ML (XGBoost) | Engineered feature vector per category | All categories |
| LLM | Language model probability estimate | Long-horizon markets |
| Ensemble | Thompson-weighted blend of above | Default, all markets |

---

## Risk Rules (Non-Negotiable)

| Rule | Limit |
|------|-------|
| Per-market max | 5% of bankroll |
| Category max | 15% of bankroll |
| Correlation-adjusted total exposure | 40% max |
| Daily loss hard stop | 3% |
| Minimum edge to trade | 5 percentage points |
| Minimum liquidity | $10k Polymarket, $5k Kalshi |
| Fractional Kelly | 0.25x baseline, max 0.5x |

Kelly formula: `kelly_raw * max(0.1, 1 - correlation_penalty(market_id, current_positions, graph))`

---

## Key Modules

| Script | Role |
|--------|------|
| `collect_polymarket.py` | Gamma API scraper, incremental |
| `collect_kalshi.py` | Kalshi REST scraper, API key required |
| `build_base_rates.py` | Feature engineering + per-category priors |
| `kelly_calculator.py` | Correlation-adjusted Kelly bet sizing |
| `regime_detector.py` | HMM market regime detection |
| `strategy_ensemble.py` | Thompson sampling + 5 strategy implementations |
| `calibration_tracker.py` | Brier score + isotonic regression calibration |
| `correlation_engine.py` | Cross-market correlation graph + arbitrage |
| `backtest_runner.py` | Walk-forward CV + Monte Carlo + parallel CPU |
| `dashboard.py` | Streamlit 5-tab live dashboard |

---

## Environment Setup

```bash
pip install pandas numpy scipy scikit-learn xgboost streamlit plotly pyarrow hmmlearn requests

# Optional (Kalshi only)
export KALSHI_API_KEY=<your_key>
```

Python 3.10+ required.

---

## Verification Checklist

After Phase 2, verify data quality:
```python
import pandas as pd, json

df = pd.read_parquet('data/features/market_features.parquet')
print(f'Markets: {len(df)}, Features: {df.shape[1]}')
print(f'Outcome balance: {df["outcome"].mean():.2%} YES')

br = json.load(open('data/base_rates.json'))
print('Categories:', list(br.keys()))
```

Expected: >1,000 markets, 8+ features, 40–55% YES rate, 5 categories.

After Phase 4, verify backtest quality:
```python
import json
summary = json.load(open('backtest/backtest_summary.json'))
print(f'Holdout AUC: {summary["holdout_auc"]:.3f}')   # Target: >0.60
print(f'Holdout Brier: {summary["holdout_brier"]:.3f}') # Target: <0.25
```
