# Prediction Market Trading Bot — Project Context

This file is auto-injected into every Claude Code session for this project.

## What This Project Is
An elite multi-agent prediction market trading bot targeting Polymarket and Kalshi with Hyperliquid delta-neutral hedging for crypto markets. Goal: maximum profitability through superforecasting-level calibration, cross-market correlation awareness, Thompson sampling strategy selection, walk-forward backtesting, and continuous self-improvement.

## Directory Structure
```
prediction-market-bot/
├── SKILL.md                    ← Master orchestration (read this first)
├── CLAUDE.md                   ← This file
├── evals/evals.json            ← 3 eval test cases
├── references/
│   ├── data-collection.md      ← APIs, features, base rates schema
│   ├── strategy-protocol.md    ← Strategy interface + 5 built-in strategies
│   ├── backtesting.md          ← Walk-forward + parallel testing
│   ├── risk-policy.md          ← Kelly rules + exposure limits
│   ├── calibration-engine.md   ← Brier score + isotonic regression
│   ├── correlation-engine.md   ← Cross-market graph + arbitrage
│   └── dashboard.md            ← 5-tab Streamlit spec
└── scripts/
    ├── collect_polymarket.py   ← Gamma API scraper (incremental)
    ├── collect_kalshi.py       ← Kalshi REST API scraper
    ├── build_base_rates.py     ← Per-category base rates → data/base_rates.json
    ├── kelly_calculator.py     ← Correlation-adjusted Kelly sizing
    ├── regime_detector.py      ← HMM for prediction market regime switching
    ├── strategy_ensemble.py    ← Thompson sampling bandit + 5 strategies
    ├── calibration_tracker.py  ← Brier score tracking + bias correction
    ├── correlation_engine.py   ← Cross-market correlation + arbitrage signals
    ├── backtest_runner.py      ← Walk-forward + Monte Carlo + parallel CPU
    └── dashboard.py            ← Streamlit 5-tab dashboard
```

## Data Locations
```
data/
├── polymarket_resolved.parquet     ← Raw resolved Polymarket markets
├── kalshi_resolved.parquet         ← Raw resolved Kalshi markets
├── features/market_features.parquet← Engineered features + outcome labels
├── base_rates.json                 ← Per-category: YES rate, median resolution time
└── market_correlations.parquet     ← Sparse cross-market correlation graph
backtest/                           ← Walk-forward results by strategy/fold
trades/live_trades.parquet          ← Live trade audit trail
```

## Key Design Decisions
- **Thompson sampling bandit** (not rolling Sharpe) for strategy selection — mathematically optimal explore/exploit
- **Walk-forward + purged K-fold CV** (not simple train/test split) — genuine out-of-sample validation
- **Isotonic regression** on raw probability outputs — provably calibrated over time
- **Correlation-adjusted Kelly**: `kelly_raw * max(0.1, 1 - correlation_penalty)` — portfolio-aware sizing
- **Fractional Kelly**: 0.25x baseline, never exceed 0.5x
- **Holdout set**: Last 20% of historical data NEVER used in training — for final model validation only
- **CPU parallelization**: `os.cpu_count()` in backtest_runner for parallel strategy evaluation
- **Data augmentation**: Gaussian noise stress testing in backtest to check robustness

## Risk Rules (Non-Negotiable)
- Per-market max: 5% of bankroll
- Category max: 15% of bankroll
- Total correlation-adjusted exposure: 40% max
- Daily loss hard stop: 3%
- Min edge to trade: 5 percentage points
- Min liquidity: $10k Polymarket, $5k Kalshi
- Max slippage: 2%

## Reusable Source Files (Desktop)
- `C:\Users\casba\Desktop\trading-strategies\kelly_criterion.py` — extended → `kelly_calculator.py`
- `C:\Users\casba\Desktop\trading-strategies\markov_regime_detector.py` — adapted → `regime_detector.py`
- `C:\Users\casba\Desktop\trading-strategies\monte_carlo_simulator.py` — integrated → `backtest_runner.py`

## Platform APIs
- **Polymarket**: `https://gamma-api.polymarket.com/markets?closed=true` (no auth)
- **Kalshi**: `https://trading.kalshi.com/v2/markets?status=settled` (API key required)
- **Hyperliquid**: Perps for delta-neutral crypto hedging

## Running the Bot
```bash
# 1. Collect data
python scripts/collect_polymarket.py
python scripts/collect_kalshi.py
python scripts/build_base_rates.py

# 2. Backtest
python scripts/backtest_runner.py

# 3. Launch dashboard
streamlit run scripts/dashboard.py
```
