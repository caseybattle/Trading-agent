# Prediction Market Trading Bot — Project Context

This file is auto-injected into every Claude Code session for this project.

## What This Project Is
An elite multi-agent prediction market trading bot targeting Kalshi BTC range markets (KXBTC series). Goal: maximum profitability through superforecasting-level calibration, cross-market correlation awareness, Thompson sampling strategy selection, walk-forward backtesting, and continuous self-improvement via automated loss postmortem.

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
│   └── dashboard.md            ← 6-tab Streamlit spec
└── scripts/
    ├── collect_kalshi.py       ← Kalshi REST API scraper
    ├── build_base_rates.py     ← Per-category base rates → data/base_rates.json
    ├── kelly_calculator.py     ← Correlation-adjusted Kelly sizing
    ├── regime_detector.py      ← HMM for prediction market regime switching
    ├── strategy_ensemble.py    ← Thompson sampling bandit + 5 strategies
    ├── calibration_tracker.py  ← Brier score tracking + bias correction
    ├── correlation_engine.py   ← Cross-market correlation + arbitrage signals
    ├── backtest_runner.py      ← Walk-forward + Monte Carlo + parallel CPU
    ├── dashboard.py            ← Streamlit 6-tab live dashboard
    ├── kalshi_btc_trader.py    ← Live BTC range market trader (RSA-PSS auth, auto-trade)
    ├── bankroll_tracker.py     ← P&L tracking, trade ledger, bankroll management
    ├── auto_resolver.py        ← Auto-resolves settled markets, triggers postmortem
    ├── strategy_optimizer.py   ← Self-improving model parameter tuning
    ├── loss_postmortem.py      ← 5-specialist parallel loss round-table assessment
    ├── kxbtc_backtest.py       ← Walk-forward backtest for BTC range strategy
    └── pull_kxbtc_history.py   ← Historical KXBTC data fetcher (17,249 markets)
```

## Data Locations
```
data/
├── kalshi_resolved.parquet         ← Raw resolved Kalshi markets
├── features/market_features.parquet← Engineered features + outcome labels
├── base_rates.json                 ← Per-category: YES rate, median resolution time
└── market_correlations.parquet     ← Sparse cross-market correlation graph
backtest/
├── kxbtc_historical.parquet    ← 60 days of settled KXBTC market data
├── kxbtc_backtest_results.json ← Latest backtest results
├── kxbtc_sweep_results.json    ← Parameter sweep results
├── strategy_config.json        ← Auto-tuned model parameters (read by trader at startup)
├── optimization_log.csv        ← Optimization history
├── loss_postmortem.json        ← Latest 5-specialist postmortem report
└── postmortem_log.csv          ← Postmortem run history
trades/
├── bankroll.json               ← Current bankroll state (daily P&L, floor $0.50)
├── live_trades.parquet         ← Trade ledger
└── signals_log.csv             ← Signal history with outcomes
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
- **5-specialist postmortem**: Vol Analyst, Timing Analyst, Market Intel, Pattern Matcher, Counterfactual — run in parallel via ThreadPoolExecutor after every resolved loss batch

## Automated Self-Improvement Loop
```
auto_resolver.py resolves settled trades
  -> loss_postmortem.run_postmortem() (5 parallel specialists)
     -> strategy_optimizer.run_optimization()
        -> updates backtest/strategy_config.json
  -> kalshi_btc_trader.py loads config at next startup
```

## Risk Rules (Non-Negotiable)
- Per-market max: 5% of bankroll
- Category max: 15% of bankroll
- Total correlation-adjusted exposure: 40% max
- Daily loss hard stop: 3% (checked at top of each scan cycle)
- Min edge to trade: 8 percentage points (configurable via strategy_config.json)
- Min liquidity: $5k Kalshi
- Max slippage: 2% (applied in backtest)
- Minimum bankroll floor: $0.50 (enforced in bankroll_tracker)
- File lock: prevents concurrent trader instances (msvcrt on Windows)
- Ticker cooldown: 30-min dedup guard on auto-traded signals

## Kalshi Authentication
- RSA-PSS signing (NOT PKCS1v15)
- Signature message: `timestamp_ms + method.upper() + path`
- Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
- Credentials in .env: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH

## Platform APIs
- **Kalshi markets**: `https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXBTC&status=open`
- **Kalshi orders**: `POST https://api.elections.kalshi.com/trade-api/v2/portfolio/orders`
- **BTC price**: `https://api.coinbase.com/v2/prices/BTC-USD/spot`

## Running the Bot
```bash
# Live trader (single cycle)
python scripts/kalshi_btc_trader.py --once --bankroll 10

# Live trader with auto-trading
python scripts/kalshi_btc_trader.py --auto-trade --bankroll 10

# Auto-resolve settled trades + trigger postmortem
python scripts/auto_resolver.py

# Backtest with parameter sweep
python scripts/kxbtc_backtest.py --sweep --bankroll 10

# Loss postmortem (dry run)
python scripts/loss_postmortem.py --dry-run

# Dashboard
streamlit run scripts/dashboard.py

# Bankroll status
python scripts/bankroll_tracker.py status
```

## Scheduled Tasks (Windows Task Scheduler)
- `kalshi-btc-morning-scan`: daily 9 AM EST — single cycle scan
- `kalshi-btc-active-monitor`: every 30 min, 11 AM–9 PM EST
- `kalshi-btc-expiry-intensive`: every 15 min, 1–6 PM EST
- `kxbtc-auto-resolve`: every 30 min, 9 AM–10 PM — resolves + triggers postmortem
- `kxbtc-strategy-optimizer-daily`: daily 10 PM — standalone optimizer run
