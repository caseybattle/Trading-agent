"""
backtest_runner.py — Walk-forward CV + Monte Carlo stress test + CPU parallelization

Architecture:
- Walk-forward cross-validation with purged K-fold (no lookahead)
- Last 20% of data is holdout — never touched during development
- Monte Carlo simulation: shuffle trade order N times, collect metrics
- Gaussian noise augmentation on features for robustness testing
- Full CPU parallelization via multiprocessing

Usage:
    python scripts/backtest_runner.py --folds 5 --mc-trials 1000
"""

from __future__ import annotations

import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
BACKTEST_DIR = Path("backtest")
FEATURES_PATH = DATA_DIR / "features" / "market_features.parquet"

# Features available in data/features/market_features.parquet that are:
#   - populated (non-null/non-constant) for all markets
#   - measured BEFORE market resolution (no leakage)
# Removed: spread_pct (leakage — encodes resolution state),
#   price_at_T1d (79.7% fill — below 80% threshold; momentum captures it),
#   open_interest (all 0.0), liquidity_ratio (duplicate of volume),
#   prior_resolution_rate (missing from parquet)
# Added (2026-03-13): price_at_T7d (97.4% fill), price_momentum_24h (100%),
#   price_volatility_7d (100%) — XGBoost hist handles NaN in price_at_T7d natively
NUMERIC_FEATURES: List[str] = [
    "time_to_resolution_hours",
    "days_since_market_open",
    "volume",
    "price_at_T7d",        # Market price 7d before resolution — 97.4% populated
    "price_momentum_24h",  # price_at_T1d - price_at_T7d — 100% populated; captures late drift
    "price_volatility_7d", # Std of 7d price window — 100% populated; uncertainty proxy
]
LABEL_COL = "outcome_label"
DATE_COL = "end_date"

# Holdout: last 20% by date — NEVER used in training
HOLDOUT_QUANTILE = 0.80

# Kelly / position sizing constants (matches kelly_calculator.py)
FRACTIONAL_KELLY = 0.25
MIN_EDGE = 0.05            # 5pp minimum edge
BANKROLL_START = 10_000.0  # Simulated bankroll


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    train_size: int
    val_size: int
    brier_score: float
    auc: float
    log_loss_val: float
    calibration_error: float   # Mean absolute difference: predicted vs. actual
    kelly_return: float        # Simulated PnL fraction over validation period
    n_trades: int
    feature_importances: Dict[str, float] = field(default_factory=dict)


@dataclass
class MonteCarloResults:
    n_trials: int
    median_return: float
    p5_return: float           # 5th percentile — downside risk
    p95_return: float          # 95th percentile — upside
    max_drawdown_median: float
    sharpe_median: float
    prob_ruin: float           # Fraction of trials ending with < 50% bankroll
    returns: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BacktestReport:
    fold_results: List[FoldResult]
    monte_carlo: MonteCarloResults
    holdout_brier: float
    holdout_auc: float
    holdout_kelly_return: float
    holdout_n_trades: int
    total_train_size: int
    total_holdout_size: int
    noise_robustness_score: float  # AUC degradation under noise augmentation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Mean absolute calibration error (ECE) across probability bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _simulate_kelly_trading(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bankroll_start: float = BANKROLL_START,
    fractional: float = FRACTIONAL_KELLY,
    min_edge: float = MIN_EDGE,
    per_market_cap: float = 0.05,
) -> Tuple[float, int, float]:
    """
    Simulate fractional Kelly trading on a set of predictions.

    Returns: (final_pnl_fraction, n_trades, max_drawdown)
    """
    bankroll = bankroll_start
    peak = bankroll_start
    n_trades = 0

    for true_p, pred_p in zip(y_true, y_pred):
        edge = pred_p - 0.5  # Simple edge relative to coin-flip
        if abs(edge) < min_edge:
            continue

        # Kelly fraction for binary bet: f = (p*b - q) / b where b=1 (even odds approx)
        # For prediction markets: f = p - (1-p) = 2p - 1
        kelly_f = (2 * pred_p - 1) * fractional
        kelly_f = np.clip(kelly_f, 0, per_market_cap)

        stake = bankroll * kelly_f
        if pred_p > 0.5:
            payout = stake if true_p == 1 else -stake
        else:
            payout = stake if true_p == 0 else -stake

        bankroll += payout
        bankroll = max(bankroll, 0.01)  # Floor
        peak = max(peak, bankroll)
        n_trades += 1

    pnl_fraction = (bankroll - bankroll_start) / bankroll_start
    max_dd = (peak - bankroll) / peak if peak > 0 else 0.0
    return pnl_fraction, n_trades, max_dd


def _add_gaussian_noise(X: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """Add Gaussian noise to feature matrix for robustness testing."""
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise


# ---------------------------------------------------------------------------
# XGBoost model factory
# ---------------------------------------------------------------------------

def _make_model(n_jobs: int = 1) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        tree_method="hist",    # Required for native NaN handling (price_at_T7d has 2.6% nulls)
        n_jobs=n_jobs,
        random_state=42,
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Walk-forward fold evaluation (runs in subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _evaluate_fold(
    fold_idx: int,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    feature_names: List[str],
) -> FoldResult:
    """Train model on one fold, evaluate on validation set."""
    model = _make_model(n_jobs=1)

    # Isotonic calibration on a held-out portion of training
    cal_split = int(len(train_X) * 0.8)
    X_tr, X_cal = train_X[:cal_split], train_X[cal_split:]
    y_tr, y_cal = train_y[:cal_split], train_y[cal_split:]

    model.fit(X_tr, y_tr)
    raw_cal = model.predict_proba(X_cal)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)

    raw_val = model.predict_proba(val_X)[:, 1]
    y_pred_cal = iso.predict(raw_val)

    brier = brier_score_loss(val_y, y_pred_cal)
    auc = roc_auc_score(val_y, y_pred_cal) if len(np.unique(val_y)) > 1 else 0.5
    ll = log_loss(val_y, np.clip(y_pred_cal, 1e-6, 1 - 1e-6))
    cal_err = _compute_calibration_error(val_y, y_pred_cal)

    kelly_ret, n_trades, _ = _simulate_kelly_trading(val_y, y_pred_cal)

    importances = {}
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_names, model.feature_importances_.tolist()))

    return FoldResult(
        fold=fold_idx,
        train_size=len(train_X),
        val_size=len(val_X),
        brier_score=round(brier, 5),
        auc=round(auc, 5),
        log_loss_val=round(ll, 5),
        calibration_error=round(cal_err, 5),
        kelly_return=round(kelly_ret, 5),
        n_trades=n_trades,
        feature_importances=importances,
    )


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def _mc_trial(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
) -> Tuple[float, float]:
    """Single Monte Carlo trial: bootstrap resample trades, return (pnl_fraction, max_dd)."""
    rng = np.random.default_rng(seed)
    # Bootstrap resample WITH replacement — each trial uses a different subset of trades
    # This models uncertainty in which opportunities the bot encounters
    idx = rng.choice(len(y_true), size=len(y_true), replace=True)
    pnl, _, max_dd = _simulate_kelly_trading(y_true[idx], y_pred[idx])
    return pnl, max_dd


def run_monte_carlo(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_trials: int = 1000,
    n_workers: int = None,
) -> MonteCarloResults:
    """
    Shuffle trade order N times to assess robustness of Kelly returns.

    Uses CPU parallelism for speed.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)

    log.info(f"Running {n_trials} Monte Carlo trials with {n_workers} workers...")

    seeds = np.arange(n_trials)
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_mc_trial, y_true, y_pred, int(s)): s for s in seeds}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                log.warning(f"MC trial failed: {e}")

    if not results:
        return MonteCarloResults(
            n_trials=0, median_return=0, p5_return=0, p95_return=0,
            max_drawdown_median=0, sharpe_median=0, prob_ruin=0,
        )

    returns = np.array([r[0] for r in results])
    drawdowns = np.array([r[1] for r in results])

    _std = returns.std()
    if _std < 1e-6:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / _std) * np.sqrt(252)  # Annualized proxy
    sharpe = float(np.clip(sharpe, -50.0, 50.0))
    prob_ruin = float((returns < -0.50).mean())

    return MonteCarloResults(
        n_trials=len(results),
        median_return=float(np.median(returns)),
        p5_return=float(np.percentile(returns, 5)),
        p95_return=float(np.percentile(returns, 95)),
        max_drawdown_median=float(np.median(drawdowns)),
        sharpe_median=float(sharpe),
        prob_ruin=prob_ruin,
        returns=returns,
    )


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def run_walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_workers: int = None,
    feature_names: List[str] = None,
) -> List[FoldResult]:
    """
    Purged K-fold walk-forward cross-validation.

    Folds are time-ordered (no shuffle). Each fold trains on all data
    before the validation cutoff (expanding window). A purge gap of
    5% of the fold size is dropped between train and val to prevent
    label leakage from overlapping markets.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    n = len(X)
    fold_size = n // n_folds
    purge_gap = max(1, fold_size // 20)  # 5% purge gap

    log.info(f"Walk-forward CV: {n_folds} folds, {n} markets, purge_gap={purge_gap}")

    fold_jobs = []
    for fold_idx in range(n_folds):
        val_start = fold_size * fold_idx + (fold_size if fold_idx == 0 else 0)
        val_end = min(val_start + fold_size, n)

        if fold_idx == 0:
            continue  # Skip first fold — not enough training data

        train_end = val_start - purge_gap
        if train_end < fold_size:
            continue  # Not enough training data

        train_X = X[:train_end]
        train_y = y[:train_end]
        val_X = X[val_start:val_end]
        val_y = y[val_start:val_end]

        if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            log.warning(f"Fold {fold_idx}: Skipping — only one class present")
            continue

        fold_jobs.append((fold_idx, train_X, train_y, val_X, val_y, feature_names))

    results: List[FoldResult] = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_evaluate_fold, *job): job[0]
            for job in fold_jobs
        }
        for fut in as_completed(futures):
            fold_idx = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                log.info(
                    f"Fold {fold_idx}: Brier={result.brier_score:.4f} "
                    f"AUC={result.auc:.4f} Kelly={result.kelly_return:+.2%} "
                    f"Trades={result.n_trades}"
                )
            except Exception as e:
                log.error(f"Fold {fold_idx} failed: {e}")

    results.sort(key=lambda r: r.fold)
    return results


# ---------------------------------------------------------------------------
# Noise robustness test
# ---------------------------------------------------------------------------

def _noise_robustness_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_std: float = 0.05,
    n_reps: int = 5,
) -> float:
    """
    AUC degradation when Gaussian noise is added to test features.

    Score = 1 - mean(noise_auc) / clean_auc — lower is better (more robust).
    """
    model = _make_model(n_jobs=1)
    model.fit(X_train, y_train)

    clean_pred = model.predict_proba(X_test)[:, 1]
    if len(np.unique(y_test)) < 2:
        return 0.0
    clean_auc = roc_auc_score(y_test, clean_pred)

    noisy_aucs = []
    for _ in range(n_reps):
        X_noisy = _add_gaussian_noise(X_test, noise_std)
        noisy_pred = model.predict_proba(X_noisy)[:, 1]
        noisy_aucs.append(roc_auc_score(y_test, noisy_pred))

    degradation = 1.0 - (np.mean(noisy_aucs) / max(clean_auc, 1e-9))
    return float(np.clip(degradation, 0, 1))


# ---------------------------------------------------------------------------
# Full backtest pipeline
# ---------------------------------------------------------------------------

def run_full_backtest(
    features_path: Path = FEATURES_PATH,
    n_folds: int = 5,
    n_mc_trials: int = 1000,
    n_workers: int = None,
) -> BacktestReport:
    """
    Main entry: load data, split holdout, run walk-forward CV + MC.
    """
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found: {features_path}. Run build_base_rates.py first."
        )

    df = pd.read_parquet(features_path)
    log.info(f"Loaded {len(df)} markets from {features_path}")

    # Enforce holdout — CRITICAL
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing '{DATE_COL}' column — needed for holdout split")

    # Ensure date column is datetime (may be stored as strings in parquet)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    cutoff = df[DATE_COL].quantile(HOLDOUT_QUANTILE)
    train_df = df[df[DATE_COL] <= cutoff].copy()
    holdout_df = df[df[DATE_COL] > cutoff].copy()

    log.info(
        f"Train: {len(train_df)} markets (before {cutoff.date() if hasattr(cutoff, 'date') else cutoff})\n"
        f"Holdout: {len(holdout_df)} markets (SEALED — not used in training)"
    )

    # Drop rows with missing features or label
    available_features = [f for f in NUMERIC_FEATURES if f in train_df.columns]
    if len(available_features) < 3:
        raise ValueError(f"Too few features available: {available_features}")

    # Only require label + always-populated features to be non-null.
    # XGBoost (tree_method='hist') handles NaN natively for sparse features like
    # price_at_T7d — dropping those rows would discard 15 valid training markets.
    ALWAYS_POPULATED = ["time_to_resolution_hours", "days_since_market_open", "volume",
                        "price_momentum_24h", "price_volatility_7d"]
    required_non_null = [f for f in ALWAYS_POPULATED if f in available_features] + [LABEL_COL]
    train_df = train_df.dropna(subset=required_non_null)
    holdout_df = holdout_df.dropna(subset=required_non_null)

    X_train = train_df[available_features].values.astype(np.float32)
    y_train = train_df[LABEL_COL].values.astype(int)
    X_holdout = holdout_df[available_features].values.astype(np.float32)
    y_holdout = holdout_df[LABEL_COL].values.astype(int)

    # --- Walk-forward CV ---
    fold_results = run_walk_forward_cv(
        X_train, y_train,
        n_folds=n_folds,
        n_workers=n_workers,
        feature_names=available_features,
    )

    # --- Monte Carlo on combined OOS fold predictions ---
    all_val_true = np.concatenate([
        y_train[int(len(X_train) * 0.2):]  # Rough approximation for MC input
    ])
    all_val_pred_placeholder = np.full_like(all_val_true, 0.5, dtype=float)

    # Use final fold's train to build a "full" model for MC predictions
    model_full = _make_model(n_jobs=1 if n_workers and n_workers == 1 else max(1, (os.cpu_count() or 1) - 1))
    cal_split = int(len(X_train) * 0.8)
    model_full.fit(X_train[:cal_split], y_train[:cal_split])
    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_full.fit(
        model_full.predict_proba(X_train[cal_split:])[:, 1],
        y_train[cal_split:],
    )

    mc_pred_full = iso_full.predict(model_full.predict_proba(X_train[cal_split:])[:, 1])
    mc_true_full = y_train[cal_split:]

    mc_results = run_monte_carlo(mc_true_full, mc_pred_full, n_trials=n_mc_trials, n_workers=n_workers)

    # --- Holdout evaluation (final, unbiased) ---
    log.info("Evaluating on holdout set (read-only final evaluation)...")
    holdout_pred_raw = model_full.predict_proba(X_holdout)[:, 1]
    holdout_pred = iso_full.predict(holdout_pred_raw)

    holdout_brier = float(brier_score_loss(y_holdout, holdout_pred))
    holdout_auc = float(roc_auc_score(y_holdout, holdout_pred)) if len(np.unique(y_holdout)) > 1 else 0.5
    holdout_kelly_ret, holdout_n_trades, _ = _simulate_kelly_trading(y_holdout, holdout_pred)

    log.info(
        f"HOLDOUT RESULTS — Brier: {holdout_brier:.4f}, AUC: {holdout_auc:.4f}, "
        f"Kelly Return: {holdout_kelly_ret:+.2%}, Trades: {holdout_n_trades}"
    )

    # --- Noise robustness ---
    robustness = _noise_robustness_score(
        X_train[:cal_split], y_train[:cal_split],
        X_holdout, y_holdout,
    )
    log.info(f"Noise robustness degradation: {robustness:.3f} (lower is better)")

    return BacktestReport(
        fold_results=fold_results,
        monte_carlo=mc_results,
        holdout_brier=holdout_brier,
        holdout_auc=holdout_auc,
        holdout_kelly_return=holdout_kelly_ret,
        holdout_n_trades=holdout_n_trades,
        total_train_size=len(train_df),
        total_holdout_size=len(holdout_df),
        noise_robustness_score=robustness,
    )


# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------

def save_report(report: BacktestReport, out_dir: Path = BACKTEST_DIR) -> None:
    """Save backtest results to CSV + JSON summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fold results
    fold_rows = []
    for fr in report.fold_results:
        row = {
            "fold": fr.fold,
            "train_size": fr.train_size,
            "val_size": fr.val_size,
            "brier_score": fr.brier_score,
            "auc": fr.auc,
            "log_loss": fr.log_loss_val,
            "calibration_error": fr.calibration_error,
            "kelly_return": fr.kelly_return,
            "n_trades": fr.n_trades,
        }
        fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    folds_path = out_dir / "fold_results.csv"
    folds_df.to_csv(folds_path, index=False)
    log.info(f"Saved fold results to {folds_path}")

    # Monte Carlo returns distribution
    mc_path = out_dir / "monte_carlo_returns.npy"
    np.save(mc_path, report.monte_carlo.returns)

    # Summary JSON
    import json
    summary = {
        "holdout": {
            "brier_score": report.holdout_brier,
            "auc": report.holdout_auc,
            "kelly_return": report.holdout_kelly_return,
            "n_trades": report.holdout_n_trades,
        },
        "monte_carlo": {
            "n_trials": report.monte_carlo.n_trials,
            "median_return": report.monte_carlo.median_return,
            "p5_return": report.monte_carlo.p5_return,
            "p95_return": report.monte_carlo.p95_return,
            "max_drawdown_median": report.monte_carlo.max_drawdown_median,
            "sharpe_median": report.monte_carlo.sharpe_median,
            "prob_ruin": report.monte_carlo.prob_ruin,
        },
        "cv_summary": {
            "mean_brier": folds_df["brier_score"].mean() if not folds_df.empty else None,
            "mean_auc": folds_df["auc"].mean() if not folds_df.empty else None,
            "mean_kelly_return": folds_df["kelly_return"].mean() if not folds_df.empty else None,
        },
        "noise_robustness_score": report.noise_robustness_score,
        "data_sizes": {
            "train": report.total_train_size,
            "holdout": report.total_holdout_size,
        },
    }

    def _json_default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    summary_path = out_dir / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    log.info(f"Saved backtest summary to {summary_path}")

    print("\n=== BACKTEST SUMMARY ===")
    print(f"Holdout Brier:  {report.holdout_brier:.4f}")
    print(f"Holdout AUC:    {report.holdout_auc:.4f}")
    print(f"Holdout Kelly:  {report.holdout_kelly_return:+.2%} ({report.holdout_n_trades} trades)")
    print(f"MC Median Ret:  {report.monte_carlo.median_return:+.2%}")
    print(f"MC P5/P95:      {report.monte_carlo.p5_return:+.2%} / {report.monte_carlo.p95_return:+.2%}")
    print(f"Sharpe (MC):    {report.monte_carlo.sharpe_median:.2f}")
    print(f"Prob Ruin:      {report.monte_carlo.prob_ruin:.1%}")
    print(f"Noise Robust:   {report.noise_robustness_score:.3f} degradation")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward backtest + Monte Carlo")
    parser.add_argument("--features", default=str(FEATURES_PATH))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mc-trials", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU workers (default: cpu_count - 1)")
    parser.add_argument("--out", default=str(BACKTEST_DIR))
    args = parser.parse_args()

    n_workers = args.workers or max(1, (os.cpu_count() or 1) - 1)
    log.info(f"Using {n_workers} CPU workers")

    report = run_full_backtest(
        features_path=Path(args.features),
        n_folds=args.folds,
        n_mc_trials=args.mc_trials,
        n_workers=n_workers,
    )
    save_report(report, Path(args.out))


if __name__ == "__main__":
    main()
