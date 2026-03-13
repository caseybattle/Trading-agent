"""
live_trader.py — Prediction Market Live Trading Engine

Scan cycle:
  1. Fetch live Polymarket markets (Gamma API)
  2. Engineer features (volume anomaly z-score, momentum, etc.)
  3. Fit ensemble on historical features (ML model refit if stale)
  4. For each candidate market:
     a. predict()  → raw probability
     b. calibrate() → isotonic-corrected probability
     c. portfolio_kelly_check() → position size / approval
     d. record trade if approved + PAPER_TRADING guard
  5. Update open positions with latest prices / resolved outcomes
  6. Enforce daily loss stop
  7. Persist trades to parquet

Usage:
    python scripts/live_trader.py --run          # single scan cycle
    python scripts/live_trader.py --daemon       # loop every 3600s
    python scripts/live_trader.py --paper        # force paper-trading mode
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Path setup — allow importing sibling scripts
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent
_PROJECT_DIR = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from kelly_calculator import portfolio_kelly_check
from strategy_ensemble import StrategyEnsemble
from calibration_tracker import CalibrationTracker
from correlation_engine import load_correlation_graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_TRADING: bool = True          # Set False only for real money execution
DEFAULT_BANKROLL: float = 10_000.0  # USD — override via env BANKROLL
MIN_EDGE: float = 0.05              # 5 percentage points minimum edge
MIN_LIQUIDITY: float = 10_000.0     # $10k Polymarket minimum
DAILY_LOSS_STOP: float = 0.03       # 3% of bankroll hard stop
MAX_NEW_TRADES_PER_RUN: int = 5     # Max new positions per scan cycle
SCAN_INTERVAL_SECONDS: int = 3_600  # 1 hour between daemon scans

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR          = _PROJECT_DIR / "data"
FEATURES_PATH     = DATA_DIR / "features" / "market_features.parquet"
BASE_RATES_PATH   = DATA_DIR / "base_rates.json"
TRADES_DIR        = _PROJECT_DIR / "trades"
TRADES_PATH       = TRADES_DIR / "live_trades.parquet"
LOGS_DIR          = _PROJECT_DIR / "logs"

TRADES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "live_trader.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trade schema — all columns required by dashboard
# ---------------------------------------------------------------------------
TRADE_COLS = [
    "trade_id",        # str UUID
    "market_id",       # str
    "question",        # str
    "category",        # str
    "platform",        # str e.g. "Polymarket"
    "direction",       # "YES" | "NO"
    "entry_price",     # float 0-1
    "current_price",   # float 0-1
    "stake_pct",       # float (fraction of bankroll)
    "kelly_fraction",  # float raw kelly sizing
    "predicted_prob",  # float raw model output
    "cal_prob",        # float after isotonic calibration
    "edge",            # float cal_prob - market_price
    "signal_source",   # str
    "status",          # "open" | "closed" | "paper"
    "outcome",         # int  1=win 0=loss -1=pending
    "pnl_pct",         # float unrealized/realized PnL as fraction
    "entered_at",      # str ISO timestamp
    "closed_at",       # str ISO timestamp or ""
    "paper",           # bool
]


# ---------------------------------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------------------------------

def _make_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Return requests.Session with exponential backoff retry."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "prediction-market-bot/1.0"})
    return session


# ---------------------------------------------------------------------------
# Category inference (mirrors collect_polymarket.infer_category)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "politics":  ["election", "president", "senate", "congress", "vote", "democrat",
                  "republican", "governor", "primary", "ballot", "legislation", "policy"],
    "crypto":    ["bitcoin", "btc", "ethereum", "eth", "crypto", "defi", "nft",
                  "solana", "binance", "coinbase", "token", "blockchain"],
    "sports":    ["nfl", "nba", "mlb", "nhl", "soccer", "championship", "super bowl",
                  "world cup", "playoffs", "match", "game", "tournament", "season"],
    "science":   ["nasa", "climate", "temperature", "co2", "earthquake", "hurricane",
                  "fda", "trial", "vaccine", "pandemic", "discovery"],
    "finance":   ["fed", "interest rate", "gdp", "inflation", "recession", "s&p",
                  "nasdaq", "dow", "earnings", "ipo", "merger"],
    "geopolitics": ["war", "russia", "ukraine", "china", "taiwan", "nato", "sanctions",
                    "ceasefire", "military", "invasion"],
}


def _categorize(question: str) -> str:
    """Classify market question into a category string."""
    q = question.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return cat
    return "other"


# ---------------------------------------------------------------------------
# Fetch live markets from Polymarket Gamma API
# ---------------------------------------------------------------------------

def fetch_live_markets(
    session: requests.Session,
    limit: int = 200,
    min_volume: float = MIN_LIQUIDITY,
) -> pd.DataFrame:
    """
    Fetch open (non-closed) Polymarket markets from Gamma API.

    Returns DataFrame with columns matching the feature engineering schema.
    Empty DataFrame on error.
    """
    url = f"{GAMMA_BASE}/markets"
    params = {
        "closed": "false",
        "active": "true",
        "limit": limit,
    }
    try:
        resp = session.get(url, params=params, timeout=20)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        log.error(f"fetch_live_markets failed: {exc}")
        return pd.DataFrame()

    records = []
    now = datetime.now(timezone.utc)
    for m in raw:
        try:
            # Parse YES price from outcomePrices JSON string
            op_raw = m.get("outcomePrices", "[]")
            if isinstance(op_raw, str):
                prices = json.loads(op_raw)
            else:
                prices = op_raw
            yes_price = float(prices[0]) if prices else 0.5

            volume = float(m.get("volume", 0) or 0)
            if volume < min_volume:
                continue

            question = m.get("question", "") or ""
            end_date_str = m.get("endDate", "") or ""
            start_date_str = m.get("startDate", "") or ""

            # Time features
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                hours_to_end = max((end_dt - now).total_seconds() / 3600, 0)
            except Exception:
                hours_to_end = 24 * 30  # default 30 days

            try:
                start_dt = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
                days_open = max((now - start_dt).total_seconds() / 86400, 0)
            except Exception:
                days_open = 0.0

            records.append({
                "market_id":              str(m.get("id", "")),
                "question":               question,
                "category":               _categorize(question),
                "yes_price":              yes_price,
                "volume":                 volume,
                "time_to_resolution_hours": hours_to_end,
                "days_since_market_open": days_open,
                "end_date":               end_date_str,
                "start_date":             start_date_str,
                "active":                 bool(m.get("active", True)),
            })
        except Exception as exc:
            log.debug(f"Skipping market {m.get('id', '?')}: {exc}")
            continue

    log.info(f"fetch_live_markets: {len(records)} markets above ${min_volume:,.0f} volume")
    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature engineering for live markets
# ---------------------------------------------------------------------------

def _engineer_features(df: pd.DataFrame, hist_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add ML feature columns to live market DataFrame.

    Features match MLStrategy.FEATURE_COLS:
        time_to_resolution_hours, days_since_market_open, volume,
        volume_anomaly_score, price_at_T7d, price_momentum_24h, price_volatility_7d

    hist_df: Optional historical features for computing category volume z-scores.
    """
    if df.empty:
        return df

    df = df.copy()

    # Volume anomaly z-score per category
    if hist_df is not None and not hist_df.empty and "volume" in hist_df.columns:
        cat_stats = (
            hist_df.groupby("category")["volume"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "vol_mean", "std": "vol_std"})
        )
        df = df.join(cat_stats, on="category", how="left")
        df["vol_std"] = df["vol_std"].fillna(df["vol_mean"].clip(lower=1000.0) * 0.25)
        df["vol_std"] = df["vol_std"].clip(lower=1000.0)
        df["vol_mean"] = df["vol_mean"].fillna(df["volume"].mean())
        df["volume_anomaly_score"] = (df["volume"] - df["vol_mean"]) / df["vol_std"]
        df.drop(columns=["vol_mean", "vol_std"], inplace=True)
    else:
        df["volume_anomaly_score"] = 0.0

    # Price stub features (live markets don't have history — use current price)
    if "yes_price" in df.columns:
        df["price_at_T7d"]       = df["yes_price"]
        df["price_momentum_24h"] = 0.0
        df["price_volatility_7d"] = 0.05  # conservative default
    else:
        df["price_at_T7d"]        = 0.5
        df["price_momentum_24h"]  = 0.0
        df["price_volatility_7d"] = 0.05

    return df


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_trades() -> pd.DataFrame:
    """Load existing trade log. Returns empty DataFrame if not found."""
    if not TRADES_PATH.exists():
        return pd.DataFrame(columns=TRADE_COLS)
    try:
        df = pd.read_parquet(TRADES_PATH)
        # Ensure all schema columns exist
        for col in TRADE_COLS:
            if col not in df.columns:
                df[col] = None
        return df
    except Exception as exc:
        log.warning(f"Could not load trades: {exc}")
        return pd.DataFrame(columns=TRADE_COLS)


def save_trades(df: pd.DataFrame) -> None:
    """Persist trade log to parquet."""
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRADES_PATH, index=False)
    log.info(f"Saved {len(df)} trades to {TRADES_PATH}")


def load_state() -> Dict:
    """
    Load runtime state: bankroll, open positions, daily PnL.

    Returns dict with keys: bankroll, daily_pnl, last_reset_date
    """
    state_path = TRADES_DIR / "state.json"
    default = {
        "bankroll": float(os.environ.get("BANKROLL", DEFAULT_BANKROLL)),
        "daily_pnl": 0.0,
        "last_reset_date": datetime.now(timezone.utc).date().isoformat(),
    }
    if not state_path.exists():
        return default
    try:
        with open(state_path) as f:
            state = json.load(f)
        # Reset daily PnL if new day
        today = datetime.now(timezone.utc).date().isoformat()
        if state.get("last_reset_date") != today:
            state["daily_pnl"] = 0.0
            state["last_reset_date"] = today
        return state
    except Exception as exc:
        log.warning(f"Could not load state: {exc} — using defaults")
        return default


def _save_state(state: Dict) -> None:
    state_path = TRADES_DIR / "state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Ensemble fitting
# ---------------------------------------------------------------------------

def fit_ensemble(ensemble: StrategyEnsemble) -> bool:
    """
    Fit ensemble on historical feature data (if available).

    Returns True if fit succeeded, False if data missing.
    """
    if not FEATURES_PATH.exists():
        log.warning(f"Features file not found: {FEATURES_PATH} — ensemble using priors only")
        return False

    try:
        hist_df = pd.read_parquet(FEATURES_PATH)
        base_rates: Dict = {}
        if BASE_RATES_PATH.exists():
            with open(BASE_RATES_PATH) as f:
                base_rates = json.load(f)

        # Use 80% training split (last 20% is holdout — never used in training)
        if "end_date" in hist_df.columns:
            # Parse dates — parquet stores end_date as strings; convert to datetime for quantile
            end_dates = pd.to_datetime(hist_df["end_date"], errors="coerce", utc=True)
            cutoff = end_dates.quantile(0.80)
            train_df = hist_df[(end_dates <= cutoff).fillna(False)]
        else:
            train_df = hist_df

        log.info(f"fit_ensemble: {len(train_df)} training rows")
        ensemble.fit(train_df, base_rates)
        return True

    except Exception as exc:
        log.error(f"fit_ensemble failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

def get_open_positions(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Return {market_id: stake_pct} for all open trades.
    """
    if trades_df.empty:
        return {}
    open_trades = trades_df[trades_df["status"] == "open"]
    return dict(zip(open_trades["market_id"], open_trades["stake_pct"].astype(float)))


def check_daily_stop(state: Dict) -> bool:
    """Return True if daily loss stop has been triggered."""
    bankroll = state.get("bankroll", DEFAULT_BANKROLL)
    daily_pnl = state.get("daily_pnl", 0.0)
    stop_threshold = -bankroll * DAILY_LOSS_STOP
    if daily_pnl <= stop_threshold:
        log.warning(
            f"Daily loss stop triggered: PnL={daily_pnl:.2f} <= threshold={stop_threshold:.2f}"
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Update open positions with latest prices
# ---------------------------------------------------------------------------

def update_open_positions(
    trades_df: pd.DataFrame,
    live_markets: pd.DataFrame,
    session: requests.Session,
    state: Dict,
) -> pd.DataFrame:
    """
    Update current_price and pnl_pct for all open positions.

    Closes positions where market has resolved (market no longer active).
    """
    if trades_df.empty or live_markets.empty:
        return trades_df

    live_price_map: Dict[str, float] = {}
    if not live_markets.empty and "market_id" in live_markets.columns:
        live_price_map = dict(zip(live_markets["market_id"], live_markets["yes_price"]))

    bankroll = state.get("bankroll", DEFAULT_BANKROLL)
    pnl_delta = 0.0

    for idx, row in trades_df.iterrows():
        if row.get("status") != "open":
            continue

        mid = row["market_id"]
        direction = row.get("direction", "YES")
        entry = float(row.get("entry_price", 0.5))
        stake = float(row.get("stake_pct", 0.0))

        if mid in live_price_map:
            cur_price = live_price_map[mid]
            trades_df.at[idx, "current_price"] = cur_price

            # Unrealized PnL as fraction of bankroll
            if direction == "YES":
                pnl = (cur_price - entry) * stake * bankroll
            else:
                pnl = (entry - cur_price) * stake * bankroll
            prev_pnl = float(row.get("pnl_pct", 0.0))
            pnl_delta += pnl - prev_pnl          # accumulate change in unrealized PnL
            trades_df.at[idx, "pnl_pct"] = round(pnl, 4)
        else:
            # Market no longer in live feed — may have resolved
            log.info(f"Market {mid} not in live feed — checking resolution")
            # Leave open; dashboard/resolve pass handles final close

    state["daily_pnl"] = state.get("daily_pnl", 0.0) + pnl_delta
    return trades_df


# ---------------------------------------------------------------------------
# Core scan-and-trade loop
# ---------------------------------------------------------------------------

def scan_and_trade(
    live_markets: pd.DataFrame,
    ensemble: StrategyEnsemble,
    tracker: CalibrationTracker,
    trades_df: pd.DataFrame,
    state: Dict,
    paper: bool = PAPER_TRADING,
) -> Tuple[pd.DataFrame, int]:
    """
    Evaluate each live market, apply Kelly sizing, record approved trades.

    Returns:
        (updated_trades_df, n_new_trades)
    """
    if live_markets.empty:
        return trades_df, 0

    if check_daily_stop(state):
        log.warning("Daily stop active — no new trades this cycle")
        return trades_df, 0

    bankroll = state.get("bankroll", DEFAULT_BANKROLL)

    # Load base rates
    base_rates: Dict = {}
    if BASE_RATES_PATH.exists():
        try:
            with open(BASE_RATES_PATH) as f:
                base_rates = json.load(f)
        except Exception:
            pass

    # Load correlation graph
    try:
        corr_graph = load_correlation_graph(path=_PROJECT_DIR / "data" / "market_correlations.parquet")
    except Exception as exc:
        log.warning(f"Could not load correlation graph: {exc}")
        from correlation_engine import CorrelationGraph
        corr_graph = CorrelationGraph()

    # Current open positions
    open_positions = get_open_positions(trades_df)

    # Build category_map for Kelly (market_id -> category for open positions)
    open_category_map: Dict[str, str] = {}
    if not trades_df.empty:
        open_trades = trades_df[trades_df["status"] == "open"]
        open_category_map = dict(zip(open_trades["market_id"], open_trades["category"].astype(str)))

    # Track already-open market IDs to avoid duplicates
    open_market_ids = set(open_positions.keys())

    n_new = 0
    new_rows: List[Dict] = []

    for _, market in live_markets.iterrows():
        if n_new >= MAX_NEW_TRADES_PER_RUN:
            break

        mid = str(market.get("market_id", ""))
        if not mid or mid in open_market_ids:
            continue

        yes_price = float(market.get("yes_price", 0.5))
        category  = str(market.get("category", "other"))
        question  = str(market.get("question", ""))

        # Skip extreme prices (near 0 or 1 — little edge available)
        if yes_price < 0.02 or yes_price > 0.98:
            continue

        # Strategy prediction
        try:
            strategy_name, signal = ensemble.predict(market, base_rates)
        except Exception as exc:
            log.debug(f"predict failed for {mid}: {exc}")
            continue

        raw_prob = float(getattr(signal, "predicted_prob", 0.5))

        # Isotonic calibration
        cal_prob = tracker.calibrate(raw_prob, strategy_name)

        # Determine direction and edge
        if cal_prob >= 0.5:
            direction   = "YES"
            p_market    = yes_price
            p_true      = cal_prob
            edge        = cal_prob - yes_price
        else:
            direction   = "NO"
            p_market    = 1.0 - yes_price
            p_true      = 1.0 - cal_prob
            edge        = (1.0 - cal_prob) - (1.0 - yes_price)

        if edge < MIN_EDGE:
            continue

        # Kelly check — risk gate
        try:
            kelly_result = portfolio_kelly_check(
                candidate_market_id=mid,
                candidate_category=category,
                p_true=p_true,
                p_market=p_market,
                open_positions=open_positions,
                category_map=open_category_map,
                correlation_matrix=corr_graph.edges,
                bankroll=bankroll,
            )
        except Exception as exc:
            log.warning(f"portfolio_kelly_check failed for {mid}: {exc}")
            continue

        if not kelly_result.get("approved", False):
            log.debug(f"Kelly rejected {mid}: {kelly_result.get('reason', '')}")
            continue

        kelly_fraction = float(kelly_result.get("kelly_fraction", 0.0))
        if kelly_fraction <= 0:
            continue

        # Record trade
        trade_id = f"{mid}_{int(time.time())}_{n_new}"
        status   = "open"  # paper flag tracked via `paper` bool column

        trade_row = {
            "trade_id":       trade_id,
            "market_id":      mid,
            "question":       question,
            "category":       category,
            "platform":       "Polymarket",
            "direction":      direction,
            "entry_price":    round(p_market, 6),
            "current_price":  round(p_market, 6),
            "stake_pct":      round(kelly_fraction, 6),
            "kelly_fraction": round(kelly_fraction, 6),
            "predicted_prob": round(raw_prob, 6),
            "cal_prob":       round(cal_prob, 6),
            "edge":           round(edge, 6),
            "signal_source":  strategy_name,
            "status":         status,
            "outcome":        -1,
            "pnl_pct":        0.0,
            "entered_at":     datetime.now(timezone.utc).isoformat(),
            "closed_at":      "",
            "paper":          paper,
        }

        new_rows.append(trade_row)
        open_positions[mid]      = kelly_fraction
        open_category_map[mid]   = category
        open_market_ids.add(mid)
        n_new += 1

        mode = "PAPER" if paper else "LIVE"
        log.info(
            f"[{mode}] Trade: {mid[:20]}... {direction} @ {p_market:.3f} | "
            f"edge={edge:.3f} kelly={kelly_fraction:.4f} strategy={strategy_name}"
        )

        # Real-money guard
        if not paper:
            log.warning("LIVE TRADING: actual order submission not implemented — paper logged")

    if new_rows:
        new_df   = pd.DataFrame(new_rows)
        trades_df = pd.concat([trades_df, new_df], ignore_index=True)

    return trades_df, n_new


# ---------------------------------------------------------------------------
# Main run cycle
# ---------------------------------------------------------------------------

def run(paper: bool = PAPER_TRADING) -> None:
    """Execute one full scan-and-trade cycle."""
    log.info(f"=== Live trader cycle start | paper={paper} ===")
    session = _make_session()

    # Load state
    state = load_state()
    if check_daily_stop(state):
        log.warning("Daily stop active — aborting cycle")
        return

    # Load historical features for ensemble fitting + anomaly z-scores
    hist_df: Optional[pd.DataFrame] = None
    if FEATURES_PATH.exists():
        try:
            hist_df = pd.read_parquet(FEATURES_PATH)
        except Exception as exc:
            log.warning(f"Could not load historical features: {exc}")

    # Fit ensemble
    ensemble = StrategyEnsemble()
    fit_ensemble(ensemble)

    # Load calibration tracker
    tracker = CalibrationTracker.load()

    # Fetch live markets
    live_markets = fetch_live_markets(session)
    if live_markets.empty:
        log.warning("No live markets fetched — aborting cycle")
        return

    # Engineer features
    live_markets = _engineer_features(live_markets, hist_df)

    # Load existing trades
    trades_df = load_trades()

    # Update open positions with latest prices
    trades_df = update_open_positions(trades_df, live_markets, session, state)

    # Scan and trade
    trades_df, n_new = scan_and_trade(
        live_markets, ensemble, tracker, trades_df, state, paper=paper
    )

    # Persist
    save_trades(trades_df)
    tracker.save()
    _save_state(state)

    log.info(f"=== Cycle complete: {n_new} new trades, {len(trades_df)} total ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prediction market live trader")
    parser.add_argument("--run",    action="store_true", help="Execute one scan cycle")
    parser.add_argument("--daemon", action="store_true", help="Loop every SCAN_INTERVAL_SECONDS")
    parser.add_argument("--paper",  action="store_true", default=PAPER_TRADING,
                        help="Force paper-trading mode (default: True)")
    parser.add_argument("--live",   action="store_true",
                        help="Enable real-money mode (overrides --paper)")
    args = parser.parse_args()

    paper = not args.live  # Default to paper unless --live explicitly passed

    if args.daemon:
        log.info(f"Daemon mode: scanning every {SCAN_INTERVAL_SECONDS}s | paper={paper}")
        while True:
            try:
                run(paper=paper)
            except KeyboardInterrupt:
                log.info("Daemon interrupted by user")
                break
            except Exception as exc:
                log.error(f"Cycle failed: {exc}", exc_info=True)
            log.info(f"Sleeping {SCAN_INTERVAL_SECONDS}s...")
            time.sleep(SCAN_INTERVAL_SECONDS)
    elif args.run:
        run(paper=paper)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
