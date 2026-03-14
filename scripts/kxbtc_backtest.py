"""
kxbtc_backtest.py — Backtest simulation engine for Kalshi BTC range market strategy.

Reads:   backtest/kxbtc_historical.parquet  (produced by pull_kxbtc_history.py)
Writes:  backtest/kxbtc_backtest_results.json

Expected parquet columns:
    market_id        (str)    unique market identifier
    ticker           (str)    e.g. "KXBTC-25MAR13-T83500"
    btc_at_open      (float)  BTC/USD spot price when market opened
    range_low        (float)  lower bound of the range (USD)
    range_high       (float)  upper bound of the range (USD)
    minutes_duration (float)  total minutes the market is open
    yes_ask_at_open  (float)  market's YES ask price at open (0-1); 0 = unknown
    close_time       (datetime) UTC timestamp when market closed/resolved
    result           (str)    "YES" or "NO"

Usage:
    python scripts/kxbtc_backtest.py --bankroll 10 --vol 0.01 --min-edge 0.08
    python scripts/kxbtc_backtest.py --bankroll 10 --vol 0.012 --min-edge 0.06
    python scripts/kxbtc_backtest.py --sweep
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DIR = _ROOT / "backtest"
HISTORICAL_PATH = BACKTEST_DIR / "kxbtc_historical.parquet"
RESULTS_PATH = BACKTEST_DIR / "kxbtc_backtest_results.json"

# ---------------------------------------------------------------------------
# Model constants (overridable via CLI)
# ---------------------------------------------------------------------------
BTC_HOURLY_VOL: float = 0.01   # 1% hourly vol
MIN_EDGE: float = 0.08          # 8 pp minimum edge
FRACTIONAL_KELLY: float = 0.25
MAX_POSITION_PCT: float = 0.05  # 5% max per trade
TRAIN_FRAC: float = 0.70        # first 70% = in-sample, last 30% = OOS

# ---------------------------------------------------------------------------
# Simulation safety caps
# ---------------------------------------------------------------------------
MAX_BANKROLL_CAP = 1000.0   # Cap at 100x starting bankroll for simulation sanity
ABS_MAX_COST = 5.0          # Never risk more than $5 per trade regardless of bankroll
SLIPPAGE_BPS = 200          # 2% slippage per CLAUDE.md risk rules


# ---------------------------------------------------------------------------
# Core pricing model
# ---------------------------------------------------------------------------
def _ncdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_fair(btc: float, low: float, high: float, minutes_left: float) -> float:
    """
    Log-normal fair value for a BTC-in-range binary contract.

    Returns probability that BTC stays (or ends) inside [low, high]
    given current spot `btc`, range bounds, and time remaining.
    """
    if btc <= 0 or minutes_left <= 0:
        return 0.0
    hours_left = max(minutes_left / 60.0, 1.0 / 60.0)
    sigma = BTC_HOURLY_VOL * math.sqrt(hours_left)

    p_high = _ncdf(math.log(high / btc) / sigma) if btc < high else 1.0
    p_low  = _ncdf(math.log(low  / btc) / sigma) if btc > low  else 0.0
    return max(0.0, min(1.0, p_high - p_low))


# ---------------------------------------------------------------------------
# Market-ask heuristic (used when yes_ask_at_open is missing/zero)
# ---------------------------------------------------------------------------
def estimate_market_ask(
    btc: float,
    low: float,
    high: float,
    fair_value: float,
    sigma: float,
) -> float:
    """
    Simple spread heuristic when live market data is unavailable.

    Three tiers based on how deeply in- or out-of-range BTC is:
      - BTC inside range:           market ask = max(0.30, fair - 0.15)
      - BTC near range (< 1 sigma): market ask = fair * 0.85
      - BTC far from range:         market ask = fair * 0.90
    """
    in_range = low <= btc <= high

    if in_range:
        return max(0.30, fair_value - 0.15)

    # Distance to nearest bound as fraction of current price
    dist_low  = abs(math.log(btc / low))   if btc > 0 and low  > 0 else float("inf")
    dist_high = abs(math.log(high / btc))  if btc > 0 and high > 0 else float("inf")
    dist = min(dist_low, dist_high)

    if dist < sigma:      # within 1-sigma of range edge
        return max(0.01, min(0.99, fair_value * 0.85))
    return max(0.01, min(0.99, fair_value * 0.90))


# ---------------------------------------------------------------------------
# Per-trade data class
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    market_id: str
    ticker: str
    direction: str        # "YES" or "NO"
    fair_value: float
    market_ask: float
    edge: float
    kelly_fraction: float
    contracts: int
    cost: float
    pnl: float
    bankroll_before: float
    bankroll_after: float
    result: str           # actual "YES" / "NO"
    close_time: pd.Timestamp
    win: bool


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    label: str
    n_markets: int = 0
    n_yes_signals: int = 0
    n_no_signals: int = 0
    n_trades: int = 0
    n_wins: int = 0
    total_pnl: float = 0.0
    starting_bankroll: float = 10.0
    ending_bankroll: float = 10.0
    return_pct: float = 0.0
    sharpe: float = float("nan")
    max_drawdown_pct: float = 0.0
    avg_edge: float = 0.0
    win_rate: float = 0.0
    calibration: Dict[str, Dict] = field(default_factory=dict)
    hourly_pnl: Dict[int, float] = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)


def simulate(
    df: pd.DataFrame,
    starting_bankroll: float,
    vol: float,
    min_edge: float,
    label: str = "full",
) -> SimResult:
    """
    Run the log-normal strategy simulation on dataframe `df`.

    The dataframe must be sorted chronologically before calling.
    """
    res = SimResult(label=label, starting_bankroll=starting_bankroll)
    res.n_markets = len(df)
    res.ending_bankroll = starting_bankroll

    bankroll = starting_bankroll
    trade_records: List[Trade] = []
    bankroll_curve: List[float] = [bankroll]
    peak_bankroll = bankroll

    # Calibration buckets: key = "30-40", "40-50", etc.
    cal_buckets: Dict[str, List[Tuple[float, bool]]] = {
        "30-40": [], "40-50": [], "50-60": [], "60-70": [], "70+": []
    }
    hourly_pnl: Dict[int, List[float]] = {h: [] for h in range(24)}

    for _, row in df.iterrows():
        btc      = float(row.get("btc_at_open", 0) or 0)
        low      = float(row.get("range_low",   0) or 0)
        high     = float(row.get("range_high",  0) or 0)
        mins     = float(row.get("minutes_duration", 60) or 60)
        ask_raw  = float(row.get("yes_ask_at_open", 0) or 0)
        result   = str(row.get("result", "")).strip().upper()
        close_t  = row.get("close_time", pd.NaT)
        mkt_id   = str(row.get("market_id", ""))
        ticker   = str(row.get("ticker", ""))

        if btc <= 0 or low <= 0 or high <= 0 or low >= high:
            continue
        if result not in ("YES", "NO"):
            continue

        hours_left = max(mins / 60.0, 1.0 / 60.0)
        sigma = vol * math.sqrt(hours_left)
        fair  = compute_fair(btc, low, high, mins)

        # Determine market ask
        if ask_raw > 0.0:
            market_ask = float(np.clip(ask_raw, 0.01, 0.99))
        else:
            market_ask = estimate_market_ask(btc, low, high, fair, sigma)

        # Apply slippage to make backtest more realistic
        market_ask_slipped = market_ask * (1 + SLIPPAGE_BPS / 10000)

        # Edges (use slipped price for conservative edge, original for cost)
        edge_yes = fair - market_ask_slipped
        yes_bid  = max(0.01, market_ask - 0.05)   # typical 5-cent spread
        edge_no  = yes_bid - fair                  # profit from selling YES (buying NO)

        # Track calibration for this market
        fv_pct = fair * 100
        if 30 <= fv_pct < 40:
            bucket = "30-40"
        elif 40 <= fv_pct < 50:
            bucket = "40-50"
        elif 50 <= fv_pct < 60:
            bucket = "50-60"
        elif 60 <= fv_pct < 70:
            bucket = "60-70"
        elif fv_pct >= 70:
            bucket = "70+"
        else:
            bucket = None

        # --------------- YES trade ---------------
        if edge_yes >= min_edge:
            if market_ask <= 0:
                continue
            res.n_yes_signals += 1
            kelly_raw = (edge_yes / max(1.0 - market_ask, 0.01)) * FRACTIONAL_KELLY
            kelly_frac = min(kelly_raw, MAX_POSITION_PCT)
            _yes_val = kelly_frac * bankroll / market_ask
            if not math.isfinite(_yes_val):
                continue
            contracts = max(1, round(_yes_val))
            cost = contracts * market_ask
            if cost > bankroll:
                contracts = max(1, math.floor(bankroll / market_ask))
                cost = contracts * market_ask

            # Cap absolute cost
            if cost > ABS_MAX_COST:
                contracts = max(1, math.floor(ABS_MAX_COST / market_ask))
                cost = contracts * market_ask

            win = result == "YES"
            if win:
                pnl = (1.0 - market_ask) * contracts
            else:
                pnl = -market_ask * contracts

            br_before = bankroll
            bankroll = max(0.0, bankroll + pnl)
            bankroll = min(MAX_BANKROLL_CAP, bankroll)
            bankroll_curve.append(bankroll)
            peak_bankroll = max(peak_bankroll, bankroll)

            t = Trade(
                market_id=mkt_id, ticker=ticker, direction="YES",
                fair_value=fair, market_ask=market_ask, edge=edge_yes,
                kelly_fraction=kelly_frac, contracts=contracts, cost=cost,
                pnl=pnl, bankroll_before=br_before, bankroll_after=bankroll,
                result=result, close_time=close_t, win=win,
            )
            trade_records.append(t)

            if bucket:
                cal_buckets[bucket].append((fair, win))
            if pd.notna(close_t):
                hour = pd.Timestamp(close_t).hour
                hourly_pnl[hour].append(pnl)

        # --------------- NO trade ---------------
        elif edge_no >= min_edge:
            res.n_no_signals += 1
            # For NO: we're selling YES at yes_bid, implying NO ask = 1 - yes_bid
            no_ask = 1.0 - yes_bid
            if no_ask <= 0:
                continue
            kelly_raw = (edge_no / max(1.0 - no_ask, 0.01)) * FRACTIONAL_KELLY
            kelly_frac = min(kelly_raw, MAX_POSITION_PCT)
            _no_val = kelly_frac * bankroll / no_ask
            if not math.isfinite(_no_val):
                continue
            contracts = max(1, round(_no_val))
            cost = contracts * no_ask
            if cost > bankroll:
                contracts = max(1, math.floor(bankroll / no_ask))
                cost = contracts * no_ask

            # Cap absolute cost
            if cost > ABS_MAX_COST:
                contracts = max(1, math.floor(ABS_MAX_COST / no_ask))
                cost = contracts * no_ask

            win = result == "NO"
            if win:
                pnl = (1.0 - no_ask) * contracts
            else:
                pnl = -no_ask * contracts

            br_before = bankroll
            bankroll = max(0.0, bankroll + pnl)
            bankroll = min(MAX_BANKROLL_CAP, bankroll)
            bankroll_curve.append(bankroll)
            peak_bankroll = max(peak_bankroll, bankroll)

            t = Trade(
                market_id=mkt_id, ticker=ticker, direction="NO",
                fair_value=fair, market_ask=market_ask, edge=edge_no,
                kelly_fraction=kelly_frac, contracts=contracts, cost=cost,
                pnl=pnl, bankroll_before=br_before, bankroll_after=bankroll,
                result=result, close_time=close_t, win=win,
            )
            trade_records.append(t)

            # Calibration: for NO trades, the "fair" probability of NO = 1 - fair
            fair_no = 1.0 - fair
            fv_no_pct = fair_no * 100
            if 30 <= fv_no_pct < 40:
                nb = "30-40"
            elif 40 <= fv_no_pct < 50:
                nb = "40-50"
            elif 50 <= fv_no_pct < 60:
                nb = "50-60"
            elif 60 <= fv_no_pct < 70:
                nb = "60-70"
            elif fv_no_pct >= 70:
                nb = "70+"
            else:
                nb = None

            if nb:
                cal_buckets[nb].append((fair_no, win))
            if pd.notna(close_t):
                hour = pd.Timestamp(close_t).hour
                hourly_pnl[hour].append(pnl)

    # --------------- Aggregate metrics ---------------
    n_trades = len(trade_records)
    res.n_trades = n_trades
    res.n_wins   = sum(1 for t in trade_records if t.win)
    res.total_pnl = bankroll - starting_bankroll
    res.ending_bankroll = round(bankroll, 4)
    res.return_pct = round((bankroll - starting_bankroll) / starting_bankroll * 100, 2) if starting_bankroll > 0 else 0.0
    res.win_rate = round(res.n_wins / n_trades * 100, 1) if n_trades > 0 else 0.0
    res.avg_edge = round(
        float(np.mean([t.edge for t in trade_records])) * 100, 2
    ) if n_trades > 0 else 0.0

    # Sharpe (annualized from per-trade returns)
    pnl_arr = np.array([t.pnl for t in trade_records])
    pnl_arr = pnl_arr[np.isfinite(pnl_arr)]  # filter out Inf/NaN
    if len(pnl_arr) >= 2:
        mean_pnl = float(np.mean(pnl_arr))
        std_pnl = float(np.std(pnl_arr, ddof=1))
        if std_pnl > 1e-10:
            res.sharpe = round(mean_pnl / std_pnl * math.sqrt(252 * 4), 3)
        else:
            res.sharpe = 0.0
    else:
        res.sharpe = 0.0

    # Max drawdown
    curve = np.array(bankroll_curve, dtype=np.float64)
    curve = np.where(np.isfinite(curve), curve, 0.0)  # replace Inf/NaN with 0
    running_max = np.maximum.accumulate(curve)
    safe_max = np.where(running_max > 0, running_max, 1.0)
    drawdowns = (running_max - curve) / safe_max
    res.max_drawdown_pct = round(float(np.max(drawdowns)) * 100, 2) if len(drawdowns) > 0 else 0.0

    # Calibration table
    cal_out: Dict[str, Dict] = {}
    for bkt, entries in cal_buckets.items():
        if entries:
            fv_vals = [e[0] for e in entries]
            wins    = [e[1] for e in entries]
            cal_out[bkt] = {
                "n": len(entries),
                "avg_fair_value_pct": round(float(np.mean(fv_vals)) * 100, 1),
                "actual_win_rate_pct": round(float(np.mean(wins)) * 100, 1),
            }
    res.calibration = cal_out

    # Best/worst hour
    hr_summary: Dict[int, float] = {}
    for h, pnls in hourly_pnl.items():
        if pnls:
            hr_summary[h] = round(float(np.sum(pnls)), 4)
    res.hourly_pnl = hr_summary

    res.trades = [
        {
            "market_id": t.market_id,
            "ticker": t.ticker,
            "direction": t.direction,
            "fair_value": round(t.fair_value, 4),
            "market_ask": round(t.market_ask, 4),
            "edge_pp": round(t.edge * 100, 2),
            "kelly_fraction": round(t.kelly_fraction, 4),
            "contracts": t.contracts,
            "cost": round(t.cost, 4),
            "pnl": round(t.pnl, 4),
            "bankroll_after": round(t.bankroll_after, 4),
            "result": t.result,
            "win": t.win,
            "close_hour": int(pd.Timestamp(t.close_time).hour) if pd.notna(t.close_time) else -1,
        }
        for t in trade_records
    ]
    return res


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------
def split_chronological(df: pd.DataFrame, train_frac: float = TRAIN_FRAC) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by close_time and split at train_frac boundary."""
    if "close_time" in df.columns:
        df = df.sort_values("close_time").reset_index(drop=True)
    n = len(df)
    cutoff = int(n * train_frac)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def _try_tabulate(headers: List[str], rows: List[List]) -> str:
    try:
        from tabulate import tabulate  # type: ignore
        return tabulate(rows, headers=headers, tablefmt="simple", floatfmt=".2f")
    except ImportError:
        lines = ["  ".join(str(h).ljust(18) for h in headers)]
        lines.append("-" * (20 * len(headers)))
        for row in rows:
            lines.append("  ".join(str(v).ljust(18) for v in row))
        return "\n".join(lines)


def print_sim_result(res: SimResult) -> None:
    """Print a formatted summary for one simulation segment."""
    print(f"\n{'='*60}")
    print(f"  Segment: {res.label}")
    print(f"{'='*60}")

    summary_rows = [
        ["Markets scanned",       res.n_markets],
        ["YES signals",           res.n_yes_signals],
        ["NO signals",            res.n_no_signals],
        ["Total trades",          res.n_trades],
        ["Wins",                  res.n_wins],
        ["Win rate (%)",          f"{res.win_rate:.1f}"],
        ["Avg edge at entry (pp)",f"{res.avg_edge:.2f}"],
        ["Starting bankroll ($)", f"{res.starting_bankroll:.2f}"],
        ["Ending bankroll ($)",   f"{res.ending_bankroll:.4f}"],
        ["Total P&L ($)",         f"{res.total_pnl:+.4f}"],
        ["Return (%)",            f"{res.return_pct:+.2f}"],
        ["Sharpe ratio",          f"{res.sharpe:.3f}" if not math.isnan(res.sharpe) else "N/A"],
        ["Max drawdown (%)",      f"{res.max_drawdown_pct:.2f}"],
    ]
    print(_try_tabulate(["Metric", "Value"], summary_rows))

    if res.calibration:
        print("\n  Calibration (fair_value bucket vs actual win rate):")
        cal_rows = []
        for bkt in ["30-40", "40-50", "50-60", "60-70", "70+"]:
            if bkt in res.calibration:
                c = res.calibration[bkt]
                cal_rows.append([
                    f"{bkt}%",
                    c["n"],
                    f"{c['avg_fair_value_pct']:.1f}%",
                    f"{c['actual_win_rate_pct']:.1f}%",
                ])
        if cal_rows:
            print(_try_tabulate(
                ["FV Bucket", "N trades", "Avg Fair Value", "Actual Win Rate"],
                cal_rows,
            ))

    if res.hourly_pnl:
        active_hours = sorted(res.hourly_pnl.items(), key=lambda x: x[1])
        best  = active_hours[-1] if active_hours else None
        worst = active_hours[0]  if active_hours else None
        if best and worst:
            print(f"\n  Best  hour of day: {best[0]:02d}:00  P&L ${best[1]:+.4f}")
            print(f"  Worst hour of day: {worst[0]:02d}:00  P&L ${worst[1]:+.4f}")


def print_sweep_table(sweep_results: List[Dict]) -> None:
    """Print parameter sweep comparison table."""
    print(f"\n{'='*70}")
    print("  Parameter Sweep Results (vol x min_edge combinations)")
    print(f"{'='*70}")
    rows = []
    for r in sweep_results:
        oos = r.get("oos", {})
        rows.append([
            r["vol"],
            r["min_edge"],
            oos.get("n_trades", 0),
            f"{oos.get('win_rate', 0):.1f}",
            f"{oos.get('return_pct', 0):+.2f}",
            f"{oos.get('sharpe', float('nan')):.3f}" if not math.isnan(oos.get("sharpe", float("nan"))) else "N/A",
            f"{oos.get('max_drawdown_pct', 0):.2f}",
        ])
    print(_try_tabulate(
        ["Vol", "Min Edge", "Trades", "Win%", "Return%", "Sharpe", "MaxDD%"],
        rows,
    ))

    # Find best by Sharpe (OOS), fallback to return
    valid = [r for r in sweep_results if not math.isnan(r.get("oos", {}).get("sharpe", float("nan")))]
    if valid:
        best = max(valid, key=lambda x: x["oos"].get("sharpe", float("-inf")))
    else:
        best = max(sweep_results, key=lambda x: x.get("oos", {}).get("return_pct", float("-inf")))
    print(f"\n  Best combination (OOS Sharpe): vol={best['vol']}, min_edge={best['min_edge']}")


# ---------------------------------------------------------------------------
# Public API: run_backtest()
# ---------------------------------------------------------------------------
def run_backtest(
    bankroll: float = 10.0,
    vol: float = BTC_HOURLY_VOL,
    min_edge: float = MIN_EDGE,
    historical_path: Path = HISTORICAL_PATH,
) -> Dict:
    """
    Load historical data and run the walk-forward simulation.

    Returns a dict suitable for JSON serialisation and dashboard use.
    Saves results to RESULTS_PATH automatically.
    """
    if not historical_path.exists():
        msg = (
            f"\n[ERROR] Historical data not found at:\n  {historical_path}\n\n"
            "Run the data collector first:\n"
            "  python scripts/pull_kxbtc_history.py\n"
        )
        print(msg, file=sys.stderr)
        return {"error": "historical data missing", "path": str(historical_path)}

    df = pd.read_parquet(historical_path)
    print(f"[INFO] Loaded {len(df)} markets from {historical_path.name}")

    # Ensure close_time is datetime
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")

    train_df, oos_df = split_chronological(df)
    print(f"[INFO] Train: {len(train_df)} markets | OOS: {len(oos_df)} markets")

    is_res  = simulate(train_df, bankroll, vol, min_edge, label="in-sample (70%)")
    oos_res = simulate(oos_df,   bankroll, vol, min_edge, label="out-of-sample (30%)")

    def _as_dict(r: SimResult) -> Dict:
        d = asdict(r)
        d.pop("trades", None)   # omit full trade list from summary JSON (too large)
        return d

    output = {
        "parameters": {"bankroll": bankroll, "vol": vol, "min_edge": min_edge},
        "in_sample":  _as_dict(is_res),
        "out_of_sample": _as_dict(oos_res),
        "all_trades_oos": oos_res.trades,
    }

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"[INFO] Results saved to {RESULTS_PATH}")

    return output


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------
def run_sweep(bankroll: float = 10.0) -> List[Dict]:
    """Try 9 combinations of vol x min_edge and return ranked results."""
    vols      = [0.008, 0.01, 0.012]
    min_edges = [0.06, 0.08, 0.10]

    if not HISTORICAL_PATH.exists():
        print(
            f"\n[ERROR] Historical data not found: {HISTORICAL_PATH}\n"
            "Run: python scripts/pull_kxbtc_history.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_parquet(HISTORICAL_PATH)
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    train_df, oos_df = split_chronological(df)

    sweep_results = []
    for v in vols:
        for me in min_edges:
            is_r  = simulate(train_df, bankroll, v, me, label=f"IS  vol={v} me={me}")
            oos_r = simulate(oos_df,   bankroll, v, me, label=f"OOS vol={v} me={me}")
            sweep_results.append({
                "vol": v,
                "min_edge": me,
                "in_sample":  asdict(is_r),
                "oos":        asdict(oos_r),
            })

    # Save sweep to its own file
    sweep_path = BACKTEST_DIR / "kxbtc_sweep_results.json"
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(sweep_path, "w", encoding="utf-8") as fh:
        json.dump(sweep_results, fh, indent=2, default=str)
    print(f"[INFO] Sweep results saved to {sweep_path}")

    return sweep_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backtest the Kalshi BTC range strategy using log-normal fair value.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bankroll", type=float, default=10.0,
                   help="Starting bankroll in USD")
    p.add_argument("--vol", type=float, default=BTC_HOURLY_VOL,
                   help="Hourly BTC volatility assumption (e.g. 0.01 = 1%%)")
    p.add_argument("--min-edge", type=float, default=MIN_EDGE,
                   help="Minimum edge in probability points (e.g. 0.08 = 8pp)")
    p.add_argument("--sweep", action="store_true",
                   help="Run 9 vol x min-edge combinations and show comparison table")
    p.add_argument("--max-cost", type=float, default=5.0,
                   help="Max cost per trade in dollars (default: 5.0)")
    p.add_argument("--max-bankroll", type=float, default=1000.0,
                   help="Max bankroll cap for simulation (default: 1000.0)")
    p.add_argument("--data", type=Path, default=HISTORICAL_PATH,
                   help="Path to kxbtc_historical.parquet")
    return p


def main() -> None:
    global ABS_MAX_COST, MAX_BANKROLL_CAP

    parser = _build_parser()
    args   = parser.parse_args()

    # Apply CLI overrides for simulation caps
    ABS_MAX_COST = args.max_cost
    MAX_BANKROLL_CAP = args.max_bankroll

    # Check data exists early
    hist_path = args.data
    if not hist_path.exists():
        print(
            f"\n[ERROR] Historical data file not found:\n  {hist_path}\n\n"
            "To generate it, run the history collector:\n"
            "  python scripts/pull_kxbtc_history.py\n\n"
            "That script pulls resolved KXBTC markets from the Kalshi API\n"
            "and saves them to backtest/kxbtc_historical.parquet.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.sweep:
        sweep_results = run_sweep(bankroll=args.bankroll)
        # Print individual OOS results
        for sr in sweep_results:
            oos_r = SimResult(**sr["oos"])
            print_sim_result(oos_r)
        print_sweep_table(sweep_results)
    else:
        output = run_backtest(
            bankroll=args.bankroll,
            vol=args.vol,
            min_edge=args.min_edge,
            historical_path=hist_path,
        )
        if "error" in output:
            sys.exit(1)

        is_r  = SimResult(**output["in_sample"])
        oos_r = SimResult(**output["out_of_sample"])

        # Reconstruct sharpe (lost float precision in dict round-trip)
        for res_obj, key in [(is_r, "in_sample"), (oos_r, "out_of_sample")]:
            s = output[key].get("sharpe", float("nan"))
            res_obj.sharpe = float(s) if s is not None else float("nan")

        print_sim_result(is_r)
        print_sim_result(oos_r)

        print(f"\n{'='*60}")
        print("  Summary: Out-of-Sample Performance")
        print(f"{'='*60}")
        print(f"  vol={args.vol}  min_edge={args.min_edge}  bankroll=${args.bankroll:.2f}")
        print(f"  Trades: {oos_r.n_trades}  |  Win rate: {oos_r.win_rate:.1f}%  |  "
              f"Return: {oos_r.return_pct:+.2f}%  |  MaxDD: {oos_r.max_drawdown_pct:.2f}%")


if __name__ == "__main__":
    main()
