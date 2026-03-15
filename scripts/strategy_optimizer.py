"""
strategy_optimizer.py
Self-improving strategy optimizer for the Kalshi BTC trading bot.

After each batch of resolved trades, this script:
  1. Loads resolved trades (outcome == WIN or LOSS) from trades/signals_log.csv
  2. Analyzes calibration, realized vol, edge threshold fitness, time-of-day,
     and classifies failure modes on losing trades
  3. Adjusts model parameters and writes updated config to backtest/strategy_config.json
  4. Appends a row to backtest/optimization_log.csv for historical tracking
  5. Prints a human-readable report

Usage:
    python scripts/strategy_optimizer.py            # run analysis + update config
    python scripts/strategy_optimizer.py --report   # report only, no writes
    python scripts/strategy_optimizer.py --reset    # reset config to defaults

Importable API:
    from strategy_optimizer import run_optimization
    updated_config = run_optimization(dry_run=False)
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

# Storage backend abstraction (Windows LocalStorage / AWS Lambda S3Storage)
from storage_backend import get_storage
_storage = get_storage()

# ---------------------------------------------------------------------------
# Path string constants for display/logging only; actual I/O uses _storage abstraction
# ---------------------------------------------------------------------------
SIGNALS_LOG   = "trades/signals_log.csv"
CONFIG_PATH   = "backtest/strategy_config.json"
OPT_LOG_PATH  = "backtest/optimization_log.csv"

# ---------------------------------------------------------------------------
# Default config -- written on first run if config file is absent
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict = {
    "btc_hourly_vol":          0.01,
    "min_edge_pp":             8.0,
    "fractional_kelly":        0.25,
    "max_position_pct":        0.05,
    "time_decay_threshold_min": 30,
    "time_decay_min_fair":     0.70,
    "last_updated":            "2026-03-13T00:00:00Z",
    "iteration":               0,
    "notes":                   "Initial parameters",
    "avoid_hours":             [],
}

MIN_TRADES_REQUIRED = 5   # do not optimize until we have this many resolved trades

# ---------------------------------------------------------------------------
# Calibration buckets: (lower_bound, upper_bound, midpoint_label)
# ---------------------------------------------------------------------------
CALIB_BUCKETS = [
    (0.30, 0.40, 0.35, "30-40%"),
    (0.40, 0.50, 0.45, "40-50%"),
    (0.50, 0.60, 0.55, "50-60%"),
    (0.60, 0.70, 0.65, "60-70%"),
    (0.70, 1.01, 0.75, "70%+"),
]

# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    # Storage backend handles directory creation via write operations
    pass


def load_config() -> Dict:
    """Load strategy_config.json; create with defaults if missing."""
    _ensure_dirs()
    if not _storage.exists("backtest/strategy_config.json"):
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    cfg = _storage.read_json("backtest/strategy_config.json")
    # Back-fill any keys added after initial creation
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg


def save_config(cfg: Dict) -> None:
    """Persist config to disk using storage backend."""
    _ensure_dirs()
    cfg["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _storage.write_json("backtest/strategy_config.json", cfg)


def reset_config() -> Dict:
    """Overwrite config with factory defaults and return it."""
    fresh = dict(DEFAULT_CONFIG)
    fresh["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh["iteration"] = 0
    save_config(fresh)
    return fresh

# ---------------------------------------------------------------------------
# Signals log loader
# ---------------------------------------------------------------------------

def load_resolved_trades() -> List[Dict]:
    """
    Return rows from signals_log.csv where outcome is WIN or LOSS.

    Handles two on-disk schemas gracefully:

    NEW schema (kalshi_btc_trader.py log_signal() output):
        timestamp, ticker, strategy, direction, range_label, range_low,
        range_high, fair_value, market_ask, market_bid, edge_pp,
        minutes_left, btc_price_at_signal, kelly_fraction,
        recommended_contracts, acted_on, outcome

    OLD schema (earlier kalshi_btc_trader.py version, no outcome column):
        timestamp, ticker, action, yes_no, fair_value, market_mid,
        edge, kelly_fraction, contracts, limit_price, btc_price,
        minutes_to_close, strategy

    Mixed files (old header + new rows appended) are also handled by
    scanning every row's raw field values for WIN/LOSS tokens and
    building a normalized dict on the fly.

    Column aliases: old -> new
        yes_no          -> direction
        market_mid      -> market_ask  (approximation)
        edge (fraction) -> edge_pp (100)
        btc_price       -> btc_price_at_signal
        minutes_to_close-> minutes_left
        limit_price     -> market_ask (preferred over market_mid)
    """
    if not _storage.exists("trades/signals_log.csv"):
        return []

    resolved: List[Dict] = []

    rows = _storage.read_csv("trades/signals_log.csv")
    if not rows:
        return []

    # Convert list of dicts (from read_csv) back to raw CSV lines for compatibility
    raw_lines = []
    if rows:
        # Add header from first row's keys
        header_keys = list(rows[0].keys())
        raw_lines.append(",".join(header_keys) + "\n")

        # Add data rows
        for row in rows:
            values = [str(row.get(k, "")).replace(",", " ") for k in header_keys]
            raw_lines.append(",".join(values) + "\n")

    if not raw_lines:
        return []

    # Detect header columns
    header_line = raw_lines[0].strip()
    header_cols = [c.strip() for c in header_line.split(",")]
    has_outcome_col = "outcome" in header_cols

    # --- Fast path: file header includes 'outcome' ---
    if has_outcome_col:
        import io
        reader = csv.DictReader(io.StringIO("".join(raw_lines)))
        for row in reader:
            outcome = row.get("outcome", "").strip().upper()
            if outcome in ("WIN", "LOSS"):
                resolved.append(_normalize_row(row))
        return resolved

    # --- Mixed-file path: header is OLD schema, rows may use NEW schema ---
    # We scan every data row (raw, comma-split) for WIN or LOSS tokens.
    # When found, we reconstruct a normalized dict using positional matching
    # against the OLD header for early rows, and heuristic matching for
    # appended new-schema rows (which will have more fields than the header).

    old_col_count = len(header_cols)

    for raw_line in raw_lines[1:]:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        fields = raw_line.split(",")

        # Scan all field values for WIN / LOSS
        outcome_val = ""
        outcome_idx = -1
        for idx, fld in enumerate(fields):
            if fld.strip().upper() in ("WIN", "LOSS"):
                outcome_val = fld.strip().upper()
                outcome_idx = idx
                break

        if not outcome_val:
            continue  # not a resolved trade

        # Try to build a dict.  If the row has the same number of fields as
        # the OLD header, use positional mapping.  Otherwise fall back to
        # positional mapping of a known NEW-schema column list.
        if len(fields) == old_col_count:
            row = dict(zip(header_cols, fields))
        else:
            # Assume new schema column order
            new_cols = [
                "timestamp","ticker","strategy","direction","range_label",
                "range_low","range_high","fair_value","market_ask","market_bid",
                "edge_pp","minutes_left","btc_price_at_signal","kelly_fraction",
                "recommended_contracts","acted_on","outcome",
            ]
            row = dict(zip(new_cols, fields))

        row["outcome"] = outcome_val
        resolved.append(_normalize_row(row))

    return resolved


def _normalize_row(row: Dict) -> Dict:
    """
    Normalize a raw CSV row dict to the canonical field names the optimizer uses.

    Canonical names:
        timestamp, ticker, strategy, direction, fair_value,
        market_ask, edge_pp, minutes_left, btc_price_at_signal,
        range_low, range_high, outcome
    """
    out = dict(row)  # copy

    # direction: old schema uses yes_no
    if "direction" not in out and "yes_no" in out:
        out["direction"] = out["yes_no"]

    # market_ask: prefer limit_price (actual cost), fall back to market_mid
    if "market_ask" not in out:
        if "limit_price" in out:
            out["market_ask"] = out["limit_price"]
        elif "market_mid" in out:
            out["market_ask"] = out["market_mid"]

    # edge_pp: old schema stores edge as a fraction (e.g. 0.31); convert to pp
    if "edge_pp" not in out and "edge" in out:
        try:
            out["edge_pp"] = str(float(out["edge"]) * 100)
        except (TypeError, ValueError):
            out["edge_pp"] = "0"

    # btc_price_at_signal: old schema uses btc_price
    if "btc_price_at_signal" not in out and "btc_price" in out:
        out["btc_price_at_signal"] = out["btc_price"]

    # minutes_left: old schema uses minutes_to_close
    if "minutes_left" not in out and "minutes_to_close" in out:
        out["minutes_left"] = out["minutes_to_close"]

    return out


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_win_rate(trades: List[Dict]) -> Tuple[int, int, float]:
    """Return (n_wins, n_losses, win_rate)."""
    wins   = sum(1 for t in trades if t.get("outcome", "").upper() == "WIN")
    losses = len(trades) - wins
    rate   = wins / len(trades) if trades else 0.0
    return wins, losses, rate


def compute_break_even_rate(trades: List[Dict]) -> float:
    """
    Break-even win rate = avg entry cost / 1.0 = avg(market_ask).
    For a binary contract that pays $1 on win, you need win_rate >= market_ask
    to break even (ignoring fees).
    """
    asks = [_safe_float(t.get("market_ask")) for t in trades if t.get("market_ask")]
    return float(np.mean(asks)) if asks else 0.45


def calibration_analysis(trades: List[Dict]) -> List[Dict]:
    """
    Group trades into fair_value buckets and compare expected vs actual win rate.

    Returns a list of dicts with keys:
        label, lower, upper, midpoint, n, expected_wr, actual_wr, delta_pp, flag
    """
    buckets: Dict[int, Dict] = {}
    for i, (lo, hi, mid, label) in enumerate(CALIB_BUCKETS):
        buckets[i] = {"label": label, "lower": lo, "upper": hi,
                      "midpoint": mid, "wins": 0, "total": 0}

    for t in trades:
        fv = _safe_float(t.get("fair_value"))
        outcome = t.get("outcome", "").upper()
        for i, (lo, hi, mid, label) in enumerate(CALIB_BUCKETS):
            if lo <= fv < hi:
                buckets[i]["total"] += 1
                if outcome == "WIN":
                    buckets[i]["wins"] += 1
                break

    results = []
    for b in buckets.values():
        n = b["total"]
        if n == 0:
            continue
        actual_wr  = b["wins"] / n
        expected_wr = b["midpoint"]
        delta_pp   = (actual_wr - expected_wr) * 100  # positive = underconfident
        flag = ""
        if delta_pp < -10:
            flag = "overconfident"
        elif delta_pp > 10:
            flag = "underconfident"
        results.append({
            "label":       b["label"],
            "lower":       b["lower"],
            "upper":       b["upper"],
            "midpoint":    expected_wr,
            "n":           n,
            "expected_wr": expected_wr,
            "actual_wr":   actual_wr,
            "delta_pp":    delta_pp,
            "flag":        flag,
        })
    return results


def compute_vol_adjustment(trades: List[Dict], current_vol: float) -> Tuple[float, str]:
    """
    Estimate realized BTC vol from btc_price_at_signal across signals.
    Uses std dev of log returns between consecutive BTC price observations.

    Returns (new_vol, note).
    """
    prices = []
    for t in trades:
        p = _safe_float(t.get("btc_price_at_signal"))
        if p > 0:
            prices.append(p)

    if len(prices) < 3:
        return current_vol, "insufficient BTC price data for vol estimation"

    # Log returns between consecutive signal prices
    log_returns = [math.log(prices[i] / prices[i - 1])
                   for i in range(1, len(prices))
                   if prices[i - 1] > 0]
    if not log_returns:
        return current_vol, "insufficient log-return data"

    realized_vol = float(np.std(log_returns))  # per-signal std dev
    # Exponential smoothing: 70% current, 30% realized
    blended = 0.7 * current_vol + 0.3 * realized_vol
    # Clamp to [0.005, 0.025]
    new_vol = max(0.005, min(0.025, blended))
    note = (f"realized_vol={realized_vol:.4f} from {len(log_returns)} log-returns; "
            f"blended={blended:.4f}, clamped={new_vol:.4f}")
    return new_vol, note


def adjust_vol_for_calibration(current_vol: float, calib_results: List[Dict]) -> Tuple[float, str]:
    """
    Secondary vol adjustment driven by calibration bias.

    If multiple buckets are consistently overconfident (actual < expected - 10pp):
        -> model underestimates uncertainty -> increase vol estimate
    If multiple buckets are consistently underconfident (actual > expected + 10pp):
        -> model overestimates uncertainty -> decrease vol estimate
    """
    overconf  = sum(1 for b in calib_results if b["flag"] == "overconfident")
    underconf = sum(1 for b in calib_results if b["flag"] == "underconfident")

    if overconf >= 2:
        # Bump vol up by 8%
        new_vol = min(0.025, current_vol * 1.08)
        return new_vol, f"calib: {overconf} overconfident buckets -> vol +8%"
    elif underconf >= 2:
        # Nudge vol down by 5%
        new_vol = max(0.005, current_vol * 0.95)
        return new_vol, f"calib: {underconf} underconfident buckets -> vol -5%"

    return current_vol, "calib: no systematic bias detected"


def adjust_edge_threshold(current_edge_pp: float, win_rate: float,
                          break_even_rate: float, n_trades: int) -> Tuple[float, str]:
    """
    Raise or lower the minimum edge threshold based on overall win-rate fitness.

    Rules:
        win_rate < break_even_rate                  -> raise by 1pp
        win_rate > break_even_rate + 0.15
            AND n_trades >= 20                      -> lower by 0.5pp (more aggressive)
        Clamped to [5pp, 15pp]
    """
    note = "no change"
    new_edge = current_edge_pp

    if win_rate < break_even_rate:
        new_edge = min(15.0, current_edge_pp + 1.0)
        note = f"win_rate {win_rate*100:.1f}% < break-even {break_even_rate*100:.1f}% -> edge +1pp"
    elif win_rate > (break_even_rate + 0.15) and n_trades >= 20:
        new_edge = max(5.0, current_edge_pp - 0.5)
        note = (f"win_rate {win_rate*100:.1f}% well above break-even ({break_even_rate*100:.1f}%) "
                f"with {n_trades} trades -> edge -0.5pp")
    else:
        note = (f"win_rate {win_rate*100:.1f}% vs break-even {break_even_rate*100:.1f}% "
                f"({n_trades} trades): within acceptable range")

    return new_edge, note


def time_of_day_analysis(trades: List[Dict]) -> Tuple[List[int], str]:
    """
    Group wins/losses by hour of day (UTC; Kalshi timestamps are UTC).
    Returns a list of hours to avoid (win rate < 40%) and a description note.
    """
    hour_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "total": 0})

    for t in trades:
        ts_str = t.get("timestamp", "")
        outcome = t.get("outcome", "").upper()
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            hr = dt.hour  # UTC; caller can note EST = UTC-5
        except Exception:
            continue
        hour_stats[hr]["total"] += 1
        if outcome == "WIN":
            hour_stats[hr]["wins"] += 1

    avoid_hours: List[int] = []
    for hr, stats in hour_stats.items():
        if stats["total"] >= 3:  # minimum sample size per hour
            wr = stats["wins"] / stats["total"]
            if wr < 0.40:
                avoid_hours.append(hr)

    avoid_hours.sort()
    if avoid_hours:
        note = f"poor-performing UTC hours (wr<40%, n>=3): {avoid_hours}"
    else:
        note = "no hours with statistically poor performance identified"
    return avoid_hours, note


def classify_failure_modes(losses: List[Dict]) -> Tuple[str, Dict[str, int]]:
    """
    Classify each losing trade into a failure mode category.

    Categories (first match wins):
        BTC_MOVED_FAR   btc_price_at_signal changed >1.5% (uses mid via range_low/range_high proxy)
        OVERCONFIDENT   fair_value > 0.70 on a loss
        POOR_ENTRY      edge_pp in (0, 10]
        LATE_ENTRY      minutes_left < 15

    Returns (top_mode, counts_dict).
    """
    counts: Counter = Counter()

    for t in losses:
        fv          = _safe_float(t.get("fair_value"))
        edge_pp     = _safe_float(t.get("edge_pp"))
        minutes_left = _safe_float(t.get("minutes_left"))
        btc_price   = _safe_float(t.get("btc_price_at_signal"))
        range_mid   = (
            (_safe_float(t.get("range_low")) + _safe_float(t.get("range_high"))) / 2
            if t.get("range_low") and t.get("range_high") else 0.0
        )

        # BTC_MOVED_FAR: BTC price at signal was far from range mid
        # We approximate "BTC moved far" by checking if btc_price was >1.5% away
        # from the center of the traded range, which would mean our entry fair_value
        # was optimistic given BTC's position.
        if btc_price > 0 and range_mid > 0:
            pct_deviation = abs(btc_price - range_mid) / btc_price
            if pct_deviation > 0.015:
                counts["BTC_MOVED_FAR"] += 1
                continue

        if fv > 0.70:
            counts["OVERCONFIDENT"] += 1
            continue

        if 0 < edge_pp <= 10:
            counts["POOR_ENTRY"] += 1
            continue

        if 0 < minutes_left < 15:
            counts["LATE_ENTRY"] += 1
            continue

        counts["OTHER"] += 1

    top_mode = counts.most_common(1)[0][0] if counts else "NONE"
    return top_mode, dict(counts)


# ---------------------------------------------------------------------------
# Optimization log writer
# ---------------------------------------------------------------------------

def append_optimization_log(
    iteration: int,
    vol: float,
    min_edge: float,
    win_rate: float,
    n_trades: int,
    change_made: str,
) -> None:
    """Append one row to backtest/optimization_log.csv using storage backend."""
    _ensure_dirs()
    fieldnames = [
        "timestamp", "iteration", "btc_hourly_vol",
        "min_edge_pp", "win_rate", "n_trades", "change_made",
    ]
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "iteration": iteration,
        "btc_hourly_vol": f"{vol:.5f}",
        "min_edge_pp": f"{min_edge:.2f}",
        "win_rate": f"{win_rate:.4f}",
        "n_trades": n_trades,
        "change_made": change_made,
    }
    _storage.append_csv("backtest/optimization_log.csv", row, fieldnames=fieldnames)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    iteration: int,
    trades: List[Dict],
    wins: int,
    losses: int,
    win_rate: float,
    break_even: float,
    calib_results: List[Dict],
    top_mode: str,
    failure_counts: Dict[str, int],
    old_vol: float,
    new_vol: float,
    vol_note: str,
    old_edge: float,
    new_edge: float,
    edge_note: str,
    avoid_hours: List[int],
    dry_run: bool,
) -> None:
    """Print the full analysis report to stdout."""
    n = len(trades)
    be_pct = break_even * 100
    wr_pct = win_rate * 100

    print()
    print(f"=== STRATEGY OPTIMIZER -- Iteration {iteration} ===")
    if dry_run:
        print("  [DRY RUN -- config will NOT be updated]")
    print(f"Resolved trades: {n} ({wins}W / {losses}L, {wr_pct:.1f}% win rate)")

    above_below = "Above" if win_rate >= break_even else "BELOW"
    marker = "[OK]" if win_rate >= break_even else "[!!]"
    print(f"Break-even rate: {be_pct:.1f}%  {marker} {above_below} break-even")

    # Calibration table
    print()
    print("Calibration:")
    if calib_results:
        for b in calib_results:
            exp_pct = b["expected_wr"] * 100
            act_pct = b["actual_wr"] * 100
            delta   = b["delta_pp"]
            flag_str = ""
            if b["flag"] == "overconfident":
                flag_str = f"  <- overconfident! ({b['n']} losses in bucket)"
            elif b["flag"] == "underconfident":
                flag_str = f"  <- underconfident in this range"
            elif b["n"] >= 3:
                flag_str = "  <- close"
            print(
                f"  Fair {b['label']:>6}: expected {exp_pct:.0f}%, "
                f"actual {act_pct:.0f}%  (n={b['n']}, delta={delta:+.0f}pp){flag_str}"
            )
    else:
        print("  No calibration data (no trades in any bucket).")

    # Failure modes
    print()
    if failure_counts:
        total_losses = sum(failure_counts.values())
        print(f"Top failure mode: {top_mode} ({failure_counts.get(top_mode, 0)}/{total_losses} losses)")
        for mode, cnt in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {cnt}")
    else:
        print("Top failure mode: NONE (no losses to analyze)")

    # Parameter changes
    print()
    vol_changed = abs(new_vol - old_vol) > 1e-6
    edge_changed = abs(new_edge - old_edge) > 1e-4
    print(
        f"Vol adjustment:  {old_vol:.4f} -> {new_vol:.4f}"
        + (" (changed)" if vol_changed else " (no change)")
    )
    print(
        f"Edge threshold:  {old_edge:.1f}pp -> {new_edge:.1f}pp"
        + (" (changed)" if edge_changed else " (no change)")
    )

    if avoid_hours:
        print(f"Avoid hours (UTC): {avoid_hours}  [poor win rate <40%]")
    else:
        print("Avoid hours:     none identified")

    print()
    action = "Next scan will use" if not dry_run else "Would use"
    print(f"Updated config {'saved' if not dry_run else 'NOT saved'}.  "
          f"{action}: vol={new_vol:.4f}, min_edge={new_edge:.1f}pp")
    print()


# ---------------------------------------------------------------------------
# Core optimization routine
# ---------------------------------------------------------------------------

def run_optimization(dry_run: bool = False) -> Dict:
    """
    Run the full self-improvement loop.

    Parameters
    ----------
    dry_run : bool
        If True, compute all adjustments but do not write config or log files.

    Returns
    -------
    dict
        The updated (or unchanged on dry_run) config dict.
    """
    cfg = load_config()
    all_resolved = load_resolved_trades()

    if len(all_resolved) < MIN_TRADES_REQUIRED:
        print(
            f"[optimizer] Only {len(all_resolved)} resolved trade(s) found "
            f"(need {MIN_TRADES_REQUIRED}). Skipping optimization."
        )
        return cfg

    # -- Core metrics --------------------------------------------------------
    wins, losses_n, win_rate = compute_win_rate(all_resolved)
    break_even = compute_break_even_rate(all_resolved)
    losses = [t for t in all_resolved if t.get("outcome", "").upper() == "LOSS"]

    # -- Calibration ---------------------------------------------------------
    calib_results = calibration_analysis(all_resolved)

    # -- Vol adjustment (realized vol -> exponential smooth) -----------------
    old_vol = cfg["btc_hourly_vol"]
    new_vol, vol_note_realized = compute_vol_adjustment(all_resolved, old_vol)

    # Secondary: calibration-driven adjustment on top
    new_vol, vol_note_calib = adjust_vol_for_calibration(new_vol, calib_results)

    # -- Edge threshold -------------------------------------------------------
    old_edge = cfg["min_edge_pp"]
    new_edge, edge_note = adjust_edge_threshold(
        old_edge, win_rate, break_even, len(all_resolved)
    )

    # -- Time-of-day ----------------------------------------------------------
    avoid_hours, tod_note = time_of_day_analysis(all_resolved)

    # -- Failure mode classification ------------------------------------------
    top_mode, failure_counts = classify_failure_modes(losses)

    # -- Build human-readable change notes ------------------------------------
    changes: List[str] = []
    if abs(new_vol - old_vol) > 1e-6:
        changes.append(f"vol {old_vol:.4f}->{new_vol:.4f} ({vol_note_calib or vol_note_realized})")
    if abs(new_edge - old_edge) > 1e-4:
        changes.append(f"edge {old_edge:.1f}->{new_edge:.1f}pp ({edge_note})")
    if avoid_hours:
        changes.append(f"avoid_hours={avoid_hours} ({tod_note})")
    if top_mode and top_mode != "NONE":
        changes.append(f"top_failure={top_mode}")

    notes_str = "; ".join(changes) if changes else "no parameter changes needed"
    change_summary = notes_str

    # -- Report ---------------------------------------------------------------
    new_iteration = cfg["iteration"] + 1
    print_report(
        iteration=new_iteration,
        trades=all_resolved,
        wins=wins,
        losses=losses_n,
        win_rate=win_rate,
        break_even=break_even,
        calib_results=calib_results,
        top_mode=top_mode,
        failure_counts=failure_counts,
        old_vol=old_vol,
        new_vol=new_vol,
        vol_note=vol_note_realized,
        old_edge=old_edge,
        new_edge=new_edge,
        edge_note=edge_note,
        avoid_hours=avoid_hours,
        dry_run=dry_run,
    )

    # -- Persist --------------------------------------------------------------
    if not dry_run:
        cfg["btc_hourly_vol"]          = round(new_vol, 5)
        cfg["min_edge_pp"]             = round(new_edge, 2)
        cfg["avoid_hours"]             = avoid_hours
        cfg["iteration"]               = new_iteration
        cfg["notes"]                   = notes_str
        save_config(cfg)

        append_optimization_log(
            iteration=new_iteration,
            vol=new_vol,
            min_edge=new_edge,
            win_rate=win_rate,
            n_trades=len(all_resolved),
            change_made=change_summary,
        )

    return cfg


# ---------------------------------------------------------------------------
# Parameter bounds (used by skill_evolver.py)
# ---------------------------------------------------------------------------

def get_config_bounds():
    """Return valid parameter ranges for the evolver."""
    return {
        "btc_hourly_vol": (0.005, 0.025),
        "min_edge_pp": (3.0, 15.0),
        "fractional_kelly": (0.10, 0.50),
        "max_position_pct": (0.02, 0.10),
        "time_decay_threshold_min": (10, 120),
        "time_decay_min_fair": (0.50, 0.95),
        "avoid_hours": "list_of_ints_0_23",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-improving strategy optimizer for the Kalshi BTC trading bot."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--report",
        action="store_true",
        help="Print analysis report without updating config files.",
    )
    group.add_argument(
        "--reset",
        action="store_true",
        help="Reset backtest/strategy_config.json to factory defaults.",
    )
    args = parser.parse_args()

    if args.reset:
        fresh = reset_config()
        print(f"[optimizer] Config reset to defaults and saved to: {CONFIG_PATH}")
        print(json.dumps(fresh, indent=2))
        return

    dry_run = args.report
    run_optimization(dry_run=dry_run)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# INTEGRATION NOTE
# ---------------------------------------------------------------------------
#
# kalshi_btc_trader.py should be updated to load its runtime parameters from
# backtest/strategy_config.json at startup, overriding its built-in constants.
# Insert the following block near the top of main(), BEFORE the argparse block:
#
#   from pathlib import Path
#   import json
#   _cfg_path = Path(__file__).parent.parent / "backtest" / "strategy_config.json"
#   if _cfg_path.exists():
#       _cfg = json.loads(_cfg_path.read_text())
#       BTC_HOURLY_VOL_PCT         = _cfg.get("btc_hourly_vol", BTC_HOURLY_VOL_PCT)
#       MIN_EDGE_PCT               = _cfg.get("min_edge_pp", 8.0) / 100.0
#       FRACTIONAL_KELLY           = _cfg.get("fractional_kelly", FRACTIONAL_KELLY)
#       MAX_POSITION_PCT           = _cfg.get("max_position_pct", MAX_POSITION_PCT)
#       TIME_DECAY_THRESHOLD_MIN   = _cfg.get("time_decay_threshold_min", TIME_DECAY_THRESHOLD_MIN)
#       TIME_DECAY_MIN_FAIR        = _cfg.get("time_decay_min_fair", TIME_DECAY_MIN_FAIR)
#       _avoid                     = set(_cfg.get("avoid_hours", []))
#       # Then, inside generate_signals(), skip markets whose signal hour is in _avoid.
#
# auto_resolver.py should call run_optimization() after resolving each batch:
#
#   from strategy_optimizer import run_optimization
#   run_optimization(dry_run=False)
#
# ---------------------------------------------------------------------------
