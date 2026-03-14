"""
bankroll_tracker.py — Kalshi BTC trading account bankroll and P&L tracker.

Maintains:
  trades/bankroll.json       — current bankroll state
  trades/live_trades.parquet — full trade ledger (one row per trade)

CLI:
  python scripts/bankroll_tracker.py init --bankroll 10.00
  python scripts/bankroll_tracker.py add-trade --ticker KXBTC-25MAR1421-B95000 \
      --direction YES --contracts 2 --entry-price 0.45 --fair-value 0.58 \
      --edge 13.0 --btc-price 94850
  python scripts/bankroll_tracker.py resolve --trade-id <uuid> --outcome WIN
  python scripts/bankroll_tracker.py resolve --trade-id <uuid> --outcome LOSS
  python scripts/bankroll_tracker.py status
  python scripts/bankroll_tracker.py trades
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
TRADES_DIR = _REPO_ROOT / "trades"
BANKROLL_FILE = TRADES_DIR / "bankroll.json"
LEDGER_FILE = TRADES_DIR / "live_trades.parquet"

# ---------------------------------------------------------------------------
# Schema for the trade ledger
# ---------------------------------------------------------------------------
LEDGER_DTYPES: dict[str, str] = {
    "trade_id": "object",
    "timestamp": "datetime64[ns, UTC]",
    "ticker": "object",
    "direction": "object",
    "range_label": "object",
    "contracts": "int64",
    "entry_price": "float64",
    "cost": "float64",
    "fair_value": "float64",
    "edge_pp": "float64",
    "btc_price_at_entry": "float64",
    "status": "object",
    "exit_price": "float64",
    "pnl": "float64",
    "resolved_at": "datetime64[ns, UTC]",
}

_EMPTY_BANKROLL: dict = {
    "starting_bankroll": 0.0,
    "current_bankroll": 0.0,
    "total_pnl": 0.0,
    "total_pnl_pct": 0.0,
    "last_updated": "",
    "n_trades": 0,
    "n_wins": 0,
    "n_losses": 0,
    "win_rate": 0.0,
    "daily_pnl": 0.0,
    "daily_pnl_date": "",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_bankroll() -> dict:
    if not BANKROLL_FILE.exists():
        raise FileNotFoundError(
            f"{BANKROLL_FILE} not found. Run `init --bankroll <amount>` first."
        )
    with BANKROLL_FILE.open() as fh:
        return json.load(fh)


def _save_bankroll(state: dict) -> None:
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = _now_iso()
    with BANKROLL_FILE.open("w") as fh:
        json.dump(state, fh, indent=2)


def _load_ledger() -> pd.DataFrame:
    if not LEDGER_FILE.exists():
        return _empty_ledger()
    df = pd.read_parquet(LEDGER_FILE)
    # Ensure all expected columns exist (forward-compat)
    for col in LEDGER_DTYPES:
        if col not in df.columns:
            df[col] = None
    return df


def _save_ledger(df: pd.DataFrame) -> None:
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEDGER_FILE, index=False)


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=dtype.replace("[ns, UTC]", "[ns]").replace("datetime64[ns, UTC]", "object"))
         for col, dtype in LEDGER_DTYPES.items()}
    )


def _infer_range_label(ticker: str) -> str:
    """
    Attempt to parse a range label from a Kalshi BTC ticker such as
    KXBTC-25MAR1421-B95000  -> '$95000+'
    KXBTC-25MAR1421-T94750  -> '$94750-'
    """
    m = re.search(r"[BT](\d+)", ticker)
    if not m:
        return ""
    price = int(m.group(1))
    prefix = ticker[m.start()]
    if prefix == "B":
        return f"${price:,}+"
    return f"${price:,}-"


# ---------------------------------------------------------------------------
# BankrollTracker class (importable by dashboard.py)
# ---------------------------------------------------------------------------

class BankrollTracker:
    """Read-only interface for the bankroll state and trade ledger."""

    def __init__(self) -> None:
        pass

    def get_state(self) -> dict:
        """Return the bankroll.json dict."""
        return _load_bankroll()

    def get_open_trades(self) -> pd.DataFrame:
        """Return DataFrame of OPEN trades."""
        df = _load_ledger()
        return df[df["status"] == "OPEN"].copy()

    def get_closed_trades(self) -> pd.DataFrame:
        """Return DataFrame of WON or LOST trades."""
        df = _load_ledger()
        return df[df["status"].isin(["WON", "LOST"])].copy()

    def daily_pnl(self) -> pd.Series:
        """Return Series indexed by date (resolved_at date) with daily P&L."""
        closed = self.get_closed_trades()
        if closed.empty or closed["pnl"].isna().all():
            return pd.Series(dtype=float, name="pnl")
        closed = closed.dropna(subset=["pnl", "resolved_at"]).copy()
        closed["resolved_at"] = pd.to_datetime(closed["resolved_at"], utc=True)
        closed["date"] = closed["resolved_at"].dt.date
        series = closed.groupby("date")["pnl"].sum()
        series.index = pd.to_datetime(series.index)
        series.name = "pnl"
        return series


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    if BANKROLL_FILE.exists():
        ans = input(
            f"bankroll.json already exists. Overwrite? [y/N] "
        ).strip().lower()
        if ans != "y":
            print("Aborted.")
            return
    state = dict(_EMPTY_BANKROLL)
    state["starting_bankroll"] = round(args.bankroll, 2)
    state["current_bankroll"] = round(args.bankroll, 2)
    state["daily_pnl"] = 0.0
    state["daily_pnl_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _save_bankroll(state)
    # Initialise empty ledger if needed
    if not LEDGER_FILE.exists():
        _save_ledger(_empty_ledger())
    print(f"Initialised bankroll at ${args.bankroll:.2f}")
    print(f"  bankroll.json -> {BANKROLL_FILE}")
    print(f"  live_trades.parquet -> {LEDGER_FILE}")


def cmd_add_trade(args: argparse.Namespace) -> None:
    state = _load_bankroll()
    df = _load_ledger()

    trade_id = str(uuid.uuid4())
    contracts = int(args.contracts)
    entry_price = float(args.entry_price)
    cost = round(contracts * entry_price, 2)
    range_label = args.range_label if args.range_label else _infer_range_label(args.ticker)

    # --- Balance validation ---
    available = state["current_bankroll"]
    MIN_BALANCE_FLOOR = 0.50  # Never let bankroll go below $0.50

    if cost > available:
        print(f"ERROR: Trade cost (${cost:.4f}) exceeds available balance (${available:.4f}).")
        max_contracts = math.floor(available / entry_price) if entry_price > 0 else 0
        print(f"  Max affordable contracts at ${entry_price:.4f}: {max_contracts}")
        sys.exit(1)

    if (available - cost) < MIN_BALANCE_FLOOR:
        print(f"ERROR: Trade would reduce bankroll below minimum floor (${MIN_BALANCE_FLOOR:.2f}).")
        print(f"  Available: ${available:.4f}, Cost: ${cost:.4f}, Would remain: ${available - cost:.4f}")
        sys.exit(1)

    row = {
        "trade_id": trade_id,
        "timestamp": pd.Timestamp.now(tz="UTC"),
        "ticker": args.ticker,
        "direction": args.direction.upper(),
        "range_label": range_label,
        "contracts": contracts,
        "entry_price": entry_price,
        "cost": cost,
        "fair_value": float(args.fair_value),
        "edge_pp": float(args.edge),
        "btc_price_at_entry": float(args.btc_price),
        "status": "OPEN",
        "exit_price": float("nan"),
        "pnl": float("nan"),
        "resolved_at": pd.NaT,
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_ledger(df)

    # Update bankroll state (cost reduces available cash)
    state["n_trades"] = int(state.get("n_trades", 0)) + 1
    state["current_bankroll"] = max(0.0, round(state["current_bankroll"] - cost, 2))
    _save_bankroll(state)

    print(f"Trade added: {trade_id}")
    print(f"  Ticker:      {args.ticker}")
    print(f"  Direction:   {args.direction.upper()}")
    print(f"  Contracts:   {contracts}")
    print(f"  Entry price: ${entry_price:.4f}")
    print(f"  Cost:        ${cost:.4f}")
    print(f"  Fair value:  ${float(args.fair_value):.4f}  (edge {float(args.edge):.1f} pp)")
    print(f"  BTC price:   ${float(args.btc_price):,.2f}")


def cmd_resolve(args: argparse.Namespace) -> None:
    state = _load_bankroll()
    df = _load_ledger()

    mask = df["trade_id"] == args.trade_id
    if not mask.any():
        print(f"Error: trade_id '{args.trade_id}' not found.", file=sys.stderr)
        sys.exit(1)

    idx = df.index[mask][0]
    row = df.loc[idx]

    if row["status"] != "OPEN":
        print(f"Error: trade {args.trade_id} is already {row['status']}.", file=sys.stderr)
        sys.exit(1)

    outcome = args.outcome.upper()
    if outcome not in ("WIN", "LOSS"):
        print("Error: --outcome must be WIN or LOSS.", file=sys.stderr)
        sys.exit(1)

    contracts = int(row["contracts"])
    entry_price = float(row["entry_price"])

    if outcome == "WIN":
        exit_price = 1.0
        new_status = "WON"
        pnl = round((exit_price - entry_price) * contracts, 2)
        state["n_wins"] = int(state.get("n_wins", 0)) + 1
    else:
        exit_price = 0.0
        new_status = "LOST"
        pnl = round((exit_price - entry_price) * contracts, 2)
        state["n_losses"] = int(state.get("n_losses", 0)) + 1

    df.at[idx, "status"] = new_status
    df.at[idx, "exit_price"] = exit_price
    df.at[idx, "pnl"] = pnl
    df.at[idx, "resolved_at"] = pd.Timestamp.now(tz="UTC")
    _save_ledger(df)

    # Restore cost to bankroll, then add/subtract pnl
    cost = float(row["cost"])
    state["current_bankroll"] = max(0.0, round(state["current_bankroll"] + cost + pnl, 2))
    state["total_pnl"] = round(state["current_bankroll"] - state["starting_bankroll"], 2)
    starting = state["starting_bankroll"]
    state["total_pnl_pct"] = round(
        state["total_pnl"] / starting * 100 if starting else 0.0, 4
    )
    n_closed = int(state.get("n_wins", 0)) + int(state.get("n_losses", 0))
    state["win_rate"] = round(
        state["n_wins"] / n_closed * 100 if n_closed else 0.0, 2
    )

    # --- Daily P&L tracking ---
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("daily_pnl_date") != today:
        state["daily_pnl"] = 0.0
        state["daily_pnl_date"] = today
    state["daily_pnl"] = round(state.get("daily_pnl", 0.0) + pnl, 2)

    _save_bankroll(state)

    sign = "+" if pnl >= 0 else ""
    print(f"Trade {args.trade_id[:8]}... resolved as {new_status}")
    print(f"  P&L: {sign}${pnl:.4f}")
    print(f"  New bankroll: ${state['current_bankroll']:.4f}")


def cmd_status(_args: argparse.Namespace) -> None:
    state = _load_bankroll()
    df = _load_ledger()

    bankroll = state["current_bankroll"]
    total_pnl = state["total_pnl"]
    total_pnl_pct = state["total_pnl_pct"]
    n_wins = int(state.get("n_wins", 0))
    n_losses = int(state.get("n_losses", 0))
    win_rate = state.get("win_rate", 0.0)

    open_trades = df[df["status"] == "OPEN"] if not df.empty else pd.DataFrame()
    n_open = len(open_trades)
    n_closed = n_wins + n_losses

    sign = "+" if total_pnl >= 0 else ""
    pct_sign = "+" if total_pnl_pct >= 0 else ""
    print(
        f"Bankroll: ${bankroll:.2f} ({sign}${total_pnl:.2f}, {pct_sign}{total_pnl_pct:.1f}%)"
    )
    print(f"Open trades: {n_open}")
    if n_closed > 0:
        print(
            f"Closed trades: {n_closed} ({n_wins}W / {n_losses}L, {win_rate:.0f}% win rate)"
        )
    else:
        print("Closed trades: 0")

    daily_pnl = state.get("daily_pnl", 0.0)
    daily_date = state.get("daily_pnl_date", "N/A")
    print(f"Daily P&L ({daily_date}): ${daily_pnl:+.2f}")


def cmd_trades(_args: argparse.Namespace) -> None:
    df = _load_ledger()
    if df.empty:
        print("No trades recorded yet.")
        return

    # Select display columns
    display_cols = [
        "trade_id",
        "timestamp",
        "ticker",
        "direction",
        "contracts",
        "entry_price",
        "cost",
        "edge_pp",
        "status",
        "pnl",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    disp = df[display_cols].copy()

    # Shorten trade_id for readability
    disp["trade_id"] = disp["trade_id"].apply(
        lambda x: str(x)[:8] + "..." if isinstance(x, str) and len(str(x)) > 8 else x
    )

    # Format timestamps
    if "timestamp" in disp.columns:
        disp["timestamp"] = pd.to_datetime(disp["timestamp"], utc=True).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    # Format floats
    for col in ("entry_price", "cost", "pnl"):
        if col in disp.columns:
            disp[col] = disp[col].apply(
                lambda x: f"${x:.4f}" if pd.notna(x) else "-"
            )
    if "edge_pp" in disp.columns:
        disp["edge_pp"] = disp["edge_pp"].apply(
            lambda x: f"{x:.1f}pp" if pd.notna(x) else "-"
        )

    try:
        from tabulate import tabulate  # type: ignore
        print(tabulate(disp, headers="keys", tablefmt="simple", showindex=False))
    except ImportError:
        print(disp.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kalshi BTC bankroll and P&L tracker"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialise bankroll")
    p_init.add_argument("--bankroll", type=float, required=True,
                        help="Starting bankroll in USD")

    # add-trade
    p_add = sub.add_parser("add-trade", help="Record a new trade")
    p_add.add_argument("--ticker", required=True)
    p_add.add_argument("--direction", required=True, choices=["YES", "NO", "yes", "no"])
    p_add.add_argument("--contracts", required=True, type=int)
    p_add.add_argument("--entry-price", required=True, type=float,
                       dest="entry_price",
                       help="Price paid per contract in dollars (e.g. 0.45)")
    p_add.add_argument("--fair-value", required=True, type=float,
                       dest="fair_value",
                       help="Model fair value (e.g. 0.58)")
    p_add.add_argument("--edge", required=True, type=float,
                       help="Edge in percentage points (e.g. 13.0)")
    p_add.add_argument("--btc-price", required=True, type=float,
                       dest="btc_price",
                       help="BTC spot price at entry")
    p_add.add_argument("--range-label", default=None, dest="range_label",
                       help='Optional range label, e.g. "$94750-$95000". '
                            'Inferred from ticker if omitted.')

    # resolve
    p_res = sub.add_parser("resolve", help="Resolve an open trade")
    p_res.add_argument("--trade-id", required=True, dest="trade_id")
    p_res.add_argument("--outcome", required=True, choices=["WIN", "LOSS", "win", "loss"])

    # status
    sub.add_parser("status", help="Print bankroll summary")

    # trades
    sub.add_parser("trades", help="Print all trades")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "init": cmd_init,
        "add-trade": cmd_add_trade,
        "resolve": cmd_resolve,
        "status": cmd_status,
        "trades": cmd_trades,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
