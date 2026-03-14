"""
auto_resolver.py — Automatic trade resolution for Kalshi KXBTC markets.

Detects when open trades have settled on Kalshi, records WIN or LOSS outcomes,
updates the bankroll tracker and signals log, and prints a portfolio summary.

CLI:
  python scripts/auto_resolver.py              # check once and exit
  python scripts/auto_resolver.py --once       # alias for default single-run
  python scripts/auto_resolver.py --watch      # loop every 5 minutes
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# sys.path: allow imports from the scripts directory regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import requests
import pandas as pd
from dotenv import load_dotenv
import os

# Load .env from repo root
_ENV_FILE = _REPO_ROOT / ".env"
load_dotenv(_ENV_FILE)

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KALSHI_BASE_URL = "https://api.elections.kalshi.com"
KALSHI_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

TRADES_DIR = _REPO_ROOT / "trades"
LEDGER_FILE = TRADES_DIR / "live_trades.parquet"
SIGNALS_LOG_FILE = TRADES_DIR / "signals_log.csv"

WATCH_INTERVAL_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# RSA auth (exact signature as specified)
# ---------------------------------------------------------------------------

def build_auth_headers(method: str, path: str) -> dict:
    if not KALSHI_KEY_PATH or not Path(KALSHI_KEY_PATH).exists():
        raise FileNotFoundError(
            f"Private key not found at: {KALSHI_KEY_PATH!r}. "
            "Set KALSHI_PRIVATE_KEY_PATH in .env"
        )
    if not KALSHI_KEY_ID:
        raise ValueError(
            "KALSHI_API_KEY_ID is empty. Set it in .env"
        )

    with open(KALSHI_KEY_PATH, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    ts_ms = str(int(time.time() * 1000))
    msg = (ts_ms + method.upper() + path).encode("utf-8")
    sig = key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Kalshi API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str) -> dict:
    """Perform authenticated GET. Returns parsed JSON dict."""
    headers = build_auth_headers("GET", path)
    url = KALSHI_BASE_URL + path
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_market(ticker: str) -> Optional[dict]:
    """
    GET /trade-api/v2/markets/{ticker}
    Returns the market dict or None on error.
    """
    path = f"/trade-api/v2/markets/{ticker}"
    try:
        data = _api_get(path)
        # Kalshi wraps the market under "market" key
        return data.get("market", data)
    except requests.HTTPError as exc:
        print(f"  [WARN] HTTP error fetching {ticker}: {exc}")
        return None
    except Exception as exc:
        print(f"  [WARN] Error fetching {ticker}: {exc}")
        return None


def fetch_open_orders() -> list[dict]:
    """
    GET /trade-api/v2/portfolio/orders?status=resting
    Returns list of open order dicts.
    """
    path = "/trade-api/v2/portfolio/orders?status=resting"
    try:
        data = _api_get(path)
        return data.get("orders", [])
    except Exception as exc:
        print(f"  [WARN] Could not fetch open orders: {exc}")
        return []


def fetch_balance() -> Optional[float]:
    """
    GET /trade-api/v2/portfolio/balance
    Returns balance in USD (Kalshi returns cents, divide by 100).
    """
    path = "/trade-api/v2/portfolio/balance"
    try:
        data = _api_get(path)
        # Kalshi returns balance in cents under various keys
        balance_cents = (
            data.get("balance")
            or data.get("available_balance_cents")
            or data.get("portfolio", {}).get("available_balance_cents")
        )
        if balance_cents is None:
            return None
        return balance_cents / 100.0
    except Exception as exc:
        print(f"  [WARN] Could not fetch balance: {exc}")
        return None


# ---------------------------------------------------------------------------
# Outcome determination
# ---------------------------------------------------------------------------

def determine_outcome(direction: str, market_result: str) -> Optional[str]:
    """
    Return "WIN" or "LOSS" based on trade direction vs market result.

    Rules:
      direction=YES + result=yes  -> WIN
      direction=YES + result=no   -> LOSS
      direction=NO  + result=no   -> WIN
      direction=NO  + result=yes  -> LOSS

    Returns None if result is not conclusive (e.g. "void", "n/a", empty).
    """
    d = direction.upper().strip()
    r = (market_result or "").lower().strip()

    if r not in ("yes", "no"):
        return None

    if d == "YES":
        return "WIN" if r == "yes" else "LOSS"
    if d == "NO":
        return "WIN" if r == "no" else "LOSS"

    return None


# ---------------------------------------------------------------------------
# Ledger helpers (directly manipulate parquet to avoid CLI subprocess overhead)
# ---------------------------------------------------------------------------

def _load_ledger() -> pd.DataFrame:
    if not LEDGER_FILE.exists():
        return pd.DataFrame()
    return pd.read_parquet(LEDGER_FILE)


def _save_ledger(df: pd.DataFrame) -> None:
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEDGER_FILE, index=False)


def _compute_pnl(direction: str, entry_price: float, contracts: int, outcome: str) -> float:
    """Replicate bankroll_tracker pnl logic: WIN=exit@1.0, LOSS=exit@0.0."""
    exit_price = 1.0 if outcome == "WIN" else 0.0
    return round((exit_price - float(entry_price)) * int(contracts), 6)


# ---------------------------------------------------------------------------
# Bankroll tracker integration
# ---------------------------------------------------------------------------

def resolve_via_bankroll_tracker(trade_id: str, outcome: str) -> None:
    """
    Call the bankroll_tracker cmd_resolve logic directly (no subprocess).
    Imports bankroll_tracker from scripts/ directory.
    """
    try:
        import bankroll_tracker as bt

        # Build a fake argparse namespace matching cmd_resolve expectations
        class _Args:
            pass

        args = _Args()
        args.trade_id = trade_id
        args.outcome = outcome  # "WIN" or "LOSS"

        bt.cmd_resolve(args)
    except Exception as exc:
        print(f"  [ERROR] bankroll_tracker.cmd_resolve failed: {exc}")
        raise


# ---------------------------------------------------------------------------
# Signals log update
# ---------------------------------------------------------------------------

def update_signals_log(ticker: str, outcome: str) -> bool:
    """
    Read signals_log.csv, find all rows where ticker matches and outcome is
    empty/missing, fill in the outcome, and write the file back.

    Returns True if at least one row was updated.
    """
    if not SIGNALS_LOG_FILE.exists():
        return False

    # Read raw lines to preserve structure
    with SIGNALS_LOG_FILE.open(newline="", encoding="utf-8") as fh:
        raw = fh.read()

    try:
        reader = csv.DictReader(io.StringIO(raw))
        fieldnames = reader.fieldnames
        if not fieldnames:
            return False

        rows = list(reader)
    except Exception as exc:
        print(f"  [WARN] Could not parse signals_log.csv: {exc}")
        return False

    if "outcome" not in fieldnames:
        # No outcome column — nothing to update
        return False

    updated = False
    for row in rows:
        row_ticker = (row.get("ticker") or "").strip()
        row_outcome = (row.get("outcome") or "").strip()
        if row_ticker == ticker and not row_outcome:
            row["outcome"] = outcome
            updated = True

    if not updated:
        return False

    with SIGNALS_LOG_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return True


# ---------------------------------------------------------------------------
# Core resolution logic
# ---------------------------------------------------------------------------

def _is_settled(market: dict) -> bool:
    """Return True if Kalshi market data indicates the market has settled."""
    status = (market.get("status") or "").lower()
    result = (market.get("result") or "").lower()
    if status == "settled":
        return True
    if result in ("yes", "no"):
        return True
    return False


def _get_result(market: dict) -> str:
    """Extract the market result string from the market dict."""
    return (market.get("result") or "").lower().strip()


def run_once() -> int:
    """
    Check all OPEN trades once, resolve any that have settled.
    Returns the count of trades resolved.
    """
    # ------------------------------------------------------------------
    # 1. Load open trades
    # ------------------------------------------------------------------
    if not LEDGER_FILE.exists():
        print("No open trades to check (live_trades.parquet not found).")
        return 0

    df = _load_ledger()

    if df.empty:
        print("No open trades to check.")
        return 0

    # Handle both "status" column variants seen in the live schema
    status_col = None
    for candidate in ("status",):
        if candidate in df.columns:
            status_col = candidate
            break

    if status_col is None:
        print("[WARN] No 'status' column found in live_trades.parquet.")
        return 0

    open_df = df[df[status_col] == "OPEN"].copy()

    if open_df.empty:
        print("No open trades to check.")
        _print_portfolio_summary()
        return 0

    print(f"Checking {len(open_df)} open trade(s)...\n")

    resolved_count = 0

    for _, trade in open_df.iterrows():
        trade_id = str(trade.get("trade_id", ""))
        ticker = str(trade.get("ticker", "")).strip()
        direction = str(trade.get("direction", "")).strip()
        entry_price = float(trade.get("entry_price", 0) or 0)
        contracts = int(trade.get("contracts", 1) or 1)

        if not ticker:
            print(f"  [SKIP] trade_id={trade_id[:8]}... has no ticker.")
            continue

        # ------------------------------------------------------------------
        # 2. Query Kalshi API
        # ------------------------------------------------------------------
        market = fetch_market(ticker)
        if market is None:
            print(f"  [SKIP] {ticker}: could not fetch market data.")
            continue

        if not _is_settled(market):
            mstatus = market.get("status", "unknown")
            print(f"  [OPEN] {ticker}: status={mstatus} — not yet settled.")
            continue

        market_result = _get_result(market)

        # ------------------------------------------------------------------
        # 3. Determine WIN / LOSS
        # ------------------------------------------------------------------
        outcome = determine_outcome(direction, market_result)
        if outcome is None:
            market_status = (market.get("status") or "").lower()
            market_result_raw = (market.get("result") or "").lower()

            if market_status in ("voided", "cancelled") or market_result_raw in ("void", "voided", "cancelled", "n/a"):
                print(f"  [VOID] {ticker}: market voided/cancelled. Refunding cost.")
                # Mark trade as VOIDED, refund cost
                try:
                    df = _load_ledger()
                    mask = df["trade_id"] == trade_id
                    if mask.any():
                        cost = float(df.loc[mask, "cost"].iloc[0])
                        df.loc[mask, "status"] = "VOIDED"
                        df.loc[mask, "pnl"] = 0.0
                        df.loc[mask, "resolved_at"] = pd.Timestamp.now(tz="UTC")
                        _save_ledger(df)

                        # Refund cost to bankroll
                        import bankroll_tracker as bt
                        state = bt.BankrollTracker().get_state()
                        state["current_bankroll"] = round(state["current_bankroll"] + cost, 2)
                        # Save bankroll
                        import json
                        bankroll_path = Path(__file__).resolve().parent.parent / "trades" / "bankroll.json"
                        with open(bankroll_path, "w") as f:
                            json.dump(state, f, indent=2)

                        update_signals_log(ticker, "VOIDED")
                        print(f"  [VOID] Refunded ${cost:.4f} to bankroll")
                        resolved_count += 1
                except Exception as e:
                    print(f"  [VOID] Error handling void: {e}")
                continue
            else:
                print(f"  [SKIP] {ticker}: result={market_result_raw!r} unrecognized, skipping")
                continue

        # ------------------------------------------------------------------
        # 4. Resolve via bankroll_tracker (updates parquet + bankroll.json)
        # ------------------------------------------------------------------
        pnl = _compute_pnl(direction, entry_price, contracts, outcome)

        try:
            resolve_via_bankroll_tracker(trade_id, outcome)
        except Exception:
            # Already printed error above — continue to next trade
            continue

        # ------------------------------------------------------------------
        # 5. Update signals_log.csv
        # ------------------------------------------------------------------
        log_updated = update_signals_log(ticker, outcome)
        log_note = "signals_log updated" if log_updated else "signals_log: no matching open row"

        # ------------------------------------------------------------------
        # 6. Print resolution line
        # ------------------------------------------------------------------
        sign = "+" if pnl >= 0 else ""
        print(
            f"RESOLVED: {ticker} -> {outcome} | PnL: {sign}${pnl:.2f} "
            f"| direction={direction} result={market_result} | {log_note}"
        )
        resolved_count += 1

    if resolved_count == 0:
        print("\nNo trades resolved this run.")
    else:
        print(f"\n{resolved_count} trade(s) resolved.")

    # ------------------------------------------------------------------
    # 7. Trigger loss postmortem (which calls strategy optimizer) if any trades resolved
    # ------------------------------------------------------------------
    if resolved_count > 0:
        print(f"\n  {resolved_count} trade(s) resolved. Running loss postmortem...")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from loss_postmortem import run_postmortem
            run_postmortem(dry_run=False)
        except Exception as exc:
            print(f"  [WARN] Loss postmortem failed: {exc}")
            # Fallback to direct optimizer
            try:
                from strategy_optimizer import run_optimization
                run_optimization(dry_run=False)
            except Exception as exc2:
                print(f"  [WARN] Strategy optimizer also failed: {exc2}")

    # ------------------------------------------------------------------
    # 8. Open orders summary
    # ------------------------------------------------------------------
    _print_open_orders()

    # ------------------------------------------------------------------
    # 9. Account balance
    # ------------------------------------------------------------------
    _print_balance()

    return resolved_count


def _print_open_orders() -> None:
    """Fetch and print a summary of any still-open resting orders."""
    print("\n--- Open Kalshi Orders (resting) ---")
    orders = fetch_open_orders()
    if not orders:
        print("  No open orders.")
        return

    for o in orders:
        ticker = o.get("ticker", "?")
        side = o.get("side", o.get("action", "?"))
        qty = o.get("remaining_count", o.get("count", "?"))
        price_cents = o.get("yes_price", o.get("no_price", None))
        price_str = f"${price_cents / 100:.2f}" if price_cents is not None else "?"
        order_id = o.get("order_id", o.get("id", "?"))
        print(f"  [{order_id}] {ticker} | side={side} qty={qty} price={price_str}")


def _print_balance() -> None:
    """Fetch and print account balance."""
    print("\n--- Kalshi Account Balance ---")
    balance = fetch_balance()
    if balance is None:
        print("  Could not retrieve balance.")
    else:
        print(f"  Available balance: ${balance:,.2f}")


def _print_portfolio_summary() -> None:
    """Print open orders and balance even when there are no open trades."""
    _print_open_orders()
    _print_balance()


# ---------------------------------------------------------------------------
# Watch loop
# ---------------------------------------------------------------------------

def run_watch() -> None:
    """Loop every WATCH_INTERVAL_SECONDS, calling run_once each iteration."""
    print(f"Watch mode: checking every {WATCH_INTERVAL_SECONDS // 60} minutes.")
    print("Press Ctrl+C to stop.\n")
    iteration = 0
    while True:
        iteration += 1
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"=== Run #{iteration} at {now_str} ===")
        try:
            run_once()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[ERROR] Unhandled exception in run_once: {exc}")

        print(f"\nSleeping {WATCH_INTERVAL_SECONDS}s...\n")
        try:
            time.sleep(WATCH_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nWatch mode stopped by user.")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-resolver: detect settled Kalshi KXBTC markets and record "
            "WIN/LOSS outcomes for open trades."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Loop every 5 minutes, continuously resolving settled trades.",
    )
    mode.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (default behaviour, alias for single-run).",
    )
    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.watch:
        run_watch()
    else:
        # --once is the default; both paths call run_once()
        run_once()


if __name__ == "__main__":
    main()
