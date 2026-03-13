"""
kalshi_btc_trader.py
Live BTC range market monitor + trade signal generator for Kalshi.
Polls every 60s. Sizes for $10–$100 bankroll. Logs signals to CSV.

Usage:
    python scripts/kalshi_btc_trader.py --bankroll 10 --interval 60
    python scripts/kalshi_btc_trader.py --bankroll 100 --interval 30 --auto-trade

Requires (pip install):
    requests scipy numpy pandas
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE_PRICE_URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"

BUCKET_WIDTH = 250          # each BTC range bucket spans $250
BTC_HOURLY_VOL_PCT = 0.01   # ~1% hourly realized vol (conservative)
FRACTIONAL_KELLY = 0.25
MAX_POSITION_PCT = 0.05     # 5% of bankroll per trade
MIN_EDGE_PCT = 0.08         # 8pp minimum edge to signal
MIN_FAIR_VALUE = 0.03       # skip near-zero-probability buckets
TIME_DECAY_THRESHOLD_MIN = 30  # flag "confirmed in range" if < this minutes
TIME_DECAY_MIN_FAIR = 0.70    # require model prob ≥ 70% for time-decay plays

SIGNAL_LOG = Path("trades/signals_log.csv")
SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)

CSV_HEADERS = [
    "timestamp", "ticker", "action", "yes_no",
    "fair_value", "market_mid", "edge",
    "kelly_fraction", "contracts", "limit_price",
    "btc_price", "minutes_to_close", "strategy",
]


# ---------------------------------------------------------------------------
# Price + market data fetchers
# ---------------------------------------------------------------------------

def get_btc_price() -> float:
    """Fetch live BTC/USD spot price from Coinbase public API."""
    try:
        r = requests.get(COINBASE_PRICE_URL, timeout=5)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception as e:
        print(f"  [WARN] BTC price fetch failed: {e}")
        return 0.0


def get_btc_range_markets() -> list[dict]:
    """
    Fetch all open BTC range bucket markets from Kalshi.

    Correct endpoint: /markets?series_ticker=KXBTC&status=open
    - series_ticker=KXBTC returns only BTC range markets (hourly + weekly)
    - status=open excludes "initialized" markets (prices not yet set)
    - Filters to B-suffix tickers only (range buckets, not T threshold markets)
    """
    url = f"{KALSHI_BASE}/markets"
    params = {
        "status": "open",
        "limit": 200,
        "series_ticker": "KXBTC",
    }
    try:
        r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
        r.raise_for_status()
        raw_markets = r.json().get("markets", [])
    except Exception as e:
        print(f"  [ERROR] Kalshi fetch failed: {e}")
        return []

    markets = []
    now = datetime.now(timezone.utc)

    for m in raw_markets:
        ticker = m.get("ticker", "")

        # Only range bucket markets: ticker must contain -B<digits> suffix
        # e.g. KXBTC-26MAR1416-B71250  (hourly, $71,250–$71,500 bucket)
        #      KXBTC-26MAR2017-B71250  (weekly)
        # Skip T-suffix threshold markets (e.g. KXBTC-26MAR1416-T62750)
        parts = ticker.split("-B")
        if len(parts) < 2 or not parts[-1].isdigit():
            continue

        try:
            range_low = int(parts[-1])
        except ValueError:
            continue

        close_str = m.get("close_time") or m.get("expiration_time", "")
        try:
            close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            minutes_left = (close_dt - now).total_seconds() / 60
        except Exception:
            minutes_left = 999

        if minutes_left < 0:
            continue  # already closed

        # yes_ask_dollars / yes_bid_dollars come back as strings ("0.0500")
        yes_ask = float(m.get("yes_ask_dollars") or 0)
        yes_bid = float(m.get("yes_bid_dollars") or 0)
        volume_24h = float(m.get("volume_24h_fp") or 0)

        if yes_ask <= 0 or yes_bid <= 0:
            continue  # skip unpriced markets

        markets.append({
            "ticker": ticker,
            "range_low": range_low,
            "range_high": range_low + BUCKET_WIDTH,
            "yes_ask": yes_ask,
            "yes_bid": yes_bid,
            "mid": round((yes_ask + yes_bid) / 2, 4),
            "spread": round(yes_ask - yes_bid, 4),
            "volume_24h": volume_24h,
            "minutes_left": round(minutes_left, 2),
        })

    return markets


# ---------------------------------------------------------------------------
# Fair-value model
# ---------------------------------------------------------------------------

def compute_fair_value(btc_price: float, range_low: float, range_high: float,
                       hours_left: float, hourly_vol_pct: float = BTC_HOURLY_VOL_PCT) -> float:
    """
    Probability that BTC closes inside [range_low, range_high].
    Uses log-normal model: ln(S_T/S_0) ~ N(0, sigma^2 * T).
    sigma = hourly_vol_pct * sqrt(hours_left)
    """
    if btc_price <= 0 or hours_left <= 0:
        # Zero time: check if BTC is in range right now
        return 1.0 if range_low <= btc_price < range_high else 0.0

    sigma = hourly_vol_pct * math.sqrt(hours_left)

    # Log-normal bounds
    ln_low = math.log(range_low / btc_price) if range_low > 0 else -np.inf
    ln_high = math.log(range_high / btc_price)

    prob = norm.cdf(ln_high / sigma) - norm.cdf(ln_low / sigma)
    return max(0.0, min(1.0, prob))


def kelly_fraction(fair_value: float, market_price: float) -> float:
    """
    Kelly fraction for YES bet.
    f* = (p*b - q) / b  where b = (1 - market_price) / market_price (net odds)
    Clipped to [0, 0.5].
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1 - market_price) / market_price
    q = 1 - fair_value
    f_full = (fair_value * b - q) / b
    f_fractional = f_full * FRACTIONAL_KELLY
    return max(0.0, min(0.5, f_fractional))


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(markets: list[dict], btc_price: float, bankroll: float) -> list[dict]:
    signals = []

    for m in markets:
        hours_left = m["minutes_left"] / 60.0
        fair = compute_fair_value(btc_price, m["range_low"], m["range_high"], hours_left)

        if fair < MIN_FAIR_VALUE:
            continue

        mid = m["mid"]
        edge_yes = fair - m["yes_ask"]   # edge buying YES
        edge_no = (1 - fair) - m["yes_bid"] / 1.0  # edge buying NO (= selling YES)
        # NO edge: fair value of NO = 1-fair, cost of NO = 1 - yes_bid
        no_cost = round(1.0 - m["yes_bid"], 4)
        edge_no_clean = (1 - fair) - no_cost

        # --- Strategy 1: Time-decay (confirmed in-range, minutes to close) ---
        in_range = m["range_low"] <= btc_price < m["range_high"]
        if in_range and m["minutes_left"] <= TIME_DECAY_THRESHOLD_MIN and fair >= TIME_DECAY_MIN_FAIR:
            edge = edge_yes
            if edge >= MIN_EDGE_PCT:
                kf = kelly_fraction(fair, m["yes_ask"])
                contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / m["yes_ask"]))
                signals.append({
                    "ticker": m["ticker"],
                    "yes_no": "YES",
                    "fair_value": round(fair, 4),
                    "market_mid": mid,
                    "edge": round(edge, 4),
                    "kelly_fraction": round(kf, 4),
                    "contracts": contracts,
                    "limit_price": m["yes_ask"],
                    "minutes_to_close": m["minutes_left"],
                    "strategy": "TIME_DECAY_IN_RANGE",
                    "range": f"${m['range_low']:,}–${m['range_high']:,}",
                })

        # --- Strategy 2: Model edge YES (overpriced probability) ---
        elif edge_yes >= MIN_EDGE_PCT and fair > mid and m["minutes_left"] > 15:
            kf = kelly_fraction(fair, m["yes_ask"])
            contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / m["yes_ask"]))
            signals.append({
                "ticker": m["ticker"],
                "yes_no": "YES",
                "fair_value": round(fair, 4),
                "market_mid": mid,
                "edge": round(edge_yes, 4),
                "kelly_fraction": round(kf, 4),
                "contracts": contracts,
                "limit_price": m["yes_ask"],
                "minutes_to_close": m["minutes_left"],
                "strategy": "MODEL_UNDERPRICED_YES",
                "range": f"${m['range_low']:,}–${m['range_high']:,}",
            })

        # --- Strategy 3: Model edge NO (market thinks too likely, buy NO) ---
        elif edge_no_clean >= MIN_EDGE_PCT and (1 - fair) > (1 - mid) and m["minutes_left"] > 15:
            fair_no = 1 - fair
            kf = kelly_fraction(fair_no, no_cost)
            contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / no_cost))
            signals.append({
                "ticker": m["ticker"],
                "yes_no": "NO",
                "fair_value": round(fair_no, 4),
                "market_mid": round(1 - mid, 4),
                "edge": round(edge_no_clean, 4),
                "kelly_fraction": round(kf, 4),
                "contracts": contracts,
                "limit_price": round(no_cost, 4),
                "minutes_to_close": m["minutes_left"],
                "strategy": "MODEL_UNDERPRICED_NO",
                "range": f"${m['range_low']:,}–${m['range_high']:,}",
            })

    # Sort by edge descending
    signals.sort(key=lambda x: x["edge"], reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_signal(sig: dict, btc_price: float):
    row = {
        "timestamp": datetime.now().isoformat(),
        "ticker": sig["ticker"],
        "action": f"BUY {sig['yes_no']}",
        "yes_no": sig["yes_no"],
        "fair_value": sig["fair_value"],
        "market_mid": sig["market_mid"],
        "edge": sig["edge"],
        "kelly_fraction": sig["kelly_fraction"],
        "contracts": sig["contracts"],
        "limit_price": sig["limit_price"],
        "btc_price": btc_price,
        "minutes_to_close": sig["minutes_to_close"],
        "strategy": sig["strategy"],
    }
    write_header = not SIGNAL_LOG.exists()
    with open(SIGNAL_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_market_table(markets: list[dict], btc_price: float):
    print(f"\n  {'TICKER':<35} {'RANGE':>22}  {'MID':>5}  {'SPREAD':>6}  {'MIN_LEFT':>8}  {'24H_VOL':>10}  {'FAIR':>5}  {'EDGE':>6}")
    print("  " + "-" * 115)
    for m in sorted(markets, key=lambda x: x["minutes_left"])[:20]:
        fair = compute_fair_value(btc_price, m["range_low"], m["range_high"], m["minutes_left"] / 60)
        edge = fair - m["yes_ask"]
        in_range = m["range_low"] <= btc_price < m["range_high"]
        flag = " <-- IN RANGE" if in_range else ""
        print(f"  {m['ticker']:<35} ${m['range_low']:>7,}–${m['range_high']:<7,}  {m['mid']:>5.2f}  {m['spread']:>6.2f}  {m['minutes_left']:>8.1f}  ${m['volume_24h']:>9,.0f}  {fair:>5.2f}  {edge:>+6.2f}{flag}")


def print_signals(signals: list[dict], btc_price: float, bankroll: float):
    if not signals:
        print("\n  No signals this cycle (no edge found above threshold).")
        return

    print(f"\n  {'#':<3} {'STRATEGY':<25} {'TICKER':<35} {'RANGE':>22}  {'FAIR':>5}  {'MID':>5}  {'EDGE':>6}  {'CONTRACTS':>9}  {'LIMIT':>6}  {'MIN_LEFT':>8}")
    print("  " + "-" * 145)
    for i, s in enumerate(signals, 1):
        cost = s["contracts"] * s["limit_price"]
        print(
            f"  {i:<3} {s['strategy']:<25} {s['ticker']:<35} {s['range']:>22}  "
            f"{s['fair_value']:>5.2f}  {s['market_mid']:>5.2f}  {s['edge']:>+6.2f}  "
            f"{s['contracts']:>9}  {s['limit_price']:>6.2f}  {s['minutes_to_close']:>8.1f}"
        )
        print(f"       --> BUY {s['contracts']}x {s['yes_no']} @ ${s['limit_price']:.2f}  (cost: ${cost:.2f} / bankroll: ${bankroll:.2f})")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalshi BTC Range Market Trader")
    parser.add_argument("--bankroll", type=float, default=10.0, help="Bankroll in USD (default: 10)")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (default: 60)")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle then exit (for scheduled agents)")
    parser.add_argument("--auto-trade", action="store_true", help="(Future) auto-place orders via Kalshi API")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_PCT, help="Min edge to signal (default: 0.08)")
    parser.add_argument("--vol", type=float, default=BTC_HOURLY_VOL_PCT, help="BTC hourly vol fraction (default: 0.01)")
    args = parser.parse_args()

    global MIN_EDGE_PCT, BTC_HOURLY_VOL_PCT
    MIN_EDGE_PCT = args.min_edge
    BTC_HOURLY_VOL_PCT = args.vol

    print("=" * 60)
    print("  KALSHI BTC RANGE MARKET TRADER")
    print(f"  Bankroll: ${args.bankroll:.2f}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Min edge: {args.min_edge*100:.0f}pp")
    print(f"  BTC hourly vol: {args.vol*100:.1f}%")
    print(f"  Signal log: {SIGNAL_LOG}")
    print("=" * 60)

    cycle = 0
    while True:
        cycle += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[Cycle {cycle}] {ts}")

        btc_price = get_btc_price()
        if btc_price <= 0:
            print("  BTC price unavailable — skipping cycle.")
            time.sleep(args.interval)
            continue

        print(f"  BTC/USD: ${btc_price:,.2f}")

        markets = get_btc_range_markets()
        if not markets:
            print("  No BTC range markets found — skipping cycle.")
            time.sleep(args.interval)
            continue

        print(f"  Markets found: {len(markets)}")
        print_market_table(markets, btc_price)

        signals = generate_signals(markets, btc_price, args.bankroll)
        print(f"\n  SIGNALS ({len(signals)} found):")
        print_signals(signals, btc_price, args.bankroll)

        # Log all signals
        for s in signals:
            log_signal(s, btc_price)
            if args.auto_trade:
                print(f"  [AUTO-TRADE] Would place: BUY {s['contracts']}x {s['yes_no']} {s['ticker']} @ ${s['limit_price']:.2f}")
                print("  [AUTO-TRADE] Kalshi RSA auth not yet configured — manual action required.")

        if signals:
            print(f"\n  ACTION REQUIRED:")
            top = signals[0]
            print(f"  On Kalshi, search: {top['ticker']}")
            print(f"  Buy {top['contracts']}x {top['yes_no']} at ${top['limit_price']:.2f} limit")
            print(f"  Cost: ${top['contracts'] * top['limit_price']:.2f}  |  Edge: {top['edge']*100:.1f}pp  |  Strategy: {top['strategy']}")

        if args.once:
            print("\n  [--once] Single cycle complete. Exiting.")
            break

        print(f"\n  Next cycle in {args.interval}s... (Ctrl+C to stop)")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped. Signal log saved to:", SIGNAL_LOG)
        sys.exit(0)
