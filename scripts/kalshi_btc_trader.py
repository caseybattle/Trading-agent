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
import base64
import csv
import math
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.stats import norm

# Kalshi RSA auth — load credentials from .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv optional; credentials can also be set as env vars

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    _CRYPTO_OK = True
except ImportError:
    _CRYPTO_OK = False

_KALSHI_KEY_ID   = os.getenv("KALSHI_API_KEY_ID", "")
_KALSHI_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")


def load_strategy_config() -> dict:
    """Load tuned params from backtest/strategy_config.json if available."""
    cfg_path = Path(__file__).resolve().parent.parent / "backtest" / "strategy_config.json"
    if not cfg_path.exists():
        return {}
    try:
        import json
        with open(cfg_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_private_key():
    """Load RSA private key from PEM file."""
    if not _KALSHI_KEY_PATH or not Path(_KALSHI_KEY_PATH).exists():
        return None
    try:
        with open(_KALSHI_KEY_PATH, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    except Exception as e:
        print(f"  [AUTH] Failed to load private key: {e}")
        return None


def build_auth_headers(method: str, path: str) -> dict:
    """
    Build Kalshi RSA authentication headers.
    Signature message: timestamp_ms + nonce + METHOD + /path
    """
    if not _CRYPTO_OK or not _KALSHI_KEY_ID:
        return {}
    key = _load_private_key()
    if key is None:
        return {}
    ts_ms = str(int(time.time() * 1000))
    nonce = ""
    msg = (ts_ms + nonce + method.upper() + path).encode("utf-8")
    sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
    sig_b64 = base64.b64encode(sig).decode("utf-8")
    return {
        "KALSHI-ACCESS-KEY":       _KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
        "Content-Type":            "application/json",
    }


def place_order(ticker: str, side: str, contracts: int, limit_price_dollars: float) -> dict | None:
    """
    Place a limit order on Kalshi.
    side: "yes" or "no"
    limit_price_dollars: e.g. 0.45 (will be converted to 45 cents)
    Returns order response dict or None on failure.
    """
    if not _CRYPTO_OK or not _KALSHI_KEY_ID:
        print("  [ORDER] Auth not available — check cryptography + .env setup.")
        return None

    path = "/trade-api/v2/portfolio/orders"
    headers = build_auth_headers("POST", path)
    if not headers:
        print("  [ORDER] Failed to build auth headers.")
        return None

    yes_price_cents = round(limit_price_dollars * 100)
    no_price_cents  = 100 - yes_price_cents

    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          "buy",
        "side":            side.lower(),
        "count":           contracts,
        "yes_price":       yes_price_cents,
        "no_price":        no_price_cents,
    }

    try:
        r = requests.post(
            f"https://api.elections.kalshi.com{path}",
            json=body,
            headers=headers,
            timeout=10,
        )
        if r.status_code in (200, 201):
            data = r.json()
            order = data.get("order", data)
            print(f"  [ORDER] Placed: {side.upper()} {contracts}x {ticker} @ ${limit_price_dollars:.2f}")
            print(f"  [ORDER] Order ID: {order.get('id', '?')} | Status: {order.get('status', '?')}")
            return order
        else:
            print(f"  [ORDER] Failed ({r.status_code}): {r.text[:200]}")
            return None
    except Exception as e:
        print(f"  [ORDER] Request error: {e}")
        return None

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
# TIME_DECAY: set to 45 min so a 30-min scan interval can catch signals
# before the window closes. At 30 min threshold + 30 min scan interval,
# the scanner would often land AFTER the window had already passed.
TIME_DECAY_THRESHOLD_MIN = 45  # flag "confirmed in range" if < this minutes
TIME_DECAY_MIN_FAIR = 0.70    # require model prob ≥ 70% for time-decay plays
# Expiry-mode: lower edge bar for time-decay plays in final 20 min (near-certain)
TIME_DECAY_EXPIRY_THRESHOLD_MIN = 20  # final-20-min window for relaxed edge
TIME_DECAY_EXPIRY_MIN_EDGE = 0.05     # 5pp edge floor (vs 8pp) for final-20-min plays

SIGNAL_LOG = Path("trades/signals_log.csv")
SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)

CSV_HEADERS = [
    "timestamp",
    "ticker",
    "strategy",
    "direction",
    "range_label",
    "range_low",
    "range_high",
    "fair_value",
    "market_ask",
    "market_bid",
    "edge_pp",
    "minutes_left",
    "btc_price_at_signal",
    "kelly_fraction",
    "recommended_contracts",
    "acted_on",
    "outcome",
]


# ---------------------------------------------------------------------------
# Price + market data fetchers
# ---------------------------------------------------------------------------

def get_btc_price() -> float:
    """Fetch live BTC/USD spot price, trying multiple sources as fallbacks."""
    sources = [
        ("Coinbase",   COINBASE_PRICE_URL,
         lambda r: float(r.json()["data"]["amount"])),
        ("Binance",    "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
         lambda r: float(r.json()["price"])),
        ("Kraken",     "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda r: float(r.json()["result"]["XXBTZUSD"]["c"][0])),
        ("CoinGecko",  "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
         lambda r: float(r.json()["bitcoin"]["usd"])),
    ]
    for name, url, parse in sources:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            price = parse(r)
            if price > 0:
                if name != "Coinbase":
                    print(f"  [INFO] BTC price from {name}: ${price:,.2f}")
                return price
        except Exception as e:
            print(f"  [WARN] BTC price fetch failed ({name}): {e}")
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
# Daily loss hard stop
# ---------------------------------------------------------------------------
DAILY_LOSS_STOP_PCT = 0.03
BANKROLL_FILE = Path(__file__).resolve().parent.parent / "trades" / "bankroll.json"

def check_daily_loss_stop() -> bool:
    """Return True if daily loss exceeds 3% of starting bankroll."""
    if not BANKROLL_FILE.exists():
        return False
    try:
        import json
        with open(BANKROLL_FILE) as f:
            state = json.load(f)
        starting = state.get("starting_bankroll", 10.0)
        current = state.get("current_bankroll", starting)
        loss = starting - current
        if loss >= starting * DAILY_LOSS_STOP_PCT:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Duplicate order guard — ticker cooldown
# ---------------------------------------------------------------------------

def get_recent_tickers(cooldown_minutes: int = 30) -> set:
    """Return tickers traded within the last cooldown_minutes."""
    if not SIGNAL_LOG.exists():
        return set()
    recent = set()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
    try:
        with open(SIGNAL_LOG) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("timestamp", "")
                ticker = row.get("ticker", "")
                acted = row.get("acted_on", "")
                if acted in ("AUTO", "MANUAL") and ts_str and ticker:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if ts >= cutoff:
                            recent.add(ticker)
                    except Exception:
                        pass
    except Exception:
        pass
    return recent


# ---------------------------------------------------------------------------
# Open position check
# ---------------------------------------------------------------------------

def get_open_position_tickers() -> set:
    """Return tickers with currently OPEN positions."""
    ledger_path = Path(__file__).resolve().parent.parent / "trades" / "live_trades.parquet"
    if not ledger_path.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_parquet(ledger_path)
        return set(df[df["status"] == "OPEN"]["ticker"].tolist())
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(markets: list[dict], btc_price: float, bankroll: float) -> list[dict]:
    current_hour = datetime.now(timezone.utc).hour
    if hasattr(sys.modules[__name__], '_avoid_hours') and current_hour in globals().get('_avoid_hours', set()):
        print(f"  [SKIP] Hour {current_hour} UTC is in avoid_hours list from optimizer")
        return []

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

        # Shared fields included in every signal for logging
        _common = {
            "range_low":          m["range_low"],
            "range_high":         m["range_high"],
            "market_bid":         m["yes_bid"],
            "btc_price_at_signal": btc_price,
        }

        # --- Strategy 1: Time-decay (confirmed in-range, minutes to close) ---
        in_range = m["range_low"] <= btc_price < m["range_high"]
        expiry_mode = globals().get('_expiry_mode', False)
        # In expiry-mode, also catch final-20-min plays with a relaxed edge bar
        _is_final_20 = expiry_mode and m["minutes_left"] <= TIME_DECAY_EXPIRY_THRESHOLD_MIN
        _td_edge_floor = TIME_DECAY_EXPIRY_MIN_EDGE if _is_final_20 else MIN_EDGE_PCT
        _strategy_label = "TIME_DECAY_EXPIRY" if _is_final_20 else "TIME_DECAY_IN_RANGE"
        if in_range and m["minutes_left"] <= TIME_DECAY_THRESHOLD_MIN and fair >= TIME_DECAY_MIN_FAIR:
            edge = edge_yes
            if edge >= _td_edge_floor:
                kf = kelly_fraction(fair, m["yes_ask"])
                contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / m["yes_ask"]))
                signals.append({
                    **_common,
                    "ticker":           m["ticker"],
                    "yes_no":           "YES",
                    "fair_value":       round(fair, 4),
                    "market_mid":       mid,
                    "edge":             round(edge, 4),
                    "kelly_fraction":   round(kf, 4),
                    "contracts":        contracts,
                    "limit_price":      m["yes_ask"],
                    "minutes_to_close": m["minutes_left"],
                    "strategy":         _strategy_label,
                    "range":            f"${m['range_low']:,}–${m['range_high']:,}",
                })

        # --- Strategy 2: Model edge YES (overpriced probability) ---
        elif edge_yes >= MIN_EDGE_PCT and fair > mid and m["minutes_left"] > 15:
            kf = kelly_fraction(fair, m["yes_ask"])
            contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / m["yes_ask"]))
            signals.append({
                **_common,
                "ticker":           m["ticker"],
                "yes_no":           "YES",
                "fair_value":       round(fair, 4),
                "market_mid":       mid,
                "edge":             round(edge_yes, 4),
                "kelly_fraction":   round(kf, 4),
                "contracts":        contracts,
                "limit_price":      m["yes_ask"],
                "minutes_to_close": m["minutes_left"],
                "strategy":         "MODEL_UNDERPRICED_YES",
                "range":            f"${m['range_low']:,}–${m['range_high']:,}",
            })

        # --- Strategy 3: Model edge NO (market thinks too likely, buy NO) ---
        elif edge_no_clean >= MIN_EDGE_PCT and (1 - fair) > (1 - mid) and m["minutes_left"] > 15:
            fair_no = 1 - fair
            kf = kelly_fraction(fair_no, no_cost)
            contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / no_cost))
            signals.append({
                **_common,
                # For NO positions, market_bid from YES perspective is still yes_bid;
                # the cost to buy NO = no_cost = 1 - yes_bid, so we override limit_price only.
                "market_bid":       round(no_cost, 4),   # cost of the NO contract
                "ticker":           m["ticker"],
                "yes_no":           "NO",
                "fair_value":       round(fair_no, 4),
                "market_mid":       round(1 - mid, 4),
                "edge":             round(edge_no_clean, 4),
                "kelly_fraction":   round(kf, 4),
                "contracts":        contracts,
                "limit_price":      round(no_cost, 4),
                "minutes_to_close": m["minutes_left"],
                "strategy":         "MODEL_UNDERPRICED_NO",
                "range":            f"${m['range_low']:,}–${m['range_high']:,}",
            })

    # Sort by edge descending
    signals.sort(key=lambda x: x["edge"], reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_signal(signal_dict: dict, bankroll: float):
    """
    Write one signal row to trades/signals_log.csv.

    Parameters
    ----------
    signal_dict : dict
        A signal produced by generate_signals().  Must contain at least:
        ticker, strategy, yes_no, range, range_low, range_high,
        fair_value, limit_price (= market_ask), market_bid,
        edge, minutes_to_close, btc_price_at_signal.
    bankroll : float
        Current bankroll in USD, used to compute recommended_contracts.
    """
    market_ask = signal_dict["limit_price"]
    direction = signal_dict["yes_no"]

    # Fractional Kelly: edge / (1 - market_ask) * FRACTIONAL_KELLY, capped at MAX_POSITION_PCT
    denom = 1.0 - market_ask
    if denom > 0:
        kf = min((signal_dict["edge"] / denom) * FRACTIONAL_KELLY, MAX_POSITION_PCT)
    else:
        kf = 0.0
    kf = max(0.0, round(kf, 6))

    # recommended_contracts = kelly_fraction * bankroll / market_ask, min 1
    if market_ask > 0 and bankroll > 0:
        rec_contracts = max(1, round(kf * bankroll / market_ask))
    else:
        rec_contracts = 1

    row = {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "ticker":               signal_dict["ticker"],
        "strategy":             signal_dict["strategy"],
        "direction":            direction,
        "range_label":          signal_dict.get("range", ""),
        "range_low":            signal_dict.get("range_low", ""),
        "range_high":           signal_dict.get("range_high", ""),
        "fair_value":           f"{signal_dict['fair_value']:.4f}",
        "market_ask":           market_ask,
        "market_bid":           signal_dict.get("market_bid", ""),
        "edge_pp":              round(signal_dict["edge"] * 100, 2),
        "minutes_left":         signal_dict["minutes_to_close"],
        "btc_price_at_signal":  signal_dict.get("btc_price_at_signal", ""),
        "kelly_fraction":       kf,
        "recommended_contracts": rec_contracts,
        "acted_on":             signal_dict.get("_acted_on", "MANUAL"),
        "outcome":              "",
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
    global BTC_HOURLY_VOL_PCT, MIN_EDGE_PCT, FRACTIONAL_KELLY, MAX_POSITION_PCT, TIME_DECAY_THRESHOLD_MIN, TIME_DECAY_MIN_FAIR
    _cfg = load_strategy_config()
    _avoid_hours = set()
    if _cfg:
        BTC_HOURLY_VOL_PCT = _cfg.get("btc_hourly_vol", BTC_HOURLY_VOL_PCT)
        _me = _cfg.get("min_edge_pp", None)
        if _me is not None:
            MIN_EDGE_PCT = _me / 100.0  # config stores pp (8.0), code uses fraction (0.08)
        FRACTIONAL_KELLY = _cfg.get("fractional_kelly", FRACTIONAL_KELLY)
        MAX_POSITION_PCT = _cfg.get("max_position_pct", MAX_POSITION_PCT)
        TIME_DECAY_THRESHOLD_MIN = _cfg.get("time_decay_threshold_min", TIME_DECAY_THRESHOLD_MIN)
        TIME_DECAY_MIN_FAIR = _cfg.get("time_decay_min_fair", TIME_DECAY_MIN_FAIR)
        _avoid_hours = set(_cfg.get("avoid_hours", []))
        print(f"  [CONFIG] Loaded strategy_config.json (iteration {_cfg.get('iteration', '?')})")
        print(f"  [CONFIG] vol={BTC_HOURLY_VOL_PCT}, min_edge={MIN_EDGE_PCT}, kelly={FRACTIONAL_KELLY}")

    # Make _avoid_hours accessible at module level for generate_signals()
    globals()['_avoid_hours'] = _avoid_hours

    # File lock to prevent concurrent execution
    LOCK_FILE = Path(__file__).resolve().parent.parent / "trades" / ".trader.lock"
    lock_fh = None
    _lock_acquired = False
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fh = open(LOCK_FILE, "w")
        try:
            import msvcrt
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
            _lock_acquired = True
        except ImportError:
            # Linux/macOS — use fcntl instead
            import fcntl
            fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_acquired = True
        except (IOError, OSError):
            print("  [LOCK] Another trader instance is running. Exiting.")
            lock_fh.close()
            sys.exit(0)
    except (IOError, OSError):
        print("  [LOCK] Another trader instance is running. Exiting.")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Kalshi BTC Range Market Trader")
    parser.add_argument("--bankroll", type=float, default=10.0, help="Bankroll in USD (default: 10)")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (default: 60)")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle then exit (for scheduled agents)")
    parser.add_argument("--auto-trade", action="store_true", help="(Future) auto-place orders via Kalshi API")
    parser.add_argument("--min-edge", type=float, default=MIN_EDGE_PCT, help="Min edge to signal (default: 0.08)")
    parser.add_argument("--vol", type=float, default=BTC_HOURLY_VOL_PCT, help="BTC hourly vol fraction (default: 0.01)")
    parser.add_argument("--expiry-mode", action="store_true",
                        help="Expiry-intensive mode: widens TIME_DECAY window and relaxes edge for final-20-min plays")
    args = parser.parse_args()
    MIN_EDGE_PCT = args.min_edge
    BTC_HOURLY_VOL_PCT = args.vol

    # Expiry-mode: widen the time-decay window + relax edge bar for near-expiry in-range plays
    if args.expiry_mode:
        globals()['_expiry_mode'] = True
        print("  [MODE] EXPIRY-INTENSIVE — TIME_DECAY window extended, edge relaxed for final-20-min plays")
    else:
        globals()['_expiry_mode'] = False

    print("=" * 60)
    print("  KALSHI BTC RANGE MARKET TRADER")
    print(f"  Bankroll: ${args.bankroll:.2f}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Min edge: {args.min_edge*100:.0f}pp")
    print(f"  BTC hourly vol: {args.vol*100:.1f}%")
    print(f"  Expiry mode: {'ON' if args.expiry_mode else 'OFF'}")
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

        if check_daily_loss_stop():
            print("  [STOP] Daily loss limit (3%) reached. Halting trading for today.")
            if args.once:
                break
            time.sleep(args.interval)
            continue

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

        # Dedup + open position guard
        recent_tickers = get_recent_tickers(cooldown_minutes=30)
        open_tickers = get_open_position_tickers()

        # Log all signals and auto-trade eligible ones
        for s in signals:
            if args.auto_trade:
                if s["ticker"] in recent_tickers:
                    print(f"  [DEDUP] Skipping {s['ticker']} — traded within last 30 min")
                    log_signal(s, args.bankroll)
                    continue
                if s["ticker"] in open_tickers:
                    print(f"  [OPEN] Skipping {s['ticker']} — already have open position")
                    log_signal(s, args.bankroll)
                    continue
                result = place_order(
                    ticker=s["ticker"],
                    side=s["yes_no"],
                    contracts=s["contracts"],
                    limit_price_dollars=s["limit_price"],
                )
                if result is not None:
                    s["_acted_on"] = "AUTO"
                else:
                    print(f"  [AUTO-TRADE] Order failed — manual action: BUY {s['contracts']}x {s['yes_no']} {s['ticker']} @ ${s['limit_price']:.2f}")
            log_signal(s, args.bankroll)

        if signals:
            print(f"\n  ACTION REQUIRED:")
            top = signals[0]
            print(f"  On Kalshi, search: {top['ticker']}")
            print(f"  Buy {top['contracts']}x {top['yes_no']} at ${top['limit_price']:.2f} limit")
            print(f"  Cost: ${top['contracts'] * top['limit_price']:.2f}  |  Edge: {top['edge']*100:.1f}pp  |  Strategy: {top['strategy']}")

        # --- Scan frequency recommendation ---
        # Prints guidance so the Windows Task Scheduler can be tuned appropriately.
        # Different assets need different frequencies:
        #   BTC hourly range (KXBTC)  → 15 min during expiry window, 30 min otherwise
        #   Daily resolution markets  → 30-60 min is sufficient
        #   Multi-day / weekly        → hourly or less
        if markets:
            min_mins_left = min(m["minutes_left"] for m in markets)
            if min_mins_left <= 20:
                rec_interval = 10
                rec_reason = f"market expiring in {min_mins_left:.0f} min — use 10-min scan"
            elif min_mins_left <= 45:
                rec_interval = 15
                rec_reason = f"market expiring in {min_mins_left:.0f} min — use 15-min scan"
            else:
                rec_interval = 30
                rec_reason = "no near-expiry markets — 30-min scan sufficient"
            print(f"\n  [SCAN_FREQ] Recommended next interval: {rec_interval} min ({rec_reason})")

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
    finally:
        # Always release the file lock so future runs aren't blocked
        try:
            if 'lock_fh' in dir() and lock_fh:
                lock_fh.close()
            LOCK_FILE = Path(__file__).resolve().parent.parent / "trades" / ".trader.lock"
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        except Exception:
            pass
