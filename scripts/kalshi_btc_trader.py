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

# Storage backend abstraction (local/S3)
try:
    from storage_backend import get_storage
    _storage = get_storage()
except ImportError:
    _storage = None

_KALSHI_KEY_ID   = os.getenv("KALSHI_API_KEY_ID", "")
_KALSHI_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

# Railway / cloud deployment: key contents supplied as env var instead of file
_KALSHI_KEY_CONTENTS = os.getenv("KALSHI_PRIVATE_KEY_CONTENTS", "")
if _KALSHI_KEY_CONTENTS and not _KALSHI_KEY_PATH:
    import tempfile as _tempfile
    _tmp_key = _tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
    _tmp_key.write(_KALSHI_KEY_CONTENTS.encode("utf-8"))
    _tmp_key.flush()
    _tmp_key.close()
    _KALSHI_KEY_PATH = _tmp_key.name


def load_strategy_config() -> dict:
    """Load tuned params from backtest/strategy_config.json if available."""
    try:
        if _storage:
            return _storage.read_json("backtest/strategy_config.json")
    except Exception:
        pass
    # Fallback: read local file directly (when storage_backend unavailable)
    try:
        import json
        local_cfg = Path(__file__).resolve().parent.parent / "backtest" / "strategy_config.json"
        if local_cfg.exists():
            with open(local_cfg) as f:
                return json.load(f)
    except Exception:
        pass
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
TIME_DECAY_THRESHOLD_MIN = 30  # flag "confirmed in range" if < this minutes
TIME_DECAY_MIN_FAIR = 0.70    # require model prob ≥ 70% for time-decay plays

# --- New filter constants (overridden from strategy_config.json) ---
MIN_CONTRACT_PRICE = 0.20   # Change 1: longshot bias filter
MIN_MINUTES_TO_EXPIRY = 30  # Change 2: expiry window floor
MAX_MINUTES_TO_EXPIRY = 90  # Change 2: expiry window ceiling
MIN_NET_EDGE_PCT = 0.08     # Change 3: net edge after fees
BTC_FEE_MULTIPLIER = 0.07   # Change 3: Kalshi BTC taker fee coefficient
SPX_FEE_MULTIPLIER = 0.035  # Change 4: SPX/Nasdaq 50% discount
SCAN_SPX = False            # Change 4: scan S&P 500 / Nasdaq markets

# SPX series tickers (prefix match)
SPX_SERIES = ["INXD", "NASDAQ"]

# Streak sizing bounds (Change 5)
_STREAK_MAX_MULTIPLIER = 2.0
_STREAK_MIN_MULTIPLIER = 0.5

SIGNAL_LOG = "trades/signals_log.csv"  # StorageBackend handles directory creation

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
            "fee_multiplier": BTC_FEE_MULTIPLIER,
            "asset": "BTC",
        })

    return markets


def get_spx_markets() -> list[dict]:
    """
    Fetch open S&P 500 and Nasdaq range markets from Kalshi.
    These carry a 50% fee discount vs BTC markets.
    Scans INXD (S&P 500 daily) and NASDAQ series.
    """
    markets = []
    now = datetime.now(timezone.utc)

    for series in SPX_SERIES:
        url = f"{KALSHI_BASE}/markets"
        params = {"status": "open", "limit": 200, "series_ticker": series}
        try:
            r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            r.raise_for_status()
            raw_markets = r.json().get("markets", [])
        except Exception as e:
            print(f"  [WARN] SPX fetch ({series}) failed: {e}")
            continue

        for m in raw_markets:
            ticker = m.get("ticker", "")
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
                continue

            yes_ask = float(m.get("yes_ask_dollars") or 0)
            yes_bid = float(m.get("yes_bid_dollars") or 0)
            volume_24h = float(m.get("volume_24h_fp") or 0)

            if yes_ask <= 0 or yes_bid <= 0:
                continue

            bucket_width = int(m.get("bucket_width") or 250)

            markets.append({
                "ticker": ticker,
                "range_low": range_low,
                "range_high": range_low + bucket_width,
                "yes_ask": yes_ask,
                "yes_bid": yes_bid,
                "mid": round((yes_ask + yes_bid) / 2, 4),
                "spread": round(yes_ask - yes_bid, 4),
                "volume_24h": volume_24h,
                "minutes_left": round(minutes_left, 2),
                "fee_multiplier": SPX_FEE_MULTIPLIER,
                "asset": series,
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
        return 1.0 if range_low <= btc_price < range_high else 0.0

    sigma = hourly_vol_pct * math.sqrt(hours_left)

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
# Change 3: Fee-aware edge calculation
# ---------------------------------------------------------------------------

def compute_net_edge(raw_edge: float, contract_price: float, fee_multiplier: float = BTC_FEE_MULTIPLIER) -> float:
    """
    Subtract round-trip taker fee from raw edge.

    Fee per contract = fee_multiplier * price * (1 - price)
    Fee as fraction of dollar invested = fee_multiplier * (1 - price)
    Round-trip (entry + exit) = 2 * fee_multiplier * (1 - price)

    At price=0.50, fee_mult=0.07: round-trip = 7pp
    At price=0.80, fee_mult=0.07: round-trip = 1.4pp
    At price=0.90, fee_mult=0.07: round-trip = 1.4pp
    """
    round_trip_fee = 2 * fee_multiplier * (1 - contract_price)
    return raw_edge - round_trip_fee


# ---------------------------------------------------------------------------
# Change 5: Streak-based position sizing
# ---------------------------------------------------------------------------

def get_streak_multiplier() -> float:
    """
    Read logs/circuit_state.json for recent win/loss streak.
    Returns a multiplier to apply to contract count:
      - 3+ consecutive wins:  +10% per win above 2, capped at 2.0x
      - 2+ consecutive losses: -20% per loss above 1, floored at 0.5x
      - Otherwise: 1.0 (neutral)

    circuit_state.json schema:
      {"consecutive_wins": N, "consecutive_losses": N, "last_updated": "..."}
    """
    try:
        circuit_path = Path(__file__).resolve().parent.parent / "logs" / "circuit_state.json"
        if not circuit_path.exists():
            return 1.0
        import json
        with open(circuit_path) as f:
            state = json.load(f)
        wins = int(state.get("consecutive_wins", 0))
        losses = int(state.get("consecutive_losses", 0))
        if wins >= 3:
            multiplier = 1.0 + 0.10 * (wins - 2)
            return min(multiplier, _STREAK_MAX_MULTIPLIER)
        elif losses >= 2:
            multiplier = 1.0 - 0.20 * (losses - 1)
            return max(multiplier, _STREAK_MIN_MULTIPLIER)
        return 1.0
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Daily loss hard stop
# ---------------------------------------------------------------------------
DAILY_LOSS_STOP_PCT = 0.03

def check_daily_loss_stop() -> bool:
    """Return True if TODAY's loss exceeds 3% of today's starting bankroll.

    Uses daily_start_bankroll / daily_start_date fields in bankroll.json.
    Resets automatically at UTC midnight.
    """
    try:
        if _storage:
            state = _storage.read_json("trades/bankroll.json")
        else:
            import json
            local_br = Path(__file__).resolve().parent.parent / "trades" / "bankroll.json"
            with open(local_br) as f:
                state = json.load(f)

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        current = float(state.get("current_bankroll", 10.0))

        # Reset daily tracking if it's a new UTC day
        if state.get("daily_start_date") != today_str:
            state["daily_start_bankroll"] = current
            state["daily_start_date"] = today_str
            if _storage:
                _storage.write_json("trades/bankroll.json", state)
            else:
                import json
                local_br = Path(__file__).resolve().parent.parent / "trades" / "bankroll.json"
                with open(local_br, "w") as f:
                    json.dump(state, f, indent=2)
            return False

        daily_start = float(state.get("daily_start_bankroll", current))
        daily_loss = daily_start - current
        if daily_loss >= daily_start * DAILY_LOSS_STOP_PCT:
            return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Duplicate order guard — ticker cooldown
# ---------------------------------------------------------------------------

def get_recent_tickers(cooldown_minutes: int = 30) -> set:
    """Return tickers traded within the last cooldown_minutes."""
    recent = set()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
    try:
        if _storage:
            rows = _storage.read_csv("trades/signals_log.csv")
            for row in rows:
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
    try:
        if _storage:
            df = _storage.read_parquet("trades/live_trades.parquet")
            return set(df[df["status"] == "OPEN"]["ticker"].tolist())
    except Exception:
        pass
    return set()


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(markets: list[dict], btc_price: float, bankroll: float,
                     ref_price: float = 0.0) -> list[dict]:
    """
    ref_price: current spot price for the asset (BTC price for BTC markets,
               index level for SPX/Nasdaq). Falls back to btc_price if 0.
    """
    current_hour = datetime.now(timezone.utc).hour
    if hasattr(sys.modules[__name__], '_avoid_hours') and current_hour in globals().get('_avoid_hours', set()):
        print(f"  [SKIP] Hour {current_hour} UTC is in avoid_hours list from optimizer")
        return []

    if ref_price <= 0:
        ref_price = btc_price

    streak_mult = get_streak_multiplier()
    signals = []

    for m in markets:
        minutes_left = m["minutes_left"]
        fee_mult = m.get("fee_multiplier", BTC_FEE_MULTIPLIER)

        # --- Change 2: Expiry window filter ---
        if minutes_left < MIN_MINUTES_TO_EXPIRY or minutes_left > MAX_MINUTES_TO_EXPIRY:
            # Exception: TIME_DECAY strategy is allowed at the 30-min boundary
            in_range = m["range_low"] <= ref_price < m["range_high"]
            if not (in_range and minutes_left <= TIME_DECAY_THRESHOLD_MIN):
                print(f"  [SKIP] {m['ticker']} EXPIRY_WINDOW_FILTER ({minutes_left:.1f}min)")
                continue

        hours_left = minutes_left / 60.0
        fair = compute_fair_value(ref_price, m["range_low"], m["range_high"], hours_left)

        if fair < MIN_FAIR_VALUE:
            continue

        mid = m["mid"]
        edge_yes = fair - m["yes_ask"]
        no_cost = round(1.0 - m["yes_bid"], 4)
        edge_no_clean = (1 - fair) - no_cost

        _common = {
            "range_low":          m["range_low"],
            "range_high":         m["range_high"],
            "market_bid":         m["yes_bid"],
            "btc_price_at_signal": btc_price,
            "fee_multiplier":     fee_mult,
        }

        # --- Strategy 1: Time-decay (confirmed in-range, minutes to close) ---
        in_range = m["range_low"] <= ref_price < m["range_high"]
        if in_range and minutes_left <= TIME_DECAY_THRESHOLD_MIN and fair >= TIME_DECAY_MIN_FAIR:
            price = m["yes_ask"]
            # Change 1: longshot bias filter
            if price < MIN_CONTRACT_PRICE:
                print(f"  [SKIP] {m['ticker']} LONGSHOT_BIAS_FILTER YES@{price:.2f}")
                continue
            edge = edge_yes
            # Change 3: fee-adjusted edge
            net_edge = compute_net_edge(edge, price, fee_mult)
            if net_edge >= MIN_NET_EDGE_PCT:
                kf = kelly_fraction(fair, price)
                raw_contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / price))
                contracts = max(1, round(raw_contracts * streak_mult))
                signals.append({
                    **_common,
                    "ticker":           m["ticker"],
                    "yes_no":           "YES",
                    "fair_value":       round(fair, 4),
                    "market_mid":       mid,
                    "edge":             round(net_edge, 4),
                    "raw_edge":         round(edge, 4),
                    "kelly_fraction":   round(kf, 4),
                    "contracts":        contracts,
                    "limit_price":      price,
                    "minutes_to_close": minutes_left,
                    "strategy":         "TIME_DECAY_IN_RANGE",
                    "range":            f"${m['range_low']:,}–${m['range_high']:,}",
                })

        # --- Strategy 2: Model edge YES (overpriced probability) ---
        elif edge_yes >= MIN_EDGE_PCT and fair > mid and minutes_left > 15:
            price = m["yes_ask"]
            # Change 1: longshot bias filter
            if price < MIN_CONTRACT_PRICE:
                print(f"  [SKIP] {m['ticker']} LONGSHOT_BIAS_FILTER YES@{price:.2f}")
                continue
            # Change 3: fee-adjusted edge
            net_edge = compute_net_edge(edge_yes, price, fee_mult)
            if net_edge >= MIN_NET_EDGE_PCT:
                kf = kelly_fraction(fair, price)
                raw_contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / price))
                contracts = max(1, round(raw_contracts * streak_mult))
                signals.append({
                    **_common,
                    "ticker":           m["ticker"],
                    "yes_no":           "YES",
                    "fair_value":       round(fair, 4),
                    "market_mid":       mid,
                    "edge":             round(net_edge, 4),
                    "raw_edge":         round(edge_yes, 4),
                    "kelly_fraction":   round(kf, 4),
                    "contracts":        contracts,
                    "limit_price":      price,
                    "minutes_to_close": minutes_left,
                    "strategy":         "MODEL_UNDERPRICED_YES",
                    "range":            f"${m['range_low']:,}–${m['range_high']:,}",
                })

        # --- Strategy 3: Model edge NO (market thinks too likely, buy NO) ---
        elif edge_no_clean >= MIN_EDGE_PCT and (1 - fair) > (1 - mid) and minutes_left > 15:
            # Change 1: longshot bias filter — apply to NO contract price
            if no_cost < MIN_CONTRACT_PRICE:
                print(f"  [SKIP] {m['ticker']} LONGSHOT_BIAS_FILTER NO@{no_cost:.2f}")
                continue
            fair_no = 1 - fair
            # Change 3: fee-adjusted edge
            net_edge = compute_net_edge(edge_no_clean, no_cost, fee_mult)
            if net_edge >= MIN_NET_EDGE_PCT:
                kf = kelly_fraction(fair_no, no_cost)
                raw_contracts = max(1, int(bankroll * min(kf, MAX_POSITION_PCT) / no_cost))
                contracts = max(1, round(raw_contracts * streak_mult))
                signals.append({
                    **_common,
                    "market_bid":       round(no_cost, 4),
                    "ticker":           m["ticker"],
                    "yes_no":           "NO",
                    "fair_value":       round(fair_no, 4),
                    "market_mid":       round(1 - mid, 4),
                    "edge":             round(net_edge, 4),
                    "raw_edge":         round(edge_no_clean, 4),
                    "kelly_fraction":   round(kf, 4),
                    "contracts":        contracts,
                    "limit_price":      round(no_cost, 4),
                    "minutes_to_close": minutes_left,
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
    """
    market_ask = signal_dict["limit_price"]
    direction = signal_dict["yes_no"]

    denom = 1.0 - market_ask
    if denom > 0:
        kf = min((signal_dict["edge"] / denom) * FRACTIONAL_KELLY, MAX_POSITION_PCT)
    else:
        kf = 0.0
    kf = max(0.0, round(kf, 6))

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

    try:
        if _storage:
            _storage.append_csv("trades/signals_log.csv", row, fieldnames=CSV_HEADERS)
    except Exception as e:
        print(f"[WARN] Failed to log signal: {e}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_market_table(markets: list[dict], btc_price: float):
    print(f"\n  {'TICKER':<35} {'RANGE':>22}  {'MID':>5}  {'SPREAD':>6}  {'MIN_LEFT':>8}  {'24H_VOL':>10}  {'FAIR':>5}  {'EDGE':>6}")
    print("  " + "-" * 115)
    for m in sorted(markets, key=lambda x: x["minutes_left"])[:20]:
        ref = btc_price
        fair = compute_fair_value(ref, m["range_low"], m["range_high"], m["minutes_left"] / 60)
        edge = fair - m["yes_ask"]
        in_range = m["range_low"] <= ref < m["range_high"]
        flag = " <-- IN RANGE" if in_range else ""
        print(f"  {m['ticker']:<35} ${m['range_low']:>7,}–${m['range_high']:<7,}  {m['mid']:>5.2f}  {m['spread']:>6.2f}  {m['minutes_left']:>8.1f}  ${m['volume_24h']:>9,.0f}  {fair:>5.2f}  {edge:>+6.2f}{flag}")


def print_signals(signals: list[dict], btc_price: float, bankroll: float):
    if not signals:
        print("\n  No signals this cycle (no edge found above threshold).")
        return

    print(f"\n  {'#':<3} {'STRATEGY':<25} {'TICKER':<35} {'RANGE':>22}  {'FAIR':>5}  {'MID':>5}  {'NET_EDGE':>8}  {'CONTRACTS':>9}  {'LIMIT':>6}  {'MIN_LEFT':>8}")
    print("  " + "-" * 150)
    for i, s in enumerate(signals, 1):
        cost = s["contracts"] * s["limit_price"]
        raw_str = f"(raw {s.get('raw_edge', s['edge'])*100:.1f}pp)" if "raw_edge" in s else ""
        print(
            f"  {i:<3} {s['strategy']:<25} {s['ticker']:<35} {s['range']:>22}  "
            f"{s['fair_value']:>5.2f}  {s['market_mid']:>5.2f}  {s['edge']*100:>+7.1f}pp  "
            f"{s['contracts']:>9}  {s['limit_price']:>6.2f}  {s['minutes_to_close']:>8.1f}  {raw_str}"
        )
        print(f"       --> BUY {s['contracts']}x {s['yes_no']} @ ${s['limit_price']:.2f}  (cost: ${cost:.2f} / bankroll: ${bankroll:.2f})")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    global BTC_HOURLY_VOL_PCT, MIN_EDGE_PCT, FRACTIONAL_KELLY, MAX_POSITION_PCT
    global TIME_DECAY_THRESHOLD_MIN, TIME_DECAY_MIN_FAIR
    global MIN_CONTRACT_PRICE, MIN_MINUTES_TO_EXPIRY, MAX_MINUTES_TO_EXPIRY
    global MIN_NET_EDGE_PCT, SCAN_SPX

    _cfg = load_strategy_config()
    _avoid_hours = set()
    if _cfg:
        BTC_HOURLY_VOL_PCT = _cfg.get("btc_hourly_vol", BTC_HOURLY_VOL_PCT)
        _me = _cfg.get("min_edge_pp", None)
        if _me is not None:
            MIN_EDGE_PCT = _me / 100.0
        FRACTIONAL_KELLY = _cfg.get("fractional_kelly", FRACTIONAL_KELLY)
        MAX_POSITION_PCT = _cfg.get("max_position_pct", MAX_POSITION_PCT)
        TIME_DECAY_THRESHOLD_MIN = _cfg.get("time_decay_threshold_min", TIME_DECAY_THRESHOLD_MIN)
        TIME_DECAY_MIN_FAIR = _cfg.get("time_decay_min_fair", TIME_DECAY_MIN_FAIR)
        _avoid_hours = set(_cfg.get("avoid_hours", []))
        # New filter params
        MIN_CONTRACT_PRICE    = _cfg.get("min_contract_price", MIN_CONTRACT_PRICE)
        MIN_MINUTES_TO_EXPIRY = _cfg.get("min_minutes_to_expiry", MIN_MINUTES_TO_EXPIRY)
        MAX_MINUTES_TO_EXPIRY = _cfg.get("max_minutes_to_expiry", MAX_MINUTES_TO_EXPIRY)
        _net = _cfg.get("min_net_edge_pp", None)
        if _net is not None:
            MIN_NET_EDGE_PCT = _net / 100.0
        SCAN_SPX = _cfg.get("scan_spx", SCAN_SPX)
        print(f"  [CONFIG] Loaded strategy_config.json (iteration {_cfg.get('iteration', '?')})")
        print(f"  [CONFIG] vol={BTC_HOURLY_VOL_PCT}, min_edge={MIN_EDGE_PCT}, kelly={FRACTIONAL_KELLY}")
        print(f"  [CONFIG] longshot_floor={MIN_CONTRACT_PRICE}, expiry={MIN_MINUTES_TO_EXPIRY}-{MAX_MINUTES_TO_EXPIRY}min, scan_spx={SCAN_SPX}")

    globals()['_avoid_hours'] = _avoid_hours

    # File lock to prevent concurrent execution
    LOCK_FILE = Path(__file__).resolve().parent.parent / "trades" / ".trader.lock"
    lock_fh = None
    try:
        LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        lock_fh = open(LOCK_FILE, "w")
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    args = parser.parse_args()
    MIN_EDGE_PCT = args.min_edge
    BTC_HOURLY_VOL_PCT = args.vol

    streak_mult = get_streak_multiplier()

    print("=" * 60)
    print("  KALSHI BTC RANGE MARKET TRADER")
    print(f"  Bankroll: ${args.bankroll:.2f}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Min edge: {args.min_edge*100:.0f}pp | Min net edge: {MIN_NET_EDGE_PCT*100:.0f}pp")
    print(f"  Longshot floor: ${MIN_CONTRACT_PRICE:.2f} | Expiry: {MIN_MINUTES_TO_EXPIRY}-{MAX_MINUTES_TO_EXPIRY}min")
    print(f"  BTC hourly vol: {args.vol*100:.1f}%")
    print(f"  Streak multiplier: {streak_mult:.2f}x")
    print(f"  SPX scanner: {'ON' if SCAN_SPX else 'OFF'}")
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

        # --- BTC markets ---
        markets = get_btc_range_markets()
        if not markets:
            print("  No BTC range markets found — skipping cycle.")
            time.sleep(args.interval)
            continue

        print(f"  Markets found: {len(markets)}")
        print_market_table(markets, btc_price)

        signals = generate_signals(markets, btc_price, args.bankroll)

        # --- Change 4: SPX/Nasdaq markets ---
        if SCAN_SPX:
            spx_markets = get_spx_markets()
            if spx_markets:
                print(f"  SPX/Nasdaq markets found: {len(spx_markets)}")
                spx_signals = generate_signals(spx_markets, btc_price, args.bankroll)
                signals = sorted(signals + spx_signals, key=lambda x: x["edge"], reverse=True)

        print(f"\n  SIGNALS ({len(signals)} found):")
        print_signals(signals, btc_price, args.bankroll)

        # Dedup + open position guard
        recent_tickers = get_recent_tickers(cooldown_minutes=30)
        open_tickers = get_open_position_tickers()

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
            print(f"  Cost: ${top['contracts'] * top['limit_price']:.2f}  |  Net Edge: {top['edge']*100:.1f}pp  |  Strategy: {top['strategy']}")

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
