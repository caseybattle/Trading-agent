"""
weather_trader.py
Weather market trading module for Kalshi.
Compares NOAA temperature forecasts against Kalshi high-temp contract prices.
Only trades NO side. Minimum 10¢ edge. Cap $0.50 per trade.

Usage:
    python scripts/weather_trader.py --bankroll 10 --dry-run
    python scripts/weather_trader.py --bankroll 10 --auto-trade

Windows scheduled task (run once to register):
    schtasks /Create /TN "kalshi-weather-scanner" /TR "python C:\\Users\\casba\\Trading agent\\prediction-market-bot\\scripts\\weather_trader.py --bankroll 10 --auto-trade" /SC HOURLY /ST 00:15 /F

Requires (pip install):
    requests scipy python-dotenv cryptography
"""

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from scipy.stats import norm

# Reuse Kalshi auth + order placement from BTC trader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalshi_btc_trader import build_auth_headers, place_order, KALSHI_BASE

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

try:
    from storage_backend import get_storage
    _storage = get_storage()
except ImportError:
    _storage = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CITIES = {
    "NYC": {
        "lat": 40.7128, "lon": -74.0060,
        "name": "New York City",
        "aliases": ["NYC", "NEWYORK", "NEWYOR", "NYNY"],
    },
    "CHI": {
        "lat": 41.8781, "lon": -87.6298,
        "name": "Chicago",
        "aliases": ["CHI", "CHICAGO", "CHIC"],
    },
    "MIA": {
        "lat": 25.7617, "lon": -80.1918,
        "name": "Miami",
        "aliases": ["MIA", "MIAMI"],
    },
    "ATX": {
        "lat": 30.2672, "lon": -97.7431,
        "name": "Austin",
        "aliases": ["ATX", "AUSTIN", "AUS"],
    },
}

# Kalshi series tickers to scan — weather market naming varies; try all known patterns
WEATHER_SERIES = [
    "KXHIGHNYC", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHATX",
    "KXHIGH",
    "HIGHNYCM", "HIGHCHIM", "HIGHMIAMI", "HIGHAUSTIN",
    "HIGHNY", "HIGHIL", "HIGHFL", "HIGHTX",
]

TEMP_FORECAST_SIGMA = 3.0   # °F — typical NWS daily-high forecast error (1-sigma)
MIN_EDGE = 0.10              # $0.10 minimum edge (dollars, not percentage points)
MAX_TRADE = 0.50             # $0.50 cap per weather trade
FRACTIONAL_KELLY = 0.25

NOAA_HEADERS = {
    "User-Agent": "KalshiWeatherTrader/1.0 prediction-market-bot",
    "Accept": "application/geo+json",
}

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "weather_trades.log"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("weather_trader")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# NOAA forecast fetcher
# ---------------------------------------------------------------------------

def _noaa_get(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=NOAA_HEADERS, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [NOAA] Request failed {url[:60]}... : {e}")
        return None


def get_noaa_forecast_high(lat: float, lon: float) -> float | None:
    """
    Two-step NOAA lookup:
      1. points endpoint → resolves to forecast office + grid coords
      2. gridpoints forecast → first daytime period temperature (°F)
    """
    data = _noaa_get(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")
    if not data:
        return None
    forecast_url = data.get("properties", {}).get("forecast")
    if not forecast_url:
        return None

    time.sleep(0.2)  # NWS rate limit: be polite

    forecast = _noaa_get(forecast_url)
    if not forecast:
        return None

    for period in forecast.get("properties", {}).get("periods", []):
        if period.get("isDaytime", False):
            temp = period.get("temperature")
            if temp is None:
                continue
            unit = period.get("temperatureUnit", "F")
            if unit == "C":
                temp = temp * 9 / 5 + 32
            return float(temp)
    return None


def fetch_all_forecasts() -> dict:
    """Return {city_code: forecast_high_f} for all configured cities."""
    result = {}
    for code, info in CITIES.items():
        high = get_noaa_forecast_high(info["lat"], info["lon"])
        if high is not None:
            result[code] = high
            print(f"  [NOAA] {code} ({info['name']}): forecast high = {high:.0f}°F")
        else:
            print(f"  [NOAA] {code}: forecast unavailable")
    return result


# ---------------------------------------------------------------------------
# Kalshi weather market scanner
# ---------------------------------------------------------------------------

def _city_from_ticker(ticker: str) -> str | None:
    t = ticker.upper()
    for code, info in CITIES.items():
        for alias in info["aliases"]:
            if alias in t:
                return code
    return None


def _threshold_from_ticker(ticker: str) -> float | None:
    """
    Parse temperature threshold from ticker.
    Patterns: ...T70, ...T68F, ...70 (last 2-3 digit suffix)
    """
    # Try explicit -T<digits> pattern first
    m = re.search(r"-T(\d{2,3})F?(?:-|$)", ticker.upper())
    if m:
        return float(m.group(1))
    # Trailing number: e.g. KXHIGHNYC70
    m = re.search(r"(\d{2,3})$", ticker)
    if m:
        val = float(m.group(1))
        if 20 <= val <= 130:   # sane °F temperature range
            return val
    return None


def get_weather_markets() -> list[dict]:
    """Scan all configured Kalshi weather series and return enriched market list."""
    url = f"{KALSHI_BASE}/markets"
    seen: set[str] = set()
    raw: list[dict] = []

    for series in WEATHER_SERIES:
        params = {"status": "open", "limit": 100, "series_ticker": series}
        try:
            r = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=10)
            if r.status_code == 200:
                for m in r.json().get("markets", []):
                    t = m.get("ticker", "")
                    if t and t not in seen:
                        seen.add(t)
                        raw.append(m)
        except Exception:
            pass
        time.sleep(0.1)

    markets = []
    for m in raw:
        ticker = m.get("ticker", "")
        city = _city_from_ticker(ticker)
        threshold = _threshold_from_ticker(ticker)
        if city is None or threshold is None:
            continue

        yes_ask = float(m.get("yes_ask_dollars") or 0)
        yes_bid = float(m.get("yes_bid_dollars") or 0)
        if yes_ask <= 0 or yes_bid <= 0:
            continue

        no_cost = round(1.0 - yes_bid, 4)
        markets.append({
            "ticker":    ticker,
            "city_code": city,
            "city_name": CITIES[city]["name"],
            "threshold": threshold,
            "yes_ask":   yes_ask,
            "yes_bid":   yes_bid,
            "mid":       round((yes_ask + yes_bid) / 2, 4),
            "no_cost":   no_cost,
        })

    return markets


# ---------------------------------------------------------------------------
# Edge detection + position sizing
# ---------------------------------------------------------------------------

def noaa_implied_prob_yes(noaa_high: float, threshold: float) -> float:
    """P(actual high > threshold | NOAA forecast = noaa_high) via Normal model."""
    return float(1.0 - norm.cdf((threshold - noaa_high) / TEMP_FORECAST_SIGMA))


def kelly_no_contracts(fair_no: float, no_cost: float, bankroll: float) -> tuple[float, int]:
    """
    Kelly fraction for NO position, capped at MAX_TRADE.
    Returns (kelly_fraction, contracts).
    """
    if no_cost <= 0 or no_cost >= 1 or fair_no <= 0:
        return 0.0, 0
    b = (1.0 - no_cost) / no_cost
    q = 1.0 - fair_no
    f_full = (fair_no * b - q) / b
    f = max(0.0, min(0.5, f_full * FRACTIONAL_KELLY))
    dollar_size = min(f * bankroll, MAX_TRADE)
    contracts = max(1, int(dollar_size / no_cost))
    return round(f, 4), contracts


def detect_edges(markets: list[dict], forecasts: dict, bankroll: float) -> list[dict]:
    """
    Return NO-side trade signals where NOAA-implied edge ≥ MIN_EDGE.
    Edge = NOAA_prob_no − kalshi_no_cost.
    """
    signals = []
    for m in markets:
        city = m["city_code"]
        if city not in forecasts:
            continue

        noaa_high = forecasts[city]
        threshold = m["threshold"]

        prob_yes = noaa_implied_prob_yes(noaa_high, threshold)
        prob_no = 1.0 - prob_yes
        edge = prob_no - m["no_cost"]

        if edge < MIN_EDGE:
            continue

        kf, contracts = kelly_no_contracts(prob_no, m["no_cost"], bankroll)
        signals.append({
            "city_code":   city,
            "city_name":   m["city_name"],
            "ticker":      m["ticker"],
            "threshold_f": threshold,
            "noaa_high_f": noaa_high,
            "prob_yes":    round(prob_yes, 4),
            "prob_no":     round(prob_no, 4),
            "no_cost":     m["no_cost"],
            "edge":        round(edge, 4),
            "kelly":       kf,
            "contracts":   contracts,
        })

    signals.sort(key=lambda x: x["edge"], reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_trade(logger: logging.Logger, s: dict, action: str):
    logger.info(
        f"CITY={s['city_code']} TICKER={s['ticker']} "
        f"NOAA={s['noaa_high_f']:.0f}F THRESHOLD={s['threshold_f']:.0f}F "
        f"PROB_NO={s['prob_no']:.3f} NO_COST={s['no_cost']:.3f} "
        f"EDGE={s['edge']:+.3f} CONTRACTS={s['contracts']} "
        f"LIMIT=${s['no_cost']:.2f} ACTION={action}"
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_signals(signals: list[dict]):
    if not signals:
        print("\n  No weather trade signals this cycle.")
        return
    print(f"\n  {'#':<3} {'CITY':<6} {'TICKER':<42} {'NOAA':>5} {'THRESH':>6} "
          f"{'PROB_NO':>7} {'NO_COST':>7} {'EDGE':>7} {'CTRCTS':>6}")
    print("  " + "-" * 102)
    for i, s in enumerate(signals, 1):
        cost = s["contracts"] * s["no_cost"]
        print(
            f"  {i:<3} {s['city_code']:<6} {s['ticker']:<42} "
            f"{s['noaa_high_f']:>5.0f}°  {s['threshold_f']:>5.0f}°  "
            f"{s['prob_no']:>7.3f}  {s['no_cost']:>7.3f}  "
            f"{s['edge']:>+7.3f}  {s['contracts']:>6}"
        )
        print(f"       --> BUY {s['contracts']}x NO @ ${s['no_cost']:.2f}  (cost: ${cost:.2f})  "
              f"[NOAA={s['noaa_high_f']:.0f}°F, threshold={s['threshold_f']:.0f}°F, edge={s['edge']*100:.1f}¢]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalshi Weather Market Trader")
    parser.add_argument("--bankroll", type=float, default=10.0, help="Bankroll in USD (default: 10)")
    parser.add_argument("--auto-trade", action="store_true", help="Place live orders via Kalshi API")
    parser.add_argument("--dry-run", action="store_true", help="Show signals without placing orders")
    args = parser.parse_args()

    logger = _setup_logger()
    mode = "DRY RUN" if args.dry_run else ("AUTO-TRADE" if args.auto_trade else "SIGNAL ONLY")

    print("=" * 60)
    print("  KALSHI WEATHER MARKET TRADER")
    print(f"  Bankroll: ${args.bankroll:.2f}  |  Mode: {mode}")
    print(f"  Cities: {', '.join(CITIES.keys())}")
    print(f"  Min edge: ${MIN_EDGE:.2f}  |  Max per trade: ${MAX_TRADE:.2f}")
    print(f"  Log: {LOG_FILE}")
    print("=" * 60)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{ts}] Fetching NOAA forecasts...")
    forecasts = fetch_all_forecasts()
    if not forecasts:
        print("  No NOAA forecasts available. Exiting.")
        logger.info("SCAN_FAILED reason=noaa_unavailable")
        sys.exit(1)

    print(f"\n[{ts}] Scanning Kalshi weather markets...")
    markets = get_weather_markets()
    if not markets:
        print("  No weather markets found on Kalshi.")
        print("  (Weather market availability varies — check series tickers seasonally)")
        logger.info("SCAN_COMPLETE markets=0 signals=0")
        sys.exit(0)

    print(f"  Markets found: {len(markets)}")
    for m in markets:
        print(f"    {m['ticker']:<46} city={m['city_code']}  threshold={m['threshold']:.0f}°F  "
              f"YES_ask=${m['yes_ask']:.2f}  NO_cost=${m['no_cost']:.2f}")

    print(f"\n  Detecting NO-side edges (min ${MIN_EDGE:.2f})...")
    signals = detect_edges(markets, forecasts, args.bankroll)
    print_signals(signals)

    for s in signals:
        if args.dry_run:
            log_trade(logger, s, "DRY_RUN")
        elif args.auto_trade:
            result = place_order(
                ticker=s["ticker"],
                side="no",
                contracts=s["contracts"],
                limit_price_dollars=s["no_cost"],
            )
            log_trade(logger, s, "AUTO" if result else "AUTO_FAILED")
        else:
            log_trade(logger, s, "SIGNAL")

    logger.info(f"SCAN_COMPLETE markets={len(markets)} signals={len(signals)}")

    if signals:
        top = signals[0]
        print(f"\n  TOP SIGNAL: {top['ticker']} — BUY {top['contracts']}x NO @ ${top['no_cost']:.2f}")
        print(f"  NOAA forecast {top['noaa_high_f']:.0f}°F vs {top['threshold_f']:.0f}°F threshold  |  "
              f"edge = {top['edge']*100:.1f}¢")


if __name__ == "__main__":
    main()
