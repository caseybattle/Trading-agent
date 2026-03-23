"""
economic_trader.py
Kalshi economic event market scanner + trader.
Targets CPI, Fed rate (FOMC), and NFP markets.
Enters when Bloomberg/Fed consensus diverges from Kalshi pricing by >8pp.
Optimal window: 24–72 hours before the event release.

Usage:
    python scripts/economic_trader.py --bankroll 10 --dry-run
    python scripts/economic_trader.py --bankroll 10 --auto-trade

Railway cron: 0 */6 * * *  (every 6 hours — add to railway.json once tested)

Requires (pip install):
    requests python-dotenv cryptography
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

# Reuse Kalshi auth + order placement from BTC trader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kalshi_btc_trader import build_auth_headers, place_order

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

KALSHI_BASE = "https://api.elections.kalshi.com"

# Kalshi series tickers to scan for economic markets
ECONOMIC_SERIES = ["CPI", "FOMC", "FED", "NONFARM", "JOBS", "CPIM", "FEDM"]

MIN_EDGE_PP     = 0.08   # 8 percentage points minimum edge
MAX_CONTRACTS   = 3      # max contracts per economic event (volatile + illiquid)
BANKROLL_PCT    = 0.02   # 2% of bankroll per trade
MIN_HOURS_OUT   = 24     # don't trade if event < 24h away
MAX_HOURS_OUT   = 72     # don't trade if event > 72h away

# FRED series IDs for latest economic data
FRED_CPI_ID  = "CPIAUCSL"   # Consumer Price Index, All Urban Consumers
FRED_NFP_ID  = "PAYEMS"     # Total Nonfarm Payrolls (thousands)
FRED_FEDFUNDS = "FEDFUNDS"  # Effective Federal Funds Rate

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "economic_trader.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED data fetching
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str) -> list[dict]:
    """Fetch the last 6 observations for a FRED series. Returns list of {date, value}."""
    try:
        r = requests.get(
            FRED_BASE,
            params={"id": series_id},
            timeout=15,
        )
        if r.status_code != 200:
            log.warning(f"FRED {series_id}: HTTP {r.status_code}")
            return []
        lines = r.text.strip().splitlines()
        # First line is header: DATE,VALUE
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) != 2:
                continue
            date_str, val_str = parts
            try:
                rows.append({"date": date_str, "value": float(val_str)})
            except ValueError:
                continue
        return rows[-6:]  # last 6 observations
    except Exception as e:
        log.warning(f"FRED {series_id} fetch error: {e}")
        return []


def get_latest_cpi() -> float | None:
    """Return the most recent CPI YoY % change (approximate from last 2 monthly values)."""
    rows = _fetch_fred_series(FRED_CPI_ID)
    if len(rows) < 13:
        # need 12 months prior for YoY
        return None
    # YoY = (current / year_ago - 1) * 100
    # FRED returns monthly, so index -1 is latest, -13 is 12 months ago
    all_rows = _fetch_fred_series.__wrapped__(FRED_CPI_ID) if hasattr(_fetch_fred_series, "__wrapped__") else None
    # Fallback: use last 2 to estimate MoM trend instead
    latest = rows[-1]["value"]
    prev   = rows[-2]["value"]
    mom_annualized = ((latest / prev) ** 12 - 1) * 100
    log.info(f"  CPI latest: {latest:.1f}, MoM annualized: {mom_annualized:.2f}%")
    return round(mom_annualized, 2)


def get_latest_cpi_yoy() -> float | None:
    """Return CPI YoY% using 13 months of FRED data."""
    try:
        r = requests.get(
            FRED_BASE,
            params={"id": FRED_CPI_ID},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        lines = r.text.strip().splitlines()
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    rows.append(float(parts[1]))
                except ValueError:
                    continue
        if len(rows) < 13:
            return None
        yoy = (rows[-1] / rows[-13] - 1) * 100
        log.info(f"  CPI YoY: {yoy:.2f}%")
        return round(yoy, 2)
    except Exception as e:
        log.warning(f"CPI YoY fetch error: {e}")
        return None


def get_latest_nfp() -> float | None:
    """Return latest NFP value (thousands of jobs added MoM)."""
    rows = _fetch_fred_series(FRED_NFP_ID)
    if len(rows) < 2:
        return None
    latest = rows[-1]["value"]
    prev   = rows[-2]["value"]
    mom_change = latest - prev  # thousands of jobs added
    log.info(f"  NFP latest level: {latest:.0f}k, MoM change: {mom_change:+.0f}k")
    return round(mom_change, 0)


def get_fed_funds_rate() -> float | None:
    """Return latest effective fed funds rate from FRED."""
    rows = _fetch_fred_series(FRED_FEDFUNDS)
    if not rows:
        return None
    rate = rows[-1]["value"]
    log.info(f"  Fed Funds Rate: {rate:.2f}%")
    return rate

# ---------------------------------------------------------------------------
# Kalshi market scanning
# ---------------------------------------------------------------------------

def fetch_economic_markets() -> list[dict]:
    """Fetch open Kalshi markets for economic event series."""
    markets = []
    for series in ECONOMIC_SERIES:
        path = f"/trade-api/v2/markets"
        params = {"series_ticker": series, "status": "open", "limit": 50}
        headers = build_auth_headers("GET", path)
        try:
            r = requests.get(
                f"{KALSHI_BASE}{path}",
                params=params,
                headers=headers,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                batch = data.get("markets", [])
                if batch:
                    log.info(f"  Series {series}: {len(batch)} open markets")
                markets.extend(batch)
            elif r.status_code == 404:
                pass  # series doesn't exist
            else:
                log.warning(f"  Series {series}: HTTP {r.status_code} — {r.text[:100]}")
        except Exception as e:
            log.warning(f"  Series {series} fetch error: {e}")
        time.sleep(0.3)  # rate limit
    return markets


def parse_event_window(market: dict) -> tuple[float, float] | None:
    """
    Return (hours_until_open, hours_until_close) for a market's close time.
    Returns None if close_time is unavailable or already closed.
    """
    close_str = market.get("close_time") or market.get("expiration_time")
    if not close_str:
        return None
    try:
        # Kalshi returns ISO8601 strings
        close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours_out = (close_dt - now).total_seconds() / 3600
        if hours_out < 0:
            return None
        return (0.0, hours_out)
    except Exception:
        return None


def get_kalshi_implied_prob(market: dict) -> float | None:
    """Extract YES implied probability from market data (0.0–1.0)."""
    yes_ask = market.get("yes_ask")
    yes_bid = market.get("yes_bid")
    if yes_ask is not None and yes_bid is not None:
        try:
            mid = (int(yes_ask) + int(yes_bid)) / 2 / 100
            return mid
        except (TypeError, ValueError):
            pass
    last = market.get("last_price")
    if last is not None:
        try:
            return int(last) / 100
        except (TypeError, ValueError):
            pass
    return None

# ---------------------------------------------------------------------------
# Consensus probability estimation
# ---------------------------------------------------------------------------

def estimate_cpi_consensus(market: dict, cpi_yoy: float | None) -> float | None:
    """
    Estimate consensus probability that market's YES outcome occurs.
    Market title example: "Will CPI exceed 3.2% YoY?"
    Uses current CPI trend + ±0.3pp uncertainty band.
    """
    if cpi_yoy is None:
        return None
    title = market.get("title", "").lower()
    subtitle = market.get("subtitle", "").lower()
    text = title + " " + subtitle

    # Try to extract threshold from title (e.g., "3.2%", "3.5%")
    import re
    match = re.search(r"(\d+\.\d+)\s*%", text)
    if not match:
        return None
    threshold = float(match.group(1))

    # "exceed" / "above" / "higher" → YES = CPI > threshold
    # "below" / "under" / "at or below" → YES = CPI ≤ threshold
    direction = "above"
    if any(w in text for w in ["below", "under", "at or below", "not exceed"]):
        direction = "below"

    # Use normal distribution: mean = cpi_yoy, std = 0.2pp (uncertainty around release)
    from scipy.stats import norm as _norm
    std = 0.2
    if direction == "above":
        prob = 1 - _norm.cdf(threshold, loc=cpi_yoy, scale=std)
    else:
        prob = _norm.cdf(threshold, loc=cpi_yoy, scale=std)
    return round(prob, 3)


def estimate_fed_consensus(market: dict, fed_rate: float | None) -> float | None:
    """
    Estimate consensus probability for a Fed rate market.
    Market titles: "Will the Fed cut rates at the [month] FOMC meeting?"
    """
    if fed_rate is None:
        return None
    title = market.get("title", "").lower()

    # Simple heuristic based on current rate + implied market odds
    # Fed cuts: more likely if rate > 4.5% and recent economic data soft
    # We'll use CME FedWatch approximation: if fed_rate >= 5.0, cut probability = 30%; if 4.5%, 45%
    if "cut" in title:
        if fed_rate >= 5.5:
            return 0.15
        elif fed_rate >= 5.0:
            return 0.30
        elif fed_rate >= 4.5:
            return 0.45
        else:
            return 0.60
    elif "raise" in title or "hike" in title:
        if fed_rate >= 5.5:
            return 0.10
        elif fed_rate >= 5.0:
            return 0.05
        else:
            return 0.02
    elif "hold" in title or "pause" in title or "unchanged" in title:
        # complement of cut + hike
        cut_p = estimate_fed_consensus({**market, "title": "cut"}, fed_rate) or 0.3
        hike_p = 0.05
        return round(max(0, 1.0 - cut_p - hike_p), 3)
    return None


def estimate_nfp_consensus(market: dict, nfp_mom: float | None) -> float | None:
    """
    Estimate consensus probability for an NFP market.
    Typical threshold: "Will NFP exceed 150k?"
    """
    if nfp_mom is None:
        return None
    import re
    title = market.get("title", "").lower()
    text = title + " " + market.get("subtitle", "").lower()

    match = re.search(r"(\d+)\s*k", text) or re.search(r"(\d[\d,]+)", text)
    if not match:
        return None
    try:
        threshold = float(match.group(1).replace(",", ""))
    except ValueError:
        return None

    direction = "above"
    if any(w in text for w in ["below", "under", "not exceed", "miss"]):
        direction = "below"

    from scipy.stats import norm as _norm
    std = 50  # ±50k uncertainty on NFP
    if direction == "above":
        prob = 1 - _norm.cdf(threshold, loc=nfp_mom, scale=std)
    else:
        prob = _norm.cdf(threshold, loc=nfp_mom, scale=std)
    return round(prob, 3)

# ---------------------------------------------------------------------------
# Signal detection + trade execution
# ---------------------------------------------------------------------------

def classify_series(market: dict) -> str:
    """Return 'cpi', 'fed', 'nfp', or 'unknown'."""
    ticker = (market.get("ticker") or "").upper()
    series = (market.get("series_ticker") or "").upper()
    title  = (market.get("title") or "").upper()
    combined = ticker + series + title
    if any(w in combined for w in ["CPI", "INFLATION", "PRICE INDEX"]):
        return "cpi"
    if any(w in combined for w in ["FOMC", "FED", "FEDERAL FUNDS", "RATE CUT", "RATE HIKE"]):
        return "fed"
    if any(w in combined for w in ["NONFARM", "NFP", "PAYROLL", "JOBS"]):
        return "nfp"
    return "unknown"


def compute_signal(
    market: dict,
    cpi_yoy: float | None,
    fed_rate: float | None,
    nfp_mom: float | None,
) -> dict | None:
    """
    Return signal dict if edge ≥ MIN_EDGE_PP and event is in 24–72h window.
    Returns None otherwise.
    """
    ticker  = market.get("ticker", "?")
    title   = market.get("title", "?")

    # Check event timing
    window = parse_event_window(market)
    if window is None:
        return None
    _, hours_out = window
    if not (MIN_HOURS_OUT <= hours_out <= MAX_HOURS_OUT):
        return None

    kalshi_prob = get_kalshi_implied_prob(market)
    if kalshi_prob is None:
        return None

    series_type = classify_series(market)
    if series_type == "cpi":
        consensus_prob = estimate_cpi_consensus(market, cpi_yoy)
    elif series_type == "fed":
        consensus_prob = estimate_fed_consensus(market, fed_rate)
    elif series_type == "nfp":
        consensus_prob = estimate_nfp_consensus(market, nfp_mom)
    else:
        return None

    if consensus_prob is None:
        return None

    edge = consensus_prob - kalshi_prob  # positive → YES underpriced; negative → NO underpriced

    if abs(edge) < MIN_EDGE_PP:
        return None

    side = "yes" if edge > 0 else "no"
    entry_price = kalshi_prob if side == "yes" else (1.0 - kalshi_prob)

    return {
        "ticker":         ticker,
        "title":          title,
        "series_type":    series_type,
        "hours_out":      round(hours_out, 1),
        "kalshi_prob":    round(kalshi_prob, 3),
        "consensus_prob": round(consensus_prob, 3),
        "edge_pp":        round(abs(edge) * 100, 1),
        "side":           side,
        "entry_price":    round(entry_price, 3),
    }


def size_trade(bankroll: float) -> int:
    """Return number of contracts (1–MAX_CONTRACTS) based on bankroll."""
    dollar_risk = bankroll * BANKROLL_PCT
    contracts   = max(1, min(MAX_CONTRACTS, int(dollar_risk)))
    return contracts


def log_signal(signal: dict, contracts: int, action: str, log_path: Path) -> None:
    """Append signal to CSV signal log."""
    fieldnames = [
        "timestamp", "ticker", "title", "series_type", "hours_out",
        "kalshi_prob", "consensus_prob", "edge_pp", "side", "entry_price",
        "contracts", "action",
    ]
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "ticker":         signal["ticker"],
            "title":          signal["title"],
            "series_type":    signal["series_type"],
            "hours_out":      signal["hours_out"],
            "kalshi_prob":    signal["kalshi_prob"],
            "consensus_prob": signal["consensus_prob"],
            "edge_pp":        signal["edge_pp"],
            "side":           signal["side"],
            "entry_price":    signal["entry_price"],
            "contracts":      contracts,
            "action":         action,
        })

# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

def run_scan(bankroll: float, auto_trade: bool, dry_run: bool) -> None:
    log.info("=" * 60)
    log.info(f"Economic trader scan — bankroll=${bankroll:.2f} auto_trade={auto_trade} dry_run={dry_run}")
    log.info("=" * 60)

    # Pull consensus data
    log.info("Fetching FRED consensus data...")
    cpi_yoy  = get_latest_cpi_yoy()
    fed_rate = get_fed_funds_rate()
    nfp_mom  = get_latest_nfp()

    log.info(f"Consensus: CPI YoY={cpi_yoy}% | Fed Funds={fed_rate}% | NFP MoM={nfp_mom}k")

    # Fetch Kalshi markets
    log.info("Scanning Kalshi economic markets...")
    markets = fetch_economic_markets()
    log.info(f"Found {len(markets)} total open economic markets")

    signals = []
    for market in markets:
        sig = compute_signal(market, cpi_yoy, fed_rate, nfp_mom)
        if sig:
            signals.append(sig)

    if not signals:
        log.info("No actionable signals found this cycle.")
        return

    log.info(f"\n{'='*60}")
    log.info(f"SIGNALS FOUND: {len(signals)}")
    log.info(f"{'='*60}")

    signal_log = LOG_DIR / "economic_signals.csv"
    contracts  = size_trade(bankroll)

    for sig in signals:
        log.info(
            f"  [{sig['series_type'].upper()}] {sig['ticker']} | {sig['title'][:60]}"
        )
        log.info(
            f"    {sig['hours_out']}h out | Kalshi={sig['kalshi_prob']:.1%} "
            f"Consensus={sig['consensus_prob']:.1%} Edge={sig['edge_pp']}pp"
        )
        log.info(f"    Action: BUY {sig['side'].upper()} @ {sig['entry_price']:.2f} x{contracts}")

        if dry_run:
            log_signal(sig, contracts, "DRY_RUN", signal_log)
            log.info("    [DRY RUN] — order not placed")
        elif auto_trade:
            result = place_order(
                ticker             = sig["ticker"],
                side               = sig["side"],
                contracts          = contracts,
                limit_price_dollars = sig["entry_price"],
            )
            action = "TRADED" if result else "FAILED"
            log_signal(sig, contracts, action, signal_log)
        else:
            log_signal(sig, contracts, "SIGNAL", signal_log)
            log.info("    [SIGNAL] — use --auto-trade to execute")


def load_config() -> dict:
    """Load strategy_config.json if available."""
    config_path = Path(__file__).resolve().parent.parent / "backtest" / "strategy_config.json"
    if _storage:
        try:
            return _storage.read_json("backtest/strategy_config.json")
        except Exception:
            pass
    if config_path.exists():
        import json
        with open(config_path) as f:
            return json.load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalshi economic event market scanner")
    parser.add_argument("--bankroll",   type=float, default=10.0, help="Current bankroll in USD")
    parser.add_argument("--auto-trade", action="store_true",      help="Execute trades automatically")
    parser.add_argument("--dry-run",    action="store_true",      help="Log signals without placing orders")
    args = parser.parse_args()

    config = load_config()
    if not config.get("scan_economic", False) and not args.auto_trade and not args.dry_run:
        log.info("scan_economic=false in strategy_config.json. Use --dry-run or --auto-trade to override.")
        return

    run_scan(
        bankroll   = args.bankroll,
        auto_trade = args.auto_trade,
        dry_run    = args.dry_run,
    )


if __name__ == "__main__":
    main()
