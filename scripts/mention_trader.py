"""
mention_trader.py
Scanner for Kalshi "mention markets" — markets about whether a person, company,
or topic will be mentioned in a specific publication or context.

Strategy:
  - Active subject (>3 articles / 48h) + YES priced < $0.70 → long edge
  - Dormant subject (0 articles / 7d)  + YES priced > $0.30 → short edge (buy NO)
  - Min edge: 8pp net of fees
  - Max position: 2 contracts (low liquidity)

Usage:
    python scripts/mention_trader.py [--auto-trade] [--bankroll 10]

Requires (pip install):
    requests
"""

import argparse
import base64
import json
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    _CRYPTO_OK = True
except ImportError:
    _CRYPTO_OK = False

_KALSHI_KEY_ID   = os.getenv("KALSHI_API_KEY_ID", "")
_KALSHI_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")

_KALSHI_KEY_CONTENTS = os.getenv("KALSHI_PRIVATE_KEY_CONTENTS", "")
if _KALSHI_KEY_CONTENTS and not _KALSHI_KEY_PATH:
    import tempfile as _tempfile
    _tmp = _tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
    _tmp.write(_KALSHI_KEY_CONTENTS.encode())
    _tmp.flush()
    _tmp.close()
    _KALSHI_KEY_PATH = _tmp.name

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
GNEWS_RSS   = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

MENTION_KEYWORDS = ["MENTION", "NAMED", "REFERENCED", "APPEAR", "CITED"]

ACTIVE_THRESHOLD  = 3     # articles in 48h → "active"
MIN_YES_LONG      = 0.70  # active  + YES below this → long
MAX_YES_SHORT     = 0.30  # dormant + YES above this → short
MIN_EDGE_PP       = 0.08  # minimum net edge in percentage points
KALSHI_FEE_PCT    = 0.07  # 7% taker fee (conservative)
MAX_CONTRACTS     = 2


# ---------------------------------------------------------------------------
# Auth helpers (identical pattern to kalshi_btc_trader.py)
# ---------------------------------------------------------------------------

def _load_private_key():
    if not _KALSHI_KEY_PATH or not Path(_KALSHI_KEY_PATH).exists():
        return None
    try:
        with open(_KALSHI_KEY_PATH, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    except Exception as e:
        print(f"[AUTH] Failed to load key: {e}")
        return None


def build_auth_headers(method: str, path: str) -> dict:
    if not _CRYPTO_OK or not _KALSHI_KEY_ID:
        return {}
    key = _load_private_key()
    if key is None:
        return {}
    ts_ms = str(int(time.time() * 1000))
    msg = (ts_ms + "" + method.upper() + path).encode()
    sig = key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       _KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json",
    }


# ---------------------------------------------------------------------------
# Kalshi market fetch
# ---------------------------------------------------------------------------

def fetch_open_markets(limit: int = 1000) -> list[dict]:
    path = f"/trade-api/v2/markets?status=open&limit={limit}"
    headers = build_auth_headers("GET", path)
    if not headers:
        headers = {"Content-Type": "application/json"}
    try:
        r = requests.get(KALSHI_BASE + path.split("v2", 1)[1], headers=headers, timeout=15)
        r.raise_for_status()
        return r.json().get("markets", [])
    except Exception as e:
        print(f"[KALSHI] Fetch failed: {e}")
        return []


def is_mention_market(market: dict) -> bool:
    title = (market.get("title") or "").upper()
    subtitle = (market.get("subtitle") or "").upper()
    series = (market.get("series_ticker") or "").upper()
    text = f"{title} {subtitle} {series}"
    return any(kw in text for kw in MENTION_KEYWORDS)


def extract_subject(market: dict) -> str:
    """Pull the likely subject from market title using simple heuristics."""
    title = market.get("title") or ""
    # Try patterns like "Will [X] be mentioned..." or "Will [X] appear in..."
    m = re.search(
        r'Will\s+"?([^"]+?)"?\s+(?:be\s+)?(?:mention|named|referenced|appear|cited)',
        title, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    # Fallback: take first 3 words after "Will"
    m2 = re.search(r'Will\s+(.+)', title, re.IGNORECASE)
    if m2:
        words = m2.group(1).split()
        return " ".join(words[:3])
    return title[:40]


def get_yes_price(market: dict) -> float | None:
    """Return best YES ask price in dollars (0–1), or None if unavailable."""
    try:
        yes_ask = market.get("yes_ask")
        if yes_ask is not None:
            return float(yes_ask) / 100.0
        # Some endpoints return last_price
        last = market.get("last_price")
        if last is not None:
            return float(last) / 100.0
    except (TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Google News RSS — no API key required
# ---------------------------------------------------------------------------

def count_recent_articles(subject: str, days: int) -> int:
    url = GNEWS_RSS.format(q=requests.utils.quote(subject))
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"[NEWS] RSS fetch failed for '{subject}': {e}")
        return -1  # -1 = unknown

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    count = 0
    for item in root.iter("item"):
        pub_date_el = item.find("pubDate")
        if pub_date_el is None or pub_date_el.text is None:
            continue
        try:
            # RFC 822 format: "Mon, 22 Mar 2026 12:00:00 GMT"
            pub = datetime.strptime(pub_date_el.text.strip(), "%a, %d %b %Y %H:%M:%S %Z")
            pub = pub.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if pub >= cutoff:
            count += 1
    return count


def classify_subject(subject: str) -> str:
    """Return 'active', 'dormant', or 'unknown'."""
    count_48h = count_recent_articles(subject, days=2)
    if count_48h == -1:
        return "unknown"
    if count_48h >= ACTIVE_THRESHOLD:
        return "active"

    count_7d = count_recent_articles(subject, days=7)
    if count_7d == -1:
        return "unknown"
    if count_7d == 0:
        return "dormant"
    return "neutral"


# ---------------------------------------------------------------------------
# Edge calculation
# ---------------------------------------------------------------------------

def calc_edge(activity: str, yes_price: float) -> tuple[str | None, float]:
    """
    Returns (side, edge_pp) or (None, 0).
    side: "yes" or "no"
    """
    if activity == "active" and yes_price < MIN_YES_LONG:
        model_prob = 0.75  # conservative estimate for active subject
        implied    = yes_price
        gross_edge = model_prob - implied
        net_edge   = gross_edge - KALSHI_FEE_PCT * yes_price
        if net_edge >= MIN_EDGE_PP:
            return "yes", net_edge

    if activity == "dormant" and yes_price > MAX_YES_SHORT:
        model_prob = 0.10  # conservative estimate for dormant subject
        implied    = 1.0 - yes_price  # NO price
        gross_edge = (1.0 - model_prob) - implied
        net_edge   = gross_edge - KALSHI_FEE_PCT * (1.0 - yes_price)
        if net_edge >= MIN_EDGE_PP:
            return "no", net_edge

    return None, 0.0


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def place_order(ticker: str, side: str, contracts: int, limit_price: float) -> dict | None:
    if not _CRYPTO_OK or not _KALSHI_KEY_ID:
        print("[ORDER] Auth unavailable — check .env and cryptography install.")
        return None
    path = "/trade-api/v2/portfolio/orders"
    headers = build_auth_headers("POST", path)
    if not headers:
        return None

    yes_cents = round(limit_price * 100)
    no_cents  = 100 - yes_cents

    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          "buy",
        "side":            side,
        "count":           contracts,
        "yes_price":       yes_cents,
        "no_price":        no_cents,
    }
    try:
        r = requests.post(
            f"https://api.elections.kalshi.com{path}",
            json=body, headers=headers, timeout=10,
        )
        if r.status_code in (200, 201):
            order = r.json().get("order", r.json())
            print(f"[ORDER] {side.upper()} {contracts}x {ticker} @ ${limit_price:.2f} | "
                  f"id={order.get('id','?')} status={order.get('status','?')}")
            return order
        else:
            print(f"[ORDER] Failed ({r.status_code}): {r.text[:200]}")
    except Exception as e:
        print(f"[ORDER] Error: {e}")
    return None


# ---------------------------------------------------------------------------
# Load strategy config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "backtest" / "strategy_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

def scan(auto_trade: bool, bankroll: float) -> None:
    cfg = load_config()
    if not cfg.get("scan_mentions", False) and not auto_trade:
        pass  # default off; --auto-trade implies user wants to run anyway

    print(f"\n[MENTION] Scan started — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"[MENTION] auto_trade={auto_trade} bankroll=${bankroll:.2f}")

    markets = fetch_open_markets()
    mention_markets = [m for m in markets if is_mention_market(m)]
    print(f"[MENTION] {len(markets)} open markets → {len(mention_markets)} mention-type")

    if not mention_markets:
        print("[MENTION] No mention markets found.")
        return

    signals = []
    for market in mention_markets:
        ticker    = market.get("ticker", "?")
        title     = market.get("title", "?")
        yes_price = get_yes_price(market)

        if yes_price is None:
            continue

        subject  = extract_subject(market)
        activity = classify_subject(subject)
        side, edge = calc_edge(activity, yes_price)

        status = (
            f"  {ticker:<30} YES=${yes_price:.2f}  activity={activity:<8}  "
            f"subject='{subject[:30]}'"
        )
        if side:
            status += f"  SIGNAL={side.upper()} edge={edge:.2%}"
            signals.append({
                "ticker":    ticker,
                "title":     title,
                "subject":   subject,
                "side":      side,
                "yes_price": yes_price,
                "activity":  activity,
                "edge":      edge,
            })
        print(status)

        time.sleep(0.5)  # gentle rate limit on Google News

    print(f"\n[MENTION] {len(signals)} actionable signal(s)")

    for sig in signals:
        limit_price = sig["yes_price"] if sig["side"] == "yes" else 1.0 - sig["yes_price"]
        contracts   = min(MAX_CONTRACTS, max(1, int(bankroll * 0.02 / limit_price)))
        print(f"\n  >> {sig['side'].upper()} {contracts}x {sig['ticker']} @ ${limit_price:.2f}"
              f"  (edge={sig['edge']:.2%}, activity={sig['activity']})")
        print(f"     Title: {sig['title'][:80]}")

        if auto_trade:
            place_order(sig["ticker"], sig["side"], contracts, limit_price)

    if signals and not auto_trade:
        print("\n[MENTION] Run with --auto-trade to execute orders.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kalshi mention market scanner")
    parser.add_argument("--auto-trade",  action="store_true", help="Place live orders")
    parser.add_argument("--bankroll",    type=float, default=10.0, help="Current bankroll ($)")
    args = parser.parse_args()

    scan(auto_trade=args.auto_trade, bankroll=args.bankroll)


if __name__ == "__main__":
    main()
