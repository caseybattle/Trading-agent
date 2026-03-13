"""
Kalshi Historical Market Scraper
Pulls settled markets from the Kalshi REST API (requires RSA key pair)
Stores raw data and engineers features compatible with Polymarket schema

Usage:
    export KALSHI_API_KEY_ID="your_key_id_here"
    export KALSHI_PRIVATE_KEY_PATH="/path/to/private_key.pem"
    python collect_kalshi.py --days 90
    python collect_kalshi.py --days 365 --max 5000

Auth: RSA PKCS1v15 SHA-256 per-request signing (Kalshi v2 API).
API docs: https://trading.kalshi.com/docs/api/v2
"""

import os
import time
import base64
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

# ─── Config ─────────────────────────────────────────────────────────────────────
BASE_URL   = "https://api.elections.kalshi.com/trade-api/v2"
DATA_DIR   = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "features").mkdir(exist_ok=True)

CATEGORY_MAP = {
    "politics": ["election", "president", "senate", "congress", "vote", "governor",
                 "democrat", "republican", "legislation", "approval", "policy"],
    "crypto":   ["bitcoin", "btc", "ethereum", "eth", "crypto", "defi", "nft",
                 "blockchain", "solana", "price", "market cap"],
    "science":  ["nasa", "space", "climate", "covid", "cancer", "fda", "vaccine",
                 "research", "temperature", "earthquake", "hurricane", "storm"],
    "sports":   ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
                 "baseball", "championship", "super bowl", "world cup", "playoffs"],
    "economics":["fed", "inflation", "gdp", "unemployment", "rate", "cpi", "jobs",
                 "recession", "interest rate", "fomc", "reserve"],
}


# ─── Auth ────────────────────────────────────────────────────────────────────────

def _load_private_key():
    """Load the RSA private key from the PEM file at KALSHI_PRIVATE_KEY_PATH."""
    pem_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    if not pem_path:
        return None
    with open(pem_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _kalshi_headers(method: str, path: str) -> dict:
    """
    Build Kalshi RSA auth headers for a single request.

    Headers required:
      KALSHI-ACCESS-KEY       — API Key ID from env KALSHI_API_KEY_ID
      KALSHI-ACCESS-TIMESTAMP — current time in milliseconds as a string
      KALSHI-ACCESS-SIGNATURE — base64(RSA-PKCS1v15-SHA256(ts + METHOD + path))
    """
    key_id      = os.environ.get("KALSHI_API_KEY_ID", "")
    private_key = _load_private_key()
    ts          = str(int(time.time() * 1000))
    headers = {
        "Content-Type":           "application/json",
        "KALSHI-ACCESS-KEY":      key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }
    if private_key and key_id:
        msg = (ts + method.upper() + path).encode("utf-8")
        sig = private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
        headers["KALSHI-ACCESS-SIGNATURE"] = base64.b64encode(sig).decode("utf-8")
    else:
        print("[!] KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY_PATH not set — unauthenticated.")
    return headers


def get_session() -> requests.Session:
    """Build a base session for Kalshi API (auth headers added per-request)."""
    session = requests.Session()
    session.headers.update({"User-Agent": "prediction-market-bot/1.0"})
    return session


# ─── Utility ─────────────────────────────────────────────────────────────────────

def infer_category(title: str) -> str:
    t = title.lower()
    for cat, kws in CATEGORY_MAP.items():
        if any(kw in t for kw in kws):
            return cat
    return "other"


def safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


# ─── Kalshi API Fetchers ─────────────────────────────────────────────────────────

def fetch_settled_markets(session: requests.Session, limit: int = 100,
                          cursor: Optional[str] = None,
                          cutoff_date: Optional[datetime] = None) -> tuple:
    """
    Fetch a page of settled markets.
    Returns (markets_list, next_cursor).
    """
    path   = "/trade-api/v2/markets"
    params: dict = {"status": "settled", "limit": limit}
    if cursor:
        params["cursor"] = cursor

    resp = session.get(
        f"{BASE_URL}/markets",
        params=params,
        headers=_kalshi_headers("GET", path),
        timeout=30,
    )
    if resp.status_code == 401:
        print("[!] Kalshi auth failed — check KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH")
        return [], None
    resp.raise_for_status()

    data        = resp.json()
    markets     = data.get("markets", [])
    next_cursor = data.get("cursor")
    return markets, next_cursor


def fetch_market_history(session: requests.Session, ticker: str) -> pd.DataFrame:
    """Fetch price/volume history for a Kalshi market ticker."""
    try:
        path = f"/trade-api/v2/markets/{ticker}/history"
        resp = session.get(
            f"{BASE_URL}/markets/{ticker}/history",
            params={"limit": 1000},
            headers=_kalshi_headers("GET", path),
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            history = data.get("history", [])
            if history:
                df = pd.DataFrame(history)
                if "ts" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
                elif "created_time" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["created_time"], utc=True)
                # Kalshi yes price is typically 0-100 cents, normalize to 0-1
                price_col = next((c for c in df.columns if "yes" in c.lower() and
                                  "price" in c.lower()), None)
                if price_col:
                    df["price"] = df[price_col].astype(float) / 100.0
                elif "price" in df.columns:
                    df["price"] = df["price"].astype(float)
                    if df["price"].max() > 1.5:
                        df["price"] = df["price"] / 100.0
                return df[["timestamp", "price"]].dropna().sort_values("timestamp")
    except Exception:
        pass
    return pd.DataFrame()


# ─── Feature Engineering ─────────────────────────────────────────────────────────

def engineer_features(market: dict, price_history: pd.DataFrame) -> dict:
    """Compute features for a single Kalshi market — same schema as Polymarket."""
    ticker   = market.get("ticker", "")
    title    = market.get("title", "")
    category = infer_category(title)

    open_time  = parse_date(market.get("open_time")  or market.get("created_time"))
    close_time = parse_date(market.get("close_time") or market.get("expiration_time"))

    volume     = safe_float(market.get("volume"))
    open_int   = safe_float(market.get("open_interest"))
    liquidity  = safe_float(market.get("liquidity", volume))

    # Kalshi prices are in cents (0–100)
    best_ask_raw = safe_float(market.get("yes_ask", market.get("best_ask", 0)))
    best_bid_raw = safe_float(market.get("yes_bid", market.get("best_bid", 0)))
    best_ask = best_ask_raw / 100.0 if best_ask_raw > 1.5 else best_ask_raw
    best_bid = best_bid_raw / 100.0 if best_bid_raw > 1.5 else best_bid_raw
    spread_pct = best_ask - best_bid

    liquidity_ratio = volume / max(open_int, 1.0)

    # Time features
    resolution_hours  = None
    days_since_open   = None
    if open_time and close_time:
        delta = close_time - open_time
        resolution_hours = delta.total_seconds() / 3600
        days_since_open  = delta.days

    # Price-derived features
    price_momentum_24h  = 0.0
    price_volatility_7d = 0.0
    price_at_T7d        = None
    price_at_T1d        = None

    if not price_history.empty and len(price_history) >= 2:
        ph = price_history.copy()
        if close_time:
            ph = ph[ph["timestamp"] <= close_time]
        if not ph.empty:
            last_price = ph["price"].iloc[-1]
            cut24h = ph["timestamp"].iloc[-1] - pd.Timedelta(hours=24)
            ph24 = ph[ph["timestamp"] >= cut24h]
            if len(ph24) >= 2:
                price_momentum_24h = last_price - ph24["price"].iloc[0]

            cut7d = ph["timestamp"].iloc[-1] - pd.Timedelta(days=7)
            ph7d = ph[ph["timestamp"] >= cut7d]
            if len(ph7d) >= 2:
                price_volatility_7d = ph7d["price"].std()
                price_at_T7d = ph7d["price"].iloc[0]

            if close_time:
                cut1d = close_time - pd.Timedelta(days=1)
                ph1d = ph[ph["timestamp"] <= cut1d]
                if not ph1d.empty:
                    price_at_T1d = ph1d["price"].iloc[-1]

    # Outcome label
    result = str(market.get("result", "")).upper()
    outcome_label = -1
    if result in ("YES", "TRUE", "1"):
        outcome_label = 1
    elif result in ("NO", "FALSE", "0"):
        outcome_label = 0

    return {
        "market_id":                 ticker,
        "question":                  title,
        "category":                  category,
        "outcome_label":             outcome_label,
        "winning_outcome":           result,
        "volume":                    volume,
        "open_interest":             open_int,
        "spread_pct":                spread_pct,
        "liquidity_ratio":           liquidity_ratio,
        "time_to_resolution_hours":  resolution_hours,
        "days_since_market_open":    days_since_open,
        "price_momentum_24h":        price_momentum_24h,
        "price_volatility_7d":       price_volatility_7d,
        "price_at_T7d":              price_at_T7d,
        "price_at_T1d":              price_at_T1d,
        "volume_anomaly_score":      0.0,  # computed cross-market
        "start_date":                str(open_time)  if open_time  else None,
        "end_date":                  str(close_time) if close_time else None,
        "platform":                  "kalshi",
    }


def add_volume_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    for cat in df["category"].unique():
        mask = df["category"] == cat
        vols = df.loc[mask, "volume"]
        std  = vols.std()
        if std and std > 0:
            df.loc[mask, "volume_anomaly_score"] = (vols - vols.mean()) / std
    return df


# ─── Main Scraper ─────────────────────────────────────────────────────────────────

def scrape_kalshi(days: int = 90, max_markets: int = 10000,
                  fetch_prices: bool = True) -> pd.DataFrame:
    """
    Scrape Kalshi settled markets, engineer features, save to parquet.
    """
    session     = get_session()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    print(f"[Kalshi] Fetching markets resolved after {cutoff_date.date()} ...")

    raw_markets: list = []
    cursor: Optional[str] = None
    page = 0

    while len(raw_markets) < max_markets:
        try:
            markets, cursor = fetch_settled_markets(session, limit=100, cursor=cursor,
                                                    cutoff_date=cutoff_date)
        except requests.RequestException as e:
            print(f"  [!] Request error on page {page}: {e}")
            break

        if not markets:
            print(f"  [✓] Done. Total markets collected: {len(raw_markets)}")
            break

        # Apply cutoff filter
        filtered = []
        stop = False
        for m in markets:
            close_time = parse_date(m.get("close_time") or m.get("expiration_time"))
            if close_time and close_time < cutoff_date:
                stop = True
                break
            result = str(m.get("result", "")).upper()
            if result in ("YES", "NO", "TRUE", "FALSE", "1", "0"):
                filtered.append(m)

        raw_markets.extend(filtered)
        print(f"  Page {page}: +{len(filtered)} → total {len(raw_markets)}")
        page += 1

        if stop or not cursor:
            break
        time.sleep(0.3)

    if not raw_markets:
        print("[!] No Kalshi markets collected. Check API key / connectivity.")
        return pd.DataFrame()

    print(f"\n[Kalshi] Engineering features for {len(raw_markets)} markets ...")
    records = []
    for i, market in enumerate(raw_markets):
        ticker        = market.get("ticker", "")
        price_history = pd.DataFrame()
        if fetch_prices and ticker:
            try:
                price_history = fetch_market_history(session, ticker)
                time.sleep(0.15)
            except Exception:
                pass
        try:
            records.append(engineer_features(market, price_history))
        except Exception as e:
            print(f"  [!] Feature error for {ticker}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(raw_markets)} ...")

    df = pd.DataFrame(records)
    df = df[df["outcome_label"].isin([0, 1])].reset_index(drop=True)
    df = add_volume_anomaly_scores(df)

    # ── Save ───────────────────────────────────────────────────────────────────
    raw_path = DATA_DIR / "kalshi_resolved.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"\n[✓] Saved Kalshi resolved → {raw_path} ({len(df)} rows)")

    feat_cols = [
        "market_id", "question", "category", "outcome_label",
        "time_to_resolution_hours", "days_since_market_open",
        "volume", "open_interest", "liquidity_ratio",
        "price_momentum_24h", "price_volatility_7d",
        "price_at_T7d", "price_at_T1d",
        "volume_anomaly_score", "spread_pct",
        "start_date", "end_date",
    ]
    available_cols = [c for c in feat_cols if c in df.columns]
    feat_path = DATA_DIR / "features" / "kalshi_features.parquet"
    df[available_cols].to_parquet(feat_path, index=False)
    print(f"[✓] Saved Kalshi features → {feat_path}")

    # ── Merge with Polymarket features if they exist ───────────────────────────
    poly_feat = DATA_DIR / "features" / "market_features.parquet"
    if poly_feat.exists():
        poly_df = pd.read_parquet(poly_feat)
        poly_df["platform"] = "polymarket"
        combined = pd.concat([poly_df, df[available_cols].assign(platform="kalshi")],
                             ignore_index=True)
        combined_path = DATA_DIR / "features" / "market_features_combined.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"[✓] Combined features saved → {combined_path} ({len(combined)} rows)")

    print("\n── Kalshi Category Distribution ────────────────────────────────")
    print(df.groupby("category")["outcome_label"].agg(count="count", yes_rate="mean")
           .round(3).to_string())
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Kalshi settled markets")
    parser.add_argument("--days",      type=int,  default=90,    help="Days of history")
    parser.add_argument("--max",       type=int,  default=10000, help="Max markets")
    parser.add_argument("--no-prices", action="store_true",      help="Skip price history")
    args = parser.parse_args()

    df = scrape_kalshi(
        days=args.days,
        max_markets=args.max,
        fetch_prices=not args.no_prices,
    )
    print(f"\nFinal Kalshi dataset: {len(df)} markets.")
