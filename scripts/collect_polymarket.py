"""
Polymarket Historical Market Scraper
Pulls resolved markets from the Gamma API (no auth required)
Stores raw data as parquet + engineers features for backtesting

Usage:
    python collect_polymarket.py --days 90       # Last 90 days
    python collect_polymarket.py --days 365      # Full year
    python collect_polymarket.py --max 5000      # Cap at 5000 markets
"""

import math
import requests
import pandas as pd
import numpy as np
import json
import time
import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_URL  = "https://gamma-api.polymarket.com"
CLOB_URL  = "https://clob.polymarket.com"
DATA_DIR  = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "features").mkdir(exist_ok=True)

# Price history cache — avoids re-fetching on reruns
PRICE_CACHE_PATH = DATA_DIR / "price_cache.parquet"

CATEGORY_MAP = {
    "politics": ["election", "president", "senate", "congress", "vote", "governor", "mayor",
                 "democrat", "republican", "biden", "trump", "harris", "legislation"],
    "crypto":   ["bitcoin", "btc", "ethereum", "eth", "crypto", "defi", "nft", "blockchain",
                 "solana", "bnb", "xrp", "price", "market cap", "altcoin"],
    "science":  ["nasa", "space", "climate", "covid", "cancer", "fda", "vaccine", "study",
                 "research", "temperature", "earthquake", "hurricane"],
    "sports":   ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball",
                 "championship", "super bowl", "world cup", "playoffs", "finals"],
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "prediction-market-bot/1.0"})


# ─── Utility Functions ──────────────────────────────────────────────────────────

def infer_category(question: str) -> str:
    """Classify a market question into a category based on keywords."""
    q = question.lower()
    for cat, keywords in CATEGORY_MAP.items():
        if any(kw in q for kw in keywords):
            return cat
    return "other"


def safe_float(value, default=0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def resolve_outcome(market: dict) -> tuple:
    """Determine the winning outcome from a resolved market.

    Handles two Gamma API market schemas:
      - AMM markets: use tokens[].winner (bool) or winnerOutcome field
      - CLOB markets: use outcomePrices JSON string ("1" = winner, "0" = loser)

    Returns:
        (winning_outcome_str, outcome_label)
        winning_outcome_str: uppercased name of winning outcome, or None
        outcome_label: 1 (YES/first wins), 0 (NO/second wins), -1 (unknown)
    """
    # 1. AMM token-based resolution
    for tok in market.get("tokens", []):
        if tok.get("winner"):
            wo = str(tok.get("outcome", "")).upper()
            if wo in ("YES", "TRUE", "1"):
                return wo, 1
            elif wo in ("NO", "FALSE", "0"):
                return wo, 0
            return wo, -1

    # 2. winnerOutcome field (AMM markets)
    wo = str(market.get("winnerOutcome", "")).upper()
    if wo in ("YES", "TRUE", "1"):
        return wo, 1
    elif wo in ("NO", "FALSE", "0"):
        return wo, 0
    elif wo:
        return wo, -1

    # 3. outcomePrices (CLOB markets) — JSON string: '["1", "0"]'
    op_raw  = market.get("outcomePrices", "[]")
    out_raw = market.get("outcomes", "[]")
    try:
        op  = json.loads(op_raw)  if isinstance(op_raw,  str) else list(op_raw)
        out = json.loads(out_raw) if isinstance(out_raw, str) else list(out_raw)
        for i, p in enumerate(op):
            if float(p) >= 0.99 and i < len(out):
                wo = str(out[i]).upper()
                if wo in ("YES", "TRUE", "1"):
                    return wo, 1
                elif wo in ("NO", "FALSE", "0"):
                    return wo, 0
                # Non-binary label (Over/Under, Team A/B, etc.)
                # Position 0 = first/YES equivalent, position 1 = second/NO equivalent
                return wo, (1 if i == 0 else 0)
    except Exception:
        pass

    return None, -1


# ─── Gamma API Fetchers ─────────────────────────────────────────────────────────

def fetch_resolved_markets(limit: int = 100, offset: int = 0) -> list:
    """Fetch a page of resolved (closed) markets from the Gamma API.

    Note: API returns oldest-first by default. Use find_start_offset() to
    jump to the right position before calling this in a loop.
    """
    params = {
        "closed": "true",
        "limit": limit,
        "offset": offset,
    }
    resp = SESSION.get(f"{BASE_URL}/markets", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def find_start_offset(cutoff_date: datetime,
                      lo: int = 0, hi: int = 600_000) -> int:
    """Binary search for the first offset whose endDateIso >= cutoff_date.

    The Gamma API returns markets oldest-first. This finds the offset that
    lands just before our desired cutoff so we collect only recent markets.
    """
    print(f"  [binary search] Finding offset for cutoff {cutoff_date.date()} ...")
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            page = fetch_resolved_markets(limit=1, offset=mid)
        except Exception:
            # If request fails, back off
            hi = mid
            continue
        if not page:
            hi = mid
            continue
        end_dt = parse_date(page[0].get("endDateIso") or page[0].get("end_date_iso"))
        if end_dt is None or end_dt < cutoff_date:
            lo = mid + 1
        else:
            hi = mid
    # Step back a bit to avoid missing edge records
    return max(0, lo - 200)



def fetch_price_history(market_id: str, resolution: int = 3600) -> pd.DataFrame:
    """Fetch hourly CLOB price history for a specific market (legacy — kept for compat)."""
    params = {"market": market_id, "resolution": resolution, "period": "max"}
    try:
        resp = SESSION.get(f"{BASE_URL}/prices-history", params=params, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            if data and "history" in data:
                df = pd.DataFrame(data["history"])
                df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
                df = df.rename(columns={"t": "timestamp", "p": "price"})
                df["price"] = df["price"].astype(float)
                return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


# ─── CLOB Price History Helpers ─────────────────────────────────────────────────

_price_cache: dict = {}   # in-memory cache: (token_id, offset_days) -> float|None
_cache_dirty: bool = False


def _load_price_cache() -> None:
    """Load persisted price cache from parquet on first use."""
    global _price_cache
    if _price_cache or not PRICE_CACHE_PATH.exists():
        return
    try:
        df = pd.read_parquet(PRICE_CACHE_PATH)
        for _, row in df.iterrows():
            key = (str(row["token_id"]), int(row["offset_days"]))
            _price_cache[key] = row["price"] if pd.notna(row["price"]) else None
    except Exception:
        pass


def _flush_price_cache() -> None:
    """Persist the in-memory price cache to parquet."""
    global _cache_dirty
    if not _cache_dirty or not _price_cache:
        return
    rows = [
        {"token_id": k[0], "offset_days": k[1], "price": v}
        for k, v in _price_cache.items()
    ]
    pd.DataFrame(rows).to_parquet(PRICE_CACHE_PATH, index=False)
    _cache_dirty = False


def fetch_price_at_offset(token_id: str, end_date, offset_days: int) -> Optional[float]:
    """
    Fetch the YES-token price at exactly `offset_days` before resolution.

    Uses the Polymarket CLOB prices-history API:
        GET https://clob.polymarket.com/prices-history
        ?market=<token_id>&startTs=<ts>&endTs=<ts>&fidelity=<n>

    Caches results in data/price_cache.parquet to avoid repeat fetches.

    Args:
        token_id:    CLOB YES-token ID (the 77-digit number from clobTokenIds[0]).
        end_date:    Market resolution datetime (UTC-aware or naive string/datetime).
        offset_days: Days before resolution to target (1 = T-1d, 7 = T-7d).

    Returns:
        Price (0.0–1.0) closest to target timestamp, or None if no data.
    """
    global _cache_dirty

    _load_price_cache()
    cache_key = (str(token_id), int(offset_days))

    # Return cached result (including cached None so we don't re-fetch failures)
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    # Normalise end_date to UTC datetime
    if isinstance(end_date, str):
        end_dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
            try:
                end_dt = datetime.strptime(end_date, fmt)
                break
            except ValueError:
                continue
        if end_dt is None:
            _price_cache[cache_key] = None
            return None
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
    elif isinstance(end_date, datetime):
        end_dt = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
    else:
        _price_cache[cache_key] = None
        return None

    # Target timestamp and window around it
    target_dt  = end_dt - timedelta(days=offset_days)
    window_sec = 43200  # ±12 h window for hourly fidelity
    start_ts   = int((target_dt - timedelta(seconds=window_sec)).timestamp())
    end_ts     = int((target_dt + timedelta(seconds=window_sec)).timestamp())

    try:
        resp = SESSION.get(
            f"{CLOB_URL}/prices-history",
            params={
                "market":   token_id,
                "startTs":  start_ts,
                "endTs":    end_ts,
                "fidelity": 100,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            _price_cache[cache_key] = None
            _cache_dirty = True
            return None

        history = resp.json().get("history", [])
        if not history:
            _price_cache[cache_key] = None
            _cache_dirty = True
            return None

        # Find the point closest to target_ts
        target_ts = int(target_dt.timestamp())
        closest = min(history, key=lambda p: abs(p["t"] - target_ts))
        price = float(closest["p"])

        _price_cache[cache_key] = price
        _cache_dirty = True
        return price

    except Exception:
        _price_cache[cache_key] = None
        _cache_dirty = True
        return None


def fetch_price_window(token_id: str, end_date,
                       lookback_days: int = 8) -> pd.DataFrame:
    """
    Fetch a multi-day price window ending at `end_date` from the CLOB API.

    Used to compute price_volatility_7d (std over 7-day window).

    Returns:
        DataFrame with columns [timestamp (UTC datetime), price (float)].
        Empty DataFrame if unavailable.
    """
    if isinstance(end_date, str):
        end_dt = None
        for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
            try:
                end_dt = datetime.strptime(end_date, fmt)
                break
            except ValueError:
                continue
        if end_dt is None:
            return pd.DataFrame()
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
    elif isinstance(end_date, datetime):
        end_dt = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
    else:
        return pd.DataFrame()

    start_ts = int((end_dt - timedelta(days=lookback_days)).timestamp())
    end_ts   = int(end_dt.timestamp())

    try:
        resp = SESSION.get(
            f"{CLOB_URL}/prices-history",
            params={
                "market":   token_id,
                "startTs":  start_ts,
                "endTs":    end_ts,
                "fidelity": 200,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            return pd.DataFrame()

        history = resp.json().get("history", [])
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["price"]     = df["p"].astype(float)
        return df[["timestamp", "price"]].sort_values("timestamp").reset_index(drop=True)

    except Exception:
        return pd.DataFrame()


# ─── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(market: dict, price_history: pd.DataFrame) -> dict:
    """Compute all features for a single market snapshot.

    Price features (price_at_T1d, price_at_T7d, price_momentum_24h,
    price_volatility_7d) are populated either from the supplied price_history
    DataFrame (legacy path) or left as None/0.0 to be backfilled by
    build_base_rates.py using the CLOB API + clobTokenIds.
    """
    question = market.get("question", "")
    category = infer_category(question)

    start_dt = parse_date(market.get("startDateIso") or market.get("created_at"))
    end_dt   = parse_date(market.get("endDateIso")   or market.get("end_date_iso"))

    volume       = safe_float(market.get("volume"))
    open_int     = safe_float(market.get("openInterest", market.get("liquidity")))
    best_ask     = safe_float(market.get("bestAsk"))
    best_bid     = safe_float(market.get("bestBid"))

    # Spread
    spread_pct = (best_ask - best_bid) if (best_ask and best_bid) else 0.0

    # Log-normalized volume as a proxy for market size/liquidity.
    # The Gamma API returns openInterest=0 for all resolved markets, so
    # volume/max(open_int,1) degenerates to raw volume — not a ratio at all.
    # Instead, use log1p-normalized volume scaled to [0, ~1] assuming a
    # practical maximum of ~$10M: log1p(0)/log1p(10M) = 0,
    # log1p(5000)/log1p(10M) ≈ 0.53, log1p(100000)/log1p(10M) ≈ 0.70.
    _LOG_MAX_VOLUME = math.log1p(10_000_000)
    liquidity_ratio = math.log1p(volume) / _LOG_MAX_VOLUME

    # Time features
    resolution_hours = None
    days_since_open  = None
    if start_dt and end_dt:
        delta_total = end_dt - start_dt
        resolution_hours = delta_total.total_seconds() / 3600
        days_since_open  = delta_total.days

    # Price-derived features
    price_momentum_24h  = 0.0
    price_volatility_7d = 0.0
    price_at_T7d        = None
    price_at_T1d        = None

    if not price_history.empty and len(price_history) >= 2:
        ph = price_history.copy()
        if end_dt:
            ph = ph[ph["timestamp"] <= end_dt]
        if not ph.empty:
            last_price = ph["price"].iloc[-1]
            # 24h momentum
            cutoff_24h = ph["timestamp"].iloc[-1] - pd.Timedelta(hours=24)
            ph_24h = ph[ph["timestamp"] >= cutoff_24h]
            if len(ph_24h) >= 2:
                price_momentum_24h = last_price - ph_24h["price"].iloc[0]

            # 7d volatility
            cutoff_7d = ph["timestamp"].iloc[-1] - pd.Timedelta(days=7)
            ph_7d = ph[ph["timestamp"] >= cutoff_7d]
            if len(ph_7d) >= 2:
                price_volatility_7d = ph_7d["price"].std()
                price_at_T7d = ph_7d["price"].iloc[0]

            # 1d before resolution
            if end_dt:
                cutoff_1d = end_dt - pd.Timedelta(days=1)
                ph_1d = ph[ph["timestamp"] <= cutoff_1d]
                if not ph_1d.empty:
                    price_at_T1d = ph_1d["price"].iloc[-1]

    # Extract YES token ID from clobTokenIds (index 0 = YES by convention)
    clob_token_ids = market.get("clobTokenIds", [])
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except Exception:
            clob_token_ids = []
    yes_token_id = str(clob_token_ids[0]) if clob_token_ids else None

    # Volume anomaly score will be computed cross-market; set placeholder
    volume_anomaly_score = 0.0

    # Outcome label: 1 = YES/first, 0 = NO/second, -1 = INVALID/unknown
    winning_outcome, outcome_label = resolve_outcome(market)
    winning_outcome = winning_outcome or ""

    return {
        # Identifiers
        "market_id":             market.get("id") or market.get("market_id", ""),
        "question":              question,
        "category":              category,
        # Outcome
        "outcome_label":         outcome_label,
        "winning_outcome":       winning_outcome,
        # Market structure
        "volume":                volume,
        "open_interest":         open_int,
        "spread_pct":            spread_pct,
        "liquidity_ratio":       liquidity_ratio,
        # Time
        "start_date":            str(start_dt) if start_dt else None,
        "end_date":              str(end_dt) if end_dt else None,
        "time_to_resolution_hours": resolution_hours,
        "days_since_market_open": days_since_open,
        # Price dynamics (populated by CLOB API in build_base_rates backfill step)
        "price_momentum_24h":    price_momentum_24h,
        "price_volatility_7d":   price_volatility_7d,
        "price_at_T7d":          price_at_T7d,
        "price_at_T1d":          price_at_T1d,
        # CLOB token ID for price-history backfill (not in feature set, stored in raw parquet)
        "yes_token_id":          yes_token_id,
        # Volume anomaly (to fill in cross-market pass)
        "volume_anomaly_score":  volume_anomaly_score,
    }


def add_volume_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score volume vs. category average (cross-market pass)."""
    for cat in df["category"].unique():
        mask = df["category"] == cat
        cat_vols = df.loc[mask, "volume"]
        mean_vol, std_vol = cat_vols.mean(), cat_vols.std()
        if std_vol and std_vol > 0:
            df.loc[mask, "volume_anomaly_score"] = (cat_vols - mean_vol) / std_vol
    return df


# ─── Main Scraper ───────────────────────────────────────────────────────────────

def scrape_polymarket(days: int = 90, max_markets: int = 15000,
                      fetch_prices: bool = True) -> pd.DataFrame:
    """
    Scrape resolved Polymarket markets, engineer features, save to parquet.

    Returns:
        DataFrame with all engineered features.
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    print(f"[Polymarket] Fetching markets resolved after {cutoff_date.date()} ...")

    raw_markets = []
    page_size = 100

    # Binary-search to the first offset with data in our date window
    offset = find_start_offset(cutoff_date)
    print(f"  [binary search] Starting collection at offset {offset}")

    while len(raw_markets) < max_markets:
        try:
            page = fetch_resolved_markets(limit=page_size, offset=offset)
        except requests.RequestException as e:
            print(f"  [!] Request error at offset {offset}: {e}")
            break

        if not page:
            print(f"  [done] No more markets at offset {offset}. Total: {len(raw_markets)}")
            break

        # Filter: has winner + end_date within our window.
        # Note: some closed markets have future endDateIso (resolved early) —
        # we skip those for the date-window filter but do NOT stop pagination.
        filtered = []
        now = datetime.now(timezone.utc)
        for m in page:
            end_dt = parse_date(m.get("endDateIso") or m.get("end_date_iso"))
            # Skip markets before our cutoff (too old)
            if end_dt and end_dt < cutoff_date:
                continue
            # Skip markets with scheduled end date still in the future
            # (resolved early — endDateIso not useful as "resolution date")
            if end_dt and end_dt > now:
                continue
            _, outcome_label = resolve_outcome(m)
            if outcome_label != -1:
                filtered.append(m)

        raw_markets.extend(filtered)
        print(f"  Fetched {len(raw_markets)} markets so far (offset={offset}) ...")
        offset += page_size
        time.sleep(0.2)  # polite rate limit

    return _process_markets(raw_markets, fetch_prices)


def _process_markets(raw_markets: list, fetch_prices: bool) -> pd.DataFrame:
    """Engineer features for all raw markets and save."""
    if not raw_markets:
        print("[!] No markets collected.")
        return pd.DataFrame()

    print(f"\n[Polymarket] Engineering features for {len(raw_markets)} markets ...")

    records = []
    for i, market in enumerate(raw_markets):
        market_id = market.get("id") or market.get("market_id", "")
        price_history = pd.DataFrame()
        if fetch_prices and market_id:
            try:
                price_history = fetch_price_history(market_id)
                time.sleep(0.1)
            except Exception:
                pass

        try:
            features = engineer_features(market, price_history)
            records.append(features)
        except Exception as e:
            print(f"  [!] Feature error for market {market_id}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(raw_markets)} ...")

    df = pd.DataFrame(records)

    # Drop markets with unknown outcome
    before = len(df)
    df = df[df["outcome_label"].isin([0, 1])].reset_index(drop=True)
    print(f"  Dropped {before - len(df)} INVALID/unknown outcome markets.")

    # Cross-market volume anomaly scoring
    df = add_volume_anomaly_scores(df)

    # ── Save raw resolved parquet (includes yes_token_id for price backfill) ──
    raw_path = DATA_DIR / "polymarket_resolved.parquet"
    df.to_parquet(raw_path, index=False)
    print(f"\n[saved] Raw resolved markets -> {raw_path} ({len(df)} rows)")

    # ── Save features parquet ──────────────────────────────────────────────────
    feature_cols = [
        "market_id", "question", "category", "outcome_label",
        "time_to_resolution_hours", "days_since_market_open",
        "volume", "open_interest", "liquidity_ratio",
        "price_momentum_24h", "price_volatility_7d",
        "price_at_T7d", "price_at_T1d",
        "volume_anomaly_score", "spread_pct",
        "start_date", "end_date",
    ]
    feat_path = DATA_DIR / "features" / "market_features.parquet"
    df[feature_cols].to_parquet(feat_path, index=False)
    print(f"[saved] Engineered features -> {feat_path}")

    # Flush any price cache entries written during this session
    _flush_price_cache()

    # ── Category summary ───────────────────────────────────────────────────────
    print("\n-- Category Distribution ------------------------------------------")
    print(df.groupby("category")["outcome_label"].agg(
        count="count",
        yes_rate="mean"
    ).round(3).to_string())
    print()

    return df


# ─── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Polymarket resolved markets")
    parser.add_argument("--days",   type=int,  default=90,    help="Days of history to pull")
    parser.add_argument("--max",    type=int,  default=15000, help="Max markets to collect")
    parser.add_argument("--no-prices", action="store_true",   help="Skip price history fetching")
    args = parser.parse_args()

    df = scrape_polymarket(
        days=args.days,
        max_markets=args.max,
        fetch_prices=not args.no_prices,
    )
    print(f"\nFinal dataset: {len(df)} resolved markets with features.")
    print(df.head(3).to_string())
