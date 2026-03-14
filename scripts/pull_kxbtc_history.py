"""
pull_kxbtc_history.py
Pull all settled KXBTC markets from Kalshi + BTC hourly price history from
Coinbase, join them, and save to backtest/kxbtc_historical.parquet.

Strategy: KXBTC has ~200 bracket markets per hourly event, most with zero
volume.  We paginate until we pass the 60-day cutoff, keeping only markets
with volume_fp > 0 (actually traded).  This keeps the dataset useful for
backtesting without storing 80k+ zero-volume rows.
"""

import base64
import time
import re
import sys
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KALSHI_KEY_ID   = "1e0449b9-0757-4877-a8d6-7496dbd3a393"
KALSHI_KEY_PATH = r"C:\Users\casba\Trading agent\prediction-market-bot\private_key.pem"
# Host only — paths are appended separately and also used for signing
KALSHI_BASE     = "https://api.elections.kalshi.com"
BACKTEST_DIR    = Path(r"C:\Users\casba\Trading agent\prediction-market-bot\backtest")
OUT_PATH        = BACKTEST_DIR / "kxbtc_historical.parquet"

DAYS_BACK = 60  # how far back to pull


# ---------------------------------------------------------------------------
# Kalshi RSA auth
# ---------------------------------------------------------------------------
def build_auth_headers(method: str, path: str) -> dict:
    with open(KALSHI_KEY_PATH, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    ts_ms = str(int(time.time() * 1000))
    msg   = (ts_ms + method.upper() + path).encode("utf-8")
    sig   = key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json",
    }


# ---------------------------------------------------------------------------
# Task 1: Pull settled KXBTC markets within the last 60 days
# ---------------------------------------------------------------------------
def pull_kalshi_markets(days_back: int) -> list[dict]:
    """
    Paginate through settled KXBTC markets.
    Keeps only markets with volume_fp > 0.
    Stops pagination once all markets in a page pre-date the cutoff.
    """
    API_PATH = "/trade-api/v2/markets"
    cutoff   = datetime.now(timezone.utc) - timedelta(days=days_back)

    params: dict = {
        "series_ticker": "KXBTC",
        "status":        "settled",
        "limit":         200,
    }

    kept:   list[dict] = []
    cursor: str | None = None
    page        = 0
    total_seen  = 0
    stop_pages  = 0   # pages seen after passing cutoff (collect a buffer)

    while True:
        page += 1
        if cursor:
            params["cursor"] = cursor

        headers = build_auth_headers("GET", API_PATH)
        resp    = requests.get(
            KALSHI_BASE + API_PATH,
            headers=headers,
            params=params,
            timeout=30,
        )

        if resp.status_code != 200:
            print(f"[WARN] Kalshi {resp.status_code}: {resp.text[:300]}")
            break

        data    = resp.json()
        markets = data.get("markets", [])
        if not markets:
            print("  Empty page — exhausted all results.")
            break

        total_seen += len(markets)
        page_kept    = 0
        page_past    = 0

        for m in markets:
            # Parse open_time to check date range
            ot_str = m.get("open_time") or ""
            try:
                open_dt = datetime.fromisoformat(ot_str.replace("Z", "+00:00"))
            except Exception:
                continue

            if open_dt < cutoff:
                page_past += 1
                continue  # outside our window

            # Only keep markets with actual trading volume
            vol = float(m.get("volume_fp", 0) or 0)
            if vol <= 0:
                continue

            kept.append(m)
            page_kept += 1

        print(
            f"  Page {page:4d}: seen={total_seen:6d} | "
            f"page_kept={page_kept:4d} | page_past_cutoff={page_past:4d} | "
            f"total_kept={len(kept):5d}"
        )

        # Stop if entire page was before the cutoff
        if page_past == len(markets):
            stop_pages += 1
            if stop_pages >= 3:
                print("  3 consecutive all-past-cutoff pages — stopping.")
                break
        else:
            stop_pages = 0

        cursor = data.get("cursor")
        if not cursor:
            print("  No cursor returned — all pages exhausted.")
            break

        time.sleep(0.15)  # polite rate limiting

    return kept


# ---------------------------------------------------------------------------
# Process a raw Kalshi market dict into our output row
# ---------------------------------------------------------------------------
def process_market(m: dict) -> dict | None:
    ticker = m.get("ticker", "")

    # Result: API returns lowercase "yes" / "no"
    raw_result = (m.get("result") or "").strip().upper()
    if raw_result == "YES":
        result = "YES"
    elif raw_result == "NO":
        result = "NO"
    else:
        return None  # unsettled or unknown

    # Range low: prefer floor_strike, fall back to parsing the ticker suffix
    floor_strike = m.get("floor_strike")
    if floor_strike is not None:
        range_low = float(floor_strike)
    else:
        # e.g. KXBTC-26MAR1323-B74625 -> 74625; KXBTC-...-T78999.99 -> 78999.99
        suffix_match = re.search(r"[-_][BT]([\d.]+)$", ticker)
        if suffix_match:
            range_low = float(suffix_match.group(1))
        else:
            nums = re.findall(r"[\d.]+", ticker)
            range_low = float(nums[-1]) if nums else 0.0

    range_high = range_low + 250.0

    # Timestamps
    def parse_dt(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    open_time  = parse_dt(m.get("open_time"))
    close_time = parse_dt(
        m.get("close_time")
        or m.get("settlement_ts")
        or m.get("expected_expiration_time")
        or m.get("expiration_time")
    )
    if open_time is None or close_time is None:
        return None

    # Financial fields — API returns dollar strings (0-1 scale) or "fp" strings
    yes_ask_at_open = float(m.get("yes_ask_dollars", 0) or 0)
    volume          = float(m.get("volume_fp", 0) or 0)
    open_interest   = float(m.get("open_interest_fp", 0) or 0)

    return {
        "ticker":          ticker,
        "range_low":       range_low,
        "range_high":      range_high,
        "open_time":       open_time,
        "close_time":      close_time,
        "result":          result,
        "yes_ask_at_open": yes_ask_at_open,
        "volume":          volume,
        "open_interest":   open_interest,
    }


# ---------------------------------------------------------------------------
# Task 2: Pull BTC hourly candles from Coinbase
# ---------------------------------------------------------------------------
def pull_coinbase_btc(days_back: int = 62) -> dict[int, float]:
    """
    Returns a dict mapping unix_hour_timestamp -> open_price.
    Coinbase limits 300 candles per call (~12.5 days at hourly).
    We page backwards in 250-hour chunks to stay under the limit.
    """
    url       = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    gran      = 3600  # 1 hour in seconds
    end_dt    = datetime.now(timezone.utc)
    start_dt  = end_dt - timedelta(days=days_back)
    price_map: dict[int, float] = {}
    chunk_end = end_dt

    print(f"\nPulling Coinbase BTC/USD hourly candles ...")
    print(f"  Range: {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')}")

    while chunk_end > start_dt:
        chunk_start = max(start_dt, chunk_end - timedelta(hours=250))
        params = {
            "granularity": gran,
            "start":       chunk_start.isoformat(),
            "end":         chunk_end.isoformat(),
        }
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 200:
            candles = resp.json()
            if candles:
                for c in candles:
                    # [timestamp, low, high, open, close, volume]
                    ts, _low, _high, open_p, _close, _vol = c
                    price_map[int(ts)] = float(open_p)
                print(
                    f"  {chunk_start.strftime('%Y-%m-%d')} -> "
                    f"{chunk_end.strftime('%Y-%m-%d')}: {len(candles)} candles"
                )
        else:
            print(f"  [WARN] Coinbase {resp.status_code}: {resp.text[:150]}")

        chunk_end = chunk_start
        time.sleep(0.25)

    print(f"  Total BTC hourly timestamps cached: {len(price_map)}")
    return price_map


# ---------------------------------------------------------------------------
# Helper: find nearest BTC price for a datetime
# ---------------------------------------------------------------------------
def nearest_btc_price(dt: datetime, price_map: dict[int, float]) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts_hour = (int(dt.timestamp()) // 3600) * 3600
    for delta in [0, 3600, -3600, 7200, -7200, 10800, -10800]:
        candidate = ts_hour + delta
        if candidate in price_map:
            return price_map[candidate]
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Task 1: Pull Kalshi markets ----
    print("=" * 60)
    print(f"Task 1: Pulling settled KXBTC markets (last {DAYS_BACK} days)")
    print("=" * 60)
    raw_markets = pull_kalshi_markets(DAYS_BACK)
    print(f"\nRaw kept markets (with volume, in window): {len(raw_markets)}")

    rows = []
    skipped = 0
    for m in raw_markets:
        row = process_market(m)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

    print(f"Processed: {len(rows)} valid rows | {skipped} skipped")

    if not rows:
        print("[ERROR] No valid markets. Exiting.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # ---- Task 2: Pull Coinbase BTC prices ----
    print("\n" + "=" * 60)
    print("Task 2: Pulling BTC hourly prices from Coinbase")
    print("=" * 60)
    price_map = pull_coinbase_btc(days_back=DAYS_BACK + 3)

    # ---- Task 3: Join ----
    print("\n" + "=" * 60)
    print("Task 3: Joining BTC prices to markets")
    print("=" * 60)

    df["btc_at_open"]      = df["open_time"].apply(lambda dt: nearest_btc_price(dt, price_map))
    df["btc_at_close"]     = df["close_time"].apply(lambda dt: nearest_btc_price(dt, price_map))
    df["minutes_duration"] = (df["close_time"] - df["open_time"]).dt.total_seconds() / 60.0

    # Enforce output column order
    final_cols = [
        "ticker", "range_low", "range_high",
        "open_time", "close_time",
        "btc_at_open", "btc_at_close",
        "minutes_duration",
        "result",
        "yes_ask_at_open",
        "volume", "open_interest",
    ]
    df = df[final_cols]
    df = df.sort_values("open_time").reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total markets saved  : {len(df)}")
    if not df.empty:
        print(f"Date range           : {df['open_time'].min().date()} -> {df['close_time'].max().date()}")
        yes_rate = (df["result"] == "YES").mean()
        print(f"YES rate             : {yes_rate:.1%}")
        print(f"Avg volume (fp)      : {df['volume'].mean():.2f}")
        print(f"Total volume (fp)    : {df['volume'].sum():.2f}")
        btc_nonzero = df.loc[df["btc_at_open"] > 0, "btc_at_open"]
        if not btc_nonzero.empty:
            print(f"BTC open price range : ${btc_nonzero.min():,.0f} - ${btc_nonzero.max():,.0f}")
        btc_missing = (df["btc_at_open"] == 0).sum()
        if btc_missing:
            print(f"[WARN] {btc_missing} rows missing BTC price (no nearby candle found)")
        print(f"\nSample rows (first 10):")
        pd.set_option("display.width", 140)
        pd.set_option("display.max_columns", 8)
        print(
            df[["ticker", "range_low", "open_time", "btc_at_open", "result", "volume"]]
            .head(10)
            .to_string()
        )


if __name__ == "__main__":
    main()
