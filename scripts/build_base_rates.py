"""
Base Rate Extractor for Prediction Markets
Computes per-category historical statistics from resolved market data.
These base rates are used by strategies for superforecasting-level
probability anchoring (Kahneman's "outside view").

Also backfills CLOB price features (price_at_T1d, price_at_T7d,
price_momentum_24h, price_volatility_7d) from the Polymarket CLOB API
using yes_token_id stored in polymarket_resolved.parquet.

Usage:
    python build_base_rates.py
    python build_base_rates.py --min-samples 20 --verbose
    python build_base_rates.py --source polymarket   # single platform
    python build_base_rates.py --skip-price-backfill # skip CLOB fetching
"""

import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from scipy.stats import pearsonr

# --- Config ------------------------------------------------------------------

DATA_DIR    = Path(__file__).parent.parent / "data"
FEATURE_DIR = DATA_DIR / "features"

COMBINED_PATH  = FEATURE_DIR / "market_features_combined.parquet"
POLY_PATH      = FEATURE_DIR / "market_features.parquet"
RAW_POLY_PATH  = DATA_DIR / "polymarket_resolved.parquet"
KALSHI_PATH    = FEATURE_DIR / "kalshi_features.parquet"
OUTPUT_PATH    = DATA_DIR / "base_rates.json"

N_RELIABILITY_BUCKETS = 10


# --- CLOB Price Backfill -----------------------------------------------------

def backfill_price_features(verbose: bool = False) -> bool:
    """
    Populate price_at_T1d, price_at_T7d, price_momentum_24h, price_volatility_7d
    in the features parquet using the Polymarket CLOB prices-history API.

    Strategy:
    1. Read yes_token_id from polymarket_resolved.parquet (stored by collect_polymarket.py).
       If that column is absent, fall back to looking up clobTokenIds from the Gamma API.
    2. For each market whose price features are still null/zero, call the CLOB API to
       fetch price windows at T-1d and T-7d.
    3. Re-save both polymarket_resolved.parquet and market_features.parquet with filled values.

    Returns True if any prices were successfully fetched, False otherwise.
    """
    # Import the helpers from collect_polymarket.py (sibling script)
    import sys, time, importlib
    scripts_dir = Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        import collect_polymarket as cp
    except ImportError as e:
        print(f"[!] Cannot import collect_polymarket: {e}")
        return False

    # ── Load raw parquet (has yes_token_id column if collected after this patch) ─
    if not RAW_POLY_PATH.exists():
        print(f"[!] Raw parquet not found: {RAW_POLY_PATH}  — run collect_polymarket.py first.")
        return False

    raw = pd.read_parquet(RAW_POLY_PATH)

    # ── Determine which markets need price backfill ───────────────────────────
    # A market needs backfill if price_at_T1d is still null
    needs_backfill_mask = raw["price_at_T1d"].isna()
    needs_backfill = raw[needs_backfill_mask].copy()
    total_needed = len(needs_backfill)

    if total_needed == 0:
        print("[price backfill] All markets already have price_at_T1d. Skipping.")
        return True

    print(f"[price backfill] {total_needed} markets need price feature population ...")

    # ── Ensure yes_token_id column exists ─────────────────────────────────────
    if "yes_token_id" not in raw.columns:
        print("[price backfill] yes_token_id column missing from raw parquet.")
        print("                 Attempting live Gamma API lookup for token IDs ...")
        raw["yes_token_id"] = None

    # ── Fetch token IDs for any rows missing them (old-format parquet) ────────
    missing_token_mask = raw["yes_token_id"].isna() | (raw["yes_token_id"] == "")
    if missing_token_mask.any():
        n_missing = missing_token_mask.sum()
        print(f"[price backfill] Fetching clobTokenIds for {n_missing} markets via Gamma API ...")
        import requests as _req, json as _json
        _session = _req.Session()
        _session.headers.update({"User-Agent": "prediction-market-bot/1.0"})

        for idx in raw.index[missing_token_mask]:
            mid = raw.at[idx, "market_id"]
            try:
                r = _session.get(
                    f"https://gamma-api.polymarket.com/markets/{mid}",
                    timeout=15,
                )
                if r.status_code == 200:
                    m = r.json()
                    ctids = m.get("clobTokenIds", [])
                    if isinstance(ctids, str):
                        ctids = _json.loads(ctids)
                    if ctids:
                        raw.at[idx, "yes_token_id"] = str(ctids[0])
            except Exception as exc:
                if verbose:
                    print(f"    [warn] market {mid}: Gamma lookup failed: {exc}")
            time.sleep(0.4)   # polite rate limit

        # Save updated token IDs back to raw parquet
        raw.to_parquet(RAW_POLY_PATH, index=False)
        print(f"[price backfill] Token IDs saved to {RAW_POLY_PATH}")

    # Reload needs_backfill with updated token IDs
    needs_backfill = raw[raw["price_at_T1d"].isna()].copy()

    # ── Pre-load the price cache ───────────────────────────────────────────────
    cp._load_price_cache()

    filled_count = 0
    total = len(needs_backfill)

    for i, (idx, row) in enumerate(needs_backfill.iterrows()):
        token_id = row.get("yes_token_id")
        end_date = row.get("end_date")

        if not token_id or not end_date or str(token_id) in ("None", "nan", ""):
            if verbose:
                print(f"  [skip] market {row.get('market_id')}: no token_id")
            continue

        token_id = str(token_id)

        # ── T-1d price ─────────────────────────────────────────────────────
        p1d = cp.fetch_price_at_offset(token_id, end_date, offset_days=1)
        time.sleep(0.5)

        # ── 7-day price window (for T-7d price AND volatility) ─────────────
        ph7 = cp.fetch_price_window(token_id, end_date, lookback_days=8)
        time.sleep(0.5)

        p7d        = None
        vol_7d     = 0.0
        momentum   = 0.0

        if not ph7.empty and len(ph7) >= 2:
            from datetime import datetime, timedelta, timezone as _tz
            end_dt = None
            if isinstance(end_date, str):
                for fmt in ("%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                            "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%d"):
                    try:
                        end_dt = datetime.strptime(end_date, fmt)
                        break
                    except ValueError:
                        continue
                if end_dt and end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=_tz.utc)
            elif isinstance(end_date, datetime):
                end_dt = end_date

            if end_dt:
                target_7d = end_dt - timedelta(days=7)
                # Pick point closest to T-7d
                ph7["dt_diff"] = (ph7["timestamp"] - target_7d).abs()
                p7d = float(ph7.loc[ph7["dt_diff"].idxmin(), "price"])

            # Volatility = std of all prices in the 7-day window
            ph_7d_window = ph7[ph7["timestamp"] >= (ph7["timestamp"].max() - pd.Timedelta(days=7))]
            if len(ph_7d_window) >= 2:
                vol_7d = float(ph_7d_window["price"].std())

            # Momentum = T-1d price minus T-7d price (approximate)
            if p1d is not None and p7d is not None:
                momentum = p1d - p7d

        # ── Write back into raw DataFrame ─────────────────────────────────
        if p1d is not None or p7d is not None:
            raw.at[idx, "price_at_T1d"]        = p1d
            raw.at[idx, "price_at_T7d"]         = p7d
            raw.at[idx, "price_momentum_24h"]   = momentum
            raw.at[idx, "price_volatility_7d"]  = vol_7d
            filled_count += 1

        if verbose or (i + 1) % 50 == 0:
            pct = (i + 1) / total * 100
            print(f"  [{i+1:4d}/{total}] {pct:5.1f}%  market={row.get('market_id')}  "
                  f"T1d={p1d}  T7d={p7d}  filled_so_far={filled_count}")

    print(f"\n[price backfill] Filled {filled_count}/{total} markets with price features.")

    # ── Flush CLOB price cache to disk ────────────────────────────────────────
    cp._flush_price_cache()

    # ── Save updated raw parquet ──────────────────────────────────────────────
    raw.to_parquet(RAW_POLY_PATH, index=False)

    # ── Re-save features parquet with updated price columns ───────────────────
    if POLY_PATH.exists():
        feat = pd.read_parquet(POLY_PATH)
        # Merge updated price columns from raw into features (join on market_id)
        price_cols = ["market_id", "price_at_T1d", "price_at_T7d",
                      "price_momentum_24h", "price_volatility_7d"]
        updates = raw[price_cols].drop_duplicates("market_id")
        # Drop old price cols from feat and merge in updated ones
        for col in ["price_at_T1d", "price_at_T7d",
                    "price_momentum_24h", "price_volatility_7d"]:
            if col in feat.columns:
                feat = feat.drop(columns=[col])
        feat = feat.merge(updates, on="market_id", how="left")
        feat.to_parquet(POLY_PATH, index=False)
        print(f"[price backfill] Updated features parquet -> {POLY_PATH}")

    return filled_count > 0


# --- Data Loading -------------------------------------------------------------

def load_features(source: Optional[str] = None) -> pd.DataFrame:
    """
    Load engineered features, preferring the combined cross-platform parquet.
    Falls back gracefully if combined doesn't exist.

    Args:
        source: 'polymarket', 'kalshi', or None (load all available)
    Returns:
        DataFrame with outcome_label column filtered to [0, 1]
    """
    if source == "polymarket":
        if not POLY_PATH.exists():
            raise FileNotFoundError(f"Polymarket features not found: {POLY_PATH}\n"
                                    "Run: python collect_polymarket.py --days 365")
        df = pd.read_parquet(POLY_PATH)
        df["platform"] = "polymarket"
        print(f"[load] Polymarket features: {len(df)} rows -> {POLY_PATH.name}")
        return df[df["outcome_label"].isin([0, 1])].reset_index(drop=True)

    if source == "kalshi":
        if not KALSHI_PATH.exists():
            raise FileNotFoundError(f"Kalshi features not found: {KALSHI_PATH}\n"
                                    "Run: python collect_kalshi.py --days 365")
        df = pd.read_parquet(KALSHI_PATH)
        df["platform"] = "kalshi"
        print(f"[load] Kalshi features: {len(df)} rows -> {KALSHI_PATH.name}")
        return df[df["outcome_label"].isin([0, 1])].reset_index(drop=True)

    # Combined (preferred)
    if COMBINED_PATH.exists():
        df = pd.read_parquet(COMBINED_PATH)
        print(f"[load] Combined features: {len(df)} rows -> {COMBINED_PATH.name}")
        return df[df["outcome_label"].isin([0, 1])].reset_index(drop=True)

    # Merge available individual files
    frames = []
    if POLY_PATH.exists():
        pf = pd.read_parquet(POLY_PATH)
        pf["platform"] = "polymarket"
        frames.append(pf)
        print(f"[load] Polymarket features: {len(pf)} rows")
    if KALSHI_PATH.exists():
        kf = pd.read_parquet(KALSHI_PATH)
        kf["platform"] = "kalshi"
        frames.append(kf)
        print(f"[load] Kalshi features: {len(kf)} rows")

    if not frames:
        raise FileNotFoundError(
            "No feature files found. Run collect_polymarket.py and/or collect_kalshi.py first.\n"
            f"Expected at: {COMBINED_PATH}\n"
            f"         or: {POLY_PATH}\n"
            f"         or: {KALSHI_PATH}"
        )

    df = pd.concat(frames, ignore_index=True)
    print(f"[load] Merged {len(frames)} sources -> {len(df)} total rows")
    return df[df["outcome_label"].isin([0, 1])].reset_index(drop=True)


# --- Reliability Diagram -----------------------------------------------------

def compute_reliability_diagram(probs: pd.Series, outcomes: pd.Series,
                                 n_buckets: int = N_RELIABILITY_BUCKETS) -> list:
    """
    Build a reliability diagram (calibration curve) over n_buckets.
    Each bucket: mean predicted probability vs. actual win rate.

    Returns:
        List of dicts: [{bucket_min, bucket_max, mean_pred_prob,
                         actual_win_rate, count, calibration_error}, ...]
    """
    buckets = []
    edges = np.linspace(0.0, 1.0, n_buckets + 1)

    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_buckets - 1:
            mask = (probs >= lo) & (probs <= hi)  # inclusive last bucket

        count = mask.sum()
        if count == 0:
            continue

        mean_pred = float(probs[mask].mean())
        actual_wr = float(outcomes[mask].mean())
        buckets.append({
            "bucket_min":       round(lo, 2),
            "bucket_max":       round(hi, 2),
            "mean_pred_prob":   round(mean_pred, 4),
            "actual_win_rate":  round(actual_wr, 4),
            "count":            int(count),
            "calibration_error": round(abs(mean_pred - actual_wr), 4),
        })

    return buckets


# --- Per-Category Stats -------------------------------------------------------

def compute_category_stats(cat_df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Compute all base-rate statistics for a single category slice.

    Args:
        cat_df: DataFrame filtered to one category, outcome_label in [0, 1]
        verbose: Print extra diagnostics
    Returns:
        Dict of statistics
    """
    n = len(cat_df)
    outcomes = cat_df["outcome_label"]

    stats: dict = {
        "sample_count": int(n),
        "base_yes_rate": round(float(outcomes.mean()), 4),
        "base_no_rate":  round(float(1.0 - outcomes.mean()), 4),
    }

    # -- Resolution time ------------------------------------------------------
    if "time_to_resolution_hours" in cat_df.columns:
        rt = cat_df["time_to_resolution_hours"].dropna()
        if len(rt) > 0:
            stats["median_time_to_resolution_hours"] = round(float(rt.median()), 1)
            stats["p25_time_to_resolution_hours"]    = round(float(rt.quantile(0.25)), 1)
            stats["p75_time_to_resolution_hours"]    = round(float(rt.quantile(0.75)), 1)
            stats["p90_time_to_resolution_hours"]    = round(float(rt.quantile(0.90)), 1)
        else:
            stats["median_time_to_resolution_hours"] = None

    # -- Volume distribution ---------------------------------------------------
    if "volume" in cat_df.columns:
        vol = cat_df["volume"].dropna()
        if len(vol) > 0:
            stats["volume_mean"]   = round(float(vol.mean()), 2)
            stats["volume_std"]    = round(float(vol.std()), 2)
            stats["volume_median"] = round(float(vol.median()), 2)
            stats["volume_p90"]    = round(float(vol.quantile(0.90)), 2)

    # -- Spread and liquidity --------------------------------------------------
    if "spread_pct" in cat_df.columns:
        sp = cat_df["spread_pct"].dropna()
        if len(sp) > 0:
            stats["spread_mean"]   = round(float(sp.mean()), 4)
            stats["spread_median"] = round(float(sp.median()), 4)

    if "liquidity_ratio" in cat_df.columns:
        lr = cat_df["liquidity_ratio"].dropna()
        if len(lr) > 0:
            stats["liquidity_ratio_mean"]   = round(float(lr.mean()), 4)
            stats["liquidity_ratio_median"] = round(float(lr.median()), 4)

    # -- Price predictiveness at T-7d -----------------------------------------
    if "price_at_T7d" in cat_df.columns:
        valid = cat_df[["price_at_T7d", "outcome_label"]].dropna()
        if len(valid) >= 10:
            try:
                r, pval = pearsonr(valid["price_at_T7d"], valid["outcome_label"])
                brier_7d = float(((valid["price_at_T7d"] - valid["outcome_label"]) ** 2).mean())
                stats["price_accuracy_at_T7d"] = {
                    "pearson_r":     round(float(r), 4),
                    "p_value":       round(float(pval), 4),
                    "brier_score":   round(brier_7d, 4),
                    "sample_count":  int(len(valid)),
                }
                stats["reliability_diagram_T7d"] = compute_reliability_diagram(
                    valid["price_at_T7d"], valid["outcome_label"]
                )
            except Exception as e:
                if verbose:
                    print(f"    [warn] T7d Pearson failed: {e}")

    # -- Price predictiveness at T-1d -----------------------------------------
    if "price_at_T1d" in cat_df.columns:
        valid = cat_df[["price_at_T1d", "outcome_label"]].dropna()
        if len(valid) >= 10:
            try:
                r, pval = pearsonr(valid["price_at_T1d"], valid["outcome_label"])
                brier_1d = float(((valid["price_at_T1d"] - valid["outcome_label"]) ** 2).mean())
                stats["price_accuracy_at_T1d"] = {
                    "pearson_r":     round(float(r), 4),
                    "p_value":       round(float(pval), 4),
                    "brier_score":   round(brier_1d, 4),
                    "sample_count":  int(len(valid)),
                }
                stats["reliability_diagram_T1d"] = compute_reliability_diagram(
                    valid["price_at_T1d"], valid["outcome_label"]
                )
            except Exception as e:
                if verbose:
                    print(f"    [warn] T1d Pearson failed: {e}")

    # -- Momentum predictiveness -----------------------------------------------
    if "price_momentum_24h" in cat_df.columns:
        valid = cat_df[["price_momentum_24h", "outcome_label"]].dropna()
        valid = valid[valid["price_momentum_24h"] != 0.0]  # drop uninformative zeros
        if len(valid) >= 10:
            try:
                r, pval = pearsonr(valid["price_momentum_24h"], valid["outcome_label"])
                stats["momentum_predictiveness"] = {
                    "pearson_r":    round(float(r), 4),
                    "p_value":      round(float(pval), 4),
                    "sample_count": int(len(valid)),
                }
            except Exception as e:
                if verbose:
                    print(f"    [warn] Momentum Pearson failed: {e}")

    # -- Category calibration (Brier score using best available predictor) ----
    # Use T1d if available, else T7d, else momentum
    category_brier = None
    for col in ("price_at_T1d", "price_at_T7d"):
        if col in cat_df.columns:
            valid = cat_df[[col, "outcome_label"]].dropna()
            if len(valid) >= 10:
                category_brier = round(
                    float(((valid[col] - valid["outcome_label"]) ** 2).mean()), 4
                )
                break

    stats["category_calibration"] = {
        "brier_score":   category_brier,
        "brier_baseline": 0.25,           # Random forecast baseline for binary
        "brier_skill":   round(1.0 - (category_brier / 0.25), 4)
                         if category_brier is not None else None,
        # Positive skill = better than random; 1.0 = perfect; <0 = worse than random
    }

    return stats


# --- Global Stats -------------------------------------------------------------

def compute_global_stats(df: pd.DataFrame) -> dict:
    """
    Compute global statistics across all categories and platforms.
    """
    outcomes = df["outcome_label"]
    global_stats: dict = {
        "total_markets": int(len(df)),
        "overall_yes_rate": round(float(outcomes.mean()), 4),
        "overall_no_rate":  round(float(1.0 - outcomes.mean()), 4),
    }

    # Category distribution
    if "category" in df.columns:
        cat_counts = df["category"].value_counts()
        global_stats["category_distribution"] = {
            cat: {"count": int(cnt), "pct": round(cnt / len(df), 4)}
            for cat, cnt in cat_counts.items()
        }
        global_stats["category_yes_rates"] = {
            cat: round(float(df[df["category"] == cat]["outcome_label"].mean()), 4)
            for cat in cat_counts.index
        }

    # Platform distribution
    if "platform" in df.columns:
        plat_counts = df["platform"].value_counts()
        global_stats["platform_distribution"] = {
            plat: int(cnt) for plat, cnt in plat_counts.items()
        }

    # Resolution time global
    if "time_to_resolution_hours" in df.columns:
        rt = df["time_to_resolution_hours"].dropna()
        if len(rt) > 0:
            global_stats["resolution_time_hours"] = {
                "p25":    round(float(rt.quantile(0.25)), 1),
                "median": round(float(rt.median()), 1),
                "p75":    round(float(rt.quantile(0.75)), 1),
                "p90":    round(float(rt.quantile(0.90)), 1),
                "mean":   round(float(rt.mean()), 1),
            }

    # Global price predictiveness (model quality ceiling)
    for col, key in [("price_at_T1d", "global_brier_T1d"),
                     ("price_at_T7d", "global_brier_T7d")]:
        if col in df.columns:
            valid = df[[col, "outcome_label"]].dropna()
            if len(valid) >= 10:
                global_stats[key] = round(
                    float(((valid[col] - valid["outcome_label"]) ** 2).mean()), 4
                )

    return global_stats


# --- Main ---------------------------------------------------------------------

def build_base_rates(min_samples: int = 20, source: Optional[str] = None,
                     verbose: bool = False,
                     skip_price_backfill: bool = False) -> dict:
    """
    Build the complete base rate database from resolved prediction market data.

    Args:
        min_samples: Minimum markets per category to include in output
        source: 'polymarket', 'kalshi', or None (all available)
        verbose: Print per-category diagnostics
        skip_price_backfill: If True, skip the CLOB API price feature population step
    Returns:
        Full base rates dictionary (also written to data/base_rates.json)
    """
    print("=" * 65)
    print("BASE RATE EXTRACTOR -- Prediction Markets")
    print("=" * 65)

    # Backfill CLOB price features (price_at_T1d, T7d, momentum, volatility)
    # Must run before load_features() so the features parquet is up to date.
    if not skip_price_backfill and (source is None or source == "polymarket"):
        print("\n-- CLOB price feature backfill ----------------------------------")
        if RAW_POLY_PATH.exists():
            backfill_price_features(verbose=verbose)
        else:
            print(f"  [skip] Raw parquet not found: {RAW_POLY_PATH}")
            print("         Run collect_polymarket.py first to enable price backfill.")
    elif skip_price_backfill:
        print("\n  [--skip-price-backfill] Skipping CLOB price feature population.")

    # Load data (reload after backfill so updated prices are included)
    df = load_features(source)
    print(f"\n[[ok]] Loaded {len(df)} markets with valid outcomes (0/1)\n")

    if len(df) == 0:
        print("[!] No data to process.")
        return {}

    # Global stats
    print("-- Computing global statistics ------------------------------")
    global_stats = compute_global_stats(df)
    if verbose:
        print(f"  Total markets:   {global_stats['total_markets']}")
        print(f"  Overall YES rate: {global_stats['overall_yes_rate']:.1%}")
        if "category_distribution" in global_stats:
            for cat, info in global_stats["category_distribution"].items():
                print(f"  {cat:12s}: {info['count']:5d} markets "
                      f"({info['pct']:.1%}) | YES rate: "
                      f"{global_stats['category_yes_rates'].get(cat, 0):.1%}")

    # Per-category stats
    print("\n-- Computing per-category base rates ------------------------")
    categories_data: dict = {}

    if "category" in df.columns:
        for cat in sorted(df["category"].unique()):
            cat_df = df[df["category"] == cat].reset_index(drop=True)
            n = len(cat_df)

            if n < min_samples:
                print(f"  [skip] {cat}: only {n} samples (min={min_samples})")
                continue

            print(f"  {cat:12s}: {n:5d} markets ... ", end="", flush=True)
            stats = compute_category_stats(cat_df, verbose=verbose)
            categories_data[cat] = stats

            brier = stats.get("category_calibration", {}).get("brier_score")
            brier_str = f"Brier={brier:.4f}" if brier else "Brier=N/A"
            print(f"YES={stats['base_yes_rate']:.1%}  {brier_str}")

    # Assemble output
    base_rates: dict = {
        "metadata": {
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "total_markets":      global_stats["total_markets"],
            "source_filter":      source or "all",
            "min_samples":        min_samples,
            "categories_included": list(categories_data.keys()),
            "n_categories":       len(categories_data),
            "reliability_buckets": N_RELIABILITY_BUCKETS,
            "notes": (
                "base_yes_rate: historical fraction of markets that resolved YES. "
                "Use as anchor for probability estimation (outside view). "
                "brier_score: 0.0=perfect, 0.25=random baseline, >0.25=worse than random. "
                "reliability_diagram: predicted probability vs actual win rate by bucket."
            ),
        },
        "global": global_stats,
        "categories": categories_data,
    }

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(base_rates, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[[ok]] Saved base rates -> {OUTPUT_PATH}")
    print(f"    Categories: {len(categories_data)}")
    print(f"    Total markets: {global_stats['total_markets']}")

    # Quick calibration summary
    print("\n-- Calibration Quality Summary ------------------------------")
    print(f"  {'Category':12s}  {'YES Rate':>9}  {'Sample N':>8}  "
          f"{'Brier T-1d':>10}  {'Skill':>8}")
    print("  " + "-" * 57)
    for cat, stats in sorted(categories_data.items()):
        cal = stats.get("category_calibration", {})
        brier = cal.get("brier_score")
        skill = cal.get("brier_skill")
        print(f"  {cat:12s}  "
              f"{stats['base_yes_rate']:>9.1%}  "
              f"{stats['sample_count']:>8d}  "
              f"{'N/A':>10}" if brier is None else
              f"  {cat:12s}  "
              f"{stats['base_yes_rate']:>9.1%}  "
              f"{stats['sample_count']:>8d}  "
              f"{brier:>10.4f}  "
              f"{skill:>+8.4f}" if skill is not None else
              f"  {cat:12s}  "
              f"{stats['base_yes_rate']:>9.1%}  "
              f"{stats['sample_count']:>8d}  "
              f"{brier:>10.4f}  "
              f"{'N/A':>8}")

    return base_rates


# --- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build prediction market base rate database from resolved market data"
    )
    parser.add_argument(
        "--min-samples", type=int, default=20,
        help="Minimum markets per category to include (default: 20)"
    )
    parser.add_argument(
        "--source", choices=["polymarket", "kalshi"], default=None,
        help="Restrict to one platform (default: use all available)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print extra diagnostics during computation"
    )
    parser.add_argument(
        "--skip-price-backfill", action="store_true",
        help="Skip CLOB API price feature population (use existing values in parquet)"
    )
    args = parser.parse_args()

    result = build_base_rates(
        min_samples=args.min_samples,
        source=args.source,
        verbose=args.verbose,
        skip_price_backfill=args.skip_price_backfill,
    )

    print(f"\nFinal base rates: {len(result.get('categories', {}))} categories written to {OUTPUT_PATH}")
