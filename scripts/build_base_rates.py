"""
Base Rate Extractor for Prediction Markets
Computes per-category historical statistics from resolved market data.
These base rates are used by strategies for superforecasting-level
probability anchoring (Kahneman's "outside view").

Usage:
    python build_base_rates.py
    python build_base_rates.py --min-samples 20 --verbose
    python build_base_rates.py --source polymarket   # single platform
"""

import json
import argparse
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
KALSHI_PATH    = FEATURE_DIR / "kalshi_features.parquet"
OUTPUT_PATH    = DATA_DIR / "base_rates.json"

N_RELIABILITY_BUCKETS = 10


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
                     verbose: bool = False) -> dict:
    """
    Build the complete base rate database from resolved prediction market data.

    Args:
        min_samples: Minimum markets per category to include in output
        source: 'polymarket', 'kalshi', or None (all available)
        verbose: Print per-category diagnostics
    Returns:
        Full base rates dictionary (also written to data/base_rates.json)
    """
    print("=" * 65)
    print("BASE RATE EXTRACTOR -- Prediction Markets")
    print("=" * 65)

    # Load data
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
    args = parser.parse_args()

    result = build_base_rates(
        min_samples=args.min_samples,
        source=args.source,
        verbose=args.verbose,
    )

    print(f"\nFinal base rates: {len(result.get('categories', {}))} categories written to {OUTPUT_PATH}")
