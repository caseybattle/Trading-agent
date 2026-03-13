"""
Calibration Tracker — Brier Score Tracking + Isotonic Regression Bias Correction

Tracks per-strategy prediction accuracy over time and applies isotonic regression
to correct systematic probability biases (overconfidence, underconfidence).

Usage:
    tracker = CalibrationTracker.load()
    tracker.record(signal, outcome=1)          # Log resolved prediction
    tracker.recalibrate()                      # Fit isotonic regression
    cal_prob = tracker.calibrate(raw_prob, strategy_name="MLStrategy")
    tracker.save()
    tracker.report()                           # Print per-strategy Brier scores
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent.parent / "data"
CAL_LOG_PATH = DATA_DIR / "calibration_log.parquet"
CAL_MDL_PATH = DATA_DIR / "calibration_models.json"


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class CalibrationRecord:
    """A single resolved prediction record."""
    strategy_name:  str
    market_id:      str
    predicted_prob: float      # Raw probability before calibration
    cal_prob:       float      # Calibrated probability (same as predicted if no model yet)
    outcome:        int        # 1 = YES won, 0 = NO won
    edge:           float      # Edge at time of prediction
    category:       str
    timestamp:      str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def brier_score(self) -> float:
        """Lower is better. Perfect = 0, random = 0.25."""
        return (self.predicted_prob - self.outcome) ** 2

    def cal_brier_score(self) -> float:
        return (self.cal_prob - self.outcome) ** 2


@dataclass
class StrategyCalibration:
    """Calibration state for one strategy."""
    strategy_name:    str
    brier_score:      float = 1.0    # Worst possible until we have data
    cal_brier_score:  float = 1.0
    n_predictions:    int   = 0
    n_correct:        float = 0.0    # Brier-weighted measure
    overconfidence:   float = 0.0    # Positive = overconfident
    mean_predicted:   float = 0.5
    mean_outcome:     float = 0.5
    # Isotonic regression breakpoints (stored as sorted lists for JSON serialization)
    iso_x: List[float] = field(default_factory=list)
    iso_y: List[float] = field(default_factory=list)

    def has_isotonic_model(self) -> bool:
        return len(self.iso_x) >= 2

    def calibrate(self, raw_prob: float) -> float:
        """Apply isotonic regression to a raw probability."""
        if not self.has_isotonic_model():
            return raw_prob
        # Linear interpolation between breakpoints
        x = np.array(self.iso_x)
        y = np.array(self.iso_y)
        return float(np.interp(raw_prob, x, y))


# ─── Main Tracker ────────────────────────────────────────────────────────────

class CalibrationTracker:
    """
    Tracks Brier scores per strategy and applies isotonic regression calibration.

    Designed to be updated incrementally after each market resolves.
    Persists state to disk so calibration improves across sessions.
    """

    STRATEGY_NAMES = [
        "MLStrategy", "MomentumStrategy", "LLMStrategy",
        "SentimentStrategy", "EnsembleStrategy",
    ]
    MIN_RECORDS_FOR_CALIBRATION = 30   # Need enough data for isotonic regression

    def __init__(self):
        self.records: List[CalibrationRecord] = []
        self.calibrations: Dict[str, StrategyCalibration] = {
            name: StrategyCalibration(strategy_name=name)
            for name in self.STRATEGY_NAMES
        }

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, strategy_name: str, market_id: str,
               predicted_prob: float, outcome: int,
               edge: float = 0.0, category: str = "other") -> CalibrationRecord:
        """
        Record a resolved prediction.

        Args:
            strategy_name:  Name of the strategy that made the prediction
            market_id:      Unique market identifier
            predicted_prob: Raw probability (0-1) before calibration
            outcome:        1 if YES won, 0 if NO won
            edge:           Edge used at trade time
            category:       Market category

        Returns:
            CalibrationRecord stored to log
        """
        cal_prob = self.calibrate(predicted_prob, strategy_name)
        rec = CalibrationRecord(
            strategy_name=strategy_name,
            market_id=market_id,
            predicted_prob=float(np.clip(predicted_prob, 1e-6, 1 - 1e-6)),
            cal_prob=float(np.clip(cal_prob, 1e-6, 1 - 1e-6)),
            outcome=int(outcome),
            edge=float(edge),
            category=category,
        )
        self.records.append(rec)
        return rec

    # ── Calibration ──────────────────────────────────────────────────────────

    def calibrate(self, raw_prob: float, strategy_name: str) -> float:
        """Apply isotonic regression calibration to a raw probability."""
        cal = self.calibrations.get(strategy_name)
        if cal is None or not cal.has_isotonic_model():
            return float(raw_prob)
        return cal.calibrate(raw_prob)

    def recalibrate(self, min_records: Optional[int] = None) -> Dict[str, float]:
        """
        Refit isotonic regression for all strategies that have enough data.

        Returns:
            Dict mapping strategy_name → new Brier score
        """
        if min_records is None:
            min_records = self.MIN_RECORDS_FOR_CALIBRATION

        df = self._records_to_df()
        if df.empty:
            return {}

        results = {}
        for name, cal in self.calibrations.items():
            mask = df["strategy_name"] == name
            sub  = df[mask].dropna(subset=["predicted_prob", "outcome"])

            if len(sub) < 2:
                continue

            probs    = sub["predicted_prob"].values.astype(float)
            outcomes = sub["outcome"].values.astype(float)

            # ── Update basic stats ───────────────────────────────────────────
            cal.n_predictions  = len(sub)
            cal.brier_score    = float(np.mean((probs - outcomes) ** 2))
            cal.mean_predicted = float(np.mean(probs))
            cal.mean_outcome   = float(np.mean(outcomes))
            cal.overconfidence = float(cal.mean_predicted - cal.mean_outcome)

            # ── Isotonic regression (needs sklearn + enough data) ────────────
            if _SKLEARN_OK and len(sub) >= min_records:
                try:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(probs, outcomes)

                    # Sample calibration curve to store as breakpoints
                    x_sample = np.linspace(0.01, 0.99, 50)
                    y_sample  = iso.predict(x_sample)

                    cal.iso_x = x_sample.tolist()
                    cal.iso_y = y_sample.tolist()

                    # Recalculate Brier on calibrated probs
                    cal_probs = iso.predict(probs)
                    cal.cal_brier_score = float(np.mean((cal_probs - outcomes) ** 2))
                except Exception:
                    pass

            results[name] = cal.brier_score

        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save records to parquet + calibration models to JSON."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save records
        df = self._records_to_df()
        if not df.empty:
            # Merge with existing log (avoid duplicate market_id + strategy)
            if CAL_LOG_PATH.exists():
                existing = pd.read_parquet(CAL_LOG_PATH)
                key_cols  = ["strategy_name", "market_id"]
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=key_cols, keep="last")
            df.to_parquet(CAL_LOG_PATH, index=False)

        # Save calibration models
        models_data = {}
        for name, cal in self.calibrations.items():
            models_data[name] = asdict(cal)
        with open(CAL_MDL_PATH, "w") as f:
            json.dump(models_data, f, indent=2)

    @classmethod
    def load(cls) -> "CalibrationTracker":
        """Load tracker state from disk."""
        tracker = cls()

        # Load records
        if CAL_LOG_PATH.exists():
            try:
                df = pd.read_parquet(CAL_LOG_PATH)
                for _, row in df.iterrows():
                    rec = CalibrationRecord(
                        strategy_name=str(row["strategy_name"]),
                        market_id=str(row["market_id"]),
                        predicted_prob=float(row["predicted_prob"]),
                        cal_prob=float(row.get("cal_prob", row["predicted_prob"])),
                        outcome=int(row["outcome"]),
                        edge=float(row.get("edge", 0.0)),
                        category=str(row.get("category", "other")),
                        timestamp=str(row.get("timestamp", "")),
                    )
                    tracker.records.append(rec)
            except Exception as e:
                print(f"[CalibrationTracker] Warning: could not load log: {e}")

        # Load calibration models
        if CAL_MDL_PATH.exists():
            try:
                with open(CAL_MDL_PATH) as f:
                    models_data = json.load(f)
                for name, data in models_data.items():
                    if name in tracker.calibrations:
                        cal = tracker.calibrations[name]
                        cal.brier_score     = data.get("brier_score", 1.0)
                        cal.cal_brier_score = data.get("cal_brier_score", 1.0)
                        cal.n_predictions   = data.get("n_predictions", 0)
                        cal.overconfidence  = data.get("overconfidence", 0.0)
                        cal.mean_predicted  = data.get("mean_predicted", 0.5)
                        cal.mean_outcome    = data.get("mean_outcome", 0.5)
                        cal.iso_x           = data.get("iso_x", [])
                        cal.iso_y           = data.get("iso_y", [])
            except Exception as e:
                print(f"[CalibrationTracker] Warning: could not load models: {e}")

        return tracker

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> pd.DataFrame:
        """Print + return per-strategy calibration summary."""
        self.recalibrate()

        rows = []
        for name, cal in self.calibrations.items():
            rows.append({
                "Strategy":        name,
                "N":               cal.n_predictions,
                "Brier":           round(cal.brier_score, 4),
                "Cal_Brier":       round(cal.cal_brier_score, 4),
                "Overconfidence":  round(cal.overconfidence, 4),
                "Mean_Pred":       round(cal.mean_predicted, 3),
                "Mean_Outcome":    round(cal.mean_outcome, 3),
                "Has_ISO_Model":   cal.has_isotonic_model(),
            })

        df = pd.DataFrame(rows).set_index("Strategy")
        print("\n── Calibration Report ──────────────────────────────────────────")
        print(df.to_string())
        print()
        return df

    def calibration_curve_data(self, strategy_name: str,
                               n_bins: int = 10) -> Optional[pd.DataFrame]:
        """
        Return reliability diagram data for a strategy.

        Returns DataFrame with columns: bin_center, mean_predicted, fraction_positive, count
        """
        df = self._records_to_df()
        if df.empty:
            return None

        sub = df[df["strategy_name"] == strategy_name].dropna()
        if len(sub) < n_bins:
            return None

        probs    = sub["predicted_prob"].values.astype(float)
        outcomes = sub["outcome"].values.astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        rows = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            rows.append({
                "bin_center":       (lo + hi) / 2,
                "mean_predicted":   probs[mask].mean(),
                "fraction_positive": outcomes[mask].mean(),
                "count":            int(mask.sum()),
            })

        return pd.DataFrame(rows) if rows else None

    def per_category_brier(self) -> pd.DataFrame:
        """Break down Brier scores by strategy × category."""
        df = self._records_to_df()
        if df.empty:
            return pd.DataFrame()

        df["brier"] = (df["predicted_prob"] - df["outcome"]) ** 2
        pivot = (
            df.groupby(["strategy_name", "category"])["brier"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "brier_score", "count": "n"})
        )
        return pivot

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _records_to_df(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([asdict(r) for r in self.records])

    def __len__(self) -> int:
        return len(self.records)


# ─── Convenience function for live trading ───────────────────────────────────

def get_calibrated_probability(raw_prob: float, strategy_name: str,
                               tracker: Optional[CalibrationTracker] = None) -> float:
    """
    Quick helper: load tracker (or use provided) and calibrate a probability.

    Args:
        raw_prob:      Raw model output probability
        strategy_name: Name of the strategy
        tracker:       Optional existing tracker (avoids disk load)

    Returns:
        Calibrated probability clipped to (0.001, 0.999)
    """
    if tracker is None:
        tracker = CalibrationTracker.load()
    cal = tracker.calibrate(raw_prob, strategy_name)
    return float(np.clip(cal, 1e-3, 1 - 1e-3))


# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibration tracker utilities")
    parser.add_argument("--report",     action="store_true", help="Print calibration report")
    parser.add_argument("--recalibrate",action="store_true", help="Refit isotonic regression models")
    parser.add_argument("--seed-demo",  action="store_true", help="Seed with synthetic data for demo")
    args = parser.parse_args()

    tracker = CalibrationTracker.load()

    if args.seed_demo:
        # Seed synthetic predictions to demo calibration
        np.random.seed(42)
        strategies = CalibrationTracker.STRATEGY_NAMES
        categories = ["politics", "crypto", "sports", "science", "other"]
        for i in range(200):
            strat    = strategies[i % len(strategies)]
            cat      = categories[i % len(categories)]
            # Simulate overconfident model: true prob is lower than predicted
            true_p   = np.random.uniform(0.1, 0.9)
            pred_p   = np.clip(true_p + np.random.normal(0.05, 0.08), 0.01, 0.99)
            outcome  = int(np.random.random() < true_p)
            tracker.record(
                strategy_name=strat,
                market_id=f"demo_{i:04d}",
                predicted_prob=pred_p,
                outcome=outcome,
                category=cat,
            )
        print(f"[Demo] Seeded {len(tracker)} synthetic records.")
        tracker.save()

    if args.recalibrate:
        scores = tracker.recalibrate()
        print("Recalibrated Brier scores:")
        for name, score in scores.items():
            print(f"  {name}: {score:.4f}")
        tracker.save()
        print("[✓] Models saved.")

    if args.report or not any([args.recalibrate, args.seed_demo]):
        tracker.report()

        print("── Per-Category Brier ──────────────────────────────────────────")
        cat_df = tracker.per_category_brier()
        if not cat_df.empty:
            print(cat_df.round(4).to_string())
        else:
            print("  (no data yet — run --seed-demo to populate)")
        print()
