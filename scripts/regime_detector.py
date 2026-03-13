"""
Prediction Market Regime Detector
Adapts HMM-based regime detection for prediction market features.

Instead of equity price/volume, operates on:
  - price_momentum_24h  (directional velocity)
  - price_volatility_7d (uncertainty level)
  - volume_anomaly_score (information arrival signal)
  - spread_pct           (liquidity / market-maker confidence)

Three regimes per category:
  - low_uncertainty  → slow drift, mature market → weight ML + Momentum higher
  - high_uncertainty → breaking news, fast-moving → weight LLM + Sentiment higher
  - trending         → strong directional move → weight Momentum + ML higher

Usage:
    from regime_detector import PredictionMarketRegimeDetector
    detector = PredictionMarketRegimeDetector()
    detector.fit(features_df, category="politics")
    regime = detector.predict(features_df, category="politics")
"""

import numpy as np
import pandas as pd
import json
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from hmmlearn import hmm as hmmlearn_hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("[regime_detector] hmmlearn not installed — falling back to threshold-based detection.")

# ─── Config ─────────────────────────────────────────────────────────────────────

CATEGORIES = ["politics", "crypto", "science", "sports", "other"]

REGIME_LABELS = {
    0: "low_uncertainty",
    1: "trending",
    2: "high_uncertainty",
}

# Strategy weights per regime — drives strategy_ensemble.py
STRATEGY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "low_uncertainty": {
        "MLStrategy":        0.40,
        "MomentumStrategy":  0.30,
        "EnsembleStrategy":  0.15,
        "SentimentStrategy": 0.10,
        "LLMStrategy":       0.05,
    },
    "trending": {
        "MomentumStrategy":  0.40,
        "MLStrategy":        0.30,
        "EnsembleStrategy":  0.15,
        "LLMStrategy":       0.10,
        "SentimentStrategy": 0.05,
    },
    "high_uncertainty": {
        "LLMStrategy":       0.40,
        "SentimentStrategy": 0.30,
        "EnsembleStrategy":  0.15,
        "MLStrategy":        0.10,
        "MomentumStrategy":  0.05,
    },
}

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = DATA_DIR / "regime_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Data Classes ────────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    """Current regime prediction for a single category."""
    category: str
    regime: str                              # "low_uncertainty" | "trending" | "high_uncertainty"
    regime_id: int
    probabilities: Dict[str, float]          # {"low_uncertainty": 0.7, ...}
    strategy_weights: Dict[str, float]       # from STRATEGY_WEIGHTS
    transition_matrix: Optional[np.ndarray] = None
    expected_duration_periods: Optional[float] = None

    def __str__(self) -> str:
        prob_str = ", ".join(f"{k}: {v:.1%}" for k, v in self.probabilities.items())
        return (
            f"[{self.category}] Regime: {self.regime} | "
            f"Probs: {prob_str} | "
            f"Expected duration: {self.expected_duration_periods:.1f} periods"
            if self.expected_duration_periods else
            f"[{self.category}] Regime: {self.regime} | Probs: {prob_str}"
        )


# ─── Feature Preparation ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "price_momentum_24h",
    "price_volatility_7d",
    "volume_anomaly_score",
    "spread_pct",
]


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and standardize HMM input features from a market features DataFrame.

    Handles missing values by forward-filling then zero-filling.
    Standardizes each feature to zero mean, unit variance.
    """
    available = [c for c in FEATURE_COLS if c in df.columns]
    if not available:
        raise ValueError(f"No regime features found. Need one of: {FEATURE_COLS}")

    X = df[available].copy().ffill().fillna(0.0)

    # Standardize
    means = X.mean()
    stds  = X.std().replace(0, 1.0)
    X = (X - means) / stds

    return X.values.astype(np.float64), available


def _label_regimes_from_means(means: np.ndarray, feature_names: List[str]) -> Dict[int, str]:
    """
    Assign semantic labels to HMM states by inspecting learned emission means.

    - High volatility mean → high_uncertainty
    - High momentum mean   → trending
    - Low both             → low_uncertainty
    """
    n = len(means)
    if "price_volatility_7d" in feature_names:
        vol_idx = feature_names.index("price_volatility_7d")
        vol_scores = means[:, vol_idx]
    else:
        vol_scores = np.zeros(n)

    if "price_momentum_24h" in feature_names:
        mom_idx = feature_names.index("price_momentum_24h")
        mom_scores = np.abs(means[:, mom_idx])
    else:
        mom_scores = np.zeros(n)

    # Combined score: high vol + high |momentum| = high_uncertainty or trending
    combined = vol_scores + mom_scores
    sorted_idx = np.argsort(combined)

    labels: Dict[int, str] = {}
    if n == 3:
        labels[sorted_idx[0]] = "low_uncertainty"
        labels[sorted_idx[1]] = "trending"
        labels[sorted_idx[2]] = "high_uncertainty"
    else:
        # Fallback for non-standard n_regimes
        for rank, idx in enumerate(sorted_idx):
            labels[idx] = f"regime_{rank}"
    return labels


# ─── Threshold Fallback ──────────────────────────────────────────────────────────

def _threshold_regime(df: pd.DataFrame) -> str:
    """Simple threshold-based regime detection when hmmlearn is unavailable."""
    if df.empty:
        return "low_uncertainty"

    last = df.iloc[-20:] if len(df) >= 20 else df

    avg_vol  = last["price_volatility_7d"].mean()  if "price_volatility_7d"  in df.columns else 0.0
    avg_mom  = last["price_momentum_24h"].abs().mean() if "price_momentum_24h"   in df.columns else 0.0
    avg_vanom = last["volume_anomaly_score"].abs().mean() if "volume_anomaly_score" in df.columns else 0.0

    # Compute global stats for relative thresholds
    global_vol = df["price_volatility_7d"].mean() if "price_volatility_7d" in df.columns else 0.01
    global_vol = max(global_vol, 1e-8)

    if avg_vol > 1.5 * global_vol or avg_vanom > 1.5:
        return "high_uncertainty"
    elif avg_mom > 0.05:
        return "trending"
    else:
        return "low_uncertainty"


# ─── Main Detector ───────────────────────────────────────────────────────────────

class PredictionMarketRegimeDetector:
    """
    Per-category HMM regime detector for prediction markets.

    Fits one model per category so crypto volatility doesn't contaminate
    the politics regime signal.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes    = n_regimes
        self.random_state = random_state
        self._models:      Dict[str, object]        = {}   # category → fitted HMM
        self._labels:      Dict[str, Dict[int, str]]= {}   # category → {state_id: label}
        self._feat_names:  Dict[str, List[str]]     = {}   # category → feature list

    # ── Fitting ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, category: str = "all") -> "PredictionMarketRegimeDetector":
        """
        Fit regime model for a single category (or all categories if category='all').

        Args:
            df: DataFrame with prediction market features (from market_features.parquet).
                Must include a 'category' column when category='all'.
            category: Category name or 'all'.
        """
        if category == "all":
            cats = df["category"].unique() if "category" in df.columns else ["other"]
            for cat in cats:
                sub = df[df["category"] == cat] if "category" in df.columns else df
                if len(sub) >= 20:
                    self._fit_single(sub, cat)
        else:
            self._fit_single(df, category)
        return self

    def _fit_single(self, df: pd.DataFrame, category: str) -> None:
        try:
            X, feat_names = _prepare_features(df)
        except ValueError as e:
            print(f"  [regime] Skipping {category}: {e}")
            return

        if len(X) < self.n_regimes * 5:
            print(f"  [regime] Not enough data for {category} ({len(X)} rows). Skipping HMM.")
            return

        if not HMM_AVAILABLE:
            # Store placeholder so predict() can fall through to threshold
            self._models[category]    = None
            self._labels[category]    = REGIME_LABELS
            self._feat_names[category] = feat_names
            return

        model = hmmlearn_hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=200,
            random_state=self.random_state,
        )
        try:
            model.fit(X)
        except Exception as e:
            print(f"  [regime] HMM fit failed for {category}: {e}")
            self._models[category]    = None
            self._labels[category]    = REGIME_LABELS
            self._feat_names[category] = feat_names
            return

        labels = _label_regimes_from_means(model.means_, feat_names)
        self._models[category]    = model
        self._labels[category]    = labels
        self._feat_names[category] = feat_names
        print(f"  [regime] Fitted {category}: {labels}")

    # ── Prediction ───────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, category: str) -> RegimeState:
        """
        Predict the current regime for a category from the most recent data.

        Returns RegimeState with regime label, probabilities, and strategy weights.
        """
        if category not in self._models:
            # Use threshold fallback
            regime = _threshold_regime(df)
            return RegimeState(
                category=category,
                regime=regime,
                regime_id=self._regime_id(regime),
                probabilities={r: (1.0 if r == regime else 0.0) for r in REGIME_LABELS.values()},
                strategy_weights=STRATEGY_WEIGHTS[regime],
            )

        model     = self._models[category]
        labels    = self._labels[category]
        feat_names = self._feat_names[category]

        if model is None:
            regime = _threshold_regime(df)
            return RegimeState(
                category=category,
                regime=regime,
                regime_id=self._regime_id(regime),
                probabilities={r: (1.0 if r == regime else 0.0) for r in REGIME_LABELS.values()},
                strategy_weights=STRATEGY_WEIGHTS[regime],
            )

        try:
            X, _ = _prepare_features(df[feat_names] if all(f in df.columns for f in feat_names) else df)
            _, posteriors  = model.score_samples(X)
            current_probs  = posteriors[-1]
            current_regime = int(np.argmax(current_probs))

            probs_named: Dict[str, float] = {}
            for state_id, label in labels.items():
                probs_named[label] = float(current_probs[state_id])

            regime_label = labels[current_regime]

            # Expected duration for current regime
            t_mat  = model.transmat_
            self_t = t_mat[current_regime, current_regime]
            exp_dur = 1.0 / (1.0 - self_t + 1e-9)

            return RegimeState(
                category=category,
                regime=regime_label,
                regime_id=current_regime,
                probabilities=probs_named,
                strategy_weights=STRATEGY_WEIGHTS.get(regime_label, STRATEGY_WEIGHTS["low_uncertainty"]),
                transition_matrix=t_mat,
                expected_duration_periods=round(exp_dur, 1),
            )

        except Exception as e:
            print(f"  [regime] Predict failed for {category}: {e} — using threshold fallback")
            regime = _threshold_regime(df)
            return RegimeState(
                category=category,
                regime=regime,
                regime_id=self._regime_id(regime),
                probabilities={r: (1.0 if r == regime else 0.0) for r in REGIME_LABELS.values()},
                strategy_weights=STRATEGY_WEIGHTS[regime],
            )

    def predict_all(self, df: pd.DataFrame) -> Dict[str, RegimeState]:
        """Predict regimes for every category found in df."""
        results: Dict[str, RegimeState] = {}
        if "category" not in df.columns:
            results["other"] = self.predict(df, "other")
            return results

        for cat in df["category"].unique():
            sub = df[df["category"] == cat]
            results[cat] = self.predict(sub, cat)
        return results

    def get_regime_history(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """
        Return a DataFrame with regime label for every row in df.

        Useful for walk-forward backtesting to annotate each market with
        the regime that was active at prediction time.
        """
        if category not in self._models or self._models[category] is None:
            regime = _threshold_regime(df)
            return df.assign(regime=regime, regime_id=self._regime_id(regime))

        model     = self._models[category]
        feat_names = self._feat_names[category]
        labels    = self._labels[category]

        try:
            X, _ = _prepare_features(df)
            regimes   = model.predict(X)
            _, posts   = model.score_samples(X)

            out = df.iloc[-len(regimes):].copy()
            out["regime_id"]    = regimes
            out["regime"]       = [labels[r] for r in regimes]

            for state_id, label in labels.items():
                out[f"prob_{label}"] = posts[:, state_id]

            return out
        except Exception as e:
            print(f"  [regime] History failed for {category}: {e}")
            return df.assign(regime="low_uncertainty", regime_id=0)

    # ── Persistence ──────────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """Save label maps and feature lists (HMM objects use joblib)."""
        try:
            import joblib
        except ImportError:
            print("[regime] joblib not available — models not saved.")
            return

        save_dir = path or MODEL_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for cat, model in self._models.items():
            if model is not None:
                joblib.dump(model, save_dir / f"{cat}_hmm.pkl")

        meta = {cat: {"labels": lbl, "feat_names": fn}
                for cat, (lbl, fn) in zip(
                    self._labels.keys(),
                    zip(self._labels.values(), self._feat_names.values())
                )}
        (save_dir / "regime_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[regime] Models saved → {save_dir}")

    def load(self, path: Optional[Path] = None) -> "PredictionMarketRegimeDetector":
        """Load persisted models."""
        try:
            import joblib
        except ImportError:
            print("[regime] joblib not available — cannot load models.")
            return self

        load_dir = path or MODEL_DIR
        meta_path = Path(load_dir) / "regime_meta.json"
        if not meta_path.exists():
            print(f"[regime] No saved models found at {load_dir}")
            return self

        meta = json.loads(meta_path.read_text())
        for cat, info in meta.items():
            pkl = Path(load_dir) / f"{cat}_hmm.pkl"
            if pkl.exists():
                self._models[cat]    = joblib.load(pkl)
                self._labels[cat]    = {int(k): v for k, v in info["labels"].items()}
                self._feat_names[cat] = info["feat_names"]
        print(f"[regime] Loaded models for: {list(self._models.keys())}")
        return self

    # ── Helpers ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _regime_id(label: str) -> int:
        inv = {v: k for k, v in REGIME_LABELS.items()}
        return inv.get(label, 0)


# ─── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    DATA_PATH = Path(__file__).parent.parent / "data" / "features" / "market_features.parquet"

    if not DATA_PATH.exists():
        print(f"[!] No feature data found at {DATA_PATH}")
        print("    Run: python scripts/collect_polymarket.py --days 90")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} markets")

    detector = PredictionMarketRegimeDetector(n_regimes=3)
    detector.fit(df, category="all")

    print("\n── Current Regime by Category ──────────────────────────────────")
    for cat in df["category"].unique():
        sub   = df[df["category"] == cat]
        state = detector.predict(sub, cat)
        print(f"  {state}")

    detector.save()
    print("\nDone. Models saved to data/regime_models/")
