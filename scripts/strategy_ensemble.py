"""
Strategy Ensemble with Thompson Sampling Multi-Armed Bandit
Implements 5 prediction market strategies + Thompson sampling for explore/exploit.

Strategies:
  - MLStrategy:        xGBoost per-category classifier on engineered features
  - MomentumStrategy:  Price momentum + volume anomaly trend-following
  - LLMStrategy:       Base-rate + resolution-time Bayesian prior
  - SentimentStrategy: Spread/volume/liquidity market microstructure signals
  - EnsembleStrategy:  Weighted blend of all other strategies

Thompson sampling selects strategy per trade using Beta(alpha, beta) posteriors
updated from each resolved trade outcome.
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"


# ─── Strategy Interface ──────────────────────────────────────────────────────

@dataclass
class PredictionSignal:
    """Output of a strategy for a single market."""
    strategy_name: str
    market_id: str
    predicted_prob: float          # Calibrated YES probability [0, 1]
    edge: float                    # predicted_prob - market_price
    confidence: float              # Internal model confidence [0, 1]
    features_used: List[str]       # Feature names used
    metadata: Dict = field(default_factory=dict)

    def should_trade(self, min_edge: float = 0.05) -> bool:
        return abs(self.edge) >= min_edge


class BaseStrategy(ABC):
    """Abstract strategy interface."""

    name: str = "BaseStrategy"

    @abstractmethod
    def predict(self, market: pd.Series, base_rates: Dict) -> PredictionSignal:
        """
        Predict YES probability for a single market row.

        Args:
            market:     Row from market_features.parquet
            base_rates: Dict of category → {yes_rate, median_hours, count}

        Returns:
            PredictionSignal
        """

    def batch_predict(self, df: pd.DataFrame,
                      base_rates: Dict) -> List[PredictionSignal]:
        return [self.predict(row, base_rates) for _, row in df.iterrows()]


# ─── Strategy Implementations ────────────────────────────────────────────────

class MLStrategy(BaseStrategy):
    """
    Category-specialised xGBoost classifier.
    Trains one model per category on engineered features.
    Falls back to base-rate if model not fitted.
    """

    name = "MLStrategy"

    FEATURE_COLS = [
        "time_to_resolution_hours", "days_since_market_open",
        "volume", "open_interest", "liquidity_ratio",
        "price_momentum_24h", "price_volatility_7d",
        "volume_anomaly_score", "spread_pct",
    ]

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "MLStrategy":
        """Fit per-category xGBoost models."""
        try:
            import xgboost as xgb
        except ImportError:
            print("[MLStrategy] xgboost not installed — ML strategy will use base rates")
            return self

        for cat in df["category"].unique():
            mask = df["category"] == cat
            sub = df[mask].dropna(subset=self.FEATURE_COLS + ["outcome_label"])
            if len(sub) < 30:
                continue
            X = sub[self.FEATURE_COLS].fillna(0)
            y = sub["outcome_label"].astype(int)
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X, y)
            self._models[cat] = model

        self._fitted = bool(self._models)
        return self

    def predict(self, market: pd.Series, base_rates: Dict) -> PredictionSignal:
        cat = market.get("category", "other")
        market_price = float(market.get("price_at_T1d") or
                             market.get("price_at_T7d") or 0.5)

        if self._fitted and cat in self._models:
            X = np.array([[market.get(f, 0.0) or 0.0
                           for f in self.FEATURE_COLS]])
            try:
                prob = float(self._models[cat].predict_proba(X)[0, 1])
                confidence = abs(prob - 0.5) * 2  # distance from 50/50
                return PredictionSignal(
                    strategy_name=self.name,
                    market_id=str(market.get("market_id", "")),
                    predicted_prob=prob,
                    edge=prob - market_price,
                    confidence=confidence,
                    features_used=self.FEATURE_COLS,
                    metadata={"category": cat, "n_features": len(self.FEATURE_COLS)},
                )
            except Exception:
                pass

        # Fallback: category base rate
        yes_rate = base_rates.get(cat, {}).get("yes_rate", 0.5)
        return PredictionSignal(
            strategy_name=self.name,
            market_id=str(market.get("market_id", "")),
            predicted_prob=yes_rate,
            edge=yes_rate - market_price,
            confidence=0.2,
            features_used=["base_rate"],
            metadata={"category": cat, "fallback": True},
        )


class MomentumStrategy(BaseStrategy):
    """
    Trend-following via price momentum + volume anomaly signals.

    Logic:
    - Strong positive momentum AND positive volume anomaly → bet YES
    - Strong negative momentum AND positive volume anomaly → bet NO
    - Weak signal → shrink toward base rate
    """

    name = "MomentumStrategy"

    MOM_THRESHOLD  = 0.03   # |momentum_24h| threshold for signal
    VOL_THRESHOLD  = 0.5    # volume_anomaly_score z-score threshold
    SIGNAL_WEIGHT  = 0.35   # max probability shift from pure signal

    def predict(self, market: pd.Series, base_rates: Dict) -> PredictionSignal:
        cat = market.get("category", "other")
        base_yes = base_rates.get(cat, {}).get("yes_rate", 0.5)
        market_price = float(market.get("price_at_T1d") or
                             market.get("price_at_T7d") or base_yes)

        momentum = float(market.get("price_momentum_24h") or 0.0)
        vol_anomaly = float(market.get("volume_anomaly_score") or 0.0)

        # Only act when both momentum and volume signals agree
        mom_signal = 0.0
        if abs(momentum) >= self.MOM_THRESHOLD:
            direction = np.sign(momentum)
            vol_amplifier = min(abs(vol_anomaly) / 2.0, 1.0) if abs(vol_anomaly) >= self.VOL_THRESHOLD else 0.3
            mom_signal = direction * self.SIGNAL_WEIGHT * vol_amplifier

        predicted_prob = float(np.clip(base_yes + mom_signal, 0.05, 0.95))
        confidence = min(abs(momentum) / 0.10, 1.0) * min(abs(vol_anomaly) / 2.0, 1.0)

        return PredictionSignal(
            strategy_name=self.name,
            market_id=str(market.get("market_id", "")),
            predicted_prob=predicted_prob,
            edge=predicted_prob - market_price,
            confidence=float(confidence),
            features_used=["price_momentum_24h", "volume_anomaly_score"],
            metadata={"momentum": momentum, "vol_anomaly": vol_anomaly,
                      "mom_signal": mom_signal},
        )


class LLMStrategy(BaseStrategy):
    """
    Bayesian base-rate + resolution-time prior strategy.

    Uses category base rates and time-to-resolution as a Bayesian prior.
    Long-horizon markets: shrink toward 50/50 (more uncertainty).
    Short-horizon markets: trust market price more.

    This is the 'LLM-inspired reasoning' layer — systematic base rate updating.
    In a production system, you would call an LLM API here for semantic analysis.
    """

    name = "LLMStrategy"

    def predict(self, market: pd.Series, base_rates: Dict) -> PredictionSignal:
        cat = market.get("category", "other")
        cat_info = base_rates.get(cat, {})
        base_yes = cat_info.get("yes_rate", 0.5)
        median_hours = cat_info.get("median_hours", 720.0)

        market_price = float(market.get("price_at_T7d") or base_yes)
        resolution_hours = float(market.get("time_to_resolution_hours") or median_hours)

        # Bayesian weight: shorter horizon → trust market price more
        if median_hours > 0:
            time_ratio = resolution_hours / max(median_hours, 1.0)
        else:
            time_ratio = 1.0

        # Short markets: w_base=0.3, w_mkt=0.7 | Long markets: w_base=0.6, w_mkt=0.4
        time_ratio_clipped = float(np.clip(time_ratio, 0.1, 5.0))
        w_base = float(np.clip(0.3 + 0.15 * np.log(time_ratio_clipped + 1), 0.2, 0.65))
        w_mkt  = 1.0 - w_base

        predicted_prob = float(np.clip(w_base * base_yes + w_mkt * market_price, 0.05, 0.95))
        confidence = 0.5 - abs(time_ratio_clipped - 1.0) * 0.1  # near-median → more confident
        confidence = float(np.clip(confidence, 0.1, 0.8))

        return PredictionSignal(
            strategy_name=self.name,
            market_id=str(market.get("market_id", "")),
            predicted_prob=predicted_prob,
            edge=predicted_prob - market_price,
            confidence=confidence,
            features_used=["base_rate", "time_to_resolution_hours"],
            metadata={"w_base": w_base, "w_mkt": w_mkt,
                      "time_ratio": float(time_ratio_clipped)},
        )


class SentimentStrategy(BaseStrategy):
    """
    Market microstructure sentiment strategy.

    Signals:
    - Tight spread → high conviction market → trust market price
    - Wide spread → uncertainty → shrink toward base rate
    - High liquidity_ratio → active trading → momentum amplifier
    - Low liquidity_ratio → illiquid → fade extremes
    """

    name = "SentimentStrategy"

    TIGHT_SPREAD = 0.03
    WIDE_SPREAD  = 0.10

    def predict(self, market: pd.Series, base_rates: Dict) -> PredictionSignal:
        cat = market.get("category", "other")
        base_yes = base_rates.get(cat, {}).get("yes_rate", 0.5)
        market_price = float(market.get("price_at_T1d") or
                             market.get("price_at_T7d") or base_yes)

        spread = float(market.get("spread_pct") or 0.05)
        liq_ratio = float(market.get("liquidity_ratio") or 1.0)
        vol_anom = float(market.get("volume_anomaly_score") or 0.0)

        # Spread-based conviction
        if spread <= self.TIGHT_SPREAD:
            # Tight spread → trust market + slight momentum tilt
            w_market = 0.75
            sentiment_adj = vol_anom * 0.03
        elif spread >= self.WIDE_SPREAD:
            # Wide spread → uncertainty, regress to mean
            w_market = 0.40
            sentiment_adj = 0.0
        else:
            # Interpolate
            frac = (spread - self.TIGHT_SPREAD) / (self.WIDE_SPREAD - self.TIGHT_SPREAD)
            w_market = 0.75 - frac * 0.35
            sentiment_adj = vol_anom * 0.03 * (1 - frac)

        w_base = 1.0 - w_market

        # Liquidity-adjusted sentiment
        liq_factor = float(np.clip(np.log1p(liq_ratio) / np.log1p(5.0), 0.3, 1.2))

        predicted_prob = float(np.clip(
            w_base * base_yes + w_market * market_price + sentiment_adj * liq_factor,
            0.05, 0.95
        ))
        confidence = (1.0 - spread / 0.20) * min(liq_factor, 1.0)
        confidence = float(np.clip(confidence, 0.05, 0.90))

        return PredictionSignal(
            strategy_name=self.name,
            market_id=str(market.get("market_id", "")),
            predicted_prob=predicted_prob,
            edge=predicted_prob - market_price,
            confidence=confidence,
            features_used=["spread_pct", "liquidity_ratio", "volume_anomaly_score"],
            metadata={"spread": spread, "liq_ratio": liq_ratio,
                      "w_market": w_market},
        )


class EnsembleStrategy(BaseStrategy):
    """
    Weighted ensemble of all other strategies.
    Weights are updated dynamically from regime detector output.
    """

    name = "EnsembleStrategy"

    DEFAULT_WEIGHTS = {
        "MLStrategy":        0.30,
        "MomentumStrategy":  0.25,
        "LLMStrategy":       0.25,
        "SentimentStrategy": 0.20,
    }

    def __init__(self, sub_strategies: Dict[str, BaseStrategy]):
        self._sub = sub_strategies  # name → strategy

    def set_weights(self, weights: Dict[str, float]):
        """Update weights (e.g. from regime detector)."""
        total = sum(weights.values())
        if total > 0:
            self._weights = {k: v / total for k, v in weights.items()}
        else:
            self._weights = self.DEFAULT_WEIGHTS.copy()

    def predict(self, market: pd.Series, base_rates: Dict) -> PredictionSignal:
        weights = getattr(self, "_weights", self.DEFAULT_WEIGHTS)
        signals = {}
        for name, strat in self._sub.items():
            try:
                sig = strat.predict(market, base_rates)
                signals[name] = sig
            except Exception:
                pass

        if not signals:
            base_yes = base_rates.get(market.get("category", "other"), {}).get("yes_rate", 0.5)
            market_price = float(market.get("price_at_T1d") or base_yes)
            return PredictionSignal(
                strategy_name=self.name,
                market_id=str(market.get("market_id", "")),
                predicted_prob=base_yes,
                edge=base_yes - market_price,
                confidence=0.1,
                features_used=[],
                metadata={"error": "all_sub_failed"},
            )

        # Weighted average of predicted probabilities
        total_w = 0.0
        prob_sum = 0.0
        conf_sum = 0.0
        for name, sig in signals.items():
            w = weights.get(name, 0.25)
            prob_sum += w * sig.predicted_prob
            conf_sum += w * sig.confidence
            total_w  += w

        predicted_prob = float(np.clip(prob_sum / max(total_w, 1e-9), 0.05, 0.95))
        market_price = float(market.get("price_at_T1d") or
                             market.get("price_at_T7d") or 0.5)
        confidence    = float(np.clip(conf_sum / max(total_w, 1e-9), 0.05, 0.95))

        return PredictionSignal(
            strategy_name=self.name,
            market_id=str(market.get("market_id", "")),
            predicted_prob=predicted_prob,
            edge=predicted_prob - market_price,
            confidence=confidence,
            features_used=list(signals.keys()),
            metadata={"weights": weights, "sub_probs": {
                k: float(s.predicted_prob) for k, s in signals.items()
            }},
        )


# ─── Thompson Sampling Bandit ────────────────────────────────────────────────

@dataclass
class BanditArm:
    """Beta distribution posterior for one strategy."""
    name: str
    alpha: float = 1.0   # successes + 1 (prior: 1)
    beta:  float = 1.0   # failures  + 1 (prior: 1)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def sample(self) -> float:
        """Draw from Beta(alpha, beta)."""
        return float(np.random.beta(self.alpha, self.beta))

    def update(self, reward: float):
        """
        Update posterior.
        reward=1.0 → correct direction; reward=0.0 → wrong; reward in (0,1) → partial.
        """
        self.alpha += reward
        self.beta  += (1.0 - reward)

    def to_dict(self) -> Dict:
        return {"name": self.name, "alpha": self.alpha, "beta": self.beta,
                "mean": self.mean}


class ThompsonSamplingBandit:
    """
    Multi-armed bandit with Thompson sampling for strategy selection.

    Each arm = one strategy. On each trade decision:
      1. Sample θ_i ~ Beta(α_i, β_i) for each arm.
      2. Select arm with highest θ_i.
      3. After resolution, update winner arm with reward.

    Reward function:
      - Correct direction (pred > 0.5 and YES wins, OR pred < 0.5 and NO wins) → 1.0
      - Wrong direction → 0.0
      - Scale by edge confidence: reward *= min(|edge| / 0.20, 1.0)
    """

    def __init__(self, strategy_names: List[str]):
        self.arms: Dict[str, BanditArm] = {
            name: BanditArm(name=name) for name in strategy_names
        }

    def select(self) -> str:
        """Thompson sample → return name of selected strategy."""
        samples = {name: arm.sample() for name, arm in self.arms.items()}
        return max(samples, key=samples.get)

    def update(self, strategy_name: str, signal: PredictionSignal,
               outcome: int):
        """
        Update arm after market resolves.

        Args:
            strategy_name: Which arm to update
            signal:        The PredictionSignal that was used
            outcome:       1 = YES resolved, 0 = NO resolved
        """
        if strategy_name not in self.arms:
            return
        pred = signal.predicted_prob
        # Correct if we predicted the right side
        correct = int((pred >= 0.5) == bool(outcome))
        # Scale reward by edge magnitude (more confident correct = bigger update)
        edge_weight = float(np.clip(abs(signal.edge) / 0.20, 0.0, 1.0))
        reward = float(correct) * (0.5 + 0.5 * edge_weight)
        self.arms[strategy_name].update(reward)

    def get_stats(self) -> pd.DataFrame:
        rows = [arm.to_dict() for arm in self.arms.values()]
        df = pd.DataFrame(rows)
        df["n_trades"] = df["alpha"] + df["beta"] - 2  # subtract priors
        df = df.sort_values("mean", ascending=False).reset_index(drop=True)
        return df

    def save(self, path: str):
        state = {name: arm.to_dict() for name, arm in self.arms.items()}
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ThompsonSamplingBandit":
        with open(path) as f:
            state = json.load(f)
        obj = cls(list(state.keys()))
        for name, d in state.items():
            obj.arms[name].alpha = d["alpha"]
            obj.arms[name].beta  = d["beta"]
        return obj


# ─── Strategy Ensemble Orchestrator ─────────────────────────────────────────

class StrategyEnsemble:
    """
    Top-level orchestrator combining all strategies + Thompson bandit.

    Usage:
        ensemble = StrategyEnsemble()
        ensemble.fit(train_df, base_rates)

        # For a single market
        signal = ensemble.predict(market_row, base_rates)

        # After resolution
        ensemble.update_bandit(strategy_name, signal, outcome=1)
    """

    BANDIT_PATH = DATA_DIR / "bandit_state.json"

    def __init__(self, regime_weights: Optional[Dict[str, float]] = None):
        # Build sub-strategies
        self.ml        = MLStrategy()
        self.momentum  = MomentumStrategy()
        self.llm       = LLMStrategy()
        self.sentiment = SentimentStrategy()
        self.ensemble  = EnsembleStrategy({
            "MLStrategy":        self.ml,
            "MomentumStrategy":  self.momentum,
            "LLMStrategy":       self.llm,
            "SentimentStrategy": self.sentiment,
        })

        all_names = [
            "MLStrategy", "MomentumStrategy", "LLMStrategy",
            "SentimentStrategy", "EnsembleStrategy",
        ]
        # Load or create bandit
        if self.BANDIT_PATH.exists():
            try:
                self.bandit = ThompsonSamplingBandit.load(str(self.BANDIT_PATH))
            except Exception:
                self.bandit = ThompsonSamplingBandit(all_names)
        else:
            self.bandit = ThompsonSamplingBandit(all_names)

        # Apply regime weights to ensemble if provided
        if regime_weights:
            self.ensemble.set_weights(regime_weights)

        self._strategies: Dict[str, BaseStrategy] = {
            "MLStrategy":        self.ml,
            "MomentumStrategy":  self.momentum,
            "LLMStrategy":       self.llm,
            "SentimentStrategy": self.sentiment,
            "EnsembleStrategy":  self.ensemble,
        }

    def fit(self, df: pd.DataFrame, base_rates: Dict) -> "StrategyEnsemble":
        """Fit trainable strategies (ML only for now)."""
        print("[StrategyEnsemble] Fitting MLStrategy ...")
        self.ml.fit(df)
        print("[StrategyEnsemble] Fit complete.")
        return self

    def set_regime_weights(self, weights: Dict[str, float]):
        """Update ensemble weights from regime detector output."""
        self.ensemble.set_weights(weights)

    def predict(self, market: pd.Series, base_rates: Dict,
                strategy_name: Optional[str] = None) -> Tuple[str, PredictionSignal]:
        """
        Predict using Thompson-selected strategy (or explicit override).

        Returns:
            (strategy_name_used, PredictionSignal)
        """
        if strategy_name is None:
            strategy_name = self.bandit.select()

        strat = self._strategies.get(strategy_name, self.ensemble)
        signal = strat.predict(market, base_rates)
        return strategy_name, signal

    def predict_all(self, market: pd.Series,
                    base_rates: Dict) -> Dict[str, PredictionSignal]:
        """Run all strategies and return dict of signals."""
        results = {}
        for name, strat in self._strategies.items():
            try:
                results[name] = strat.predict(market, base_rates)
            except Exception as e:
                print(f"  [!] {name} failed for {market.get('market_id')}: {e}")
        return results

    def update_bandit(self, strategy_name: str, signal: PredictionSignal,
                      outcome: int):
        """Update Thompson sampling posteriors after a market resolves."""
        self.bandit.update(strategy_name, signal, outcome)
        # Persist bandit state
        try:
            self.BANDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.bandit.save(str(self.BANDIT_PATH))
        except Exception:
            pass

    def bandit_stats(self) -> pd.DataFrame:
        return self.bandit.get_stats()


# ─── CLI / Demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run strategy ensemble on feature data")
    parser.add_argument("--fit",   action="store_true", help="Fit ML models")
    parser.add_argument("--stats", action="store_true", help="Print bandit stats")
    args = parser.parse_args()

    # Load data
    feat_path = DATA_DIR / "features" / "market_features.parquet"
    base_path = DATA_DIR / "base_rates.json"

    if not feat_path.exists():
        print(f"[!] No feature data at {feat_path}. Run collect_polymarket.py first.")
        exit(1)

    df = pd.read_parquet(feat_path)
    base_rates = {}
    if base_path.exists():
        with open(base_path) as f:
            base_rates = json.load(f)

    print(f"Loaded {len(df)} markets, {len(base_rates)} category base rates")

    ensemble = StrategyEnsemble()

    if args.fit:
        ensemble.fit(df, base_rates)

    if args.stats:
        print("\n── Thompson Sampling Bandit State ──")
        print(ensemble.bandit_stats().to_string(index=False))
        exit(0)

    # Demo: predict on first 5 markets
    print("\n── Sample Predictions ──")
    for i, (_, row) in enumerate(df.head(5).iterrows()):
        strategy_used, signal = ensemble.predict(row, base_rates)
        print(f"\n  Market: {signal.market_id[:20]:20s} | Cat: {row.get('category','?'):10s}")
        print(f"  Strategy: {strategy_used:20s} | Prob: {signal.predicted_prob:.3f} | Edge: {signal.edge:+.3f} | Conf: {signal.confidence:.2f}")
        print(f"  Trade: {'YES' if signal.edge > 0 else 'NO '} | Should trade: {signal.should_trade()}")

    print("\n── All-Strategy Comparison (market 0) ──")
    row0 = df.iloc[0]
    all_signals = ensemble.predict_all(row0, base_rates)
    for name, sig in all_signals.items():
        print(f"  {name:22s}: prob={sig.predicted_prob:.3f}  edge={sig.edge:+.3f}  conf={sig.confidence:.2f}")
