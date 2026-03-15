"""
fitness_scorer.py -- Composite fitness score [0-100] for the trading bot.

Combines backtest performance metrics with test suite results into a single
score that drives the skill evolver's keep/discard decisions.

Components and weights:
  - OOS Return (25%): Sigmoid mapping -- 0%->0.5, +50%->0.9, -50%->0.1
  - OOS Sharpe (20%): Sigmoid -- 0->0.3, 1.0->0.7, 2.0->0.95
  - Calibration (15%): 1 - (mean_abs_error / 50), clipped [0,1]
  - Win Rate vs Break-even (15%): Sigmoid on margin above break-even
  - Max Drawdown (10%): 1 - (drawdown / 100), clipped [0,1]
  - Test Pass Rate (10%): passed / total
  - Safeguard Integrity (5%): Binary 1.0 or 0.0

CRITICAL: If safeguard_integrity == 0, entire score forced to 0.

Usage:
    from fitness_scorer import compute_fitness, FitnessResult
    result = compute_fitness(backtest_results, test_results)
    print(f"Fitness: {result.total_score:.1f}/100")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FitnessResult:
    """Complete fitness evaluation result."""
    total_score: float                  # Composite [0, 100]
    components: Dict[str, float]        # Individual component scores [0, 1]
    safeguard_veto: bool                # True if safeguard integrity failed
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def sigmoid_map(x: float, center: float, scale: float) -> float:
    """
    Sigmoid mapping: maps x to (0, 1) with configurable center and scale.

    Parameters
    ----------
    x : float
        Input value.
    center : float
        The x value that maps to 0.5.
    scale : float
        Controls the steepness. Larger = more gradual.

    Returns
    -------
    float in (0, 1)
    """
    z = (x - center) / max(abs(scale), 1e-10)
    # Clamp to prevent overflow
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------
# Component scoring functions
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "oos_return":           0.25,
    "oos_sharpe":           0.20,
    "calibration":          0.15,
    "win_rate_margin":      0.15,
    "max_drawdown":         0.10,
    "test_pass_rate":       0.10,
    "safeguard_integrity":  0.05,
}


def _score_oos_return(return_pct: float) -> float:
    """Map OOS return percentage to [0, 1].  0% -> 0.5, +50% -> ~0.9, -50% -> ~0.1."""
    # sigmoid_map(0, 0, 22) = 0.5
    # sigmoid_map(50, 0, 22) ~ 0.91
    # sigmoid_map(-50, 0, 22) ~ 0.10
    return sigmoid_map(return_pct, center=0.0, scale=22.0)


def _score_oos_sharpe(sharpe: float) -> float:
    """Map Sharpe ratio to [0, 1].  0 -> ~0.33, 1.0 -> ~0.67, 2.0 -> ~0.93."""
    if not math.isfinite(sharpe):
        return 0.3
    # sigmoid_map(0, 0.5, 0.7) ~ 0.33
    # sigmoid_map(1.0, 0.5, 0.7) ~ 0.67
    # sigmoid_map(2.0, 0.5, 0.7) ~ 0.93
    return sigmoid_map(sharpe, center=0.5, scale=0.7)


def _score_calibration(mean_abs_error_pp: float) -> float:
    """Map calibration error (pp) to [0, 1].  0pp -> 1.0, 50pp -> 0.0."""
    return max(0.0, min(1.0, 1.0 - (mean_abs_error_pp / 50.0)))


def _score_win_rate_margin(win_rate: float, break_even: float) -> float:
    """Map win rate vs break-even margin to [0, 1].  Margin in pp."""
    margin_pp = (win_rate - break_even) * 100.0
    return sigmoid_map(margin_pp, center=0.0, scale=8.0)


def _score_max_drawdown(drawdown_pct: float) -> float:
    """Map max drawdown % to [0, 1].  0% -> 1.0, 100% -> 0.0."""
    return max(0.0, min(1.0, 1.0 - (drawdown_pct / 100.0)))


def _score_test_pass_rate(passed: int, total: int) -> float:
    """Map test pass rate to [0, 1]."""
    if total == 0:
        return 1.0  # No tests = assume pass
    return passed / total


def _compute_calibration_error(calibration: Dict) -> float:
    """
    Compute mean absolute calibration error in percentage points.

    Expects calibration dict with bucket keys mapping to dicts containing
    'avg_fair_value_pct' and 'actual_win_rate_pct'.
    """
    if not calibration:
        return 25.0  # Default: assume moderate error
    errors = []
    for bucket_name, bucket_data in calibration.items():
        if not isinstance(bucket_data, dict):
            continue
        expected = bucket_data.get("avg_fair_value_pct", 50)
        actual = bucket_data.get("actual_win_rate_pct", 50)
        errors.append(abs(expected - actual))
    return sum(errors) / len(errors) if errors else 25.0


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_fitness(
    backtest_results: Optional[Dict] = None,
    test_results: Optional[Dict] = None,
) -> FitnessResult:
    """
    Compute composite fitness score [0, 100].

    Parameters
    ----------
    backtest_results : dict
        Output from kxbtc_backtest.run_backtest(). Expected keys:
        - out_of_sample: {return_pct, sharpe, win_rate, max_drawdown_pct,
                          calibration, n_trades, avg_edge}
    test_results : dict
        Pytest results: {passed, failed, errors, total, safeguard_passed}

    Returns
    -------
    FitnessResult with total_score, components breakdown, and safeguard_veto flag.
    """
    # Default empty inputs
    bt = backtest_results or {}
    tr = test_results or {
        "passed": 0, "failed": 0, "errors": 0, "total": 0,
        "safeguard_passed": True,
    }

    # Extract OOS metrics
    oos = bt.get("out_of_sample", {})
    return_pct = oos.get("return_pct", 0.0)
    sharpe = oos.get("sharpe", 0.0)
    if sharpe is None or (isinstance(sharpe, float) and not math.isfinite(sharpe)):
        sharpe = 0.0
    win_rate = oos.get("win_rate", 50.0) / 100.0  # Convert from percentage
    n_trades = oos.get("n_trades", 0)
    max_dd = oos.get("max_drawdown_pct", 0.0)
    calibration = oos.get("calibration", {})

    # Compute break-even rate (approximate from avg edge)
    avg_edge = oos.get("avg_edge", 0.0)
    break_even = max(0.3, win_rate - (avg_edge / 100.0)) if n_trades > 0 else 0.45

    # Compute component scores
    calib_error = _compute_calibration_error(calibration)

    safeguard_ok = 1.0 if tr.get("safeguard_passed", True) else 0.0

    components = {
        "oos_return":           _score_oos_return(return_pct),
        "oos_sharpe":           _score_oos_sharpe(sharpe),
        "calibration":          _score_calibration(calib_error),
        "win_rate_margin":      _score_win_rate_margin(win_rate, break_even),
        "max_drawdown":         _score_max_drawdown(max_dd),
        "test_pass_rate":       _score_test_pass_rate(
                                    tr.get("passed", 0), tr.get("total", 0)),
        "safeguard_integrity":  safeguard_ok,
    }

    # Weighted sum -> [0, 1] -> scale to [0, 100]
    raw_score = sum(components[k] * WEIGHTS[k] for k in components)

    # CRITICAL: Safeguard veto -- if any safeguard test fails, score = 0
    safeguard_veto = components["safeguard_integrity"] < 1.0
    if safeguard_veto:
        raw_score = 0.0

    total_score = round(raw_score * 100, 2)

    return FitnessResult(
        total_score=total_score,
        components={k: round(v, 4) for k, v in components.items()},
        safeguard_veto=safeguard_veto,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_fitness(result: FitnessResult, path: Path) -> None:
    """Append a FitnessResult to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result.to_dict(), default=str) + "\n")


def load_fitness_history(path: Path) -> List[FitnessResult]:
    """Load all FitnessResult entries from a JSONL file."""
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            results.append(FitnessResult(
                total_score=d.get("total_score", 0.0),
                components=d.get("components", {}),
                safeguard_veto=d.get("safeguard_veto", False),
                timestamp=d.get("timestamp", ""),
            ))
    return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick demo with synthetic data
    bt = {
        "out_of_sample": {
            "return_pct": 15.0,
            "sharpe": 1.2,
            "win_rate": 58.0,
            "n_trades": 50,
            "max_drawdown_pct": 12.0,
            "avg_edge": 10.0,
            "calibration": {
                "30-40": {"avg_fair_value_pct": 35, "actual_win_rate_pct": 32, "n": 10},
                "50-60": {"avg_fair_value_pct": 55, "actual_win_rate_pct": 52, "n": 15},
                "70+":   {"avg_fair_value_pct": 75, "actual_win_rate_pct": 60, "n": 20},
            },
        }
    }
    tr = {"passed": 28, "failed": 2, "errors": 0, "total": 30, "safeguard_passed": True}
    result = compute_fitness(bt, tr)
    print(f"Fitness Score: {result.total_score:.1f}/100")
    print(f"Safeguard Veto: {result.safeguard_veto}")
    print(f"Components: {result.components}")
