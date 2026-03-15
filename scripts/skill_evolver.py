"""
skill_evolver.py -- Autonomous self-improving parameter evolution loop.

Inspired by Karpathy's autoresearch: mutate -> test -> measure -> keep/discard -> repeat.

Instead of modifying code, we mutate strategy parameters (strategy_config.json).
Instead of val_bpb, our metric is a composite fitness score from fitness_scorer.py.
Instead of 5-min GPU training, our experiment is a backtest run.

Usage:
    python scripts/skill_evolver.py --iterations 20
    python scripts/skill_evolver.py --daemon
    python scripts/skill_evolver.py --report
    python scripts/skill_evolver.py --iterations 10 --no-tests
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

BACKTEST_DIR = PROJECT_ROOT / "backtest"
RESULTS_TSV = BACKTEST_DIR / "evolution_results.tsv"
FITNESS_HISTORY = BACKTEST_DIR / "fitness_history.jsonl"

# ---------------------------------------------------------------------------
# Imports from sibling scripts
# ---------------------------------------------------------------------------
from strategy_optimizer import (
    load_config, save_config, DEFAULT_CONFIG, CONFIG_PATH, get_config_bounds,
)
from fitness_scorer import compute_fitness, FitnessResult, save_fitness

# Import backtest -- handle missing historical data gracefully
try:
    from kxbtc_backtest import run_backtest, HISTORICAL_PATH
    _BACKTEST_AVAILABLE = HISTORICAL_PATH.exists()
except Exception:
    _BACKTEST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Parameter bounds and perturbation magnitudes
# ---------------------------------------------------------------------------
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "btc_hourly_vol":           (0.005, 0.025),
    "min_edge_pp":              (3.0,   15.0),
    "fractional_kelly":         (0.10,  0.50),
    "max_position_pct":         (0.02,  0.10),
    "time_decay_threshold_min": (10.0,  120.0),
    "time_decay_min_fair":      (0.50,  0.95),
}

PERTURBATION: Dict[str, Tuple[float, float]] = {
    "btc_hourly_vol":           (0.001, 0.003),
    "min_edge_pp":              (0.5,   3.0),
    "fractional_kelly":         (0.02,  0.10),
    "max_position_pct":         (0.005, 0.02),
    "time_decay_threshold_min": (5.0,   20.0),
    "time_decay_min_fair":      (0.05,  0.15),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    param_name: str
    old_value: Any
    new_value: Any
    rationale: str

    def __str__(self) -> str:
        return f"{self.param_name}: {self.old_value} -> {self.new_value} ({self.rationale})"


@dataclass
class ExperimentResult:
    experiment_id: int
    timestamp: str
    mutation: Dict
    baseline_fitness: float
    mutated_fitness: float
    delta: float
    status: str  # "keep", "discard", "crash"
    config_snapshot: Dict
    backtest_summary: Dict
    test_pass_rate: float
    duration_seconds: float


# ---------------------------------------------------------------------------
# Mutation strategies
# ---------------------------------------------------------------------------

def _mutate_numeric(config: Dict, param: str) -> Mutation:
    """Perturb a single numeric parameter within valid bounds."""
    old_val = config.get(param, DEFAULT_CONFIG.get(param, 0.0))
    lo_pert, hi_pert = PERTURBATION[param]
    lo_bound, hi_bound = PARAM_BOUNDS[param]

    delta = random.uniform(lo_pert, hi_pert)
    if random.random() < 0.5:
        delta = -delta

    new_val = old_val + delta

    # Integer params
    if param == "time_decay_threshold_min":
        new_val = round(new_val)

    # Clamp to bounds
    new_val = max(lo_bound, min(hi_bound, new_val))

    # Round for readability
    if param in ("btc_hourly_vol",):
        new_val = round(new_val, 4)
    elif param in ("min_edge_pp", "time_decay_min_fair"):
        new_val = round(new_val, 2)
    elif param in ("fractional_kelly", "max_position_pct"):
        new_val = round(new_val, 3)

    return Mutation(
        param_name=param,
        old_value=old_val,
        new_value=new_val,
        rationale=f"perturb {'+' if delta >= 0 else ''}{delta:.4f}",
    )


def _mutate_avoid_hours(config: Dict) -> Mutation:
    """Add or remove a random UTC hour from avoid_hours."""
    current = list(config.get("avoid_hours", []))
    all_hours = list(range(24))
    available = [h for h in all_hours if h not in current]

    if current and (not available or random.random() < 0.5):
        hour = random.choice(current)
        new_hours = [h for h in current if h != hour]
        rationale = f"remove hour {hour}"
    elif available:
        hour = random.choice(available)
        new_hours = sorted(current + [hour])
        rationale = f"add hour {hour}"
    else:
        return Mutation("avoid_hours", current, current, "no change possible")

    return Mutation("avoid_hours", current, new_hours, rationale)


def _mutate_multi_param(config: Dict) -> List[Mutation]:
    """Simultaneously mutate 2-3 parameters."""
    n = random.choice([2, 3])
    params = random.sample(list(PARAM_BOUNDS.keys()), min(n, len(PARAM_BOUNDS)))
    return [_mutate_numeric(config, p) for p in params]


def select_mutation(config: Dict) -> List[Mutation]:
    """Select a random mutation strategy. Returns list of Mutation(s)."""
    r = random.random()

    if r < 0.15:
        # Multi-param mutation (15% chance)
        return _mutate_multi_param(config)
    elif r < 0.25:
        # Avoid hours mutation (10% chance)
        return [_mutate_avoid_hours(config)]
    else:
        # Single numeric param (75% chance)
        param = random.choice(list(PARAM_BOUNDS.keys()))
        return [_mutate_numeric(config, param)]


def apply_mutations(config: Dict, mutations: List[Mutation]) -> Dict:
    """Apply mutations to a config copy and return the mutated config."""
    mutated = copy.deepcopy(config)
    for m in mutations:
        mutated[m.param_name] = m.new_value
    return mutated


# ---------------------------------------------------------------------------
# Simplicity criterion
# ---------------------------------------------------------------------------

def _config_complexity(config: Dict) -> float:
    """Lower is simpler. Counts deviations from defaults + avoid_hours length."""
    score = 0.0
    for key, default_val in DEFAULT_CONFIG.items():
        if key in ("last_updated", "iteration", "notes", "source"):
            continue
        current = config.get(key, default_val)
        if key == "avoid_hours":
            score += len(current) * 0.1
        elif isinstance(default_val, (int, float)) and isinstance(current, (int, float)):
            if default_val != 0:
                score += abs(current - default_val) / abs(default_val)
    return score


# ---------------------------------------------------------------------------
# Test runner helper
# ---------------------------------------------------------------------------

def _run_tests_fast() -> Dict:
    """Run pytest in fast mode and parse results."""
    test_results = {
        "passed": 0, "failed": 0, "errors": 0, "total": 0,
        "safeguard_passed": True,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
             "--timeout=30", "-m", "not slow"],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        output = result.stdout + result.stderr
        lines = output.strip().split("\n")
        summary_line = lines[-1] if lines else ""

        passed = failed = errors = 0
        m = re.search(r"(\d+) passed", summary_line)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", summary_line)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", summary_line)
        if m:
            errors = int(m.group(1))

        total = passed + failed + errors
        safeguard_passed = (
            "FAILED" not in output or "test_risk_safeguards" not in output
        )

        test_results = {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": max(total, 1),
            "safeguard_passed": safeguard_passed,
        }
    except Exception:
        pass

    return test_results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    config: Dict,
    mutations: List[Mutation],
    experiment_id: int,
    run_tests: bool = True,
) -> Tuple[FitnessResult, ExperimentResult]:
    """
    Run one experiment: apply mutations, backtest, score, return result.
    """
    start_time = time.time()
    mutated_config = apply_mutations(config, mutations)

    # Run backtest with mutated parameters
    bt_result = {}
    if _BACKTEST_AVAILABLE:
        try:
            bt_result = run_backtest(
                bankroll=10.0,
                vol=mutated_config.get("btc_hourly_vol", 0.01),
                min_edge=mutated_config.get("min_edge_pp", 8.0) / 100.0,
            )
        except Exception as e:
            duration = time.time() - start_time
            report = compute_fitness(None, None)
            return report, ExperimentResult(
                experiment_id=experiment_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                mutation={
                    "mutations": [
                        {"param_name": m.param_name, "old_value": m.old_value,
                         "new_value": m.new_value, "rationale": m.rationale}
                        for m in mutations
                    ]
                },
                baseline_fitness=0,
                mutated_fitness=0,
                delta=0,
                status="crash",
                config_snapshot=mutated_config,
                backtest_summary={"error": str(e)},
                test_pass_rate=0,
                duration_seconds=round(duration, 2),
            )

    # Run tests (fast mode)
    test_results = {
        "passed": 0, "failed": 0, "errors": 0, "total": 0,
        "safeguard_passed": True,
    }
    if run_tests:
        test_results = _run_tests_fast()

    # Compute fitness
    report = compute_fitness(bt_result, test_results)
    duration = time.time() - start_time

    mutation_desc = "; ".join(str(m) for m in mutations)
    bt_summary = {}
    oos = bt_result.get("out_of_sample", {})
    if oos:
        bt_summary = {
            "return_pct": oos.get("return_pct", 0),
            "sharpe": oos.get("sharpe", 0),
            "win_rate": oos.get("win_rate", 0),
            "max_drawdown_pct": oos.get("max_drawdown_pct", 0),
        }

    experiment = ExperimentResult(
        experiment_id=experiment_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        mutation={
            "description": mutation_desc,
            "params": {m.param_name: m.new_value for m in mutations},
        },
        baseline_fitness=0,  # filled by caller
        mutated_fitness=report.total_score,
        delta=0,  # filled by caller
        status="pending",  # filled by caller
        config_snapshot=mutated_config,
        backtest_summary=bt_summary,
        test_pass_rate=test_results["passed"] / max(test_results["total"], 1),
        duration_seconds=round(duration, 2),
    )

    return report, experiment


# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

TSV_HEADERS = [
    "experiment_id", "timestamp", "param_changed", "old_value", "new_value",
    "baseline_fitness", "mutated_fitness", "delta", "status",
    "oos_return_pct", "oos_sharpe", "oos_win_rate", "oos_max_drawdown",
    "test_pass_rate", "duration_seconds",
]


def _log_result(exp: ExperimentResult, mutations: List[Mutation]) -> None:
    """Append experiment result to evolution_results.tsv."""
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_TSV.exists()

    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_HEADERS, delimiter="\t")
        if write_header:
            writer.writeheader()

        param_changed = "; ".join(m.param_name for m in mutations)
        old_vals = "; ".join(str(m.old_value) for m in mutations)
        new_vals = "; ".join(str(m.new_value) for m in mutations)

        bt = exp.backtest_summary
        writer.writerow({
            "experiment_id": exp.experiment_id,
            "timestamp": exp.timestamp,
            "param_changed": param_changed,
            "old_value": old_vals,
            "new_value": new_vals,
            "baseline_fitness": round(exp.baseline_fitness, 2),
            "mutated_fitness": round(exp.mutated_fitness, 2),
            "delta": round(exp.delta, 2),
            "status": exp.status,
            "oos_return_pct": bt.get("return_pct", ""),
            "oos_sharpe": bt.get("sharpe", ""),
            "oos_win_rate": bt.get("win_rate", ""),
            "oos_max_drawdown": bt.get("max_drawdown_pct", ""),
            "test_pass_rate": round(exp.test_pass_rate, 3),
            "duration_seconds": exp.duration_seconds,
        })


# ---------------------------------------------------------------------------
# Core evolution loop
# ---------------------------------------------------------------------------

def run_evolution(
    n_iterations: int = 20,
    run_tests: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Run the autonomous evolution loop.

    Core loop:
      1. Load current strategy_config.json -> baseline_config
      2. Run test suite + backtest -> baseline_fitness
      3. For each iteration:
         a. Select random mutation strategy
         b. Apply mutation to config copy, validate ranges
         c. Run kxbtc_backtest with mutated params
         d. Run test suite (fast mode, skip slow tests)
         e. Compute mutated_fitness
         f. KEEP if mutated_fitness > baseline_fitness
            KEEP if equal but simpler (fewer avoid_hours, closer to defaults)
            DISCARD otherwise
         g. Log to evolution_results.tsv
      4. Print final report

    Returns a summary dict with best config, total improvements, etc.
    """
    if not _BACKTEST_AVAILABLE:
        print("[evolver] WARNING: Historical data not found. "
              "Backtest will return empty results.")
        print("[evolver] Run: python scripts/pull_kxbtc_history.py first")

    # Load baseline config
    baseline_config = load_config()
    baseline_iteration = baseline_config.get("iteration", 0)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  SKILL EVOLVER -- {n_iterations} iterations")
        print(f"  Starting config: vol={baseline_config.get('btc_hourly_vol')}, "
              f"edge={baseline_config.get('min_edge_pp')}pp, "
              f"kelly={baseline_config.get('fractional_kelly')}")
        print(f"{'='*60}\n")

    # Compute baseline fitness
    baseline_report, _ = run_experiment(
        baseline_config, [], 0, run_tests=run_tests
    )
    baseline_fitness = baseline_report.total_score
    save_fitness(baseline_report, FITNESS_HISTORY)

    if verbose:
        print(f"  Baseline fitness: {baseline_fitness:.1f}/100\n")

    # Track stats
    n_kept = 0
    n_discarded = 0
    n_crashed = 0
    best_fitness = baseline_fitness
    best_config = copy.deepcopy(baseline_config)

    for i in range(1, n_iterations + 1):
        # Conflict resolution: check if iteration changed since last read
        try:
            current_disk_config = load_config()
            disk_iteration = current_disk_config.get("iteration", 0)
            if disk_iteration != baseline_iteration:
                # Another process (strategy_optimizer) updated config
                baseline_config = current_disk_config
                baseline_iteration = disk_iteration
                # Recompute baseline fitness with new config
                baseline_report, _ = run_experiment(
                    baseline_config, [], 0, run_tests=run_tests
                )
                baseline_fitness = baseline_report.total_score
                if verbose:
                    print(f"  [RELOAD] Config changed externally "
                          f"(iteration {disk_iteration}). "
                          f"New baseline: {baseline_fitness:.1f}/100")
        except Exception:
            pass  # If read fails, continue with current baseline

        # Select and apply mutation
        mutations = select_mutation(baseline_config)
        mutation_desc = "; ".join(str(m) for m in mutations)

        if verbose:
            print(f"  [{i}/{n_iterations}] Trying: {mutation_desc}")

        # Run experiment
        report, experiment = run_experiment(
            baseline_config, mutations, i, run_tests=run_tests,
        )
        experiment.baseline_fitness = baseline_fitness
        experiment.delta = report.total_score - baseline_fitness

        # Decision: keep or discard
        if experiment.status == "crash":
            n_crashed += 1
        elif report.total_score > baseline_fitness:
            experiment.status = "keep"
            n_kept += 1
            baseline_config = apply_mutations(baseline_config, mutations)
            baseline_config["iteration"] = baseline_config.get("iteration", 0) + 1
            baseline_config["source"] = "evolver"
            baseline_config["notes"] = (
                f"evolver iteration {i}: {mutation_desc}"
            )
            baseline_iteration = baseline_config["iteration"]
            save_config(baseline_config)
            baseline_fitness = report.total_score
            if report.total_score > best_fitness:
                best_fitness = report.total_score
                best_config = copy.deepcopy(baseline_config)
        elif abs(report.total_score - baseline_fitness) < 0.01:
            # Equal fitness -- check simplicity
            mutated = apply_mutations(baseline_config, mutations)
            if _config_complexity(mutated) < _config_complexity(baseline_config):
                experiment.status = "keep"
                n_kept += 1
                baseline_config = mutated
                baseline_config["iteration"] = (
                    baseline_config.get("iteration", 0) + 1
                )
                baseline_config["source"] = "evolver"
                baseline_config["notes"] = (
                    f"evolver iteration {i} (simplicity): {mutation_desc}"
                )
                baseline_iteration = baseline_config["iteration"]
                save_config(baseline_config)
            else:
                experiment.status = "discard"
                n_discarded += 1
        else:
            experiment.status = "discard"
            n_discarded += 1

        # Log
        _log_result(experiment, mutations)
        save_fitness(report, FITNESS_HISTORY)

        status_icon = {
            "keep": "+", "discard": "-", "crash": "X",
        }
        if verbose:
            icon = status_icon.get(experiment.status, "?")
            print(f"         [{icon}] {experiment.status.upper()} | "
                  f"fitness: {report.total_score:.1f} "
                  f"(delta: {experiment.delta:+.1f}) | "
                  f"{experiment.duration_seconds:.1f}s")

    # Final report
    if verbose:
        print(f"\n{'='*60}")
        print(f"  EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Iterations:   {n_iterations}")
        print(f"  Kept:         {n_kept}")
        print(f"  Discarded:    {n_discarded}")
        print(f"  Crashed:      {n_crashed}")
        print(f"  Best fitness: {best_fitness:.1f}/100")
        print(f"  Final config: vol={baseline_config.get('btc_hourly_vol')}, "
              f"edge={baseline_config.get('min_edge_pp')}pp, "
              f"kelly={baseline_config.get('fractional_kelly')}")
        print(f"  Results log:  {RESULTS_TSV}")
        print(f"{'='*60}\n")

    return {
        "n_iterations": n_iterations,
        "n_kept": n_kept,
        "n_discarded": n_discarded,
        "n_crashed": n_crashed,
        "best_fitness": best_fitness,
        "best_config": best_config,
        "final_config": baseline_config,
    }


def print_report() -> None:
    """Print summary of evolution results from TSV."""
    if not RESULTS_TSV.exists():
        print("[evolver] No evolution results found.")
        return

    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        print("[evolver] Evolution results file is empty.")
        return

    n_keep = sum(1 for r in rows if r.get("status") == "keep")
    n_discard = sum(1 for r in rows if r.get("status") == "discard")
    n_crash = sum(1 for r in rows if r.get("status") == "crash")

    fitnesses = [
        float(r["mutated_fitness"])
        for r in rows if r.get("mutated_fitness")
    ]
    best = max(fitnesses) if fitnesses else 0

    print(f"\n{'='*60}")
    print(f"  EVOLUTION REPORT ({len(rows)} experiments)")
    print(f"{'='*60}")
    print(f"  Kept:      {n_keep}")
    print(f"  Discarded: {n_discard}")
    print(f"  Crashed:   {n_crash}")
    print(f"  Best fitness: {best:.1f}/100")

    # Show kept mutations
    kept = [r for r in rows if r.get("status") == "keep"]
    if kept:
        print(f"\n  Accepted mutations:")
        for r in kept[-10:]:
            print(f"    #{r['experiment_id']}: {r['param_changed']} "
                  f"({r['old_value']} -> {r['new_value']}) "
                  f"fitness: {r['mutated_fitness']} ({r['delta']:+s})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous self-improving parameter evolution "
                    "for the trading bot."
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=20,
        help="Number of experiments to run (default: 20)",
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run indefinitely until Ctrl+C",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print summary of past evolution results",
    )
    parser.add_argument(
        "--no-tests", action="store_true",
        help="Skip pytest suite (backtest-only, faster)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    if args.report:
        print_report()
        return

    if args.daemon:
        print("[evolver] Running in daemon mode. Press Ctrl+C to stop.")
        iteration_batch = 0
        try:
            while True:
                iteration_batch += 1
                print(f"\n[evolver] Batch {iteration_batch}")
                run_evolution(
                    n_iterations=10,
                    run_tests=not args.no_tests,
                    verbose=not args.quiet,
                )
        except KeyboardInterrupt:
            print("\n[evolver] Stopped by user.")
            print_report()
    else:
        run_evolution(
            n_iterations=args.iterations,
            run_tests=not args.no_tests,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
