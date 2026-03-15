"""
run_test_suite.py -- CLI orchestrator for the full test suite + fitness score.

Runs pytest, optionally runs kxbtc_backtest, computes composite fitness,
and saves to backtest/fitness_history.jsonl.

Usage:
    python scripts/run_test_suite.py                    # pytest only
    python scripts/run_test_suite.py --with-backtest    # pytest + backtest
    python scripts/run_test_suite.py --slow              # include slow tests
    python scripts/run_test_suite.py --bankroll 25       # custom bankroll
    python scripts/run_test_suite.py --with-backtest --slow --bankroll 10
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

BACKTEST_DIR = PROJECT_ROOT / "backtest"
FITNESS_HISTORY = BACKTEST_DIR / "fitness_history.jsonl"


def run_tests(include_slow: bool = False) -> dict:
    """
    Run pytest programmatically and return results dict.

    Parameters
    ----------
    include_slow : bool
        If True, include tests marked @pytest.mark.slow.
    """
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--timeout=60"]
    if not include_slow:
        cmd.extend(["-m", "not slow"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": 0, "failed": 0, "errors": 0, "total": 0,
            "safeguard_passed": True, "returncode": -1,
            "output": "pytest timed out after 300s",
        }

    output = result.stdout + result.stderr
    lines = output.strip().split("\n")
    summary = lines[-1] if lines else ""

    passed = failed = errors = 0
    m = re.search(r"(\d+) passed", summary)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", summary)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", summary)
    if m:
        errors = int(m.group(1))

    safeguard_passed = (
        "FAILED" not in output or "test_risk_safeguards" not in output
    )

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": passed + failed + errors,
        "safeguard_passed": safeguard_passed,
        "returncode": result.returncode,
        "output": output[-2000:],  # Last 2000 chars for debugging
    }


def run_backtest_evaluation(bankroll: float = 10.0) -> dict:
    """Run kxbtc_backtest and return results."""
    try:
        from kxbtc_backtest import run_backtest
        return run_backtest(bankroll=bankroll)
    except Exception as e:
        print(f"[WARN] Backtest failed: {e}")
        return {}


def compute_and_report(
    with_backtest: bool = False,
    include_slow: bool = False,
    bankroll: float = 10.0,
    save: bool = True,
) -> dict:
    """
    Run full evaluation pipeline: pytest + optional backtest + fitness score.

    Returns the fitness report dict.
    """
    from fitness_scorer import compute_fitness, save_fitness
    from strategy_optimizer import load_config

    config = load_config()

    print("=" * 60)
    print("  SELF-EVALUATION TEST SUITE")
    print("=" * 60)

    # Run tests
    slow_label = " (including slow)" if include_slow else ""
    print(f"\n  Running pytest{slow_label}...")
    test_results = run_tests(include_slow=include_slow)
    print(f"  Tests: {test_results['passed']} passed, "
          f"{test_results['failed']} failed, "
          f"{test_results['errors']} errors")

    # Optionally run backtest
    bt_results = {}
    if with_backtest:
        print(f"\n  Running backtest (bankroll=${bankroll:.2f})...")
        bt_results = run_backtest_evaluation(bankroll=bankroll)
        oos = bt_results.get("out_of_sample", {})
        if oos:
            print(f"  OOS: {oos.get('n_trades', 0)} trades, "
                  f"return={oos.get('return_pct', 0):+.1f}%, "
                  f"Sharpe={oos.get('sharpe', 'N/A')}")
    else:
        print("\n  Backtest: skipped (use --with-backtest to include)")

    # Compute fitness
    report = compute_fitness(bt_results, test_results)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  FITNESS SCORE: {report.total_score:.1f} / 100")
    if report.safeguard_veto:
        print(f"  *** SAFEGUARD VETO: Score forced to 0 ***")
    print(f"{'='*60}")

    from fitness_scorer import WEIGHTS
    for name, value in report.components.items():
        weight = WEIGHTS.get(name, 0) * 100
        scaled = value * weight
        print(f"  {name:<25} {value:.3f}  x {weight:4.0f}% = {scaled:6.2f}")

    print(f"  {'':25} {'':5}         ------")
    print(f"  {'Total':25} {'':5}         {report.total_score:6.1f}")

    # Save to history
    if save:
        save_fitness(report, FITNESS_HISTORY)
        print(f"\n  Saved to: {FITNESS_HISTORY}")

    return report.to_dict()


def show_last_report():
    """Print the most recent fitness evaluation."""
    if not FITNESS_HISTORY.exists():
        print("[INFO] No fitness history found. Run the full suite first.")
        return

    with open(FITNESS_HISTORY) as f:
        lines = f.readlines()

    if not lines:
        print("[INFO] Fitness history is empty.")
        return

    last = json.loads(lines[-1])
    score = last.get("total_score", last.get("score", 0))
    print(f"\n  Last fitness score: {score:.1f}/100")
    print(f"  Timestamp: {last.get('timestamp', 'unknown')}")
    print(f"  Safeguard veto: {last.get('safeguard_veto', False)}")
    for name, value in last.get("components", {}).items():
        print(f"    {name}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Run self-evaluation test suite and compute fitness score."
    )
    parser.add_argument(
        "--with-backtest", action="store_true",
        help="Also run kxbtc_backtest (slower but more complete)",
    )
    parser.add_argument(
        "--slow", action="store_true",
        help="Include tests marked @pytest.mark.slow",
    )
    parser.add_argument(
        "--bankroll", type=float, default=10.0,
        help="Backtest starting bankroll (default: 10.0)",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Show last fitness score without running anything",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to fitness_history.jsonl",
    )

    args = parser.parse_args()

    if args.report_only:
        show_last_report()
        return

    compute_and_report(
        with_backtest=args.with_backtest,
        include_slow=args.slow,
        bankroll=args.bankroll,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
