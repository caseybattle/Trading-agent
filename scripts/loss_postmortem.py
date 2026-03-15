"""
loss_postmortem.py
5-specialist loss postmortem coordinator for the Kalshi BTC trading bot.

Runs 5 independent specialist analyzers in parallel via ThreadPoolExecutor
whenever trades are resolved. Each specialist analyzes losses from a different
angle. The coordinator merges findings, writes a postmortem report, then calls
strategy_optimizer.run_optimization() with enriched context.

Usage:
    python scripts/loss_postmortem.py              # run postmortem + optimizer
    python scripts/loss_postmortem.py --dry-run    # report only, no writes
    python scripts/loss_postmortem.py --report     # alias for --dry-run
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional

from storage_backend import get_storage

# ---------------------------------------------------------------------------
# Paths — all relative to project root regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent

# Initialize storage backend
_storage = get_storage()

# BTC vol constant (1% per hour, same as kalshi_btc_trader.py)
BTC_HOURLY_VOL = 0.01

# Minimum losses required before running full postmortem
MIN_LOSSES_REQUIRED = 3

# ---------------------------------------------------------------------------
# Postmortem log CSV columns
# ---------------------------------------------------------------------------
POSTMORTEM_LOG_COLS = [
    "timestamp",
    "n_losses",
    "top_finding",
    "vol_adj",
    "edge_adj",
    "worst_strategy",
    "wrong_direction_pct",
]

# ---------------------------------------------------------------------------
# Helper: safe float cast
# ---------------------------------------------------------------------------

def _float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Specialist 1: Vol Analyst
# ---------------------------------------------------------------------------

def analyze_vol(losses: List[Dict]) -> Dict:
    """
    Check whether actual BTC volatility during the market window exceeded
    our assumed BTC_HOURLY_VOL. High actual moves on losses suggest vol
    was systematically underestimated.
    """
    if not losses:
        return {
            "specialist": "vol_analyst",
            "vol_underestimated_pct": 0.0,
            "avg_actual_move": 0.0,
            "recommended_vol_adj": 1.0,
            "summary": "No losses to analyze.",
        }

    try:
        underestimated_count = 0
        actual_moves = []

        for trade in losses:
            minutes_left = _float(trade.get("minutes_left", 60), 60)
            btc_price    = _float(trade.get("btc_price_at_signal", 0))
            range_low    = _float(trade.get("range_low", 0))
            range_high   = _float(trade.get("range_high", 0))

            if btc_price <= 0 or range_low <= 0 or range_high <= 0:
                continue

            range_mid = (range_low + range_high) / 2.0

            # Our predicted 1-sigma move over remaining window
            predicted_sigma = BTC_HOURLY_VOL * math.sqrt(max(minutes_left, 1) / 60.0)

            # Actual % distance of BTC from range midpoint at signal time
            actual_pct_from_mid = abs(btc_price - range_mid) / btc_price

            actual_moves.append(actual_pct_from_mid)

            if actual_pct_from_mid > predicted_sigma:
                underestimated_count += 1

        n = len(actual_moves)
        if n == 0:
            avg_actual  = 0.0
            under_pct   = 0.0
            vol_adj     = 1.0
            summary     = "Insufficient price data in losses to assess vol."
        else:
            avg_actual  = sum(actual_moves) / n
            under_pct   = underestimated_count / n * 100.0
            # Recommend scaling vol assumption up proportionally
            vol_ratio   = avg_actual / max(BTC_HOURLY_VOL, 1e-9)
            vol_adj     = round(max(1.0, vol_ratio), 3)

            if under_pct >= 60:
                summary = (
                    f"Vol UNDERESTIMATED in {under_pct:.0f}% of losses. "
                    f"Avg actual move {avg_actual*100:.2f}% vs assumed "
                    f"{BTC_HOURLY_VOL*100:.2f}%/hr. Recommend x{vol_adj} vol multiplier."
                )
            elif under_pct <= 20:
                summary = (
                    f"BTC barely moved in most losses ({under_pct:.0f}% exceeded sigma). "
                    "Vol not the primary culprit — check timing or edge quality."
                )
            else:
                summary = (
                    f"Vol underestimated in {under_pct:.0f}% of losses. "
                    f"Avg actual move {avg_actual*100:.2f}%. Marginal vol issue."
                )

        return {
            "specialist": "vol_analyst",
            "vol_underestimated_pct": round(under_pct if n > 0 else 0.0, 2),
            "avg_actual_move": round(avg_actual, 6),
            "recommended_vol_adj": vol_adj if n > 0 else 1.0,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "specialist": "vol_analyst",
            "vol_underestimated_pct": 0.0,
            "avg_actual_move": 0.0,
            "recommended_vol_adj": 1.0,
            "summary": f"Specialist error: {exc}",
        }


# ---------------------------------------------------------------------------
# Specialist 2: Timing Analyst
# ---------------------------------------------------------------------------

def analyze_timing(losses: List[Dict]) -> Dict:
    """
    Identify which entry-time buckets (by minutes_left and hour of day)
    are associated with the highest loss density.
    """
    if not losses:
        return {
            "specialist": "timing_analyst",
            "worst_time_bucket": "N/A",
            "worst_hour_utc": -1,
            "early_entry_losses_pct": 0.0,
            "late_entry_losses_pct": 0.0,
            "summary": "No losses to analyze.",
        }

    try:
        buckets = defaultdict(int)
        hour_counts = defaultdict(int)
        early_count = 0  # >120 min
        late_count  = 0  # <15 min

        BUCKET_LABELS = [
            ("<15",    0,   15),
            ("15-30",  15,  30),
            ("30-60",  30,  60),
            ("60-120", 60, 120),
            (">120",  120, 1e9),
        ]

        for trade in losses:
            minutes_left = _float(trade.get("minutes_left", 60), 60)
            ts_str = trade.get("timestamp", "")

            for label, lo, hi in BUCKET_LABELS:
                if lo <= minutes_left < hi:
                    buckets[label] += 1
                    break

            if minutes_left > 120:
                early_count += 1
            if minutes_left < 15:
                late_count += 1

            # Parse hour from timestamp
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    hour_counts[dt.hour] += 1
                except Exception:
                    pass

        n = len(losses)
        worst_bucket = max(buckets, key=buckets.get) if buckets else "N/A"
        worst_hour   = max(hour_counts, key=hour_counts.get) if hour_counts else -1

        early_pct = early_count / n * 100.0
        late_pct  = late_count  / n * 100.0

        if worst_bucket == ">120":
            timing_note = "Most losses from EARLY entries (>120 min) — BTC drifted out over time."
        elif worst_bucket == "<15":
            timing_note = "Most losses from LATE entries (<15 min) — likely adverse selection near settlement."
        else:
            timing_note = f"Worst time bucket is {worst_bucket} minutes remaining."

        hour_note = f" UTC hour {worst_hour} is worst by loss count." if worst_hour >= 0 else ""

        summary = (
            f"{timing_note}{hour_note} "
            f"Early-entry losses: {early_pct:.0f}%, late-entry losses: {late_pct:.0f}%."
        )

        return {
            "specialist": "timing_analyst",
            "worst_time_bucket": worst_bucket,
            "worst_hour_utc": worst_hour,
            "early_entry_losses_pct": round(early_pct, 2),
            "late_entry_losses_pct": round(late_pct, 2),
            "summary": summary,
        }

    except Exception as exc:
        return {
            "specialist": "timing_analyst",
            "worst_time_bucket": "N/A",
            "worst_hour_utc": -1,
            "early_entry_losses_pct": 0.0,
            "late_entry_losses_pct": 0.0,
            "summary": f"Specialist error: {exc}",
        }


# ---------------------------------------------------------------------------
# Specialist 3: Market Intelligence Analyst
# ---------------------------------------------------------------------------

def analyze_market_intelligence(losses: List[Dict], wins: List[Dict]) -> Dict:
    """
    Compare our fair_value vs market_ask on losses vs wins.
    If perceived edge on losses < perceived edge on wins, we took bad setups.
    """
    if not losses:
        return {
            "specialist": "market_intel",
            "avg_perceived_edge_wins": 0.0,
            "avg_perceived_edge_losses": 0.0,
            "market_was_right_pct": 0.0,
            "summary": "No losses to analyze.",
        }

    try:
        def _perceived_edges(trade_list: List[Dict]) -> List[float]:
            edges = []
            for t in trade_list:
                fv  = _float(t.get("fair_value", 0) or t.get("fair_value", 0))
                ask = _float(t.get("market_ask",  0) or t.get("limit_price", 0))
                if fv > 0 and ask > 0:
                    edges.append(fv - ask)
            return edges

        loss_edges = _perceived_edges(losses)
        win_edges  = _perceived_edges(wins)

        avg_loss_edge = sum(loss_edges) / len(loss_edges) if loss_edges else 0.0
        avg_win_edge  = sum(win_edges)  / len(win_edges)  if win_edges  else 0.0

        # "Market was right" = market_ask was already above or near fair_value
        # i.e., we paid more than we thought something was worth (negative edge)
        market_right = sum(1 for e in loss_edges if e <= 0)
        market_right_pct = (market_right / len(loss_edges) * 100.0) if loss_edges else 0.0

        if avg_loss_edge < avg_win_edge * 0.5:
            summary = (
                f"Poor setup quality: avg perceived edge on losses ({avg_loss_edge*100:.1f}pp) "
                f"was less than half of wins ({avg_win_edge*100:.1f}pp). "
                f"Market was already right {market_right_pct:.0f}% of the time."
            )
        elif market_right_pct >= 50:
            summary = (
                f"Market was correct {market_right_pct:.0f}% of loss cases — "
                "we were overconfident in our fair value estimate."
            )
        else:
            summary = (
                f"Edge on losses ({avg_loss_edge*100:.1f}pp) vs wins ({avg_win_edge*100:.1f}pp). "
                f"Market right {market_right_pct:.0f}% — coin-flip quality setups."
            )

        return {
            "specialist": "market_intel",
            "avg_perceived_edge_wins": round(avg_win_edge, 6),
            "avg_perceived_edge_losses": round(avg_loss_edge, 6),
            "market_was_right_pct": round(market_right_pct, 2),
            "summary": summary,
        }

    except Exception as exc:
        return {
            "specialist": "market_intel",
            "avg_perceived_edge_wins": 0.0,
            "avg_perceived_edge_losses": 0.0,
            "market_was_right_pct": 0.0,
            "summary": f"Specialist error: {exc}",
        }


# ---------------------------------------------------------------------------
# Specialist 4: Pattern Matcher
# ---------------------------------------------------------------------------

def analyze_patterns(losses: List[Dict], all_trades: List[Dict]) -> Dict:
    """
    Identify systematic loss clusters by strategy, direction, and fair_value range.
    """
    if not losses:
        return {
            "specialist": "pattern_matcher",
            "worst_strategy": "N/A",
            "worst_strategy_loss_rate": 0.0,
            "loss_clusters": [],
            "summary": "No losses to analyze.",
        }

    try:
        # Loss rate per strategy
        strategy_wins   = defaultdict(int)
        strategy_losses = defaultdict(int)

        for t in all_trades:
            outcome  = t.get("outcome", "").strip().upper()
            strategy = t.get("strategy", "UNKNOWN").strip()
            if outcome == "WIN":
                strategy_wins[strategy] += 1
            elif outcome == "LOSS":
                strategy_losses[strategy] += 1

        all_strategies = set(list(strategy_wins.keys()) + list(strategy_losses.keys()))
        loss_rates = {}
        for s in all_strategies:
            total = strategy_wins[s] + strategy_losses[s]
            if total >= 2:
                loss_rates[s] = strategy_losses[s] / total

        worst_strategy      = max(loss_rates, key=loss_rates.get) if loss_rates else "N/A"
        worst_strategy_rate = loss_rates.get(worst_strategy, 0.0)

        # Loss clusters: group by (strategy, direction)
        cluster_counts = defaultdict(int)
        for t in losses:
            strategy  = t.get("strategy",  "UNKNOWN").strip()
            direction = t.get("direction", "?").strip()
            cluster_counts[(strategy, direction)] += 1

        clusters_sorted = sorted(
            [{"strategy": s, "direction": d, "loss_count": c}
             for (s, d), c in cluster_counts.items()],
            key=lambda x: x["loss_count"],
            reverse=True,
        )
        top_clusters = clusters_sorted[:5]

        # Build summary
        if worst_strategy != "N/A":
            summary = (
                f"Worst strategy: {worst_strategy} "
                f"({worst_strategy_rate*100:.0f}% loss rate). "
            )
        else:
            summary = "Insufficient data per strategy. "

        if top_clusters:
            top = top_clusters[0]
            summary += (
                f"Top loss cluster: {top['strategy']} {top['direction']} "
                f"({top['loss_count']} losses)."
            )

        return {
            "specialist": "pattern_matcher",
            "worst_strategy": worst_strategy,
            "worst_strategy_loss_rate": round(worst_strategy_rate, 4),
            "loss_clusters": top_clusters,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "specialist": "pattern_matcher",
            "worst_strategy": "N/A",
            "worst_strategy_loss_rate": 0.0,
            "loss_clusters": [],
            "summary": f"Specialist error: {exc}",
        }


# ---------------------------------------------------------------------------
# Specialist 5: Counterfactual Analyst
# ---------------------------------------------------------------------------

def analyze_counterfactual(losses: List[Dict], all_trades: List[Dict]) -> Dict:
    """
    For each loss, check if the opposite direction would have won.
    Also count signals that were not acted on (acted_on == MANUAL, outcome empty)
    that settled as wins (inferred from other resolved rows for same ticker).
    """
    if not losses:
        return {
            "specialist": "counterfactual",
            "wrong_direction_pct": 0.0,
            "would_win_if_flipped": 0,
            "missed_opportunities": 0,
            "summary": "No losses to analyze.",
        }

    try:
        # Build a lookup: ticker -> market result (WIN side direction)
        # We infer "correct direction" from winning trades for same ticker
        ticker_win_direction: Dict[str, str] = {}
        for t in all_trades:
            outcome = t.get("outcome", "").strip().upper()
            if outcome == "WIN":
                ticker = t.get("ticker", "")
                direction = t.get("direction", "").strip().upper()
                if ticker:
                    ticker_win_direction[ticker] = direction

        wrong_direction_count = 0
        would_win_count = 0

        for t in losses:
            ticker    = t.get("ticker", "")
            direction = t.get("direction", "").strip().upper()

            # If another trade on the same ticker won with opposite direction
            opp = "NO" if direction == "YES" else "YES"
            correct = ticker_win_direction.get(ticker)
            if correct and correct == opp:
                wrong_direction_count += 1
                would_win_count += 1

        wrong_pct = wrong_direction_count / len(losses) * 100.0

        # Count missed opportunities: acted_on != YES (or == MANUAL) AND outcome is empty
        # These are signals we saw but did not trade
        missed = 0
        for t in all_trades:
            acted   = t.get("acted_on", "").strip().upper()
            outcome = t.get("outcome", "").strip().upper()
            if acted in ("MANUAL", "NO", "") and outcome == "":
                missed += 1

        if wrong_pct >= 40:
            summary = (
                f"Direction error in {wrong_pct:.0f}% of losses — "
                f"{would_win_count} would have won if flipped. "
                "Consider inverting signal logic or adding direction filter."
            )
        elif missed > 0:
            summary = (
                f"Wrong direction in {wrong_pct:.0f}% of losses. "
                f"{missed} untaken signals may have been winners."
            )
        else:
            summary = (
                f"Wrong direction in {wrong_pct:.0f}% of losses. "
                "Direction bias not the dominant issue."
            )

        return {
            "specialist": "counterfactual",
            "wrong_direction_pct": round(wrong_pct, 2),
            "would_win_if_flipped": would_win_count,
            "missed_opportunities": missed,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "specialist": "counterfactual",
            "wrong_direction_pct": 0.0,
            "would_win_if_flipped": 0,
            "missed_opportunities": 0,
            "summary": f"Specialist error: {exc}",
        }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_round_table_report(findings: Dict, n_losses: int) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    border  = "=" * 60

    vol     = findings.get("vol_analyst",      {})
    timing  = findings.get("timing_analyst",   {})
    intel   = findings.get("market_intel",     {})
    pattern = findings.get("pattern_matcher",  {})
    counter = findings.get("counterfactual",   {})

    top_finding = findings.get("top_finding",  "See specialist summaries above.")
    action_note = findings.get("action_taken", "strategy_optimizer.run_optimization() called.")

    print()
    print(border)
    print("         LOSS POSTMORTEM -- ROUND TABLE ASSESSMENT")
    print(border)
    print(f"Losses analyzed: {n_losses}  |  Date: {now_str}")
    print()
    print(f"[VOL ANALYST]      {vol.get('summary', '')}")
    print(f"[TIMING ANALYST]   {timing.get('summary', '')}")
    print(f"[MARKET INTEL]     {intel.get('summary', '')}")
    print(f"[PATTERN MATCHER]  {pattern.get('summary', '')}")
    print(f"[COUNTERFACTUAL]   {counter.get('summary', '')}")
    print()
    print(f"-> Top finding: {top_finding}")
    print(f"-> Action taken: {action_note}")
    print(border)


# ---------------------------------------------------------------------------
# Determine top finding from merged specialist data
# ---------------------------------------------------------------------------

def _determine_top_finding(findings: Dict) -> str:
    """Pick the most impactful issue from the 5 specialist results."""
    vol     = findings.get("vol_analyst",     {})
    timing  = findings.get("timing_analyst",  {})
    intel   = findings.get("market_intel",    {})
    pattern = findings.get("pattern_matcher", {})
    counter = findings.get("counterfactual",  {})

    candidates = []

    vu = _float(vol.get("vol_underestimated_pct", 0))
    if vu >= 60:
        candidates.append((vu, f"Volatility underestimated in {vu:.0f}% of losses"))

    wd = _float(counter.get("wrong_direction_pct", 0))
    if wd >= 40:
        candidates.append((wd, f"Wrong direction in {wd:.0f}% of losses"))

    mr = _float(intel.get("market_was_right_pct", 0))
    if mr >= 50:
        candidates.append((mr, f"Market was already right {mr:.0f}% of loss cases (overconfident)"))

    wsl = _float(pattern.get("worst_strategy_loss_rate", 0)) * 100
    if wsl >= 60:
        ws = pattern.get("worst_strategy", "unknown")
        candidates.append((wsl, f"Strategy '{ws}' has {wsl:.0f}% loss rate"))

    wtb = timing.get("worst_time_bucket", "N/A")
    early = _float(timing.get("early_entry_losses_pct", 0))
    if early >= 50:
        candidates.append((early, f"Early entry (>120 min) accounts for {early:.0f}% of losses"))

    if not candidates:
        return "No dominant single cause — losses are distributed across factors."

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Postmortem log writer
# ---------------------------------------------------------------------------

def _append_postmortem_log(
    n_losses: int,
    top_finding: str,
    findings: Dict,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    vol     = findings.get("vol_analyst",     {})
    intel   = findings.get("market_intel",    {})
    pattern = findings.get("pattern_matcher", {})
    counter = findings.get("counterfactual",  {})

    row = {
        "timestamp":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_losses":           n_losses,
        "top_finding":        top_finding,
        "vol_adj":            vol.get("recommended_vol_adj", 1.0),
        "edge_adj":           round(_float(intel.get("avg_perceived_edge_losses", 0)), 6),
        "worst_strategy":     pattern.get("worst_strategy", "N/A"),
        "wrong_direction_pct": counter.get("wrong_direction_pct", 0.0),
    }

    _storage.append_csv("backtest/postmortem_log.csv", row, fieldnames=POSTMORTEM_LOG_COLS)


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

def run_postmortem(dry_run: bool = False) -> Dict:
    """
    Run all 5 specialists in parallel, merge findings, write report,
    then call strategy_optimizer.run_optimization().

    Parameters
    ----------
    dry_run : bool
        If True, print the report but do NOT write any files or call the optimizer.

    Returns
    -------
    dict
        Merged findings from all specialists.
    """
    # -----------------------------------------------------------------------
    # 1. Load resolved trades via strategy_optimizer
    # -----------------------------------------------------------------------
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        from strategy_optimizer import load_resolved_trades
    except ImportError as exc:
        print(f"[postmortem] Cannot import strategy_optimizer: {exc}")
        return {}

    all_trades = load_resolved_trades()

    # -----------------------------------------------------------------------
    # 2. Filter losses
    # -----------------------------------------------------------------------
    losses = [t for t in all_trades if t.get("outcome", "").strip().upper() == "LOSS"]
    wins   = [t for t in all_trades if t.get("outcome", "").strip().upper() == "WIN"]
    n_losses = len(losses)

    # -----------------------------------------------------------------------
    # 3. Guard: too few losses
    # -----------------------------------------------------------------------
    if n_losses < MIN_LOSSES_REQUIRED:
        print(
            f"[postmortem] Insufficient losses for postmortem "
            f"({n_losses} found, need {MIN_LOSSES_REQUIRED})."
        )
        if not dry_run:
            _call_optimizer(dry_run=False)
        return {}

    # -----------------------------------------------------------------------
    # 4. Run 5 specialists in parallel
    # -----------------------------------------------------------------------
    specialist_results: Dict[str, Dict] = {}

    def _run_vol():
        return "vol_analyst", analyze_vol(losses)

    def _run_timing():
        return "timing_analyst", analyze_timing(losses)

    def _run_intel():
        return "market_intel", analyze_market_intelligence(losses, wins)

    def _run_patterns():
        return "pattern_matcher", analyze_patterns(losses, all_trades)

    def _run_counter():
        return "counterfactual", analyze_counterfactual(losses, all_trades)

    tasks = [_run_vol, _run_timing, _run_intel, _run_patterns, _run_counter]

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn): fn.__name__ for fn in tasks}
        for future in as_completed(futures):
            try:
                key, result = future.result()
                specialist_results[key] = result
            except Exception as exc:
                fn_name = futures[future]
                print(f"[postmortem] Specialist {fn_name} raised: {exc}")

    # -----------------------------------------------------------------------
    # 5. Merge all findings
    # -----------------------------------------------------------------------
    merged = dict(specialist_results)
    top_finding = _determine_top_finding(merged)
    merged["top_finding"]  = top_finding
    merged["n_losses"]     = n_losses
    merged["timestamp"]    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    merged["action_taken"] = "strategy_optimizer.run_optimization() called." if not dry_run else "Dry run — no action taken."

    # -----------------------------------------------------------------------
    # 6. Write postmortem JSON
    # -----------------------------------------------------------------------
    if not dry_run:
        _storage.write_json("backtest/loss_postmortem.json", merged)
        print(f"[postmortem] Report written to backtest/loss_postmortem.json")

    # -----------------------------------------------------------------------
    # 7. Append to postmortem log CSV
    # -----------------------------------------------------------------------
    _append_postmortem_log(n_losses, top_finding, merged, dry_run)

    # -----------------------------------------------------------------------
    # 8. Print human-readable round table report
    # -----------------------------------------------------------------------
    _print_round_table_report(merged, n_losses)

    # -----------------------------------------------------------------------
    # 9. Call strategy optimizer
    # -----------------------------------------------------------------------
    if not dry_run:
        _call_optimizer(dry_run=False)

    # -----------------------------------------------------------------------
    # 10. Return merged findings
    # -----------------------------------------------------------------------
    return merged


def _call_optimizer(dry_run: bool = False) -> None:
    """Call strategy_optimizer.run_optimization(), handling import failures."""
    try:
        sys.path.insert(0, str(_SCRIPT_DIR))
        from strategy_optimizer import run_optimization
        print("[postmortem] Calling strategy_optimizer.run_optimization()...")
        run_optimization(dry_run=dry_run)
    except Exception as exc:
        print(f"[postmortem] strategy_optimizer.run_optimization() failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Loss postmortem coordinator: runs 5 specialist analyzers in parallel "
            "then calls strategy_optimizer."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report only; do not write files or call the optimizer.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Alias for --dry-run.",
    )
    args = parser.parse_args(argv)

    dry = args.dry_run or args.report
    run_postmortem(dry_run=dry)


if __name__ == "__main__":
    main()
