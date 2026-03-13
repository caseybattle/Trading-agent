"""
Correlation-Adjusted Kelly Criterion for Prediction Market Position Sizing
Extends Ed Thorp's Kelly Criterion with portfolio-aware correlation penalty.

Original Kelly formula:  f* = (bp - q) / b
Correlation-adjusted:    f_adj = f_raw * max(0.1, 1 - correlation_penalty)

Usage:
    from kelly_calculator import (
        kelly_criterion, correlation_adjusted_kelly,
        portfolio_kelly_check, calculate_position_size
    )

    # Standard Kelly for a single trade
    result = kelly_criterion(win_rate=0.60, avg_win=1.8, avg_loss=1.0)

    # Portfolio-aware sizing (prevents overconcentration in correlated markets)
    adj = correlation_adjusted_kelly(
        p_true=0.65,
        b=1.8,
        open_positions={'POLY-XYZ': 0.04},
        correlation_row={'POLY-XYZ': 0.72},
        market_id='POLY-ABC',
        category='crypto',
    )
    print(f"Adjusted fraction: {adj.kelly_adjusted:.2%}")
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


# ─── Base Kelly ──────────────────────────────────────────────────────────────

@dataclass
class KellyResult:
    """Container for standard Kelly Criterion results."""
    full_kelly:    float
    half_kelly:    float
    third_kelly:   float
    quarter_kelly: float
    recommended:   float   # half Kelly by default (Thorp's advice)
    edge:          float   # expected value per unit risked

    def __str__(self) -> str:
        return (
            f"Kelly Results:\n"
            f"  Full Kelly:          {self.full_kelly:.2%}\n"
            f"  Half Kelly:          {self.half_kelly:.2%}\n"
            f"  Third Kelly:         {self.third_kelly:.2%}\n"
            f"  Quarter Kelly:       {self.quarter_kelly:.2%}\n"
            f"  Recommended (Half):  {self.recommended:.2%}\n"
            f"  Edge:                {self.edge:.4f}"
        )


def kelly_criterion(win_rate: float,
                    avg_win: float,
                    avg_loss: float,
                    fractional: float = 0.5) -> KellyResult:
    """
    Calculate optimal position size using Kelly Criterion.

    Formula: f* = (bp - q) / b
        b = avg_win / avg_loss  (odds received on a win)
        p = win_rate            (probability of winning)
        q = 1 - p               (probability of losing)

    Args:
        win_rate:   Probability of winning (0.0 – 1.0)
        avg_win:    Average win as multiple of risk (e.g., 2.0 for 2:1 R:R)
        avg_loss:   Average loss in same units (typically 1.0)
        fractional: Fraction of full Kelly used as 'recommended' (default 0.5)

    Returns:
        KellyResult with full/half/third/quarter fractions and edge.
    """
    if not (0 < win_rate < 1):
        raise ValueError("win_rate must be strictly between 0 and 1")
    if avg_loss <= 0:
        raise ValueError("avg_loss must be positive")

    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p

    edge       = p * avg_win - q * avg_loss
    full_kelly = max(0.0, (b * p - q) / b) if b != 0 else 0.0

    return KellyResult(
        full_kelly    = full_kelly,
        half_kelly    = full_kelly * 0.5,
        third_kelly   = full_kelly * (1 / 3),
        quarter_kelly = full_kelly * 0.25,
        recommended   = full_kelly * fractional,
        edge          = edge,
    )


def calculate_position_size(account_balance: float,
                             kelly_fraction: float,
                             stop_loss_pct: float,
                             leverage: float = 1.0) -> dict:
    """
    Convert a Kelly fraction into a concrete position size.

    Args:
        account_balance: Total account equity in currency units.
        kelly_fraction:  Kelly percentage (e.g., 0.02 → risk 2% of account).
        stop_loss_pct:   Stop loss as fraction of position (e.g., 0.01 → 1%).
        leverage:        Leverage multiplier (1.0 for spot).

    Returns:
        Dict with risk_amount, position_size, adjusted_position, max_loss.
    """
    if stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be positive")

    risk_amount       = account_balance * kelly_fraction
    position_size     = risk_amount / stop_loss_pct
    adjusted_position = position_size / leverage

    return {
        "account_balance":   account_balance,
        "kelly_fraction":    kelly_fraction,
        "risk_amount":       risk_amount,
        "stop_loss_pct":     stop_loss_pct,
        "position_size":     position_size,
        "leverage":          leverage,
        "adjusted_position": adjusted_position,
        "max_loss":          risk_amount,
    }


def kelly_from_history(trades: List[float],
                       fractional: float = 0.5) -> KellyResult:
    """
    Derive Kelly sizing from a list of historical trade returns.

    Args:
        trades:     List of returns as decimals (e.g., 0.02 win, -0.01 loss).
        fractional: Fraction of full Kelly to use as 'recommended'.

    Returns:
        KellyResult calibrated to historical performance.
    """
    if len(trades) < 10:
        raise ValueError("Need at least 10 trades for a reliable Kelly estimate")

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]

    win_rate = len(wins) / len(trades)
    avg_win  = float(np.mean(wins))  if wins   else 0.0
    avg_loss = float(abs(np.mean(losses))) if losses else 0.01

    # Express in R-multiples relative to average loss
    avg_win_r  = avg_win / avg_loss if avg_loss > 0 else avg_win / 0.01
    avg_loss_r = 1.0

    return kelly_criterion(win_rate, avg_win_r, avg_loss_r, fractional)


def simulate_kelly_growth(initial_capital: float,
                          win_rate: float,
                          avg_win: float,
                          avg_loss: float,
                          num_trades: int = 100,
                          fractional: float = 0.5) -> np.ndarray:
    """
    Simulate capital growth using Kelly position sizing.

    Args:
        initial_capital: Starting equity.
        win_rate:        Probability of a winning trade.
        avg_win:         Average win (R-multiple).
        avg_loss:        Average loss (R-multiple).
        num_trades:      Number of trades to simulate.
        fractional:      Kelly fraction to use.

    Returns:
        Array of shape (num_trades + 1,) with capital values over time.
    """
    result = kelly_criterion(win_rate, avg_win, avg_loss, fractional)
    f = result.recommended

    capital    = np.zeros(num_trades + 1)
    capital[0] = initial_capital

    for i in range(1, num_trades + 1):
        if np.random.random() < win_rate:
            trade_return = f * avg_win
        else:
            trade_return = -f * avg_loss
        capital[i] = capital[i - 1] * (1.0 + trade_return)

    return capital


# ─── Correlation-Adjusted Kelly ──────────────────────────────────────────────

@dataclass
class CorrelationAdjustedKellyResult:
    """
    Kelly result with cross-market correlation penalty applied.

    Extends KellyResult with fields that show how the correlation adjustment
    was computed — useful for dashboard reporting and audit trails.
    """
    # Standard Kelly fields (mirrors KellyResult)
    full_kelly:    float
    half_kelly:    float
    third_kelly:   float
    quarter_kelly: float
    recommended:   float
    edge:          float

    # Correlation-adjustment fields
    kelly_raw:           float   # raw Kelly before correlation penalty
    correlation_penalty: float   # weighted sum of correlated exposure
    kelly_adjusted:      float   # final fraction after adjustment

    # Metadata
    market_id:  str
    category:   str

    def __str__(self) -> str:
        return (
            f"Correlation-Adjusted Kelly [{self.market_id}]:\n"
            f"  Base Kelly (raw):       {self.kelly_raw:.2%}\n"
            f"  Correlation Penalty:    {self.correlation_penalty:.4f}\n"
            f"  Adjusted Kelly:         {self.kelly_adjusted:.2%}\n"
            f"  Edge:                   {self.edge:.4f}\n"
            f"  Category:               {self.category}"
        )


def correlation_adjusted_kelly(p_true: float,
                                b: float,
                                open_positions: Dict[str, float],
                                correlation_row: Dict[str, float],
                                market_id: str = "",
                                category: str = "",
                                fractional: float = 0.25) -> CorrelationAdjustedKellyResult:
    """
    Compute Kelly fraction penalised by portfolio correlation.

    Formula (from plan):
        kelly_raw         = (p_true * b - (1 - p_true)) / b
        correlation_penalty = sum(open_positions[m] * correlation_row[m]
                                  for m in open_positions)
        kelly_adjusted    = kelly_raw * max(0.1, 1 - correlation_penalty)

    The penalty shrinks the Kelly fraction whenever we hold positions in
    markets that move with the candidate market.  At full correlation
    (penalty → 1.0) the adjusted fraction floors at 10% of raw Kelly.

    Args:
        p_true:          Our estimated true probability of YES outcome (0–1).
        b:               Odds = (1 - p_market) / p_market — what the market
                         pays per unit risked.  Alternatively pass avg_win/avg_loss.
        open_positions:  {market_id → current Kelly fraction deployed}.
        correlation_row: {market_id → correlation with candidate market} (–1 to 1).
        market_id:       Identifier of the candidate market (for logging).
        category:        Market category (politics / crypto / sports / …).
        fractional:      Baseline Kelly fraction (default 0.25 — conservative
                         for prediction markets per risk policy).

    Returns:
        CorrelationAdjustedKellyResult with full audit trail.
    """
    if not (0 < p_true < 1):
        raise ValueError("p_true must be strictly between 0 and 1")
    if b <= 0:
        raise ValueError("b (odds) must be positive")

    q = 1.0 - p_true

    # Standard Kelly components
    edge      = p_true * b - q
    full_raw  = max(0.0, (p_true * b - q) / b)

    # Apply fractional multiplier for recommended size
    kelly_raw = full_raw * fractional

    # Correlation penalty: weighted sum of existing exposure × correlation
    penalty = 0.0
    for mkt_id, fraction_deployed in open_positions.items():
        corr = correlation_row.get(mkt_id, 0.0)
        penalty += abs(fraction_deployed) * abs(corr)   # abs() → no sign cancellation

    kelly_adjusted = kelly_raw * max(0.1, 1.0 - penalty)

    return CorrelationAdjustedKellyResult(
        full_kelly          = full_raw,
        half_kelly          = full_raw * 0.5,
        third_kelly         = full_raw * (1 / 3),
        quarter_kelly       = full_raw * 0.25,
        recommended         = kelly_raw,
        edge                = edge,
        kelly_raw           = kelly_raw,
        correlation_penalty = penalty,
        kelly_adjusted      = kelly_adjusted,
        market_id           = market_id,
        category            = category,
    )


# ─── Portfolio Risk Gate ──────────────────────────────────────────────────────

_DEFAULT_RISK_RULES = {
    "max_per_market_pct":   0.05,   # 5%  of bankroll per market
    "max_per_category_pct": 0.15,   # 15% of bankroll per category
    "max_total_exposure":   0.40,   # 40% total correlation-adjusted exposure
    "min_edge":             0.05,   # 5 percentage-point minimum edge
    "fractional_kelly":     0.25,   # baseline Kelly fraction
    "max_kelly":            0.50,   # hard cap on any single position
}


def portfolio_kelly_check(candidate_market_id: str,
                          candidate_category: str,
                          p_true: float,
                          p_market: float,
                          open_positions: Dict[str, float],
                          category_map: Dict[str, str],
                          correlation_matrix: Dict[str, Dict[str, float]],
                          bankroll: float,
                          risk_rules: Optional[Dict] = None) -> Dict:
    """
    Full risk gate for a candidate trade:
        1. Edge check (p_true - p_market > min_edge)
        2. Per-market cap (5% bankroll)
        3. Per-category cap (15% bankroll)
        4. Total correlation-adjusted exposure cap (40%)
        5. Return approved size with full audit trail

    Args:
        candidate_market_id:  Market ID of candidate trade.
        candidate_category:   Category string (politics / crypto / …).
        p_true:               Our calibrated probability estimate.
        p_market:             Current market price (implied probability).
        open_positions:       {market_id → Kelly fraction currently deployed}.
        category_map:         {market_id → category} for all open positions.
        correlation_matrix:   {market_id → {market_id → correlation}}.
        bankroll:             Total account equity.
        risk_rules:           Override default risk parameters.

    Returns:
        Dict with keys:
            approved       (bool)
            kelly_fraction (float) — adjusted size to deploy
            reason         (str)   — approval or rejection rationale
            checks         (dict)  — per-rule pass/fail details
    """
    rules = {**_DEFAULT_RISK_RULES, **(risk_rules or {})}

    checks: Dict[str, dict] = {}

    # ── 1. Edge check ────────────────────────────────────────────────────────
    edge = p_true - p_market
    checks["edge"] = {
        "value":    round(edge, 4),
        "required": rules["min_edge"],
        "pass":     edge >= rules["min_edge"],
    }

    # ── 2. Per-market cap ────────────────────────────────────────────────────
    checks["per_market_cap"] = {
        "limit": rules["max_per_market_pct"],
        "pass":  True,   # will be updated after Kelly computation
    }

    # ── 3. Per-category exposure ─────────────────────────────────────────────
    cat_exposure = sum(
        frac for mkt, frac in open_positions.items()
        if category_map.get(mkt) == candidate_category
    )
    checks["category_cap"] = {
        "category":       candidate_category,
        "current":        round(cat_exposure, 4),
        "limit":          rules["max_per_category_pct"],
        "remaining":      round(rules["max_per_category_pct"] - cat_exposure, 4),
        "pass":           cat_exposure < rules["max_per_category_pct"],
    }

    # ── 4. Total correlation-adjusted exposure ───────────────────────────────
    total_exposure = sum(abs(f) for f in open_positions.values())
    checks["total_exposure"] = {
        "current": round(total_exposure, 4),
        "limit":   rules["max_total_exposure"],
        "pass":    total_exposure < rules["max_total_exposure"],
    }

    # Fail fast before computing Kelly
    hard_fails = [k for k, v in checks.items() if not v["pass"]]
    if hard_fails:
        return {
            "approved":       False,
            "kelly_fraction": 0.0,
            "reason":         f"Failed: {', '.join(hard_fails)}",
            "checks":         checks,
        }

    # ── 5. Correlation-adjusted Kelly ────────────────────────────────────────
    if p_market <= 0 or p_market >= 1:
        return {
            "approved":       False,
            "kelly_fraction": 0.0,
            "reason":         "Invalid market price (must be 0 < p_market < 1)",
            "checks":         checks,
        }

    b = (1.0 - p_market) / p_market   # implied odds from market price

    correlation_row = correlation_matrix.get(candidate_market_id, {})

    adj_result = correlation_adjusted_kelly(
        p_true          = p_true,
        b               = b,
        open_positions  = open_positions,
        correlation_row = correlation_row,
        market_id       = candidate_market_id,
        category        = candidate_category,
        fractional      = rules["fractional_kelly"],
    )

    # Apply hard caps
    kelly_fraction = min(
        adj_result.kelly_adjusted,
        rules["max_per_market_pct"],
        rules["max_kelly"],
        rules["max_per_category_pct"] - cat_exposure,       # category headroom
        rules["max_total_exposure"]   - total_exposure,     # total headroom
    )
    kelly_fraction = max(0.0, kelly_fraction)

    checks["per_market_cap"]["value"] = round(kelly_fraction, 4)
    checks["per_market_cap"]["pass"]  = kelly_fraction > 0

    checks["kelly"] = {
        "raw":               round(adj_result.kelly_raw, 4),
        "correlation_penalty": round(adj_result.correlation_penalty, 4),
        "adjusted":          round(adj_result.kelly_adjusted, 4),
        "after_caps":        round(kelly_fraction, 4),
        "edge":              round(adj_result.edge, 4),
    }

    if kelly_fraction <= 0:
        return {
            "approved":       False,
            "kelly_fraction": 0.0,
            "reason":         "Kelly fraction reduced to zero after caps",
            "checks":         checks,
        }

    risk_dollars = bankroll * kelly_fraction
    return {
        "approved":       True,
        "kelly_fraction": round(kelly_fraction, 6),
        "risk_dollars":   round(risk_dollars, 2),
        "reason":         (
            f"Approved: edge={edge:.2%}, kelly={kelly_fraction:.2%} "
            f"(corr_penalty={adj_result.correlation_penalty:.3f})"
        ),
        "checks":         checks,
        "adj_result":     adj_result,
    }


# ─── CLI Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("PREDICTION MARKET KELLY CALCULATOR — CORRELATION-ADJUSTED SIZING")
    print("=" * 70)

    # ── Example 1: Standard Kelly for a Polymarket politics market ────────────
    print("\n1.  Standard Kelly — Political Market (55% win, 1.8:1 payout)")
    r = kelly_criterion(win_rate=0.55, avg_win=1.8, avg_loss=1.0)
    print(r)

    # ── Example 2: Correlation-adjusted sizing ────────────────────────────────
    print("\n" + "=" * 70)
    print("2.  Correlation-Adjusted Kelly — Crypto Election Market")
    print("=" * 70)

    # Suppose we have two open positions in correlated crypto markets
    open_pos = {
        "POLY-BTC-50K": 0.04,   # 4% of bankroll in BTC-$50k market
        "POLY-ETH-3K":  0.03,   # 3% in ETH-$3k market
    }
    # Correlation of our candidate market with existing positions
    corr_row = {
        "POLY-BTC-50K": 0.82,   # highly correlated (both crypto)
        "POLY-ETH-3K":  0.71,
    }

    adj = correlation_adjusted_kelly(
        p_true          = 0.65,      # we estimate 65% probability
        b               = 1.857,     # market prices at 35¢ → odds = 0.65/0.35
        open_positions  = open_pos,
        correlation_row = corr_row,
        market_id       = "POLY-SOL-200",
        category        = "crypto",
        fractional      = 0.25,
    )
    print(adj)

    # ── Example 3: Full portfolio risk gate ───────────────────────────────────
    print("\n" + "=" * 70)
    print("3.  Portfolio Risk Gate — Full Approval Flow")
    print("=" * 70)

    all_positions = {
        "POLY-BTC-50K":    0.04,
        "POLY-ETH-3K":     0.03,
        "KALSHI-FED-PAUSE": 0.05,   # at cap already
    }
    cat_lookup = {
        "POLY-BTC-50K":    "crypto",
        "POLY-ETH-3K":     "crypto",
        "KALSHI-FED-PAUSE":"economics",
    }
    corr_matrix = {
        "POLY-SOL-200": {
            "POLY-BTC-50K":    0.82,
            "POLY-ETH-3K":     0.71,
            "KALSHI-FED-PAUSE":0.12,
        }
    }

    result = portfolio_kelly_check(
        candidate_market_id = "POLY-SOL-200",
        candidate_category  = "crypto",
        p_true              = 0.65,
        p_market            = 0.52,
        open_positions      = all_positions,
        category_map        = cat_lookup,
        correlation_matrix  = corr_matrix,
        bankroll            = 10_000,
    )

    print(f"\nApproved:       {result['approved']}")
    print(f"Reason:         {result['reason']}")
    if result["approved"]:
        print(f"Kelly Fraction: {result['kelly_fraction']:.2%}")
        print(f"Risk $:         ${result['risk_dollars']:,.2f}")
    print("\nDetailed Checks:")
    for check_name, vals in result["checks"].items():
        status = "✓" if vals.get("pass", True) else "✗"
        print(f"  [{status}] {check_name}: {vals}")

    # ── Example 4: Simulate capital growth ────────────────────────────────────
    print("\n" + "=" * 70)
    print("4.  Capital Growth Simulation (100 trades, Quarter-Kelly)")
    print("=" * 70)
    np.random.seed(42)
    trajectory = simulate_kelly_growth(
        initial_capital = 10_000,
        win_rate        = 0.55,
        avg_win         = 1.8,
        avg_loss        = 1.0,
        num_trades      = 100,
        fractional      = 0.25,
    )
    print(f"  Start:      ${trajectory[0]:>10,.2f}")
    print(f"  End:        ${trajectory[-1]:>10,.2f}")
    print(f"  Peak:       ${trajectory.max():>10,.2f}")
    print(f"  Trough:     ${trajectory.min():>10,.2f}")
    print(f"  Return:     {(trajectory[-1] / trajectory[0] - 1):>10.1%}")
    print(f"  Max DD:     {((trajectory / np.maximum.accumulate(trajectory)) - 1).min():.1%}")
