"""Unit tests for Kelly sizing in kalshi_btc_trader.py and kelly_calculator.py."""
import pytest
import numpy as np

from scripts.kalshi_btc_trader import kelly_fraction, FRACTIONAL_KELLY
from scripts.kelly_calculator import (
    kelly_criterion,
    correlation_adjusted_kelly,
    portfolio_kelly_check,
    calculate_position_size,
    KellyResult,
    CorrelationAdjustedKellyResult,
)


# ---------------------------------------------------------------------------
# kelly_fraction() from kalshi_btc_trader.py
# ---------------------------------------------------------------------------

class TestKellyFractionTrader:
    """Tests for kelly_fraction() -- the simple YES-bet sizing function."""

    def test_positive_edge_returns_positive(self):
        """When fair_value > market_price, fraction should be > 0."""
        frac = kelly_fraction(0.70, 0.50)
        assert frac > 0.0

    def test_no_edge_returns_zero(self):
        """When fair_value <= market_price, fraction should be 0."""
        frac = kelly_fraction(0.40, 0.50)
        assert frac == 0.0

    def test_equal_value_returns_zero(self):
        """When fair_value == market_price, fraction should be 0."""
        frac = kelly_fraction(0.50, 0.50)
        assert frac == 0.0

    def test_never_exceeds_half(self):
        """Kelly fraction should never exceed 0.5 (hard cap)."""
        frac = kelly_fraction(0.99, 0.01)
        assert frac <= 0.5

    def test_invalid_market_price_zero(self):
        """market_price = 0 should return 0."""
        assert kelly_fraction(0.70, 0.0) == 0.0

    def test_invalid_market_price_one(self):
        """market_price = 1 should return 0."""
        assert kelly_fraction(0.70, 1.0) == 0.0

    def test_fractional_kelly_applied(self):
        """Result should be FRACTIONAL_KELLY (0.25x) of full Kelly."""
        # fair=0.80, ask=0.50: b=1.0, full Kelly = (0.8*1 - 0.2)/1 = 0.6
        # Fractional = 0.6 * 0.25 = 0.15
        frac = kelly_fraction(0.80, 0.50)
        assert 0.10 < frac < 0.20, f"Expected ~0.15 (quarter Kelly), got {frac}"

    def test_various_edge_sizes(self):
        """Larger edges should produce larger fractions."""
        frac_small = kelly_fraction(0.60, 0.50)
        frac_large = kelly_fraction(0.85, 0.50)
        assert frac_large > frac_small, "Larger edge should produce larger fraction"

    def test_high_ask_small_fraction(self):
        """When ask is high (near fair), fraction should be small."""
        frac = kelly_fraction(0.55, 0.50)
        assert 0.0 < frac < 0.10


# ---------------------------------------------------------------------------
# kelly_criterion() from kelly_calculator.py
# ---------------------------------------------------------------------------

class TestKellyCriterion:
    """Tests for kelly_criterion() -- standard Kelly formula."""

    def test_basic_positive_edge(self):
        result = kelly_criterion(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
        assert isinstance(result, KellyResult)
        assert result.full_kelly > 0
        assert result.edge > 0

    def test_half_kelly_is_half(self):
        result = kelly_criterion(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
        assert abs(result.half_kelly - result.full_kelly * 0.5) < 1e-10

    def test_quarter_kelly_is_quarter(self):
        result = kelly_criterion(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
        assert abs(result.quarter_kelly - result.full_kelly * 0.25) < 1e-10

    def test_recommended_uses_fractional(self):
        """recommended should equal full_kelly * fractional."""
        result = kelly_criterion(win_rate=0.60, avg_win=2.0, avg_loss=1.0, fractional=0.3)
        assert abs(result.recommended - result.full_kelly * 0.3) < 1e-10

    def test_invalid_win_rate_zero(self):
        with pytest.raises(ValueError):
            kelly_criterion(win_rate=0.0, avg_win=2.0, avg_loss=1.0)

    def test_invalid_win_rate_one(self):
        with pytest.raises(ValueError):
            kelly_criterion(win_rate=1.0, avg_win=2.0, avg_loss=1.0)

    def test_bad_odds_returns_zero_kelly(self):
        """Low win_rate with even odds should give full_kelly=0."""
        result = kelly_criterion(win_rate=0.30, avg_win=1.0, avg_loss=1.0)
        assert result.full_kelly == 0.0
        assert result.edge < 0

    def test_win_rate_above_breakeven_positive(self):
        """win_rate > q/b + q should give positive Kelly."""
        # b = 2.0/1.0 = 2.0, breakeven = 1/(b+1) = 0.333
        result = kelly_criterion(win_rate=0.50, avg_win=2.0, avg_loss=1.0)
        assert result.full_kelly > 0


# ---------------------------------------------------------------------------
# correlation_adjusted_kelly() from kelly_calculator.py
# ---------------------------------------------------------------------------

class TestCorrelationAdjustedKelly:
    """Tests for correlation_adjusted_kelly()."""

    def test_no_positions_no_penalty(self):
        result = correlation_adjusted_kelly(
            p_true=0.65, b=1.857, open_positions={},
            correlation_row={}, market_id="test", category="crypto",
        )
        assert result.correlation_penalty == 0.0
        assert result.kelly_adjusted == result.kelly_raw

    def test_high_correlation_reduces_kelly(self):
        result = correlation_adjusted_kelly(
            p_true=0.65, b=1.857,
            open_positions={"market_a": 0.04},
            correlation_row={"market_a": 0.90},
            market_id="test", category="crypto",
        )
        assert result.correlation_penalty > 0
        assert result.kelly_adjusted < result.kelly_raw

    def test_penalty_floors_at_10_percent(self):
        """Even with massive correlation, adjusted Kelly floors at 10% of raw."""
        result = correlation_adjusted_kelly(
            p_true=0.65, b=1.857,
            open_positions={"m1": 0.50, "m2": 0.50},
            correlation_row={"m1": 1.0, "m2": 1.0},
            market_id="test", category="crypto",
        )
        assert result.kelly_adjusted >= result.kelly_raw * 0.1 - 1e-10

    def test_invalid_p_true_raises(self):
        with pytest.raises(ValueError):
            correlation_adjusted_kelly(p_true=0.0, b=1.5, open_positions={},
                                       correlation_row={})
        with pytest.raises(ValueError):
            correlation_adjusted_kelly(p_true=1.0, b=1.5, open_positions={},
                                       correlation_row={})

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError):
            correlation_adjusted_kelly(p_true=0.5, b=-1.0, open_positions={},
                                       correlation_row={})

    def test_result_dataclass_fields(self):
        result = correlation_adjusted_kelly(
            p_true=0.65, b=1.857, open_positions={},
            correlation_row={}, market_id="M1", category="crypto",
        )
        assert isinstance(result, CorrelationAdjustedKellyResult)
        assert result.market_id == "M1"
        assert result.category == "crypto"
        assert result.full_kelly >= result.kelly_raw  # full_kelly >= fractional


# ---------------------------------------------------------------------------
# portfolio_kelly_check() from kelly_calculator.py
# ---------------------------------------------------------------------------

class TestPortfolioKellyCheck:
    """Tests for portfolio_kelly_check() -- full risk gate."""

    def test_edge_below_minimum_rejected(self):
        result = portfolio_kelly_check(
            candidate_market_id="test",
            candidate_category="crypto",
            p_true=0.53,
            p_market=0.50,
            open_positions={},
            category_map={},
            correlation_matrix={},
            bankroll=1000,
        )
        assert not result["approved"]

    def test_good_trade_approved(self):
        result = portfolio_kelly_check(
            candidate_market_id="test",
            candidate_category="crypto",
            p_true=0.70,
            p_market=0.50,
            open_positions={},
            category_map={},
            correlation_matrix={},
            bankroll=1000,
        )
        assert result["approved"]
        assert result["kelly_fraction"] > 0

    def test_per_market_cap_5_percent(self):
        """Kelly fraction should not exceed 5% per market."""
        result = portfolio_kelly_check(
            candidate_market_id="test",
            candidate_category="crypto",
            p_true=0.95,
            p_market=0.10,
            open_positions={},
            category_map={},
            correlation_matrix={},
            bankroll=1000,
        )
        if result["approved"]:
            assert result["kelly_fraction"] <= 0.05

    def test_category_cap_15_percent(self):
        """15% category cap prevents new trade when category is maxed."""
        result = portfolio_kelly_check(
            candidate_market_id="new_trade",
            candidate_category="crypto",
            p_true=0.70,
            p_market=0.50,
            open_positions={"m1": 0.05, "m2": 0.05, "m3": 0.05},
            category_map={"m1": "crypto", "m2": "crypto", "m3": "crypto"},
            correlation_matrix={},
            bankroll=1000,
        )
        assert not result["approved"]

    def test_total_exposure_cap_40_percent(self):
        """40% total exposure cap enforced."""
        # 8 positions x 5% = 40% already at cap
        pos = {f"m{i}": 0.05 for i in range(8)}
        cats = {f"m{i}": f"cat{i}" for i in range(8)}
        result = portfolio_kelly_check(
            candidate_market_id="new",
            candidate_category="new_cat",
            p_true=0.70,
            p_market=0.50,
            open_positions=pos,
            category_map=cats,
            correlation_matrix={},
            bankroll=1000,
        )
        assert not result["approved"]


# ---------------------------------------------------------------------------
# calculate_position_size() from kelly_calculator.py
# ---------------------------------------------------------------------------

class TestCalculatePositionSize:
    """Tests for calculate_position_size()."""

    def test_basic_position_sizing(self):
        result = calculate_position_size(
            account_balance=10000,
            kelly_fraction=0.02,
            stop_loss_pct=0.01,
        )
        assert result["risk_amount"] == 200.0
        assert result["position_size"] == 20000.0

    def test_zero_stop_loss_raises(self):
        with pytest.raises(ValueError):
            calculate_position_size(10000, 0.02, 0.0)
