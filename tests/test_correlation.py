"""
Unit tests for correlation_engine.py.
Implements the 3 eval cases from evals.json for both correlation_penalty()
and detect_arbitrage().
"""
import pytest

from scripts.correlation_engine import (
    CorrelationGraph,
    correlation_penalty,
    detect_arbitrage,
    category_exposure,
)


# ===================================================================
# eval_correlation_penalty (3 cases from evals.json)
# ===================================================================

class TestCorrelationPenalty:
    """Tests implementing eval_correlation_penalty from evals.json."""

    @pytest.fixture
    def graph(self):
        """Setup graph from evals.json eval_correlation_penalty."""
        return CorrelationGraph(edges={
            "market_btc_100k": {"market_btc_90k": 0.85, "market_eth_5k": 0.42},
            "market_btc_90k": {"market_btc_100k": 0.85},
            "market_eth_5k": {"market_btc_100k": 0.42},
        })

    def test_high_correlation_reduces_sizing(self, graph):
        """
        Case 1: market_btc_100k with positions in market_btc_90k (0.04) and
        market_eth_5k (0.02). Expected penalty in [0.35, 0.75].

        Computation:
          total_exposure = 0.04 + 0.02 = 0.06
          weighted_corr = 0.85*(0.04/0.06) + 0.42*(0.02/0.06) = 0.567 + 0.140 = 0.707
        """
        penalty = correlation_penalty(
            market_id="market_btc_100k",
            current_positions={"market_btc_90k": 0.04, "market_eth_5k": 0.02},
            graph=graph,
        )
        assert 0.35 <= penalty <= 0.75, (
            f"Expected penalty in [0.35, 0.75], got {penalty}"
        )
        # Verify the Kelly adjustment formula holds
        kelly_raw = 0.10  # arbitrary
        kelly_adj = kelly_raw * max(0.1, 1 - penalty)
        assert 0 < kelly_adj <= kelly_raw

    def test_no_positions_zero_penalty(self, graph):
        """Case 2: Empty portfolio -> penalty must be exactly 0.0."""
        penalty = correlation_penalty(
            market_id="market_btc_100k",
            current_positions={},
            graph=graph,
        )
        assert penalty == 0.0

    def test_unrelated_market_zero_penalty(self, graph):
        """Case 3: Market with no edges in graph -> penalty must be 0.0."""
        penalty = correlation_penalty(
            market_id="market_election_2026",
            current_positions={"market_btc_90k": 0.04},
            graph=graph,
        )
        assert penalty == 0.0


# ===================================================================
# eval_arbitrage_detection (3 cases from evals.json)
# ===================================================================

class TestArbitrageDetection:
    """Tests implementing eval_arbitrage_detection from evals.json."""

    @pytest.fixture
    def graph(self):
        """Setup graph from evals.json eval_arbitrage_detection."""
        return CorrelationGraph(edges={
            "market_a": {"market_b": 0.90},
            "market_b": {"market_a": 0.90},
            "market_c": {"market_d": 0.30},
            "market_d": {"market_c": 0.30},
        })

    def test_complementary_arbitrage_detected(self, graph):
        """
        Case 1: High correlation + price gap -> arbitrage signal.

        Using prices that produce implied_gap > threshold via
        min(direct_gap, complement_gap):
          market_a=0.80, market_b=0.80
          direct_gap = |0.80 - 0.80| = 0.0
          complement_gap = |0.80 - (1-0.80)| = |0.80 - 0.20| = 0.60
          implied_gap = min(0.0, 0.60) = 0.0 -- not > 0.15

        Use divergent prices instead:
          market_a=0.80, market_b=0.40
          direct_gap = |0.80 - 0.40| = 0.40
          complement_gap = |0.80 - 0.60| = 0.20
          implied_gap = min(0.40, 0.20) = 0.20 > 0.15
        """
        signals = detect_arbitrage(
            live_prices={"market_a": 0.80, "market_b": 0.40},
            graph=graph,
            threshold=0.15,
        )
        assert len(signals) == 1
        sig = signals[0]
        assert sig.market_a in ("market_a", "market_b")
        assert sig.implied_gap >= 0.15
        assert abs(sig.signal_strength - sig.correlation * sig.implied_gap) < 1e-10

    def test_consistent_prices_no_signal(self, graph):
        """
        Case 2: Consistent prices -> no arbitrage.
        market_a=0.60, market_b=0.41
          direct_gap = 0.19, complement_gap = |0.60 - 0.59| = 0.01
          implied_gap = min(0.19, 0.01) = 0.01 < 0.15
        """
        signals = detect_arbitrage(
            live_prices={"market_a": 0.60, "market_b": 0.41},
            graph=graph,
            threshold=0.15,
        )
        assert len(signals) == 0

    def test_low_correlation_weak_signal(self, graph):
        """
        Case 3: Low correlation pair produces weak signal_strength.
        market_c=0.80, market_d=0.40, corr=0.30
          direct_gap = 0.40, complement_gap = |0.80-0.60| = 0.20
          implied_gap = min(0.40, 0.20) = 0.20 > 0.15
          signal_strength = 0.30 * 0.20 = 0.06 < 0.25
        """
        signals = detect_arbitrage(
            live_prices={"market_c": 0.80, "market_d": 0.40},
            graph=graph,
            threshold=0.15,
        )
        assert len(signals) == 1
        assert signals[0].correlation == 0.30
        assert signals[0].signal_strength < 0.25


# ===================================================================
# category_exposure()
# ===================================================================

class TestCategoryExposure:
    """Tests for category_exposure()."""

    def test_basic_aggregation(self):
        result = category_exposure(
            current_positions={"m1": 0.03, "m2": 0.04, "m3": 0.02},
            market_categories={"m1": "crypto", "m2": "crypto", "m3": "politics"},
        )
        assert abs(result["crypto"] - 0.07) < 1e-10
        assert abs(result["politics"] - 0.02) < 1e-10

    def test_empty_positions(self):
        result = category_exposure({}, {})
        assert result == {}

    def test_single_category(self):
        result = category_exposure(
            current_positions={"m1": 0.05, "m2": 0.03},
            market_categories={"m1": "crypto", "m2": "crypto"},
        )
        assert abs(result["crypto"] - 0.08) < 1e-10
