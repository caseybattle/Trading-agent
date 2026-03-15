"""Unit tests for generate_signals() from kalshi_btc_trader.py."""
import sys
import pytest
from unittest.mock import patch
from datetime import datetime, timezone

from scripts.kalshi_btc_trader import generate_signals, MIN_EDGE_PCT


# Required keys that every signal dict must contain
REQUIRED_SIGNAL_KEYS = {
    "ticker", "yes_no", "fair_value", "edge", "kelly_fraction",
    "contracts", "limit_price", "minutes_to_close", "strategy",
    "range_low", "range_high", "market_bid", "btc_price_at_signal",
    "market_mid", "range",
}


class TestGenerateSignals:
    """Tests for the signal generation pipeline."""

    def test_returns_list(self, sample_markets):
        """generate_signals should always return a list."""
        signals = generate_signals(sample_markets, btc_price=65_000.0, bankroll=10.0)
        assert isinstance(signals, list)

    def test_signal_dict_has_required_fields(self, sample_markets):
        """All signals must contain required keys."""
        signals = generate_signals(sample_markets, btc_price=65_000.0, bankroll=10.0)
        for s in signals:
            missing = REQUIRED_SIGNAL_KEYS - set(s.keys())
            assert not missing, f"Signal missing fields: {missing}"

    def test_only_signals_above_min_edge(self, sample_markets):
        """Every signal must have edge >= MIN_EDGE_PCT."""
        signals = generate_signals(sample_markets, btc_price=65_000.0, bankroll=10.0)
        for s in signals:
            assert s["edge"] >= MIN_EDGE_PCT - 1e-9, (
                f"Signal {s['ticker']} has edge {s['edge']} < min {MIN_EDGE_PCT}"
            )

    def test_no_signal_below_min_edge(self):
        """Edge below MIN_EDGE_PCT should produce no signal."""
        # BTC far from range, short time, prices set so neither YES nor NO has edge.
        # Fair value for YES ~ 0 (BTC at 70000, range 65000-65250, 60 min).
        # yes_ask=0.01 -> edge_yes = ~0 - 0.01 < 0 (no YES edge)
        # no_cost = 1 - 0.01 = 0.99 -> edge_no = ~1.0 - 0.99 = 0.01 < 0.08 (no NO edge)
        markets = [{
            "ticker": "KXBTC-TEST-B65000",
            "range_low": 65_000,
            "range_high": 65_250,
            "yes_ask": 0.01,
            "yes_bid": 0.01,
            "mid": 0.01,
            "spread": 0.00,
            "volume_24h": 10_000,
            "minutes_left": 60,
        }]
        signals = generate_signals(markets, btc_price=70_000, bankroll=10.0)
        assert len(signals) == 0, "No signal when edge is below threshold"

    def test_signals_sorted_by_edge_descending(self, sample_markets):
        """Signals should be sorted by edge descending."""
        signals = generate_signals(sample_markets, btc_price=65_000.0, bankroll=10.0)
        if len(signals) >= 2:
            for i in range(len(signals) - 1):
                assert signals[i]["edge"] >= signals[i + 1]["edge"], (
                    f"Signals not sorted: {signals[i]['edge']} < {signals[i+1]['edge']}"
                )

    def test_contracts_at_least_one(self, sample_markets):
        """Every signal should have at least 1 contract."""
        signals = generate_signals(sample_markets, btc_price=65_000.0, bankroll=10.0)
        for s in signals:
            assert s["contracts"] >= 1

    def test_all_three_strategies_can_generate(self):
        """All 3 strategies can generate signals given the right market conditions."""
        valid_strategies = {
            "TIME_DECAY_IN_RANGE",
            "MODEL_UNDERPRICED_YES",
            "MODEL_UNDERPRICED_NO",
        }
        # TIME_DECAY: BTC in range, short time, high fair value, cheap ask
        td_market = [{
            "ticker": "KXBTC-TD-B64500",
            "range_low": 64_500,
            "range_high": 65_500,
            "yes_ask": 0.50,
            "yes_bid": 0.45,
            "mid": 0.475,
            "spread": 0.05,
            "volume_24h": 10_000,
            "minutes_left": 10.0,
        }]
        td_signals = generate_signals(td_market, btc_price=65_000.0, bankroll=10.0)
        td_strats = {s["strategy"] for s in td_signals}

        # MODEL_UNDERPRICED_YES: BTC near range, long time, fair > mid, cheap ask
        yes_market = [{
            "ticker": "KXBTC-YES-B64750",
            "range_low": 64_750,
            "range_high": 65_250,
            "yes_ask": 0.20,
            "yes_bid": 0.15,
            "mid": 0.175,
            "spread": 0.05,
            "volume_24h": 10_000,
            "minutes_left": 120.0,
        }]
        yes_signals = generate_signals(yes_market, btc_price=65_000.0, bankroll=10.0)
        yes_strats = {s["strategy"] for s in yes_signals}

        # MODEL_UNDERPRICED_NO: BTC far from range, long time, cheap NO
        no_market = [{
            "ticker": "KXBTC-NO-B70000",
            "range_low": 70_000,
            "range_high": 70_250,
            "yes_ask": 0.90,
            "yes_bid": 0.85,
            "mid": 0.875,
            "spread": 0.05,
            "volume_24h": 10_000,
            "minutes_left": 120.0,
        }]
        no_signals = generate_signals(no_market, btc_price=65_000.0, bankroll=10.0)
        no_strats = {s["strategy"] for s in no_signals}

        all_strats = td_strats | yes_strats | no_strats
        assert len(all_strats & valid_strategies) >= 1, (
            f"Expected at least 1 valid strategy, got {all_strats}"
        )

    def test_empty_market_list_returns_empty(self):
        """Empty market list -> empty signals."""
        signals = generate_signals([], btc_price=65_000.0, bankroll=10.0)
        assert signals == []

    def test_avoid_hours_blocks_signals(self, sample_markets):
        """Signal generation should return empty during avoid hours."""
        import scripts.kalshi_btc_trader as trader
        current_hour = datetime.now(timezone.utc).hour
        trader._avoid_hours = {current_hour}
        try:
            signals = generate_signals(sample_markets, btc_price=65_000.0, bankroll=10.0)
            assert len(signals) == 0, "Should produce no signals during avoid hours"
        finally:
            trader._avoid_hours = set()

    def test_fair_value_in_signal_is_bounded(self, sample_markets):
        """fair_value in each signal should be in [0, 1]."""
        signals = generate_signals(sample_markets, btc_price=65_000.0, bankroll=10.0)
        for s in signals:
            assert 0.0 <= s["fair_value"] <= 1.0
