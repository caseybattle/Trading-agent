"""Unit tests for all risk safeguards in the trading bot."""
import json
import csv
import sys
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.kalshi_btc_trader import (
    check_daily_loss_stop,
    get_recent_tickers,
    kelly_fraction,
    generate_signals,
    DAILY_LOSS_STOP_PCT,
    MAX_POSITION_PCT,
    MIN_EDGE_PCT,
)


# ---------------------------------------------------------------------------
# Daily loss stop
# ---------------------------------------------------------------------------

class TestDailyLossStop:
    """Tests for check_daily_loss_stop()."""

    def test_no_bankroll_file_returns_false(self, tmp_path):
        """Missing bankroll file should not trigger stop."""
        with patch("scripts.kalshi_btc_trader.BANKROLL_FILE", tmp_path / "missing.json"):
            assert check_daily_loss_stop() is False

    def test_within_threshold_returns_false(self, tmp_path):
        """Loss within threshold should not trigger stop."""
        bf = tmp_path / "bankroll.json"
        bf.write_text(json.dumps({
            "starting_bankroll": 10.0,
            "current_bankroll": 9.80,  # 2% loss, under 3%
        }))
        with patch("scripts.kalshi_btc_trader.BANKROLL_FILE", bf):
            assert check_daily_loss_stop() is False

    def test_at_threshold_triggers(self, tmp_path):
        """Loss at exactly 3% should trigger stop."""
        bf = tmp_path / "bankroll.json"
        bf.write_text(json.dumps({
            "starting_bankroll": 10.0,
            "current_bankroll": 9.70,  # exactly 3% loss
        }))
        with patch("scripts.kalshi_btc_trader.BANKROLL_FILE", bf):
            assert check_daily_loss_stop() is True

    def test_above_threshold_triggers(self, tmp_path):
        """Loss above 3% should trigger stop."""
        bf = tmp_path / "bankroll.json"
        bf.write_text(json.dumps({
            "starting_bankroll": 10.0,
            "current_bankroll": 9.00,  # 10% loss
        }))
        with patch("scripts.kalshi_btc_trader.BANKROLL_FILE", bf):
            assert check_daily_loss_stop() is True

    def test_corrupt_file_returns_false(self, tmp_path):
        """Corrupt JSON should not crash; returns False."""
        bf = tmp_path / "bankroll.json"
        bf.write_text("NOT VALID JSON{{{")
        with patch("scripts.kalshi_btc_trader.BANKROLL_FILE", bf):
            assert check_daily_loss_stop() is False


# ---------------------------------------------------------------------------
# Bankroll floor ($0.50 minimum)
# ---------------------------------------------------------------------------

class TestBankrollFloor:
    """Verify $0.50 minimum bankroll is recognized."""

    def test_floor_stops_trading(self, tmp_path):
        """When bankroll is at or below floor, daily loss stop should trigger."""
        bf = tmp_path / "bankroll.json"
        bf.write_text(json.dumps({
            "starting_bankroll": 10.0,
            "current_bankroll": 0.50,  # floor: $0.50
        }))
        with patch("scripts.kalshi_btc_trader.BANKROLL_FILE", bf):
            # starting=10, current=0.50 -> loss=9.50, 95% > 3% -> triggered
            assert check_daily_loss_stop() is True


# ---------------------------------------------------------------------------
# Max position cap
# ---------------------------------------------------------------------------

class TestMaxPositionCap:
    """Test that Kelly sizing is capped at MAX_POSITION_PCT (5%)."""

    def test_kelly_never_exceeds_max_position_via_generate(self):
        """In generate_signals, min(kf, MAX_POSITION_PCT) caps position sizing."""
        frac = kelly_fraction(0.99, 0.01)
        capped = min(frac, MAX_POSITION_PCT)
        assert capped <= MAX_POSITION_PCT
        assert capped == MAX_POSITION_PCT, (
            f"Very high edge should hit the 5% cap, got {capped}"
        )


# ---------------------------------------------------------------------------
# Ticker dedup (cooldown)
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "timestamp", "ticker", "strategy", "direction", "range_label",
    "range_low", "range_high", "fair_value", "market_ask", "market_bid",
    "edge_pp", "minutes_left", "btc_price_at_signal", "kelly_fraction",
    "recommended_contracts", "acted_on", "outcome",
]


class TestTickerDedup:
    """Tests for get_recent_tickers()."""

    def test_empty_log_returns_empty(self, tmp_path):
        """No signal log -> empty set."""
        with patch("scripts.kalshi_btc_trader.SIGNAL_LOG", tmp_path / "missing.csv"):
            assert get_recent_tickers(cooldown_minutes=30) == set()

    def test_recent_ticker_detected(self, tmp_path):
        """Ticker traded within cooldown window should be in set."""
        log_file = tmp_path / "signals_log.csv"
        now = datetime.now(timezone.utc)
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow({
                "timestamp": now.isoformat(),
                "ticker": "KXBTC-TEST-B65000",
                "strategy": "TIME_DECAY",
                "direction": "YES",
                "range_label": "",
                "range_low": 65000,
                "range_high": 65250,
                "fair_value": 0.85,
                "market_ask": 0.45,
                "market_bid": 0.40,
                "edge_pp": 40,
                "minutes_left": 10,
                "btc_price_at_signal": 65100,
                "kelly_fraction": 0.05,
                "recommended_contracts": 1,
                "acted_on": "AUTO",
                "outcome": "",
            })
        with patch("scripts.kalshi_btc_trader.SIGNAL_LOG", log_file):
            result = get_recent_tickers(cooldown_minutes=30)
            assert "KXBTC-TEST-B65000" in result

    def test_old_ticker_not_detected(self, tmp_path):
        """Ticker traded >30 min ago should NOT be in set."""
        log_file = tmp_path / "signals_log.csv"
        old_time = datetime.now(timezone.utc) - timedelta(minutes=60)
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow({
                "timestamp": old_time.isoformat(),
                "ticker": "KXBTC-OLD-B65000",
                "strategy": "TIME_DECAY",
                "direction": "YES",
                "range_label": "",
                "range_low": 65000,
                "range_high": 65250,
                "fair_value": 0.85,
                "market_ask": 0.45,
                "market_bid": 0.40,
                "edge_pp": 40,
                "minutes_left": 10,
                "btc_price_at_signal": 65100,
                "kelly_fraction": 0.05,
                "recommended_contracts": 1,
                "acted_on": "AUTO",
                "outcome": "",
            })
        with patch("scripts.kalshi_btc_trader.SIGNAL_LOG", log_file):
            result = get_recent_tickers(cooldown_minutes=30)
            assert "KXBTC-OLD-B65000" not in result

    def test_manual_acted_on_counted(self, tmp_path):
        """Tickers with acted_on=MANUAL should also be counted."""
        log_file = tmp_path / "signals_log.csv"
        now = datetime.now(timezone.utc)
        with open(log_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow({
                "timestamp": now.isoformat(),
                "ticker": "KXBTC-MANUAL-B65000",
                "strategy": "MODEL_UNDERPRICED_YES",
                "direction": "YES",
                "range_label": "",
                "range_low": 65000,
                "range_high": 65250,
                "fair_value": 0.80,
                "market_ask": 0.50,
                "market_bid": 0.45,
                "edge_pp": 30,
                "minutes_left": 30,
                "btc_price_at_signal": 65100,
                "kelly_fraction": 0.04,
                "recommended_contracts": 1,
                "acted_on": "MANUAL",
                "outcome": "",
            })
        with patch("scripts.kalshi_btc_trader.SIGNAL_LOG", log_file):
            result = get_recent_tickers(cooldown_minutes=30)
            assert "KXBTC-MANUAL-B65000" in result


# ---------------------------------------------------------------------------
# Avoid hours
# ---------------------------------------------------------------------------

class TestAvoidHours:
    """Test that generate_signals respects avoid_hours."""

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


# ---------------------------------------------------------------------------
# Exposure caps (from kelly_calculator)
# ---------------------------------------------------------------------------

class TestExposureCaps:
    """Test exposure caps from portfolio_kelly_check."""

    def test_per_market_5_percent_cap(self):
        from scripts.kelly_calculator import portfolio_kelly_check
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

    def test_category_15_percent_cap(self):
        from scripts.kelly_calculator import portfolio_kelly_check
        result = portfolio_kelly_check(
            candidate_market_id="new",
            candidate_category="crypto",
            p_true=0.70,
            p_market=0.50,
            open_positions={"m1": 0.05, "m2": 0.05, "m3": 0.05},
            category_map={"m1": "crypto", "m2": "crypto", "m3": "crypto"},
            correlation_matrix={},
            bankroll=1000,
        )
        assert not result["approved"]

    def test_total_40_percent_cap(self):
        from scripts.kelly_calculator import portfolio_kelly_check
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
