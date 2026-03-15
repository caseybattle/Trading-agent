"""Integration tests for config loading, optimizer, and module imports."""
import json
import pytest
from pathlib import Path

from scripts.strategy_optimizer import (
    load_config, save_config, DEFAULT_CONFIG, reset_config,
    compute_win_rate, compute_break_even_rate, calibration_analysis,
    run_optimization,
)


# ===================================================================
# Config loading
# ===================================================================

class TestConfigLoading:
    """Test strategy config loading and saving."""

    def test_load_config_returns_valid_dict(self, tmp_path):
        """load_config() returns a dict with all required keys."""
        import scripts.strategy_optimizer as opt
        orig = opt.CONFIG_PATH
        try:
            opt.CONFIG_PATH = tmp_path / "backtest" / "strategy_config.json"
            cfg = load_config()
            assert isinstance(cfg, dict)
            required = {
                "btc_hourly_vol", "min_edge_pp", "fractional_kelly",
                "max_position_pct", "time_decay_threshold_min",
                "time_decay_min_fair", "avoid_hours", "iteration",
            }
            assert required.issubset(set(cfg.keys())), (
                f"Missing keys: {required - set(cfg.keys())}"
            )
        finally:
            opt.CONFIG_PATH = orig

    def test_save_and_load_roundtrip(self, tmp_path):
        """Config should survive save/load roundtrip with modified values."""
        import scripts.strategy_optimizer as opt
        orig = opt.CONFIG_PATH
        try:
            opt.CONFIG_PATH = tmp_path / "backtest" / "strategy_config.json"
            cfg = dict(DEFAULT_CONFIG)
            cfg["btc_hourly_vol"] = 0.015
            cfg["min_edge_pp"] = 10.0
            save_config(cfg)
            loaded = load_config()
            assert loaded["btc_hourly_vol"] == 0.015
            assert loaded["min_edge_pp"] == 10.0
        finally:
            opt.CONFIG_PATH = orig

    def test_default_config_has_all_required_fields(self):
        """DEFAULT_CONFIG must have every field the trader reads."""
        required = {
            "btc_hourly_vol", "min_edge_pp", "fractional_kelly",
            "max_position_pct", "time_decay_threshold_min",
            "time_decay_min_fair", "avoid_hours", "iteration",
        }
        assert required.issubset(set(DEFAULT_CONFIG.keys()))

    def test_reset_config_restores_defaults(self, tmp_path):
        """reset_config() should restore factory defaults."""
        import scripts.strategy_optimizer as opt
        orig = opt.CONFIG_PATH
        try:
            opt.CONFIG_PATH = tmp_path / "backtest" / "strategy_config.json"
            # Save modified config
            cfg = dict(DEFAULT_CONFIG)
            cfg["min_edge_pp"] = 99.0
            save_config(cfg)
            # Reset
            fresh = reset_config()
            assert fresh["min_edge_pp"] == DEFAULT_CONFIG["min_edge_pp"]
            assert fresh["iteration"] == 0
        finally:
            opt.CONFIG_PATH = orig


# ===================================================================
# Optimizer analysis functions
# ===================================================================

class TestOptimizerAnalysis:
    """Test optimizer analysis functions with synthetic data."""

    def test_compute_win_rate(self):
        trades = [
            {"outcome": "WIN"}, {"outcome": "WIN"}, {"outcome": "LOSS"},
            {"outcome": "WIN"}, {"outcome": "LOSS"},
        ]
        wins, losses, rate = compute_win_rate(trades)
        assert wins == 3
        assert losses == 2
        assert abs(rate - 0.6) < 1e-10

    def test_compute_break_even(self):
        trades = [
            {"market_ask": "0.50"}, {"market_ask": "0.60"}, {"market_ask": "0.40"},
        ]
        be = compute_break_even_rate(trades)
        assert abs(be - 0.50) < 1e-10

    def test_calibration_analysis(self):
        trades = [
            {"fair_value": "0.55", "outcome": "WIN"},
            {"fair_value": "0.55", "outcome": "LOSS"},
            {"fair_value": "0.55", "outcome": "WIN"},
            {"fair_value": "0.75", "outcome": "LOSS"},
            {"fair_value": "0.75", "outcome": "LOSS"},
        ]
        results = calibration_analysis(trades)
        assert len(results) > 0
        bucket_50_60 = [r for r in results if r["label"] == "50-60%"]
        if bucket_50_60:
            assert bucket_50_60[0]["n"] == 3

    def test_optimizer_skips_with_few_trades(self, tmp_path):
        """Optimizer should skip when fewer than MIN_TRADES_REQUIRED."""
        import scripts.strategy_optimizer as opt
        orig_log = opt.SIGNALS_LOG
        orig_cfg = opt.CONFIG_PATH
        try:
            opt.SIGNALS_LOG = tmp_path / "empty_log.csv"
            opt.CONFIG_PATH = tmp_path / "cfg.json"
            cfg = run_optimization(dry_run=True)
            assert "btc_hourly_vol" in cfg
        finally:
            opt.SIGNALS_LOG = orig_log
            opt.CONFIG_PATH = orig_cfg


# ===================================================================
# Module import checks
# ===================================================================

class TestModuleImports:
    """Verify key modules can be imported and have expected functions."""

    def test_backtest_runner_importable(self):
        """backtest_runner should be importable."""
        try:
            from scripts import backtest_runner
            assert hasattr(backtest_runner, "run_full_backtest") or hasattr(backtest_runner, "run_backtest"), (
                "backtest_runner must have run_full_backtest or run_backtest function"
            )
        except ImportError:
            pytest.skip("backtest_runner not available (optional dependency)")

    def test_kxbtc_backtest_importable(self):
        """kxbtc_backtest should be importable."""
        try:
            from scripts import kxbtc_backtest
            assert hasattr(kxbtc_backtest, "run_backtest"), (
                "kxbtc_backtest must have run_backtest function"
            )
        except ImportError:
            pytest.skip("kxbtc_backtest not available (optional dependency)")

    def test_bankroll_tracker_importable(self):
        """bankroll_tracker should be importable with BankrollTracker class."""
        from scripts.bankroll_tracker import BankrollTracker
        assert callable(BankrollTracker)

    def test_correlation_engine_importable(self):
        """correlation_engine key functions should be importable."""
        from scripts.correlation_engine import (
            CorrelationGraph,
            correlation_penalty,
            detect_arbitrage,
        )
        assert callable(correlation_penalty)
        assert callable(detect_arbitrage)
