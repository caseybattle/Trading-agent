"""
Shared pytest fixtures for prediction-market-bot test suite.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

# ---------------------------------------------------------------------------
# Ensure `from scripts.X import Y` works regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def sample_btc_price() -> float:
    """Current BTC price for tests."""
    return 65_000.0


@pytest.fixture
def sample_markets() -> list:
    """
    List of dicts mimicking parsed Kalshi API market response.
    Covers time-decay, model-underpriced YES/NO, and out-of-range scenarios.
    BTC assumed at 65,000.
    """
    return [
        {
            "ticker": "KXBTC-26MAR1416-B64750",
            "range_low": 64_750,
            "range_high": 65_250,
            "yes_ask": 0.50,
            "yes_bid": 0.45,
            "mid": 0.475,
            "spread": 0.05,
            "volume_24h": 12_000,
            "minutes_left": 20.0,       # short -- time-decay candidate
        },
        {
            "ticker": "KXBTC-26MAR1416-B64500",
            "range_low": 64_500,
            "range_high": 64_750,
            "yes_ask": 0.30,
            "yes_bid": 0.25,
            "mid": 0.275,
            "spread": 0.05,
            "volume_24h": 8_000,
            "minutes_left": 120.0,      # 2 hours out
        },
        {
            "ticker": "KXBTC-26MAR1416-B66000",
            "range_low": 66_000,
            "range_high": 66_250,
            "yes_ask": 0.10,
            "yes_bid": 0.05,
            "mid": 0.075,
            "spread": 0.05,
            "volume_24h": 6_000,
            "minutes_left": 60.0,       # BTC well below this range
        },
        {
            "ticker": "KXBTC-26MAR1416-B63000",
            "range_low": 63_000,
            "range_high": 63_250,
            "yes_ask": 0.05,
            "yes_bid": 0.02,
            "mid": 0.035,
            "spread": 0.03,
            "volume_24h": 3_000,
            "minutes_left": 45.0,       # BTC well above this range
        },
        {
            "ticker": "KXBTC-26MAR1416-B65250",
            "range_low": 65_250,
            "range_high": 65_500,
            "yes_ask": 0.20,
            "yes_bid": 0.15,
            "mid": 0.175,
            "spread": 0.05,
            "volume_24h": 10_000,
            "minutes_left": 180.0,      # 3 hours, BTC just below range
        },
    ]


@pytest.fixture
def strategy_config():
    """Load current strategy_config.json or return defaults."""
    cfg_path = PROJECT_ROOT / "backtest" / "strategy_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {
        "btc_hourly_vol": 0.01,
        "min_edge_pp": 8.0,
        "fractional_kelly": 0.25,
        "max_position_pct": 0.05,
        "time_decay_threshold_min": 30,
        "time_decay_min_fair": 0.70,
        "avoid_hours": [],
        "iteration": 0,
    }


@pytest.fixture
def tmp_config(tmp_path) -> Path:
    """Write a temporary strategy_config.json and return its path."""
    cfg = {
        "btc_hourly_vol": 0.01,
        "min_edge_pp": 8.0,
        "fractional_kelly": 0.25,
        "max_position_pct": 0.05,
        "time_decay_threshold_min": 30,
        "time_decay_min_fair": 0.70,
        "last_updated": "2026-03-13T00:00:00Z",
        "iteration": 0,
        "notes": "Test config",
        "avoid_hours": [],
    }
    cfg_path = tmp_path / "strategy_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


@pytest.fixture
def tmp_bankroll(tmp_path) -> Path:
    """Write a temporary bankroll.json and return its path."""
    state = {
        "starting_bankroll": 10.0,
        "current_bankroll": 10.0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "n_trades": 0,
        "n_wins": 0,
        "n_losses": 0,
        "win_rate": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    bankroll_path = tmp_path / "bankroll.json"
    bankroll_path.write_text(json.dumps(state, indent=2))
    return bankroll_path


@pytest.fixture
def correlation_graph():
    """Build a CorrelationGraph for penalty tests (from evals.json setup)."""
    from scripts.correlation_engine import CorrelationGraph
    return CorrelationGraph(
        edges={
            "market_btc_100k": {"market_btc_90k": 0.85, "market_eth_5k": 0.42},
            "market_btc_90k": {"market_btc_100k": 0.85},
            "market_eth_5k": {"market_btc_100k": 0.42},
        }
    )


@pytest.fixture
def arbitrage_graph():
    """CorrelationGraph for arbitrage detection tests (from evals.json)."""
    from scripts.correlation_engine import CorrelationGraph
    return CorrelationGraph(
        edges={
            "market_a": {"market_b": 0.90},
            "market_b": {"market_a": 0.90},
            "market_c": {"market_d": 0.30},
            "market_d": {"market_c": 0.30},
        }
    )
