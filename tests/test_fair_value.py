"""Unit tests for compute_fair_value() from kalshi_btc_trader.py."""
import math
import pytest

from scripts.kalshi_btc_trader import compute_fair_value


class TestComputeFairValue:
    """Tests for the log-normal fair value model."""

    def test_btc_well_inside_range_near_one(self):
        """BTC centered in wide range with low time left -> fair value near 1.0."""
        # Use wider range (1000 pts) so log-normal model gives high probability
        fv = compute_fair_value(65_000, 64_500, 65_500, hours_left=0.1)
        assert fv > 0.85, f"Expected > 0.85, got {fv}"

    def test_btc_well_outside_range_near_zero(self):
        """BTC far from range -> fair value near 0.0."""
        fv = compute_fair_value(70_000, 65_000, 65_250, hours_left=0.5)
        assert fv < 0.01, f"Expected < 0.01, got {fv}"

    def test_btc_at_range_boundary_approx_half(self):
        """BTC at range boundary with time remaining -> fair value around 0.5."""
        # BTC at the lower boundary of the range
        fv = compute_fair_value(65_000, 65_000, 65_250, hours_left=1.0)
        # Should be roughly 0.3-0.7 range (not extremes)
        assert 0.1 < fv < 0.9, f"Expected boundary value, got {fv}"

    def test_zero_hours_in_range(self):
        """hours_left=0, BTC in range -> exactly 1.0."""
        fv = compute_fair_value(65_100, 65_000, 65_250, hours_left=0)
        assert fv == 1.0

    def test_zero_hours_out_of_range(self):
        """hours_left=0, BTC out of range -> exactly 0.0."""
        fv = compute_fair_value(66_000, 65_000, 65_250, hours_left=0)
        assert fv == 0.0

    def test_high_vol_increases_fair_value_for_far_out_of_range(self):
        """Higher vol should increase fair value when BTC is far outside range."""
        # BTC well above range; higher vol means more probability mass reaches range
        fv_low = compute_fair_value(70_000, 65_000, 65_500, hours_left=2.0, hourly_vol_pct=0.005)
        fv_high = compute_fair_value(70_000, 65_000, 65_500, hours_left=2.0, hourly_vol_pct=0.03)
        assert fv_high > fv_low, "Higher vol should increase fair value for far out-of-range BTC"

    def test_high_vol_decreases_fair_value_centered(self):
        """Higher vol should reduce fair value when BTC is centered in range."""
        fv_low = compute_fair_value(65_125, 65_000, 65_250, hours_left=1.0, hourly_vol_pct=0.005)
        fv_high = compute_fair_value(65_125, 65_000, 65_250, hours_left=1.0, hourly_vol_pct=0.02)
        assert fv_low > fv_high, "Higher vol should reduce in-range probability when centered"

    def test_all_outputs_in_zero_one(self):
        """Fair value must always be in [0, 1] for any valid inputs."""
        test_cases = [
            (65_000, 64_000, 65_000, 1.0),
            (65_100, 65_000, 65_250, 0.5),
            (100_000, 65_000, 65_250, 2.0),
            (50_000, 65_000, 65_250, 0.1),
            (65_125, 65_000, 65_250, 0.001),
            (65_125, 65_000, 65_250, 24.0),
            (1, 65_000, 65_250, 1.0),
        ]
        for btc, low, high, hours in test_cases:
            fv = compute_fair_value(btc, low, high, hours)
            assert 0.0 <= fv <= 1.0, (
                f"OOB: fv={fv} for btc={btc}, range=[{low},{high}], hours={hours}"
            )

    def test_symmetry_same_distance(self):
        """BTC equidistant above vs below range center should give similar fair values."""
        center = 65_125  # midpoint of [65000, 65250]
        offset = 100
        fv_above = compute_fair_value(center + offset, 65_000, 65_250, hours_left=1.0)
        fv_below = compute_fair_value(center - offset, 65_000, 65_250, hours_left=1.0)
        # Not exactly equal due to log-normal skew, but should be close
        assert abs(fv_above - fv_below) < 0.15, (
            f"Symmetry broken: above={fv_above:.4f}, below={fv_below:.4f}"
        )

    def test_more_time_reduces_certainty(self):
        """More time remaining should reduce certainty for in-range BTC."""
        fv_short = compute_fair_value(65_125, 65_000, 65_250, hours_left=0.1)
        fv_long = compute_fair_value(65_125, 65_000, 65_250, hours_left=5.0)
        assert fv_short > fv_long, "Less time should mean higher in-range probability"

    def test_zero_btc_price(self):
        """BTC price = 0 with hours_left > 0 should return valid value (0 or 1)."""
        fv = compute_fair_value(0, 65_000, 65_250, hours_left=1.0)
        assert 0.0 <= fv <= 1.0

    def test_at_range_low_boundary_zero_time(self):
        """BTC exactly at range_low with zero time should be in range."""
        fv = compute_fair_value(65_000, 65_000, 65_250, hours_left=0)
        assert fv == 1.0, "range_low <= btc < range_high should hold"

    def test_at_range_high_boundary_zero_time(self):
        """BTC exactly at range_high with zero time should be out of range."""
        fv = compute_fair_value(65_250, 65_000, 65_250, hours_left=0)
        assert fv == 0.0, "btc == range_high should be out of range"
