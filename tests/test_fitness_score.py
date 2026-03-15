"""Tests for the composite fitness scorer."""
import math
import pytest

from fitness_scorer import compute_fitness, FitnessResult, sigmoid_map


# ---------------------------------------------------------------------------
# Score range tests
# ---------------------------------------------------------------------------

class TestFitnessScoreRange:
    """Fitness score should always be in [0, 100]."""

    def test_score_with_good_results(self):
        bt = {"out_of_sample": {
            "return_pct": 20, "sharpe": 1.5, "win_rate": 60,
            "n_trades": 50, "max_drawdown_pct": 10, "avg_edge": 10,
            "calibration": {},
        }}
        tr = {"passed": 30, "failed": 0, "total": 30, "safeguard_passed": True}
        result = compute_fitness(bt, tr)
        assert 0 <= result.total_score <= 100

    def test_score_with_bad_results(self):
        bt = {"out_of_sample": {
            "return_pct": -50, "sharpe": -2.0, "win_rate": 20,
            "n_trades": 50, "max_drawdown_pct": 80, "avg_edge": -5,
            "calibration": {},
        }}
        tr = {"passed": 10, "failed": 20, "total": 30, "safeguard_passed": True}
        result = compute_fitness(bt, tr)
        assert 0 <= result.total_score <= 100

    def test_score_with_no_data(self):
        result = compute_fitness(None, None)
        assert 0 <= result.total_score <= 100

    def test_score_with_extreme_positive(self):
        bt = {"out_of_sample": {
            "return_pct": 500, "sharpe": 10.0, "win_rate": 99,
            "n_trades": 1000, "max_drawdown_pct": 0, "avg_edge": 50,
            "calibration": {},
        }}
        tr = {"passed": 100, "failed": 0, "total": 100, "safeguard_passed": True}
        result = compute_fitness(bt, tr)
        assert 0 <= result.total_score <= 100

    def test_score_with_extreme_negative(self):
        bt = {"out_of_sample": {
            "return_pct": -99, "sharpe": -10.0, "win_rate": 1,
            "n_trades": 1000, "max_drawdown_pct": 99, "avg_edge": -50,
            "calibration": {},
        }}
        tr = {"passed": 0, "failed": 100, "total": 100, "safeguard_passed": True}
        result = compute_fitness(bt, tr)
        assert 0 <= result.total_score <= 100


# ---------------------------------------------------------------------------
# Safeguard veto tests
# ---------------------------------------------------------------------------

class TestSafeguardVeto:
    """Safeguard failure should zero the score."""

    def test_safeguard_failure_zeroes_score(self):
        bt = {"out_of_sample": {
            "return_pct": 50, "sharpe": 3.0, "win_rate": 80,
            "n_trades": 100, "max_drawdown_pct": 5, "avg_edge": 15,
            "calibration": {},
        }}
        tr = {"passed": 29, "failed": 1, "total": 30, "safeguard_passed": False}
        result = compute_fitness(bt, tr)
        assert result.total_score == 0.0, "Safeguard failure must zero the score"
        assert result.safeguard_veto is True

    def test_safeguard_pass_allows_score(self):
        bt = {"out_of_sample": {
            "return_pct": 10, "sharpe": 0.5, "win_rate": 55,
            "n_trades": 30, "max_drawdown_pct": 20, "avg_edge": 8,
            "calibration": {},
        }}
        tr = {"passed": 30, "failed": 0, "total": 30, "safeguard_passed": True}
        result = compute_fitness(bt, tr)
        assert result.total_score > 0, "With safeguards passing, score should be > 0"
        assert result.safeguard_veto is False

    def test_safeguard_veto_with_perfect_scores(self):
        """Even perfect performance should be zeroed with safeguard failure."""
        bt = {"out_of_sample": {
            "return_pct": 100, "sharpe": 5.0, "win_rate": 95,
            "n_trades": 200, "max_drawdown_pct": 1, "avg_edge": 20,
            "calibration": {
                "50-60": {"avg_fair_value_pct": 55, "actual_win_rate_pct": 55, "n": 50},
            },
        }}
        tr = {"passed": 49, "failed": 1, "total": 50, "safeguard_passed": False}
        result = compute_fitness(bt, tr)
        assert result.total_score == 0.0


# ---------------------------------------------------------------------------
# Perfect / terrible input tests
# ---------------------------------------------------------------------------

class TestPerfectAndTerrible:
    """Perfect inputs should give high score, terrible should give low."""

    def test_perfect_inputs_high_score(self):
        bt = {"out_of_sample": {
            "return_pct": 100, "sharpe": 3.0, "win_rate": 80,
            "n_trades": 200, "max_drawdown_pct": 2, "avg_edge": 15,
            "calibration": {
                "30-40": {"avg_fair_value_pct": 35, "actual_win_rate_pct": 35, "n": 20},
                "50-60": {"avg_fair_value_pct": 55, "actual_win_rate_pct": 55, "n": 30},
                "70+":   {"avg_fair_value_pct": 75, "actual_win_rate_pct": 75, "n": 25},
            },
        }}
        tr = {"passed": 50, "failed": 0, "total": 50, "safeguard_passed": True}
        result = compute_fitness(bt, tr)
        assert result.total_score >= 75, (
            f"Perfect inputs should score >= 75, got {result.total_score}"
        )

    def test_terrible_inputs_low_score(self):
        bt = {"out_of_sample": {
            "return_pct": -80, "sharpe": -3.0, "win_rate": 10,
            "n_trades": 100, "max_drawdown_pct": 90, "avg_edge": -20,
            "calibration": {
                "70+": {"avg_fair_value_pct": 80, "actual_win_rate_pct": 10, "n": 50},
            },
        }}
        tr = {"passed": 5, "failed": 25, "total": 30, "safeguard_passed": True}
        result = compute_fitness(bt, tr)
        assert result.total_score <= 35, (
            f"Terrible inputs should score <= 35, got {result.total_score}"
        )


# ---------------------------------------------------------------------------
# Individual component bound tests
# ---------------------------------------------------------------------------

class TestComponentBounds:
    """Each component score should be in [0, 1]."""

    def test_all_components_bounded(self):
        bt = {"out_of_sample": {
            "return_pct": 15, "sharpe": 1.2, "win_rate": 58,
            "n_trades": 50, "max_drawdown_pct": 12, "avg_edge": 10,
            "calibration": {},
        }}
        tr = {"passed": 28, "failed": 2, "total": 30, "safeguard_passed": True}
        result = compute_fitness(bt, tr)

        for name, value in result.components.items():
            assert 0 <= value <= 1, (
                f"Component {name} = {value} is out of [0, 1] range"
            )

    def test_safeguard_integrity_is_binary(self):
        """Safeguard should be exactly 0.0 or 1.0."""
        tr_pass = {"passed": 30, "failed": 0, "total": 30, "safeguard_passed": True}
        tr_fail = {"passed": 29, "failed": 1, "total": 30, "safeguard_passed": False}

        r_pass = compute_fitness(None, tr_pass)
        r_fail = compute_fitness(None, tr_fail)

        assert r_pass.components["safeguard_integrity"] == 1.0
        assert r_fail.components["safeguard_integrity"] == 0.0

    def test_report_has_all_components(self):
        result = compute_fitness()
        assert isinstance(result.components, dict)
        assert len(result.components) == 7
        expected_keys = {
            "oos_return", "oos_sharpe", "calibration", "win_rate_margin",
            "max_drawdown", "test_pass_rate", "safeguard_integrity",
        }
        assert set(result.components.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Sigmoid mapping tests
# ---------------------------------------------------------------------------

class TestSigmoidMapping:
    """Test the sigmoid_map helper function."""

    def test_center_maps_to_half(self):
        assert abs(sigmoid_map(0.0, center=0.0, scale=1.0) - 0.5) < 1e-6

    def test_positive_offset_above_half(self):
        assert sigmoid_map(5.0, center=0.0, scale=1.0) > 0.5

    def test_negative_offset_below_half(self):
        assert sigmoid_map(-5.0, center=0.0, scale=1.0) < 0.5

    def test_output_always_between_0_and_1(self):
        for x in [-1000, -100, -1, 0, 1, 100, 1000]:
            val = sigmoid_map(x, center=0.0, scale=10.0)
            assert 0.0 <= val <= 1.0, f"sigmoid_map({x}) = {val}"  # inclusive bounds

    def test_larger_scale_more_gradual(self):
        """Larger scale should produce values closer to 0.5 for same offset."""
        narrow = sigmoid_map(10.0, center=0.0, scale=1.0)
        wide = sigmoid_map(10.0, center=0.0, scale=100.0)
        assert abs(wide - 0.5) < abs(narrow - 0.5)

    def test_monotonic(self):
        """sigmoid_map should be monotonically increasing in x."""
        prev = 0.0
        for x in range(-10, 11):
            val = sigmoid_map(float(x), center=0.0, scale=2.0)
            assert val > prev or x == -10
            prev = val


# ---------------------------------------------------------------------------
# Comparative scoring tests
# ---------------------------------------------------------------------------

class TestComparativeScoring:
    """Higher quality inputs should produce higher scores."""

    def test_higher_sharpe_higher_score(self):
        bt_low = {"out_of_sample": {
            "return_pct": 10, "sharpe": 0.5, "win_rate": 55,
            "n_trades": 30, "max_drawdown_pct": 20, "avg_edge": 8,
            "calibration": {},
        }}
        bt_high = {"out_of_sample": {
            "return_pct": 10, "sharpe": 2.0, "win_rate": 55,
            "n_trades": 30, "max_drawdown_pct": 20, "avg_edge": 8,
            "calibration": {},
        }}
        tr = {"passed": 30, "failed": 0, "total": 30, "safeguard_passed": True}
        r_low = compute_fitness(bt_low, tr)
        r_high = compute_fitness(bt_high, tr)
        assert r_high.total_score > r_low.total_score

    def test_higher_return_higher_score(self):
        tr = {"passed": 30, "failed": 0, "total": 30, "safeguard_passed": True}
        r_low = compute_fitness(
            {"out_of_sample": {"return_pct": -20, "sharpe": 0, "win_rate": 50,
             "n_trades": 30, "max_drawdown_pct": 20, "avg_edge": 5, "calibration": {}}},
            tr,
        )
        r_high = compute_fitness(
            {"out_of_sample": {"return_pct": 40, "sharpe": 0, "win_rate": 50,
             "n_trades": 30, "max_drawdown_pct": 20, "avg_edge": 5, "calibration": {}}},
            tr,
        )
        assert r_high.total_score > r_low.total_score

    def test_lower_drawdown_higher_score(self):
        tr = {"passed": 30, "failed": 0, "total": 30, "safeguard_passed": True}
        r_bad = compute_fitness(
            {"out_of_sample": {"return_pct": 10, "sharpe": 1, "win_rate": 55,
             "n_trades": 30, "max_drawdown_pct": 60, "avg_edge": 8, "calibration": {}}},
            tr,
        )
        r_good = compute_fitness(
            {"out_of_sample": {"return_pct": 10, "sharpe": 1, "win_rate": 55,
             "n_trades": 30, "max_drawdown_pct": 5, "avg_edge": 8, "calibration": {}}},
            tr,
        )
        assert r_good.total_score > r_bad.total_score


# ---------------------------------------------------------------------------
# FitnessResult dataclass tests
# ---------------------------------------------------------------------------

class TestFitnessResultDataclass:
    """Test the FitnessResult dataclass behavior."""

    def test_result_is_dataclass(self):
        result = compute_fitness()
        assert isinstance(result, FitnessResult)
        assert hasattr(result, "total_score")
        assert hasattr(result, "components")
        assert hasattr(result, "safeguard_veto")
        assert hasattr(result, "timestamp")

    def test_to_dict(self):
        result = compute_fitness()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_score" in d
        assert "components" in d
        assert "safeguard_veto" in d
        assert "timestamp" in d

    def test_timestamp_auto_populated(self):
        result = compute_fitness()
        assert len(result.timestamp) > 0
