"""Tests for synthdet.quality.control_chart — pure math, no mocking needed."""

from __future__ import annotations

import pytest

from synthdet.quality.control_chart import (
    _rule_1_beyond_3sigma,
    _rule_2_two_of_three_beyond_2sigma,
    _rule_3_consecutive_same_side,
    build_control_chart,
    check_western_electric_rules,
)


# ---------------------------------------------------------------------------
# build_control_chart
# ---------------------------------------------------------------------------


class TestBuildControlChart:
    def test_center_and_limits_from_baseline(self):
        baseline = [10.0, 12.0, 11.0, 13.0, 9.0]
        chart = build_control_chart("test_metric", baseline, [11.0])
        assert chart.metric_name == "test_metric"
        assert chart.center_line == pytest.approx(11.0)
        assert chart.sigma > 0
        assert chart.ucl == pytest.approx(chart.center_line + 3 * chart.sigma)
        assert chart.lcl == pytest.approx(chart.center_line - 3 * chart.sigma)

    def test_single_baseline_value(self):
        chart = build_control_chart("m", [5.0], [5.0])
        assert chart.center_line == 5.0
        assert chart.sigma == 0.0
        assert chart.ucl == 5.0
        assert chart.lcl == 5.0

    def test_empty_baseline(self):
        chart = build_control_chart("m", [], [1.0])
        assert chart.center_line == 0.0
        assert chart.sigma == 0.0

    def test_new_values_stored(self):
        chart = build_control_chart("m", [1.0, 2.0, 3.0], [10.0, 20.0])
        assert chart.values == [10.0, 20.0]

    def test_no_alerts_when_within_limits(self):
        baseline = [10.0] * 20
        # All identical → sigma = 0, so only exact matches are in-control
        # Use a spread baseline instead
        baseline = [9.0, 10.0, 11.0, 10.0, 9.5, 10.5, 10.0, 9.8, 10.2, 10.1]
        chart = build_control_chart("m", baseline, [10.0, 10.1, 9.9])
        assert chart.out_of_control_indices == []

    def test_alerts_when_beyond_limits(self):
        baseline = [10.0, 10.0, 10.0, 11.0, 9.0]
        chart = build_control_chart("m", baseline, [50.0])
        assert 0 in chart.out_of_control_indices

    def test_custom_sigma_multiplier(self):
        baseline = [0.0, 1.0]  # mean=0.5, stdev ~0.707
        chart = build_control_chart("m", baseline, [5.0], sigma_multiplier=1.0)
        # With 1-sigma limits, 5.0 should be way out
        assert 0 in chart.out_of_control_indices


# ---------------------------------------------------------------------------
# Rule 1: beyond 3σ
# ---------------------------------------------------------------------------


class TestRule1:
    def test_point_beyond_3sigma_flagged(self):
        # center=0, sigma=1: value 3.5 is beyond 3σ
        flagged = _rule_1_beyond_3sigma([0.0, 3.5, -3.5], 0.0, 1.0)
        assert 1 in flagged
        assert 2 in flagged
        assert 0 not in flagged

    def test_point_at_exactly_3sigma_not_flagged(self):
        # Exactly 3σ = boundary, not flagged (> not >=)
        flagged = _rule_1_beyond_3sigma([3.0], 0.0, 1.0)
        assert len(flagged) == 0

    def test_zero_sigma_no_flags(self):
        flagged = _rule_1_beyond_3sigma([1.0, 2.0, 3.0], 0.0, 0.0)
        assert len(flagged) == 0

    def test_empty_values(self):
        flagged = _rule_1_beyond_3sigma([], 0.0, 1.0)
        assert len(flagged) == 0


# ---------------------------------------------------------------------------
# Rule 2: 2 of 3 beyond 2σ on same side
# ---------------------------------------------------------------------------


class TestRule2:
    def test_two_of_three_above_flagged(self):
        # center=0, sigma=1, 2σ=2.0
        values = [2.5, 0.0, 2.5]  # indices 0, 2 are beyond 2σ above
        flagged = _rule_2_two_of_three_beyond_2sigma(values, 0.0, 1.0)
        assert 0 in flagged
        assert 2 in flagged

    def test_two_of_three_below_flagged(self):
        values = [-2.5, 0.0, -2.5]
        flagged = _rule_2_two_of_three_beyond_2sigma(values, 0.0, 1.0)
        assert 0 in flagged
        assert 2 in flagged

    def test_opposite_sides_not_flagged(self):
        values = [2.5, -2.5, 0.0]  # one above, one below — different sides
        flagged = _rule_2_two_of_three_beyond_2sigma(values, 0.0, 1.0)
        assert len(flagged) == 0

    def test_zero_sigma_no_flags(self):
        flagged = _rule_2_two_of_three_beyond_2sigma([5.0, 5.0, 5.0], 0.0, 0.0)
        assert len(flagged) == 0

    def test_too_few_values(self):
        flagged = _rule_2_two_of_three_beyond_2sigma([2.5, 2.5], 0.0, 1.0)
        assert len(flagged) == 0

    def test_at_exactly_2sigma_not_flagged(self):
        values = [2.0, 2.0, 2.0]  # exactly 2σ, not beyond
        flagged = _rule_2_two_of_three_beyond_2sigma(values, 0.0, 1.0)
        assert len(flagged) == 0


# ---------------------------------------------------------------------------
# Rule 3: consecutive same side
# ---------------------------------------------------------------------------


class TestRule3:
    def test_exactly_trend_window_consecutive_flagged(self):
        # 7 consecutive above center
        values = [1.0] * 7
        flagged = _rule_3_consecutive_same_side(values, 0.0, 7)
        assert flagged == set(range(7))

    def test_fewer_than_trend_window_not_flagged(self):
        values = [1.0] * 6
        flagged = _rule_3_consecutive_same_side(values, 0.0, 7)
        assert len(flagged) == 0

    def test_mixed_sides_not_flagged(self):
        values = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        flagged = _rule_3_consecutive_same_side(values, 0.0, 7)
        assert len(flagged) == 0

    def test_below_center_consecutive(self):
        values = [-1.0] * 8
        flagged = _rule_3_consecutive_same_side(values, 0.0, 7)
        assert len(flagged) == 8

    def test_on_center_breaks_run(self):
        values = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        flagged = _rule_3_consecutive_same_side(values, 0.0, 4)
        # Run of 4 at indices 4-7
        assert {4, 5, 6, 7}.issubset(flagged)
        # First run of 3 is not enough
        assert 0 not in flagged

    def test_empty_values(self):
        flagged = _rule_3_consecutive_same_side([], 0.0, 7)
        assert len(flagged) == 0


# ---------------------------------------------------------------------------
# check_western_electric_rules (combined)
# ---------------------------------------------------------------------------


class TestCombinedRules:
    def test_all_rules_combine(self):
        # center=0, sigma=1
        # Index 0: beyond 3σ (rule 1)
        # Indices 1-7: 7 consecutive above center (rule 3)
        values = [4.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ooc = check_western_electric_rules(values, 0.0, 1.0, trend_window=7)
        assert 0 in ooc  # rule 1
        for i in range(1, 8):
            assert i in ooc  # rule 3

    def test_clean_data_no_flags(self):
        # center=10, sigma=2, values all within 1σ
        values = [10.5, 9.5, 10.2, 9.8, 10.1]
        ooc = check_western_electric_rules(values, 10.0, 2.0, trend_window=7)
        assert ooc == []

    def test_negative_sigma_returns_empty(self):
        ooc = check_western_electric_rules([1.0, 2.0], 0.0, -1.0)
        assert ooc == []

    def test_returns_sorted_unique(self):
        # A value that triggers multiple rules
        values = [4.0, 4.0, 4.0]  # beyond 3σ and 2-of-3 beyond 2σ
        ooc = check_western_electric_rules(values, 0.0, 1.0)
        assert ooc == sorted(set(ooc))
