"""Shewhart X-bar control chart with Western Electric rules.

Pure math — no model dependencies.
"""

from __future__ import annotations

import statistics

from synthdet.types import QualityControlChart


def build_control_chart(
    metric_name: str,
    baseline_values: list[float],
    new_values: list[float],
    sigma_multiplier: float = 3.0,
) -> QualityControlChart:
    """Build a Shewhart X-bar control chart.

    The center line and sigma are computed from *baseline_values*.
    Out-of-control indices are detected in *new_values* using Western Electric rules.

    Args:
        metric_name: Human-readable metric name (e.g. "backbone.stage3_mean").
        baseline_values: Historical values used to establish center line and sigma.
        new_values: New observations to check for drift.
        sigma_multiplier: Number of sigma for UCL/LCL (default 3.0).

    Returns:
        A populated QualityControlChart.
    """
    if len(baseline_values) < 2:
        center = baseline_values[0] if baseline_values else 0.0
        sigma = 0.0
    else:
        center = statistics.mean(baseline_values)
        sigma = statistics.stdev(baseline_values)

    ucl = center + sigma_multiplier * sigma
    lcl = center - sigma_multiplier * sigma

    ooc = check_western_electric_rules(new_values, center, sigma)

    return QualityControlChart(
        metric_name=metric_name,
        center_line=center,
        ucl=ucl,
        lcl=lcl,
        sigma=sigma,
        values=list(new_values),
        out_of_control_indices=ooc,
    )


def check_western_electric_rules(
    values: list[float],
    center_line: float,
    sigma: float,
    trend_window: int = 7,
) -> list[int]:
    """Apply Western Electric rules to detect out-of-control points.

    Rules:
        1. Any point beyond 3σ from center.
        2. Two of three consecutive points beyond 2σ on the same side.
        3. ``trend_window`` or more consecutive points on the same side of center.

    Returns:
        Sorted list of unique out-of-control indices.
    """
    if not values or sigma < 0:
        return []

    ooc: set[int] = set()
    ooc |= _rule_1_beyond_3sigma(values, center_line, sigma)
    ooc |= _rule_2_two_of_three_beyond_2sigma(values, center_line, sigma)
    ooc |= _rule_3_consecutive_same_side(values, center_line, trend_window)
    return sorted(ooc)


def _rule_1_beyond_3sigma(
    values: list[float], center_line: float, sigma: float
) -> set[int]:
    """Rule 1: any single point beyond 3σ."""
    if sigma == 0:
        return set()
    flagged: set[int] = set()
    for i, v in enumerate(values):
        if abs(v - center_line) > 3 * sigma:
            flagged.add(i)
    return flagged


def _rule_2_two_of_three_beyond_2sigma(
    values: list[float], center_line: float, sigma: float
) -> set[int]:
    """Rule 2: 2 of 3 consecutive points beyond 2σ on the same side."""
    if sigma == 0 or len(values) < 3:
        return set()
    flagged: set[int] = set()
    two_sigma = 2 * sigma
    for i in range(len(values) - 2):
        window = values[i : i + 3]
        # Check upper side
        above = [j for j, v in enumerate(window) if v - center_line > two_sigma]
        if len(above) >= 2:
            for j in above:
                flagged.add(i + j)
        # Check lower side
        below = [j for j, v in enumerate(window) if center_line - v > two_sigma]
        if len(below) >= 2:
            for j in below:
                flagged.add(i + j)
    return flagged


def _rule_3_consecutive_same_side(
    values: list[float], center_line: float, trend_window: int
) -> set[int]:
    """Rule 3: ``trend_window`` or more consecutive points on the same side of center."""
    if len(values) < trend_window or trend_window < 1:
        return set()
    flagged: set[int] = set()
    # Classify each point as above (+1), below (-1), or on center (0)
    sides = []
    for v in values:
        if v > center_line:
            sides.append(1)
        elif v < center_line:
            sides.append(-1)
        else:
            sides.append(0)

    run_start = 0
    run_side = sides[0]
    for i in range(1, len(sides)):
        if sides[i] == run_side and run_side != 0:
            # Continue run
            pass
        else:
            # Check if completed run is long enough
            if run_side != 0 and (i - run_start) >= trend_window:
                for j in range(run_start, i):
                    flagged.add(j)
            run_start = i
            run_side = sides[i]
    # Check final run
    if run_side != 0 and (len(sides) - run_start) >= trend_window:
        for j in range(run_start, len(sides)):
            flagged.add(j)

    return flagged
