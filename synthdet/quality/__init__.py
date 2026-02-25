"""SPC quality monitoring for synthetic data generation."""

from synthdet.quality.control_chart import build_control_chart, check_western_electric_rules
from synthdet.quality.monitor import QualityMonitor

__all__ = ["build_control_chart", "check_western_electric_rules", "QualityMonitor"]
