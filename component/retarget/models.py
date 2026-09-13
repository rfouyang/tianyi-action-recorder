"""Results returned by the G1-to-Tianyi pose retargeter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArmRetargetReport:
    side: str
    converged: bool
    iterations: int
    hand_position_error_m: float
    hand_orientation_error_rad: float
    elbow_position_error_m: float
    upper_arm_direction_error_rad: float
    forearm_direction_error_rad: float
    reached_joint_limits: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PoseRetargetResult:
    source_name: str
    source_pose_type: str
    source_model_id: str
    target_model_id: str
    source_joint_values: Mapping[str, float]
    joint_values: Mapping[str, float]
    arm_reports: tuple[ArmRetargetReport, ...]
    contact_count: int

    @property
    def converged(self) -> bool:
        return all(report.converged for report in self.arm_reports)
