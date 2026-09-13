"""Transport schemas for pose retargeting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetargetPoseCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file_name: str
    source_pose_json: str = Field(max_length=262144)


class SaveRetargetedPoseCommand(RetargetPoseCommand):
    output_pose_type: Literal["left_arm", "right_arm"]
    name: str
    notes: str = ""
    overwrite: bool = False


class ArmRetargetReportResponse(BaseModel):
    side: Literal["left", "right"]
    converged: bool
    iterations: int
    hand_position_error_m: float
    hand_orientation_error_rad: float
    elbow_position_error_m: float
    upper_arm_direction_error_rad: float
    forearm_direction_error_rad: float
    reached_joint_limits: tuple[str, ...]


class RetargetPoseResponse(BaseModel):
    source_pose_type: str
    source_name: str
    source_model_id: str
    target_model_id: str
    converged: bool
    arm_reports: tuple[ArmRetargetReportResponse, ...]
    contact_count: int
    joint_positions: dict[str, float]
    revision: int
    saved: bool
    saved_pose_name: str | None = None
    saved_pose_type: Literal["left_arm", "right_arm"] | None = None
