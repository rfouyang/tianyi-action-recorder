"""Validate uploaded G1 pose JSON against the vendored model contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from util.g1_asset_helper import G1AssetHelper

G1_MODEL_ID = "unitree_g1_29dof_rev_1_0_fake_hand"
G1_POSE_TYPES = ("base", "left_arm", "right_arm", "composed")
G1_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
G1_LEFT_ARM_JOINT_NAMES = G1_JOINT_NAMES[3:10]
G1_RIGHT_ARM_JOINT_NAMES = G1_JOINT_NAMES[10:17]
G1_VISUAL_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    *G1_JOINT_NAMES,
)


@dataclass(frozen=True, slots=True)
class G1Pose:
    """Validated two-arm G1 source pose."""

    name: str
    pose_type: str
    robot_model_id: str
    joint_values: Mapping[str, float]


class G1PoseAdapter:
    """Parse untrusted browser uploads without importing the G1 application."""

    MAX_POSE_BYTES = 256 * 1024

    def __init__(self, *, asset_dir: Path) -> None:
        self.assets = G1AssetHelper(asset_dir=asset_dir)
        self.assets.validate()
        self.mjcf_path = self.assets.mjcf_path
        self.retarget_baseline_path = self.assets.retarget_baseline_path

    def parse_pose(self, *, content: str, file_name: str) -> G1Pose:
        if not isinstance(content, str):
            raise ValueError("Uploaded G1 pose content must be text")
        if len(content.encode("utf-8")) > self.MAX_POSE_BYTES:
            raise ValueError("Uploaded G1 pose JSON cannot exceed 256 KiB")
        normalized_file_name = self._validate_file_name(file_name)
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Uploaded G1 pose {normalized_file_name!r} is not valid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError("G1 pose JSON root must be an object")
        if payload.get("schema_version") != 1:
            raise ValueError("Only G1 pose schema version 1 is supported")
        normalized_name = self._validate_name(payload.get("name"))
        normalized_type = self._validate_pose_type(payload.get("pose_type"))
        if payload.get("robot_model_id") != G1_MODEL_ID:
            raise ValueError(
                f"G1 pose model {payload.get('robot_model_id')} does not match {G1_MODEL_ID}"
            )
        joint_values = payload.get("joint_values")
        if not isinstance(joint_values, dict):
            raise ValueError("G1 pose joint_values must be an object")
        expected_joint_names = self._joint_names(normalized_type)
        if set(joint_values) != set(expected_joint_names):
            missing = sorted(set(expected_joint_names) - set(joint_values))
            unknown = sorted(set(joint_values) - set(expected_joint_names))
            raise ValueError(
                f"Invalid G1 two-arm pose joint set: missing={missing}, unknown={unknown}"
            )
        normalized_values: dict[str, float] = {}
        for joint_name in expected_joint_names:
            value = joint_values[joint_name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"G1 joint {joint_name} must be numeric")
            normalized_value = float(value)
            if not math.isfinite(normalized_value):
                raise ValueError(f"G1 joint {joint_name} must be finite")
            normalized_values[joint_name] = normalized_value
        return G1Pose(
            name=normalized_name,
            pose_type=normalized_type,
            robot_model_id=G1_MODEL_ID,
            joint_values=normalized_values,
        )

    def load_retarget_baseline(self) -> G1Pose:
        return self.parse_pose(
            content=self.retarget_baseline_path.read_text(encoding="utf-8"),
            file_name=self.retarget_baseline_path.name,
        )

    @staticmethod
    def _validate_pose_type(pose_type: object) -> str:
        if pose_type not in G1_POSE_TYPES:
            raise ValueError(
                "G1 retargeting supports base, composed, left_arm, and right_arm poses"
            )
        return pose_type

    @staticmethod
    def _joint_names(pose_type: str) -> tuple[str, ...]:
        if pose_type == "left_arm":
            return G1_LEFT_ARM_JOINT_NAMES
        if pose_type == "right_arm":
            return G1_RIGHT_ARM_JOINT_NAMES
        return G1_JOINT_NAMES

    @staticmethod
    def _validate_name(name: object) -> str:
        if not isinstance(name, str):
            raise ValueError("G1 pose name must be a string")
        normalized = name.strip()
        if not normalized or len(normalized) > 100:
            raise ValueError("G1 pose name must contain 1 to 100 characters")
        if normalized in {".", ".."} or any(
            character in normalized for character in '<>:"/\\|?*'
        ):
            raise ValueError("G1 pose name contains a path-unsafe character")
        return normalized

    @staticmethod
    def _validate_file_name(file_name: str) -> str:
        if not isinstance(file_name, str):
            raise ValueError("Uploaded G1 pose filename must be a string")
        normalized = file_name.strip()
        if not normalized or Path(normalized).name != normalized:
            raise ValueError("Uploaded G1 pose filename is invalid")
        if Path(normalized).suffix.lower() != ".json":
            raise ValueError("Select a G1 pose file with a .json extension")
        return normalized
