"""Canonical Tianyi body-joint groups and saved-pose shapes."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType

from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper

LOGGER = logging.getLogger(__name__)


class PoseType(str, Enum):
    """Supported saved pose shapes."""

    BASE = "base"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    COMPOSED = "composed"


class JointGroup(str, Enum):
    """Physical groups in the pinned Tianyi 21-joint body model."""

    LEG_MECHANISM = "leg_mechanism"
    WAIST = "waist"
    HEAD = "head"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"


@dataclass(frozen=True, slots=True)
class JointDefinition:
    """One named joint and its position in the fixed-base MuJoCo state."""

    name: str
    model_index: int
    group: JointGroup
    axis: tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    effort_limit: float
    velocity_limit: float
    authored: bool


@dataclass(frozen=True, slots=True)
class ArmMirrorRule:
    """Map one anatomical left-arm joint to its right-arm counterpart."""

    left_joint: str
    right_joint: str
    sign: float


class TianyiJointSchema:
    """Application-level view of Tianyi's body and saved-pose joint contract."""

    MODEL_JOINT_NAMES = (
        "first_leg_pitch_joint",
        "second_leg_pitch_joint",
        "waist_pitch_joint",
        "waist_yaw_joint",
        "head_yaw_joint",
        "head_pitch_joint",
        "head_roll_joint",
        "shoulder_pitch_l_joint",
        "shoulder_roll_l_joint",
        "shoulder_yaw_l_joint",
        "elbow_pitch_l_joint",
        "elbow_yaw_l_joint",
        "wrist_pitch_l_joint",
        "wrist_roll_l_joint",
        "shoulder_pitch_r_joint",
        "shoulder_roll_r_joint",
        "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint",
        "elbow_yaw_r_joint",
        "wrist_pitch_r_joint",
        "wrist_roll_r_joint",
    )
    LEG_MECHANISM_JOINT_NAMES = MODEL_JOINT_NAMES[:2]
    WAIST_JOINT_NAMES = MODEL_JOINT_NAMES[2:4]
    HEAD_JOINT_NAMES = MODEL_JOINT_NAMES[4:7]
    LEFT_ARM_JOINT_NAMES = MODEL_JOINT_NAMES[7:14]
    RIGHT_ARM_JOINT_NAMES = MODEL_JOINT_NAMES[14:21]
    ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
    BASE_JOINT_NAMES = HEAD_JOINT_NAMES + ARM_JOINT_NAMES
    LOCKED_JOINT_NAMES = LEG_MECHANISM_JOINT_NAMES + WAIST_JOINT_NAMES
    LOCKED_JOINT_MOTOR_IDS = MappingProxyType(
        {
            "first_leg_pitch_joint": 51,
            "second_leg_pitch_joint": 52,
            "waist_pitch_joint": 32,
            "waist_yaw_joint": 31,
        }
    )
    LOCKED_JOINT_VALUES = MappingProxyType(
        {
            "first_leg_pitch_joint": 0.14,
            "second_leg_pitch_joint": -0.4,
            "waist_pitch_joint": 0.24,
            "waist_yaw_joint": 0.0,
        }
    )
    ARM_MIRROR_RULES = (
        ArmMirrorRule("shoulder_pitch_l_joint", "shoulder_pitch_r_joint", 1.0),
        ArmMirrorRule("shoulder_roll_l_joint", "shoulder_roll_r_joint", -1.0),
        ArmMirrorRule("shoulder_yaw_l_joint", "shoulder_yaw_r_joint", -1.0),
        ArmMirrorRule("elbow_pitch_l_joint", "elbow_pitch_r_joint", 1.0),
        ArmMirrorRule("elbow_yaw_l_joint", "elbow_yaw_r_joint", -1.0),
        ArmMirrorRule("wrist_pitch_l_joint", "wrist_pitch_r_joint", 1.0),
        ArmMirrorRule("wrist_roll_l_joint", "wrist_roll_r_joint", -1.0),
    )

    def __init__(self, *, asset_helper: TianyiAssetHelper) -> None:
        asset_helper.validate()
        metadata = asset_helper.load_metadata()
        model_joint_specs = asset_helper.load_joint_specs()
        if tuple(model_joint_specs) != self.MODEL_JOINT_NAMES:
            raise ValueError(
                "Pinned Tianyi model joint names or order do not match the application schema"
            )

        self.model_id = str(metadata["model_id"])
        self._definitions = tuple(
            JointDefinition(
                name=joint_name,
                model_index=model_index,
                group=self._joint_group(model_index),
                axis=model_joint_specs[joint_name].axis,
                lower_limit=model_joint_specs[joint_name].lower_limit,
                upper_limit=model_joint_specs[joint_name].upper_limit,
                effort_limit=model_joint_specs[joint_name].effort_limit,
                velocity_limit=model_joint_specs[joint_name].velocity_limit,
                authored=joint_name in self.BASE_JOINT_NAMES,
            )
            for model_index, joint_name in enumerate(self.MODEL_JOINT_NAMES)
        )
        self._definitions_by_name = {
            definition.name: definition for definition in self._definitions
        }

    @property
    def definitions(self) -> tuple[JointDefinition, ...]:
        return self._definitions

    def definition(self, joint_name: str) -> JointDefinition:
        try:
            return self._definitions_by_name[joint_name]
        except KeyError as error:
            raise ValueError(f"Unknown Tianyi joint: {joint_name}") from error

    def joint_names(self, pose_type: PoseType) -> tuple[str, ...]:
        """Return the exact, canonical joint order for one saved pose shape."""
        if not isinstance(pose_type, PoseType):
            raise ValueError("pose_type must be a PoseType")
        if pose_type is PoseType.LEFT_ARM:
            return self.LEFT_ARM_JOINT_NAMES
        if pose_type is PoseType.RIGHT_ARM:
            return self.RIGHT_ARM_JOINT_NAMES
        return self.BASE_JOINT_NAMES

    def extract_from_model_positions(
        self,
        *,
        model_positions: Sequence[float],
        pose_type: PoseType,
    ) -> dict[str, float]:
        """Extract one pose from positions ordered like the pinned fixed-base model."""
        if len(model_positions) != len(self.MODEL_JOINT_NAMES):
            raise ValueError(
                f"Model state has {len(model_positions)} positions; "
                f"expected {len(self.MODEL_JOINT_NAMES)}"
            )
        extracted_values = {
            joint_name: model_positions[self.definition(joint_name).model_index]
            for joint_name in self.joint_names(pose_type)
        }
        return self.validate_joint_values(
            pose_type=pose_type,
            joint_values=extracted_values,
        )

    def validate_joint_values(
        self,
        *,
        pose_type: PoseType,
        joint_values: Mapping[str, object],
    ) -> dict[str, float]:
        """Validate an exact pose joint set, finite values, and URDF limits."""
        if not isinstance(joint_values, Mapping):
            raise ValueError("joint_values must be a mapping")
        expected_joint_names = self.joint_names(pose_type)
        expected_joint_set = set(expected_joint_names)
        actual_joint_set = set(joint_values)
        if actual_joint_set != expected_joint_set:
            missing = sorted(expected_joint_set - actual_joint_set)
            unknown = sorted(actual_joint_set - expected_joint_set)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ValueError(f"Invalid {pose_type.value} joint set: {', '.join(details)}")

        validated_values: dict[str, float] = {}
        for joint_name in expected_joint_names:
            validated_values[joint_name] = self.validate_joint_value(
                joint_name=joint_name,
                value=joint_values[joint_name],
            )
        return validated_values

    def validate_joint_value(self, *, joint_name: str, value: object) -> float:
        """Validate one named body-joint value against the pinned model."""
        definition = self.definition(joint_name)
        normalized_value = self._finite_joint_value(joint_name=joint_name, value=value)
        if not definition.lower_limit <= normalized_value <= definition.upper_limit:
            raise ValueError(
                f"Joint {joint_name} value {normalized_value} is outside "
                f"[{definition.lower_limit}, {definition.upper_limit}]"
            )
        return normalized_value

    def calibration_limits(self, joint_name: str) -> tuple[float, float]:
        """Return the URDF range expanded to include the configured baseline."""
        if joint_name not in self.LOCKED_JOINT_VALUES:
            raise ValueError(f"Joint {joint_name} is not a calibration joint")
        definition = self.definition(joint_name)
        baseline = self.LOCKED_JOINT_VALUES[joint_name]
        return (
            min(definition.lower_limit, baseline),
            max(definition.upper_limit, baseline),
        )

    def validate_calibration_joint_value(
        self,
        *,
        joint_name: str,
        value: object,
    ) -> float:
        """Validate one UI-only leg/waist calibration value."""
        lower_limit, upper_limit = self.calibration_limits(joint_name)
        normalized_value = self._finite_joint_value(joint_name=joint_name, value=value)
        if not lower_limit <= normalized_value <= upper_limit:
            raise ValueError(
                f"Calibration joint {joint_name} value {normalized_value} is outside "
                f"[{lower_limit}, {upper_limit}]"
            )
        return normalized_value

    def neutral_values(self, pose_type: PoseType) -> dict[str, float]:
        """Return a zero pose in canonical order for the requested shape."""
        return {joint_name: 0.0 for joint_name in self.joint_names(pose_type)}

    def locked_values(self) -> dict[str, float]:
        """Return the immutable SDK-ID calibration used for locked body joints."""
        return dict(self.LOCKED_JOINT_VALUES)

    def mirror_arm_values(
        self,
        *,
        source_pose_type: PoseType,
        joint_values: Mapping[str, object],
    ) -> tuple[PoseType, dict[str, float]]:
        """Validate and reflect one exact arm across the robot's XZ plane."""
        if source_pose_type not in {PoseType.LEFT_ARM, PoseType.RIGHT_ARM}:
            raise ValueError("Only left_arm and right_arm values can be mirrored")
        validated = self.validate_joint_values(
            pose_type=source_pose_type,
            joint_values=joint_values,
        )
        if source_pose_type is PoseType.LEFT_ARM:
            target_pose_type = PoseType.RIGHT_ARM
            mirrored = {
                rule.right_joint: rule.sign * validated[rule.left_joint]
                for rule in self.ARM_MIRROR_RULES
            }
        else:
            target_pose_type = PoseType.LEFT_ARM
            mirrored = {
                rule.left_joint: rule.sign * validated[rule.right_joint]
                for rule in self.ARM_MIRROR_RULES
            }
        return target_pose_type, self.validate_joint_values(
            pose_type=target_pose_type,
            joint_values=mirrored,
        )

    @staticmethod
    def _finite_joint_value(*, joint_name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Joint {joint_name} must be a real number")
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            raise ValueError(f"Joint {joint_name} must be finite")
        return normalized_value

    @staticmethod
    def _joint_group(model_index: int) -> JointGroup:
        if model_index < 2:
            return JointGroup.LEG_MECHANISM
        if model_index < 4:
            return JointGroup.WAIST
        if model_index < 7:
            return JointGroup.HEAD
        if model_index < 14:
            return JointGroup.LEFT_ARM
        return JointGroup.RIGHT_ARM


def demo_tianyi_joint_schema() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = AppSettings()
    schema = TianyiJointSchema(
        asset_helper=TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
    )
    base_values = schema.extract_from_model_positions(
        model_positions=[0.0] * len(schema.MODEL_JOINT_NAMES),
        pose_type=PoseType.BASE,
    )
    LOGGER.info(
        "Loaded %d Tianyi body joints and extracted %d base-pose joints for model %s",
        len(schema.definitions),
        len(base_values),
        schema.model_id,
    )


def main() -> None:
    demo_tianyi_joint_schema()


if __name__ == "__main__":
    main()
