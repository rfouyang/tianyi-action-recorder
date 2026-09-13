"""View data and commands for the interactive pose recorder."""

from __future__ import annotations

from dataclasses import dataclass

from app.ui_tianyi_3d.context import UIContext
from component.pose import JointGroup, PoseDefinition, PoseType


@dataclass(frozen=True, slots=True)
class JointRow:
    """One authored or UI-only calibration joint shown by the pose recorder."""

    name: str
    label: str
    group: str
    lower_limit: float
    upper_limit: float
    target: float
    actual: float
    reset_value: float
    motor_id: int | None
    recordable: bool


class PoseRecorderPanel:
    """Build and apply the simulation-only Pose Recorder workspace."""

    EDITABLE_POSE_TYPES = (PoseType.BASE, PoseType.LEFT_ARM, PoseType.RIGHT_ARM)

    @classmethod
    def template_context(cls, context: UIContext) -> dict[str, object]:
        snapshot = context.app.simulation.snapshot()
        current_positions = snapshot.joint_position_map()
        schema = context.app.joint_schema
        default_pose_values = context.app.simulation.default_pose_values()
        editable_names = set(schema.BASE_JOINT_NAMES)
        calibration_names = set(schema.LOCKED_JOINT_NAMES)
        rows = tuple(
            JointRow(
                name=definition.name,
                label=cls._joint_label(definition.name),
                group=(
                    "calibration"
                    if definition.name in calibration_names
                    else definition.group.value
                ),
                lower_limit=(
                    schema.calibration_limits(definition.name)[0]
                    if definition.name in calibration_names
                    else definition.lower_limit
                ),
                upper_limit=(
                    schema.calibration_limits(definition.name)[1]
                    if definition.name in calibration_names
                    else definition.upper_limit
                ),
                target=current_positions[definition.name],
                actual=current_positions[definition.name],
                reset_value=(
                    schema.LOCKED_JOINT_VALUES[definition.name]
                    if definition.name in calibration_names
                    else default_pose_values[definition.name]
                ),
                motor_id=schema.LOCKED_JOINT_MOTOR_IDS.get(definition.name),
                recordable=definition.name in editable_names,
            )
            for definition in schema.definitions
            if definition.name in editable_names | calibration_names
        )
        return {
            "joint_rows": rows,
            "editable_joint_count": len(editable_names),
            "initial_revision": snapshot.revision,
            "default_pose_name": context.app.simulation.default_pose_name,
            "recorder_pose_names": context.app.pose_service.list_pose_names(),
        }

    @staticmethod
    def load_saved_pose(
        *,
        context: UIContext,
        pose_type: PoseType,
        name: str,
    ) -> tuple[PoseDefinition, PoseType, int]:
        """Load and apply a saved pose as an editable MuJoCo starting point."""
        pose = context.app.pose_service.load_pose(pose_type=pose_type, name=name)
        snapshot = context.app.simulation.apply_pose(pose)
        edit_pose_type = PoseType.BASE if pose_type is PoseType.COMPOSED else pose_type
        return pose, edit_pose_type, snapshot.revision

    @classmethod
    def save_simulation_pose(
        cls,
        *,
        context: UIContext,
        pose_type: PoseType,
        name: str,
        notes: str,
        joint_positions: dict[str, float],
        overwrite: bool,
    ) -> tuple[PoseDefinition, int]:
        """Validate, apply, and save the selected simulated pose group."""
        if pose_type not in cls.EDITABLE_POSE_TYPES:
            raise ValueError(f"Pose type {pose_type.value} cannot be recorded")

        schema = context.app.joint_schema
        validated_positions = schema.validate_joint_values(
            pose_type=pose_type,
            joint_values=joint_positions,
        )
        snapshot = context.app.simulation.update_joint_positions(validated_positions)
        actual_positions = snapshot.joint_position_map()
        pose = context.app.pose_service.create_simulated_pose(
            name=name,
            pose_type=pose_type,
            joint_values={
                joint_name: actual_positions[joint_name]
                for joint_name in schema.joint_names(pose_type)
            },
            notes=notes,
        )
        context.app.pose_service.save_pose(pose=pose, overwrite=overwrite)
        return pose, snapshot.revision

    @staticmethod
    def mirror_arm(
        *,
        context: UIContext,
        source_pose_type: PoseType,
    ) -> tuple[PoseType, dict[str, float], dict[str, float], int]:
        """Capture live MuJoCo values and mirror only the destination arm."""
        target_pose_type, source_positions, mirrored_positions, snapshot = (
            context.app.simulation.mirror_current_arm(
                source_pose_type=source_pose_type,
            )
        )
        return (
            target_pose_type,
            source_positions,
            mirrored_positions,
            snapshot.revision,
        )

    @staticmethod
    def _joint_label(joint_name: str) -> str:
        if joint_name.startswith("elbow_yaw_"):
            return "Wrist Yaw"
        label = joint_name.removesuffix("_joint")
        if label.endswith("_l") or label.endswith("_r"):
            label = label[:-2]
        return label.replace("_", " ").title()

    @staticmethod
    def group_label(group: JointGroup) -> str:
        labels = {
            JointGroup.HEAD: "Head",
            JointGroup.LEFT_ARM: "Robot-left",
            JointGroup.RIGHT_ARM: "Robot-right",
        }
        return labels[group]
