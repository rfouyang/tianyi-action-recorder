"""Pose Retargeting workspace logic for the browser UI."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.ui_tianyi_3d.context import UIContext
from component.pose import PoseDefinition, PoseType
from component.retarget import PoseRetargetResult


class RetargetPoseCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_file_name: str
    source_pose_json: str = Field(max_length=262144)
    output_pose_type: PoseType | None = None
    name: str = ""
    notes: str = ""
    overwrite: bool = False


class PoseRetargetingPanel:
    """Retarget a saved G1 two-arm pose into shared Tianyi MuJoCo state."""

    @staticmethod
    def preview(
        *,
        context: UIContext,
        command: RetargetPoseCommand,
    ) -> tuple[PoseRetargetResult, int]:
        result = context.app.pose_retargeting.retarget(
            source_pose_json=command.source_pose_json,
            source_file_name=command.source_file_name,
        )
        snapshot = context.app.simulation.apply_pose_values(
            pose_type=PoseType.BASE,
            joint_values=result.joint_values,
        )
        context.viser.update_retarget_preview(result.source_joint_values)
        return result, snapshot.revision

    @classmethod
    def save(
        cls,
        *,
        context: UIContext,
        command: RetargetPoseCommand,
    ) -> tuple[PoseRetargetResult, PoseDefinition, int]:
        if not command.name.strip():
            raise ValueError("Enter a Tianyi arm pose name")
        if command.output_pose_type not in {PoseType.LEFT_ARM, PoseType.RIGHT_ARM}:
            raise ValueError("Select robot-left arm or robot-right arm as the output")
        result = context.app.pose_retargeting.retarget(
            source_pose_json=command.source_pose_json,
            source_file_name=command.source_file_name,
        )
        source_reference = (
            f"G1 {result.source_pose_type}/{result.source_name} "
            f"from {command.source_file_name}"
        )
        side_label = (
            "robot-left" if command.output_pose_type is PoseType.LEFT_ARM else "robot-right"
        )
        notes = f"Retargeted {side_label} arm in MuJoCo from {source_reference}."
        if command.notes.strip():
            notes = f"{notes} {command.notes.strip()}"
        arm_values = context.app.joint_schema.validate_joint_values(
            pose_type=command.output_pose_type,
            joint_values={
                joint_name: result.joint_values[joint_name]
                for joint_name in context.app.joint_schema.joint_names(
                    command.output_pose_type
                )
            },
        )
        pose = context.app.pose_service.create_retargeted_arm_pose(
            name=command.name,
            pose_type=command.output_pose_type,
            joint_values=arm_values,
            notes=notes,
        )
        context.app.pose_service.save_pose(pose=pose, overwrite=command.overwrite)
        snapshot = context.app.simulation.apply_pose(pose)
        context.viser.update_retarget_preview(result.source_joint_values)
        return result, pose, snapshot.revision
