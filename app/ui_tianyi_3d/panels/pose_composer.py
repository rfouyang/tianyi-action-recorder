"""Pose Composer workspace logic for the Tianyi browser UI."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.ui_tianyi_3d.context import UIContext
from component.pose import PoseDefinition


class ComposePoseCommand(BaseModel):
    """Browser command for previewing or saving a composed pose."""

    model_config = ConfigDict(extra="forbid")

    base_pose: str
    left_arm_pose: str | None = None
    right_arm_pose: str | None = None
    name: str = ""
    notes: str = ""
    overwrite: bool = False


class PoseComposerPanel:
    """Compose persisted pose parts and apply them to shared MuJoCo state."""

    @staticmethod
    def template_context(context: UIContext) -> dict[str, object]:
        pose_names = context.app.pose_service.list_pose_names()
        return {
            "base_pose_names": pose_names["base"],
            "left_arm_pose_names": pose_names["left_arm"],
            "right_arm_pose_names": pose_names["right_arm"],
            "composed_pose_names": pose_names["composed"],
        }

    @staticmethod
    def pose_names(context: UIContext) -> dict[str, tuple[str, ...]]:
        return context.app.pose_service.list_pose_names()

    @classmethod
    def preview(
        cls,
        *,
        context: UIContext,
        command: ComposePoseCommand,
    ) -> tuple[PoseDefinition, int]:
        pose = cls._compose(
            context=context,
            command=command,
            fallback_name="unsaved_composed_preview",
        )
        snapshot = context.app.simulation.apply_pose(pose)
        return pose, snapshot.revision

    @classmethod
    def save(
        cls,
        *,
        context: UIContext,
        command: ComposePoseCommand,
    ) -> tuple[PoseDefinition, int]:
        pose = cls._compose(context=context, command=command)
        context.app.pose_service.save_pose(pose=pose, overwrite=command.overwrite)
        snapshot = context.app.simulation.apply_pose(pose)
        return pose, snapshot.revision

    @staticmethod
    def _compose(
        *,
        context: UIContext,
        command: ComposePoseCommand,
        fallback_name: str | None = None,
    ) -> PoseDefinition:
        name = command.name.strip() or fallback_name
        if name is None:
            raise ValueError("Enter a composed pose name")
        return context.app.pose_service.compose_pose_from_names(
            name=name,
            base_pose_name=command.base_pose,
            left_arm_pose_name=command.left_arm_pose,
            right_arm_pose_name=command.right_arm_pose,
            notes=command.notes,
        )
