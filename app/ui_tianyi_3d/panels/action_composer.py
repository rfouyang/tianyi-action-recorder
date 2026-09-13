"""Action Composer presentation logic for the Tianyi 3D UI."""

from __future__ import annotations

from app.api_tianyi_3d.actions.schemas import ActionDefinitionCommand, CompileActionCommand
from app.ui_tianyi_3d.context import UIContext
from component.action import (
    ActionDefinition,
    ActionPoseReference,
    ActionTrajectory,
    ActionTransition,
)
from component.pose import PoseType


class ActionComposerPanel:
    """Translate browser commands into shared action capability calls."""

    default_transition_seconds = 1.0
    default_sample_frequency_hz = 25.0

    @classmethod
    def template_context(cls, context: UIContext) -> dict[str, object]:
        return {
            **cls.sources(context),
            "default_action_transition_seconds": cls.default_transition_seconds,
            "default_action_sample_frequency_hz": cls.default_sample_frequency_hz,
        }

    @staticmethod
    def sources(context: UIContext) -> dict[str, object]:
        choices = []
        for pose_type, label in ((PoseType.BASE, "Base"), (PoseType.COMPOSED, "Composed")):
            for pose in context.app.pose_service.list_poses(pose_type=pose_type):
                choices.append(
                    {
                        "value": f"{pose_type.value}/{pose.name}",
                        "label": f"{label} · {pose.name}",
                    }
                )
        return {
            "action_home_pose_name": ActionDefinition.HOME_POSE_NAME,
            "action_pose_choices": tuple(choices),
            "saved_action_names": tuple(
                action.name for action in context.app.action_service.list_actions()
            ),
            "compiled_action_names": context.app.action_service.list_trajectories(),
        }

    @classmethod
    def create_action(
        cls,
        *,
        context: UIContext,
        command: ActionDefinitionCommand,
    ) -> ActionDefinition:
        frames = tuple(
            ActionTransition(
                target_pose=ActionPoseReference(frame.pose_type, frame.name),
                duration_seconds=frame.duration_seconds,
                hold_seconds=frame.hold_seconds,
            )
            for frame in command.frames
        )
        return context.app.action_service.create_action(
            name=command.name,
            frames=frames,
            return_duration_seconds=command.return_duration_seconds,
            notes=command.notes,
        )

    @classmethod
    def save(
        cls,
        *,
        context: UIContext,
        command: ActionDefinitionCommand,
    ) -> ActionDefinition:
        action = cls.create_action(context=context, command=command)
        context.app.action_service.save_action(action=action, overwrite=command.overwrite)
        return action

    @classmethod
    def compile(
        cls,
        *,
        context: UIContext,
        command: CompileActionCommand,
    ) -> ActionTrajectory:
        action = cls.create_action(context=context, command=command)
        trajectory = context.app.action_service.generate_trajectory(
            action=action,
            sample_frequency_hz=command.sample_frequency_hz,
        )
        context.app.action_service.save_trajectory(
            trajectory=trajectory,
            overwrite=command.overwrite,
        )
        return trajectory

    @staticmethod
    def load(*, context: UIContext, name: str) -> dict[str, object]:
        action = context.app.action_service.load_action(name=name)
        return {
            "name": action.name,
            "home_pose": action.initial_pose.name,
            "notes": action.notes,
            "frames": tuple(
                {
                    "pose_type": item.target_pose.pose_type.value,
                    "name": item.target_pose.name,
                    "duration_seconds": item.duration_seconds,
                    "hold_seconds": item.hold_seconds,
                }
                for item in action.transitions[:-1]
            ),
            "return_duration_seconds": action.transitions[-1].duration_seconds,
            "total_duration_seconds": action.total_duration_seconds,
        }

    @staticmethod
    def preview_pose(*, context: UIContext, pose_type: PoseType, name: str) -> int:
        reference = ActionPoseReference(pose_type, name)
        pose = context.app.pose_service.load_pose(
            pose_type=reference.pose_type,
            name=reference.name,
        )
        return context.app.simulation.apply_pose(pose).revision
