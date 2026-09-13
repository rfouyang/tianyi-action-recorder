"""Persist Tianyi actions and compile direct joint-space MuJoCo trajectories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from component.action.models import (
    ActionDefinition,
    ActionPoseReference,
    ActionTrajectory,
    ActionTransition,
)
from component.pose import PoseDefinition, PoseService, PoseType, TianyiJointSchema
from util.numpy_archive_helper import NumpyArchiveHelper
from util.pose_file_helper import PoseFileHelper


@dataclass(frozen=True, slots=True)
class ResolvedAction:
    """An action whose complete pose references all resolve."""

    definition: ActionDefinition
    poses: tuple[PoseDefinition, ...]


class ActionService:
    """Own action rules, JSON definitions, and 17-joint NPZ trajectories."""

    TRAJECTORY_SCHEMA_VERSION = 1
    TRAJECTORY_ARRAY_NAMES = frozenset(
        {
            "schema_version",
            "action_name",
            "robot_model_id",
            "fps",
            "joint_names",
            "timestamps",
            "joint_positions",
            "keyframe_sample_indices",
            "source_pose_names",
            "keyframe_hold_seconds",
        }
    )

    def __init__(
        self,
        *,
        schema: TianyiJointSchema,
        pose_service: PoseService,
        definition_dir: Path,
        trajectory_dir: Path,
        file_helper: PoseFileHelper,
        archive_helper: NumpyArchiveHelper,
    ) -> None:
        self.schema = schema
        self.pose_service = pose_service
        self.definition_dir = definition_dir
        self.trajectory_dir = trajectory_dir
        self.file_helper = file_helper
        self.archive_helper = archive_helper

    def create_action(
        self,
        *,
        name: str,
        frames: Sequence[ActionTransition],
        return_duration_seconds: float,
        notes: str = "",
    ) -> ActionDefinition:
        """Create an action fixed to concierge_init at both boundaries."""
        home = ActionPoseReference(PoseType.BASE, ActionDefinition.HOME_POSE_NAME)
        transitions = tuple(frames) + (
            ActionTransition(home, duration_seconds=return_duration_seconds),
        )
        action = ActionDefinition.create(
            schema=self.schema,
            name=name,
            initial_pose=home,
            transitions=transitions,
            notes=notes,
        )
        self.resolve_action(action=action)
        return action

    def save_action(self, *, action: ActionDefinition, overwrite: bool = False) -> Path:
        self.resolve_action(action=action)
        return self.file_helper.write_json(
            path=self._definition_path(action.name),
            payload=action.to_dict(),
            overwrite=overwrite,
        )

    def load_action(self, *, name: str) -> ActionDefinition:
        normalized = PoseDefinition.validate_name(name)
        action = ActionDefinition.from_dict(
            schema=self.schema,
            payload=self.file_helper.read_json(path=self._definition_path(normalized)),
        )
        if action.name != normalized:
            raise ValueError("Action file contains a different action name")
        return action

    def list_actions(self) -> tuple[ActionDefinition, ...]:
        return tuple(
            self.load_action(name=path.stem)
            for path in self.file_helper.list_json_files(directory=self.definition_dir)
        )

    def resolve_action(self, *, action: ActionDefinition) -> ResolvedAction:
        if action.robot_model_id != self.schema.model_id:
            raise ValueError(
                f"Action model {action.robot_model_id} does not match {self.schema.model_id}"
            )
        poses = []
        for reference in action.pose_sequence:
            try:
                pose = self.pose_service.load_pose(
                    pose_type=reference.pose_type,
                    name=reference.name,
                )
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(
                    "Could not resolve action pose "
                    f"{reference.pose_type.value}/{reference.name}: {error}"
                ) from error
            if tuple(pose.joint_values) != self.schema.BASE_JOINT_NAMES:
                raise ValueError(
                    f"Action pose {pose.name} does not contain exactly 17 authored joints"
                )
            poses.append(pose)
        return ResolvedAction(definition=action, poses=tuple(poses))

    def generate_trajectory(
        self,
        *,
        action: ActionDefinition,
        sample_frequency_hz: float = 25.0,
    ) -> ActionTrajectory:
        """Linearly interpolate complete poses; zero hold adds no samples."""
        if isinstance(sample_frequency_hz, bool) or not isinstance(
            sample_frequency_hz,
            (int, float),
        ):
            raise ValueError("Trajectory frequency must be a real number")
        frequency = float(sample_frequency_hz)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("Trajectory frequency must be finite and greater than zero")
        resolved = self.resolve_action(action=action)
        vectors = [
            np.asarray(
                [pose.joint_values[name] for name in self.schema.BASE_JOINT_NAMES],
                dtype=np.float64,
            )
            for pose in resolved.poses
        ]
        samples = [vectors[0].copy()]
        timestamps = [0.0]
        keyframe_indices = [0]
        elapsed = 0.0
        for transition, start, target in zip(
            action.transitions,
            vectors[:-1],
            vectors[1:],
            strict=True,
        ):
            travel_steps = max(1, int(round(transition.duration_seconds * frequency)))
            for step in range(1, travel_steps + 1):
                fraction = step / travel_steps
                samples.append(start + ((target - start) * fraction))
                timestamps.append(elapsed + transition.duration_seconds * fraction)
            elapsed += transition.duration_seconds
            keyframe_indices.append(len(samples) - 1)
            if transition.hold_seconds > 0.0:
                hold_steps = max(1, int(round(transition.hold_seconds * frequency)))
                for step in range(1, hold_steps + 1):
                    samples.append(target.copy())
                    timestamps.append(elapsed + transition.hold_seconds * (step / hold_steps))
                elapsed += transition.hold_seconds
        trajectory = ActionTrajectory(
            action_name=action.name,
            robot_model_id=action.robot_model_id,
            joint_names=self.schema.BASE_JOINT_NAMES,
            timestamps=np.asarray(timestamps, dtype=np.float64),
            joint_positions=np.asarray(samples, dtype=np.float64),
            keyframe_sample_indices=tuple(keyframe_indices),
            source_pose_names=tuple(reference.name for reference in action.pose_sequence),
            keyframe_hold_seconds=(0.0,) + tuple(
                transition.hold_seconds for transition in action.transitions
            ),
            requested_sample_frequency_hz=frequency,
        )
        self._validate_trajectory(trajectory)
        return trajectory

    def save_trajectory(
        self,
        *,
        trajectory: ActionTrajectory,
        overwrite: bool = False,
    ) -> Path:
        self._validate_trajectory(trajectory)
        return self.archive_helper.write_npz(
            path=self._trajectory_path(trajectory.action_name),
            arrays={
                "schema_version": np.asarray(self.TRAJECTORY_SCHEMA_VERSION, dtype=np.int64),
                "action_name": np.asarray(trajectory.action_name),
                "robot_model_id": np.asarray(trajectory.robot_model_id),
                "fps": np.asarray(trajectory.requested_sample_frequency_hz, dtype=np.float64),
                "joint_names": np.asarray(trajectory.joint_names),
                "timestamps": trajectory.timestamps,
                "joint_positions": trajectory.joint_positions,
                "keyframe_sample_indices": np.asarray(
                    trajectory.keyframe_sample_indices, dtype=np.int64
                ),
                "source_pose_names": np.asarray(trajectory.source_pose_names),
                "keyframe_hold_seconds": np.asarray(
                    trajectory.keyframe_hold_seconds, dtype=np.float64
                ),
            },
            overwrite=overwrite,
        )

    def load_trajectory(self, *, name: str) -> ActionTrajectory:
        normalized = PoseDefinition.validate_name(name)
        arrays = self.archive_helper.read_npz(path=self._trajectory_path(normalized))
        if set(arrays) != self.TRAJECTORY_ARRAY_NAMES:
            raise ValueError("Trajectory archive does not contain the exact schema-1 arrays")
        if self._scalar_int(arrays["schema_version"], "schema_version") != 1:
            raise ValueError("Unsupported trajectory schema version")
        trajectory = ActionTrajectory(
            action_name=self._scalar_string(arrays["action_name"], "action_name"),
            robot_model_id=self._scalar_string(arrays["robot_model_id"], "robot_model_id"),
            joint_names=tuple(str(item) for item in arrays["joint_names"].tolist()),
            timestamps=np.asarray(arrays["timestamps"], dtype=np.float64),
            joint_positions=np.asarray(arrays["joint_positions"], dtype=np.float64),
            keyframe_sample_indices=tuple(
                int(item) for item in arrays["keyframe_sample_indices"].tolist()
            ),
            source_pose_names=tuple(str(item) for item in arrays["source_pose_names"].tolist()),
            keyframe_hold_seconds=tuple(
                float(item) for item in arrays["keyframe_hold_seconds"].tolist()
            ),
            requested_sample_frequency_hz=self._scalar_float(arrays["fps"], "fps"),
        )
        self._validate_trajectory(trajectory)
        if trajectory.action_name != normalized:
            raise ValueError("Trajectory archive contains a different action name")
        return trajectory

    def list_trajectories(self) -> tuple[str, ...]:
        return tuple(
            path.stem
            for path in self.archive_helper.list_npz_files(directory=self.trajectory_dir)
        )

    def trajectory_path(self, *, name: str) -> Path:
        path = self._trajectory_path(PoseDefinition.validate_name(name))
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _validate_trajectory(self, trajectory: ActionTrajectory) -> None:
        if trajectory.robot_model_id != self.schema.model_id:
            raise ValueError("Trajectory belongs to a different robot model")
        if trajectory.joint_names != self.schema.BASE_JOINT_NAMES:
            raise ValueError("Trajectory must contain exactly the 17 authored joints")
        if trajectory.timestamps.ndim != 1 or trajectory.sample_count < 2:
            raise ValueError("Trajectory timestamps must contain at least two samples")
        if trajectory.joint_positions.shape != (
            trajectory.sample_count,
            len(self.schema.BASE_JOINT_NAMES),
        ):
            raise ValueError("Trajectory joint_positions shape does not match its contract")
        if not np.all(np.isfinite(trajectory.timestamps)) or not np.all(
            np.isfinite(trajectory.joint_positions)
        ):
            raise ValueError("Trajectory values must be finite")
        if trajectory.timestamps[0] != 0.0 or np.any(np.diff(trajectory.timestamps) <= 0.0):
            raise ValueError("Trajectory timestamps must strictly increase from zero")
        if len(trajectory.keyframe_sample_indices) != len(trajectory.source_pose_names):
            raise ValueError("Trajectory keyframe metadata lengths do not match")
        if len(trajectory.keyframe_hold_seconds) != len(trajectory.source_pose_names):
            raise ValueError("Trajectory hold metadata lengths do not match")
        if trajectory.keyframe_sample_indices[0] != 0:
            raise ValueError("Trajectory first keyframe must be sample zero")
        for joint_index, joint_name in enumerate(trajectory.joint_names):
            definition = self.schema.definition(joint_name)
            values = trajectory.joint_positions[:, joint_index]
            if np.any(values < definition.lower_limit) or np.any(values > definition.upper_limit):
                raise ValueError(f"Trajectory joint {joint_name} exceeds its model limit")

    def _definition_path(self, name: str) -> Path:
        return self.definition_dir / f"{PoseDefinition.validate_name(name)}.json"

    def _trajectory_path(self, name: str) -> Path:
        return self.trajectory_dir / f"{PoseDefinition.validate_name(name)}.npz"

    @staticmethod
    def _scalar_string(array: np.ndarray, name: str) -> str:
        if array.shape != ():
            raise ValueError(f"Trajectory {name} must be a scalar")
        return str(array.item())

    @staticmethod
    def _scalar_float(array: np.ndarray, name: str) -> float:
        if array.shape != ():
            raise ValueError(f"Trajectory {name} must be a scalar")
        return float(array.item())

    @staticmethod
    def _scalar_int(array: np.ndarray, name: str) -> int:
        if array.shape != ():
            raise ValueError(f"Trajectory {name} must be a scalar")
        return int(array.item())
