"""Create and persist typed Tianyi poses captured from MuJoCo."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from component.pose.models import PoseDefinition, PoseSource
from component.pose.tianyi_joint_schema import PoseType, TianyiJointSchema
from util.pose_file_helper import PoseFileHelper


class PoseService:
    """Apply pose business rules and persist validated pose definitions."""

    def __init__(
        self,
        *,
        schema: TianyiJointSchema,
        pose_dir: Path,
        file_helper: PoseFileHelper,
    ) -> None:
        self.schema = schema
        self.pose_dir = pose_dir
        self.file_helper = file_helper

    def create_simulated_pose(
        self,
        *,
        name: str,
        pose_type: PoseType,
        joint_values: Mapping[str, object],
        notes: str = "",
    ) -> PoseDefinition:
        """Create a validated pose from the current MuJoCo editor values."""
        if pose_type is PoseType.COMPOSED:
            raise ValueError("A composed pose must be created by pose composition")
        return PoseDefinition.create(
            schema=self.schema,
            name=name,
            pose_type=pose_type,
            joint_values=joint_values,
            source=PoseSource.SIMULATION,
            notes=notes,
        )

    def create_retargeted_arm_pose(
        self,
        *,
        name: str,
        pose_type: PoseType,
        joint_values: Mapping[str, object],
        notes: str = "",
    ) -> PoseDefinition:
        """Create one exact Tianyi arm pose from a simulated retargeting solve."""
        if pose_type not in {PoseType.LEFT_ARM, PoseType.RIGHT_ARM}:
            raise ValueError("A retargeted pose can be saved only as left_arm or right_arm")
        return PoseDefinition.create(
            schema=self.schema,
            name=name,
            pose_type=pose_type,
            joint_values=joint_values,
            source=PoseSource.RETARGETING,
            notes=notes,
        )

    def save_pose(self, *, pose: PoseDefinition, overwrite: bool = False) -> Path:
        """Persist one pose after re-validating the pinned-model contract."""
        self._validate_pose(pose)
        return self.file_helper.write_json(
            path=self._pose_path(pose_type=pose.pose_type, name=pose.name),
            payload=pose.to_dict(),
            overwrite=overwrite,
        )

    def load_pose(self, *, pose_type: PoseType, name: str) -> PoseDefinition:
        """Load one typed pose and reject folder or metadata mismatches."""
        normalized_name = PoseDefinition.validate_name(name)
        path = self._pose_path(pose_type=pose_type, name=normalized_name)
        pose = PoseDefinition.from_dict(
            schema=self.schema,
            payload=self.file_helper.read_json(path=path),
        )
        if pose.pose_type is not pose_type:
            raise ValueError(
                f"Pose file {path} contains {pose.pose_type.value}, expected {pose_type.value}"
            )
        if pose.name != normalized_name:
            raise ValueError(f"Pose file {path} contains a different pose name: {pose.name}")
        return pose

    def list_poses(self, *, pose_type: PoseType) -> tuple[PoseDefinition, ...]:
        """Return all valid saved poses of one exact pose type."""
        directory = self.pose_dir / pose_type.value
        return tuple(
            self.load_pose(pose_type=pose_type, name=path.stem)
            for path in self.file_helper.list_json_files(directory=directory)
        )

    def list_pose_names(self) -> dict[str, tuple[str, ...]]:
        """Return saved pose names grouped by their exact pose type."""
        return {
            pose_type.value: tuple(
                pose.name for pose in self.list_poses(pose_type=pose_type)
            )
            for pose_type in PoseType
        }

    def compose_pose(
        self,
        *,
        name: str,
        base: PoseDefinition,
        left_arm: PoseDefinition | None = None,
        right_arm: PoseDefinition | None = None,
        notes: str = "",
    ) -> PoseDefinition:
        """Copy a base pose, then replace only the selected anatomical arms."""
        self._require_pose_type(base, PoseType.BASE)
        if left_arm is not None:
            self._require_pose_type(left_arm, PoseType.LEFT_ARM)
        if right_arm is not None:
            self._require_pose_type(right_arm, PoseType.RIGHT_ARM)

        composed_values = dict(base.joint_values)
        source_parts = {"base": base.name}
        if left_arm is not None:
            composed_values.update(left_arm.joint_values)
            source_parts["left_arm"] = left_arm.name
        if right_arm is not None:
            composed_values.update(right_arm.joint_values)
            source_parts["right_arm"] = right_arm.name

        return PoseDefinition.create(
            schema=self.schema,
            name=name,
            pose_type=PoseType.COMPOSED,
            joint_values=composed_values,
            source=PoseSource.COMPOSITION,
            source_parts=source_parts,
            notes=notes,
        )

    def compose_pose_from_names(
        self,
        *,
        name: str,
        base_pose_name: str,
        left_arm_pose_name: str | None = None,
        right_arm_pose_name: str | None = None,
        notes: str = "",
    ) -> PoseDefinition:
        """Load named source parts and compose them under the pinned model contract."""
        if not base_pose_name:
            raise ValueError("Select a base pose")
        base = self.load_pose(pose_type=PoseType.BASE, name=base_pose_name)
        left_arm = (
            self.load_pose(pose_type=PoseType.LEFT_ARM, name=left_arm_pose_name)
            if left_arm_pose_name
            else None
        )
        right_arm = (
            self.load_pose(pose_type=PoseType.RIGHT_ARM, name=right_arm_pose_name)
            if right_arm_pose_name
            else None
        )
        return self.compose_pose(
            name=name,
            base=base,
            left_arm=left_arm,
            right_arm=right_arm,
            notes=notes,
        )

    def mirror_arm_values(
        self,
        *,
        source_pose_type: PoseType,
        joint_values: Mapping[str, object],
    ) -> tuple[PoseType, dict[str, float]]:
        """Reflect one exact seven-joint arm edit across the robot sagittal plane."""
        return self.schema.mirror_arm_values(
            source_pose_type=source_pose_type,
            joint_values=joint_values,
        )

    def _validate_pose(self, pose: PoseDefinition) -> None:
        if not isinstance(pose, PoseDefinition):
            raise ValueError("pose must be a PoseDefinition")
        if pose.robot_model_id != self.schema.model_id:
            raise ValueError(
                f"Pose model {pose.robot_model_id} does not match {self.schema.model_id}"
            )
        self.schema.validate_joint_values(
            pose_type=pose.pose_type,
            joint_values=pose.joint_values,
        )

    def _require_pose_type(
        self,
        pose: PoseDefinition,
        expected_type: PoseType,
    ) -> None:
        self._validate_pose(pose)
        if pose.pose_type is not expected_type:
            raise ValueError(
                f"Pose {pose.name} is {pose.pose_type.value}; expected {expected_type.value}"
            )

    def _pose_path(self, *, pose_type: PoseType, name: str) -> Path:
        normalized_name = PoseDefinition.validate_name(name)
        return self.pose_dir / pose_type.value / f"{normalized_name}.json"
