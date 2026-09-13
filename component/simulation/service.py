"""Own the fixed-base Tianyi MuJoCo state independently of presentation code."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco

from component.pose import PoseDefinition, PoseType, TianyiJointSchema
from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    """Immutable copy of one complete fixed-base MuJoCo state revision."""

    revision: int
    updated_at: float
    robot_model_id: str
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    joint_position_targets: tuple[float, ...]
    base_position: tuple[float, float, float]
    base_wxyz: tuple[float, float, float, float]

    def joint_position_map(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.joint_positions, strict=True))

    def joint_position_target_map(self) -> dict[str, float]:
        return dict(zip(self.joint_names, self.joint_position_targets, strict=True))


class SimulationService:
    """Apply authored poses while enforcing Tianyi's fixed-body authoring boundary."""

    BASE_BODY_NAME = "base"

    def __init__(
        self,
        *,
        schema: TianyiJointSchema,
        mjcf_path: Path,
        default_pose: PoseDefinition | None = None,
    ) -> None:
        self.schema = schema
        self.mjcf_path = mjcf_path
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self._data = mujoco.MjData(self.model)
        self._condition = threading.Condition()
        self._revision = 0
        self._updated_at = time.time()
        self._joint_qpos_addresses = self._load_joint_qpos_addresses()
        self._joint_actuator_ids = self._load_joint_actuator_ids()
        self._base_body_id = self._load_base_body_id()
        self._calibration_joint_positions = self.schema.locked_values()
        self._default_pose_name, self._default_pose_values = self._validate_default_pose(
            default_pose
        )
        self._validate_fixed_base_contract()

        with self._condition:
            self._set_all_default_unlocked()
            mujoco.mj_forward(self.model, self._data)

    def snapshot(self) -> SimulationSnapshot:
        """Copy the latest complete state without exposing mutable MuJoCo data."""
        with self._condition:
            return self._snapshot_unlocked()

    def capture_pose_values(self, pose_type: PoseType) -> dict[str, float]:
        """Capture one exact pose shape from the current MuJoCo configuration."""
        with self._condition:
            return self.schema.extract_from_model_positions(
                model_positions=self._joint_positions_unlocked(),
                pose_type=pose_type,
            )

    def update_joint_positions(
        self,
        joint_positions: Mapping[str, object],
    ) -> SimulationSnapshot:
        """Apply a partial edit containing authored head or arm joints only."""
        if not isinstance(joint_positions, Mapping):
            raise ValueError("joint_positions must be a mapping")
        if not joint_positions:
            raise ValueError("At least one authored joint position is required")

        requested_names = set(joint_positions)
        locked = sorted(requested_names & set(self.schema.LOCKED_JOINT_NAMES))
        if locked:
            raise ValueError(f"Locked Tianyi joints cannot be authored: {locked}")
        unknown = sorted(requested_names - set(self.schema.BASE_JOINT_NAMES))
        if unknown:
            raise ValueError(f"Unknown Tianyi authored joints: {unknown}")
        validated = {
            joint_name: self.schema.validate_joint_value(
                joint_name=joint_name,
                value=value,
            )
            for joint_name, value in joint_positions.items()
        }
        return self._apply_validated_positions(validated)

    def update_calibration_joint_positions(
        self,
        joint_positions: Mapping[str, object],
    ) -> SimulationSnapshot:
        """Apply UI-only leg/waist calibration without changing the pose contract."""
        if not isinstance(joint_positions, Mapping):
            raise ValueError("calibration_joint_positions must be a mapping")
        if not joint_positions:
            raise ValueError("At least one calibration joint position is required")
        unknown = sorted(set(joint_positions) - set(self.schema.LOCKED_JOINT_NAMES))
        if unknown:
            raise ValueError(f"Unknown Tianyi calibration joints: {unknown}")
        validated = {
            joint_name: self.schema.validate_calibration_joint_value(
                joint_name=joint_name,
                value=value,
            )
            for joint_name, value in joint_positions.items()
        }
        with self._condition:
            self._calibration_joint_positions.update(validated)
            for joint_name, value in validated.items():
                self._set_joint_position_unlocked(joint_name=joint_name, value=value)
            return self._finish_update_unlocked()

    def calibration_joint_positions(self) -> dict[str, float]:
        """Return the current session calibration used by the locked-state invariant."""
        with self._condition:
            return dict(self._calibration_joint_positions)

    @property
    def default_pose_name(self) -> str:
        """Name of the base pose used for startup and default resets."""
        return self._default_pose_name

    def default_pose_values(self) -> dict[str, float]:
        """Return a copy of the validated base-pose reset targets."""
        return dict(self._default_pose_values)

    def apply_pose_values(
        self,
        *,
        pose_type: PoseType,
        joint_values: Mapping[str, object],
    ) -> SimulationSnapshot:
        """Apply an exact base, arm, or composed pose to the current state."""
        validated = self.schema.validate_joint_values(
            pose_type=pose_type,
            joint_values=joint_values,
        )
        return self._apply_validated_positions(validated)

    def apply_pose(self, pose: PoseDefinition) -> SimulationSnapshot:
        """Apply a typed pose belonging to this pinned Tianyi model."""
        if not isinstance(pose, PoseDefinition):
            raise ValueError("pose must be a PoseDefinition")
        if pose.robot_model_id != self.schema.model_id:
            raise ValueError(
                f"Pose model {pose.robot_model_id} does not match {self.schema.model_id}"
            )
        return self.apply_pose_values(
            pose_type=pose.pose_type,
            joint_values=pose.joint_values,
        )

    def mirror_current_arm(
        self,
        *,
        source_pose_type: PoseType,
    ) -> tuple[PoseType, dict[str, float], dict[str, float], SimulationSnapshot]:
        """Atomically capture one live arm and reflect it into the opposite arm."""
        if source_pose_type not in {PoseType.LEFT_ARM, PoseType.RIGHT_ARM}:
            raise ValueError("Only left_arm and right_arm values can be mirrored")
        with self._condition:
            source_values = self.schema.extract_from_model_positions(
                model_positions=self._joint_positions_unlocked(),
                pose_type=source_pose_type,
            )
            target_pose_type, mirrored_values = self.schema.mirror_arm_values(
                source_pose_type=source_pose_type,
                joint_values=source_values,
            )
            for joint_name, value in mirrored_values.items():
                self._set_joint_position_unlocked(joint_name=joint_name, value=value)
            self._enforce_locked_state_unlocked()
            snapshot = self._finish_update_unlocked()
        return target_pose_type, source_values, mirrored_values, snapshot

    def reset_to_default(self) -> SimulationSnapshot:
        """Restore the configured base pose and confirmed calibration baseline."""
        with self._condition:
            self._calibration_joint_positions = self.schema.locked_values()
            self._set_all_default_unlocked()
            return self._finish_update_unlocked()

    def reset_to_neutral(self) -> SimulationSnapshot:
        """Explicitly zero authored joints and restore the calibration baseline."""
        with self._condition:
            self._calibration_joint_positions = self.schema.locked_values()
            self._set_all_neutral_unlocked()
            return self._finish_update_unlocked()

    def wait_for_revision(
        self,
        *,
        after_revision: int,
        timeout: float,
    ) -> SimulationSnapshot:
        """Wait for a newer state or return the current state after a timeout."""
        if after_revision < 0:
            raise ValueError("after_revision cannot be negative")
        if timeout < 0:
            raise ValueError("timeout cannot be negative")
        with self._condition:
            self._condition.wait_for(
                lambda: self._revision > after_revision,
                timeout=timeout,
            )
            return self._snapshot_unlocked()

    def _apply_validated_positions(
        self,
        joint_positions: Mapping[str, float],
    ) -> SimulationSnapshot:
        with self._condition:
            for joint_name, value in joint_positions.items():
                self._set_joint_position_unlocked(joint_name=joint_name, value=value)
            self._enforce_locked_state_unlocked()
            return self._finish_update_unlocked()

    def _finish_update_unlocked(self) -> SimulationSnapshot:
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self._data)
        self._revision += 1
        self._updated_at = time.time()
        snapshot = self._snapshot_unlocked()
        self._condition.notify_all()
        return snapshot

    def _set_all_neutral_unlocked(self) -> None:
        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        self._enforce_locked_state_unlocked()

    def _set_all_default_unlocked(self) -> None:
        self._set_all_neutral_unlocked()
        for joint_name, value in self._default_pose_values.items():
            self._set_joint_position_unlocked(joint_name=joint_name, value=value)

    def _validate_default_pose(
        self,
        default_pose: PoseDefinition | None,
    ) -> tuple[str, dict[str, float]]:
        if default_pose is None:
            return "neutral", self.schema.neutral_values(PoseType.BASE)
        if not isinstance(default_pose, PoseDefinition):
            raise ValueError("default_pose must be a PoseDefinition")
        if default_pose.pose_type is not PoseType.BASE:
            raise ValueError("The simulation default pose must be a base pose")
        if default_pose.robot_model_id != self.schema.model_id:
            raise ValueError(
                f"Default pose model {default_pose.robot_model_id} does not match "
                f"{self.schema.model_id}"
            )
        values = self.schema.validate_joint_values(
            pose_type=PoseType.BASE,
            joint_values=default_pose.joint_values,
        )
        return default_pose.name, values

    def _enforce_locked_state_unlocked(self) -> None:
        for joint_name, value in self._calibration_joint_positions.items():
            self._set_joint_position_unlocked(joint_name=joint_name, value=value)

    def _set_joint_position_unlocked(self, *, joint_name: str, value: float) -> None:
        model_index = self.schema.definition(joint_name).model_index
        self._data.qpos[self._joint_qpos_addresses[model_index]] = value
        self._data.ctrl[self._joint_actuator_ids[model_index]] = value

    def _joint_positions_unlocked(self) -> tuple[float, ...]:
        return tuple(
            float(self._data.qpos[address]) for address in self._joint_qpos_addresses
        )

    def _snapshot_unlocked(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            revision=self._revision,
            updated_at=self._updated_at,
            robot_model_id=self.schema.model_id,
            joint_names=self.schema.MODEL_JOINT_NAMES,
            joint_positions=self._joint_positions_unlocked(),
            joint_position_targets=tuple(
                float(self._data.ctrl[actuator_id])
                for actuator_id in self._joint_actuator_ids
            ),
            base_position=tuple(
                float(value) for value in self._data.xpos[self._base_body_id]
            ),
            base_wxyz=tuple(
                float(value) for value in self._data.xquat[self._base_body_id]
            ),
        )

    def _load_joint_qpos_addresses(self) -> tuple[int, ...]:
        addresses = []
        for joint_name in self.schema.MODEL_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing Tianyi joint: {joint_name}")
            addresses.append(int(self.model.jnt_qposadr[joint_id]))
        return tuple(addresses)

    def _load_joint_actuator_ids(self) -> tuple[int, ...]:
        actuator_ids_by_joint: dict[str, int] = {}
        for actuator_id, joint_id in enumerate(self.model.actuator_trnid[:, 0]):
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(joint_id),
            )
            if joint_name is None:
                raise ValueError(f"MuJoCo actuator {actuator_id} references an unnamed joint")
            if joint_name in actuator_ids_by_joint:
                raise ValueError(f"MuJoCo model has duplicate actuators for {joint_name}")
            actuator_ids_by_joint[joint_name] = actuator_id
        if set(actuator_ids_by_joint) != set(self.schema.MODEL_JOINT_NAMES):
            raise ValueError("MuJoCo actuators do not cover the exact Tianyi body joint set")
        return tuple(
            actuator_ids_by_joint[joint_name]
            for joint_name in self.schema.MODEL_JOINT_NAMES
        )

    def _load_base_body_id(self) -> int:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.BASE_BODY_NAME,
        )
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing body: {self.BASE_BODY_NAME}")
        return body_id

    def _validate_fixed_base_contract(self) -> None:
        expected_count = len(self.schema.MODEL_JOINT_NAMES)
        if self.model.nq != expected_count or self.model.nv != expected_count:
            raise ValueError(
                f"Tianyi MuJoCo model must have {expected_count} fixed-base positions "
                f"and velocities; got nq={self.model.nq}, nv={self.model.nv}"
            )
        joint_types = tuple(int(value) for value in self.model.jnt_type)
        if any(joint_type == int(mujoco.mjtJoint.mjJNT_FREE) for joint_type in joint_types):
            raise ValueError("Tianyi MuJoCo authoring model must not contain a free joint")
        if len(set(self._joint_qpos_addresses)) != expected_count:
            raise ValueError("Tianyi MuJoCo joints do not have unique position addresses")


def demo_simulation_service() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = AppSettings()
    assets = TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
    schema = TianyiJointSchema(asset_helper=assets)
    service = SimulationService(schema=schema, mjcf_path=assets.mjcf_path)
    snapshot = service.update_joint_positions(
        {"head_yaw_joint": 0.2, "elbow_pitch_l_joint": -0.8}
    )
    LOGGER.info(
        "MuJoCo simulation revision %d has %d body positions",
        snapshot.revision,
        len(snapshot.joint_positions),
    )


def main() -> None:
    demo_simulation_service()


if __name__ == "__main__":
    main()
