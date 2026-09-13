"""Retarget G1 two-arm poses into Tianyi with constrained MuJoCo IK."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

import mujoco
import numpy as np

from component.pose import PoseType, TianyiJointSchema
from component.retarget.models import ArmRetargetReport, PoseRetargetResult
from util.g1_pose_adapter import (
    G1_JOINT_NAMES,
    G1_LEFT_ARM_JOINT_NAMES,
    G1_RIGHT_ARM_JOINT_NAMES,
    G1PoseAdapter,
)


@dataclass(frozen=True, slots=True)
class _ArmContract:
    side: str
    g1_joints: tuple[str, ...]
    tianyi_joints: tuple[str, ...]
    g1_shoulder_joint: str
    g1_elbow_body: str
    g1_hand_body: str
    g1_hand_offset: tuple[float, float, float]
    tianyi_shoulder_joint: str
    tianyi_elbow_body: str
    tianyi_hand_body: str


@dataclass(frozen=True, slots=True)
class _ArmTarget:
    hand_position: np.ndarray
    hand_rotation: np.ndarray
    elbow_position: np.ndarray


class PoseRetargetService:
    """Map expressive G1 arm motion to Tianyi while keeping Tianyi's body fixed."""

    G1_FRAME_BODY = "torso_link"
    TIANYI_FRAME_BODY = "waist_yaw_link"
    MAX_ITERATIONS = 500
    POSITION_TOLERANCE_M = 0.02
    ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
    ELBOW_TOLERANCE_M = 0.03
    HAND_POSITION_WEIGHT = 1.0
    HAND_ORIENTATION_WEIGHT = 0.12
    ELBOW_POSITION_WEIGHT = 1.2
    POSTURE_WEIGHT = 0.01
    DAMPING = 1e-4

    def __init__(
        self,
        *,
        schema: TianyiJointSchema,
        g1_poses: G1PoseAdapter,
        tianyi_mjcf_path,
        tianyi_default_values: dict[str, float],
        tianyi_locked_values: dict[str, float],
    ) -> None:
        self.schema = schema
        self.g1_poses = g1_poses
        self._g1_model = mujoco.MjModel.from_xml_path(str(g1_poses.mjcf_path))
        self._tianyi_model = mujoco.MjModel.from_xml_path(str(tianyi_mjcf_path))
        self._tianyi_default_values = schema.validate_joint_values(
            pose_type=PoseType.BASE,
            joint_values=tianyi_default_values,
        )
        self._tianyi_locked_values = dict(tianyi_locked_values)
        self._arms = self._arm_contracts()
        self._validate_contract()

    def retarget(
        self,
        *,
        source_pose_json: str,
        source_file_name: str,
    ) -> PoseRetargetResult:
        source = self.g1_poses.parse_pose(
            content=source_pose_json,
            file_name=source_file_name,
        )
        source_joint_values = dict.fromkeys(G1_JOINT_NAMES, 0.0)
        source_joint_values.update(source.joint_values)
        g1_source_data = self._pose_data(
            self._g1_model,
            source_joint_values,
        )
        tianyi_reference_data = self._pose_data(
            self._tianyi_model,
            self._tianyi_locked_values,
        )
        tianyi_data = self._pose_data(
            self._tianyi_model,
            self._tianyi_locked_values,
        )

        reports: list[ArmRetargetReport] = []
        solved_values = dict(self._tianyi_default_values)
        for arm in self._active_arms(source.pose_type):
            target = self._build_target(
                arm=arm,
                g1_source_data=g1_source_data,
                tianyi_reference_data=tianyi_reference_data,
            )
            report, values = self._solve_arm(
                arm=arm,
                data=tianyi_data,
                target=target,
                seed_values=self._arm_seed_values(
                    arm=arm,
                    source_joint_values=source_joint_values,
                ),
            )
            reports.append(report)
            solved_values.update(values)

        validated = self.schema.validate_joint_values(
            pose_type=PoseType.BASE,
            joint_values=solved_values,
        )
        final_data = self._pose_data(
            self._tianyi_model,
            {**self._tianyi_locked_values, **validated},
        )
        return PoseRetargetResult(
            source_name=source.name,
            source_pose_type=source.pose_type,
            source_model_id=source.robot_model_id,
            target_model_id=self.schema.model_id,
            source_joint_values=MappingProxyType(source_joint_values),
            joint_values=MappingProxyType(validated),
            arm_reports=tuple(reports),
            contact_count=int(final_data.ncon),
        )

    def _build_target(
        self,
        *,
        arm: _ArmContract,
        g1_source_data: mujoco.MjData,
        tianyi_reference_data: mujoco.MjData,
    ) -> _ArmTarget:
        g1_source = self._arm_geometry(
            model=self._g1_model,
            data=g1_source_data,
            frame_body=self.G1_FRAME_BODY,
            shoulder_joint=arm.g1_shoulder_joint,
            elbow_body=arm.g1_elbow_body,
            hand_body=arm.g1_hand_body,
            hand_offset=arm.g1_hand_offset,
        )
        tianyi_reference = self._arm_geometry(
            model=self._tianyi_model,
            data=tianyi_reference_data,
            frame_body=self.TIANYI_FRAME_BODY,
            shoulder_joint=arm.tianyi_shoulder_joint,
            elbow_body=arm.tianyi_elbow_body,
            hand_body=arm.tianyi_hand_body,
            hand_offset=(0.0, 0.0, 0.0),
        )
        g1_upper, g1_fore = self._segment_lengths(g1_source)
        tianyi_upper, tianyi_fore = self._segment_lengths(tianyi_reference)
        elbow_local = tianyi_reference["shoulder_position"] + (
            tianyi_upper
            / g1_upper
            * (g1_source["elbow_position"] - g1_source["shoulder_position"])
        )
        hand_local = elbow_local + (
            tianyi_fore
            / g1_fore
            * (g1_source["hand_position"] - g1_source["elbow_position"])
        )
        hand_rotation_local = self._functional_hand_rotation(
            forearm_direction=hand_local - elbow_local,
            source_hand_rotation=g1_source["hand_rotation"],
        )
        frame_id = self._body_id(self._tianyi_model, self.TIANYI_FRAME_BODY)
        frame_position = tianyi_reference_data.xpos[frame_id]
        frame_rotation = tianyi_reference_data.xmat[frame_id].reshape(3, 3)
        return _ArmTarget(
            hand_position=frame_position + frame_rotation @ hand_local,
            hand_rotation=frame_rotation @ hand_rotation_local,
            elbow_position=frame_position + frame_rotation @ elbow_local,
        )

    def _solve_arm(
        self,
        *,
        arm: _ArmContract,
        data: mujoco.MjData,
        target: _ArmTarget,
        seed_values: tuple[float, ...],
    ) -> tuple[ArmRetargetReport, dict[str, float]]:
        model = self._tianyi_model
        joint_ids = [self._joint_id(model, name) for name in arm.tianyi_joints]
        qpos_addresses = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
        dof_addresses = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
        lower = np.array([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
        upper = np.array([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
        seed = np.clip(np.asarray(seed_values), lower + 1e-5, upper - 1e-5)
        data.qpos[qpos_addresses] = seed
        hand_id = self._body_id(model, arm.tianyi_hand_body)
        elbow_id = self._body_id(model, arm.tianyi_elbow_body)
        shoulder_id = self._joint_id(model, arm.tianyi_shoulder_joint)
        jac_hand_position = np.zeros((3, model.nv))
        jac_hand_rotation = np.zeros((3, model.nv))
        jac_elbow_position = np.zeros((3, model.nv))
        jac_elbow_rotation = np.zeros((3, model.nv))

        iterations = 0
        for iteration_index in range(1, self.MAX_ITERATIONS + 1):
            iterations = iteration_index
            mujoco.mj_forward(model, data)
            hand_position = data.xpos[hand_id]
            hand_rotation = data.xmat[hand_id].reshape(3, 3)
            elbow_position = data.xpos[elbow_id]
            position_error = target.hand_position - hand_position
            orientation_error = self._rotation_vector(
                target.hand_rotation @ hand_rotation.T
            )
            elbow_error = target.elbow_position - elbow_position
            mujoco.mj_jacBody(
                model,
                data,
                jac_hand_position,
                jac_hand_rotation,
                hand_id,
            )
            mujoco.mj_jacBody(
                model,
                data,
                jac_elbow_position,
                jac_elbow_rotation,
                elbow_id,
            )
            arm_hand_position = jac_hand_position[:, dof_addresses]
            arm_hand_rotation = jac_hand_rotation[:, dof_addresses]
            arm_elbow_position = jac_elbow_position[:, dof_addresses]
            current = data.qpos[qpos_addresses].copy()
            matrix = np.vstack(
                (
                    self.HAND_POSITION_WEIGHT * arm_hand_position,
                    self.HAND_ORIENTATION_WEIGHT * arm_hand_rotation,
                    self.ELBOW_POSITION_WEIGHT * arm_elbow_position,
                    self.POSTURE_WEIGHT * np.eye(len(joint_ids)),
                )
            )
            residual = np.concatenate(
                (
                    self.HAND_POSITION_WEIGHT * position_error,
                    self.HAND_ORIENTATION_WEIGHT * orientation_error,
                    self.ELBOW_POSITION_WEIGHT * elbow_error,
                    self.POSTURE_WEIGHT * (seed - current),
                )
            )
            normal = matrix.T @ matrix + self.DAMPING * np.eye(len(joint_ids))
            step = np.linalg.solve(normal, matrix.T @ residual)
            if np.linalg.norm(step) <= 1e-7:
                break
            data.qpos[qpos_addresses] = np.clip(
                current + np.clip(step, -0.08, 0.08),
                lower + 1e-5,
                upper - 1e-5,
            )

        mujoco.mj_forward(model, data)
        position_error = target.hand_position - data.xpos[hand_id]
        orientation_error = self._rotation_vector(
            target.hand_rotation @ data.xmat[hand_id].reshape(3, 3).T
        )
        elbow_error = target.elbow_position - data.xpos[elbow_id]
        shoulder_position = data.xanchor[shoulder_id]
        upper_arm_direction_error = self._direction_error(
            target.elbow_position - shoulder_position,
            data.xpos[elbow_id] - shoulder_position,
        )
        forearm_direction_error = self._direction_error(
            target.hand_position - target.elbow_position,
            data.xpos[hand_id] - data.xpos[elbow_id],
        )
        current = data.qpos[qpos_addresses].copy()
        at_limit = tuple(
            name
            for name, value, minimum, maximum in zip(
                arm.tianyi_joints,
                current,
                lower,
                upper,
                strict=True,
            )
            if min(abs(value - minimum), abs(value - maximum)) < 5e-4
        )
        report = ArmRetargetReport(
            side=arm.side,
            converged=bool(
                self._within_tolerance(
                    position_error,
                    orientation_error,
                    elbow_error,
                )
            ),
            iterations=iterations,
            hand_position_error_m=float(np.linalg.norm(position_error)),
            hand_orientation_error_rad=float(np.linalg.norm(orientation_error)),
            elbow_position_error_m=float(np.linalg.norm(elbow_error)),
            upper_arm_direction_error_rad=upper_arm_direction_error,
            forearm_direction_error_rad=forearm_direction_error,
            reached_joint_limits=at_limit,
        )
        values = {
            name: float(value)
            for name, value in zip(arm.tianyi_joints, current, strict=True)
        }
        return report, values

    def _pose_data(self, model: mujoco.MjModel, values) -> mujoco.MjData:
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        for joint_name, value in values.items():
            joint_id = self._joint_id(model, joint_name)
            lower, upper = model.jnt_range[joint_id]
            if model.jnt_limited[joint_id] and not lower <= value <= upper:
                raise ValueError(
                    f"Source joint {joint_name} value {value} is outside "
                    f"[{lower}, {upper}]"
                )
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)
        return data

    def _arm_geometry(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        frame_body: str,
        shoulder_joint: str,
        elbow_body: str,
        hand_body: str,
        hand_offset: tuple[float, float, float],
    ) -> dict[str, np.ndarray]:
        frame_id = self._body_id(model, frame_body)
        hand_id = self._body_id(model, hand_body)
        elbow_id = self._body_id(model, elbow_body)
        shoulder_id = self._joint_id(model, shoulder_joint)
        frame_position = data.xpos[frame_id]
        frame_rotation = data.xmat[frame_id].reshape(3, 3)
        hand_rotation = data.xmat[hand_id].reshape(3, 3)
        hand_position = data.xpos[hand_id] + hand_rotation @ np.asarray(hand_offset)
        to_local = frame_rotation.T
        return {
            "shoulder_position": to_local @ (data.xanchor[shoulder_id] - frame_position),
            "elbow_position": to_local @ (data.xpos[elbow_id] - frame_position),
            "hand_position": to_local @ (hand_position - frame_position),
            "hand_rotation": to_local @ hand_rotation,
        }

    @staticmethod
    def _segment_lengths(geometry: dict[str, np.ndarray]) -> tuple[float, float]:
        upper = np.linalg.norm(
            geometry["elbow_position"] - geometry["shoulder_position"]
        )
        fore = np.linalg.norm(geometry["hand_position"] - geometry["elbow_position"])
        return float(upper), float(fore)

    @staticmethod
    def _direction_error(target: np.ndarray, actual: np.ndarray) -> float:
        denominator = float(np.linalg.norm(target) * np.linalg.norm(actual))
        if denominator < 1e-12:
            raise ValueError("Cannot compare a zero-length arm segment")
        cosine = float(np.clip(np.dot(target, actual) / denominator, -1.0, 1.0))
        return math.acos(cosine)

    @staticmethod
    def _functional_hand_rotation(
        *,
        forearm_direction: np.ndarray,
        source_hand_rotation: np.ndarray,
    ) -> np.ndarray:
        """Build a Tianyi hand frame from arm direction and G1 palm roll."""
        target_z = -forearm_direction / np.linalg.norm(forearm_direction)
        source_y = source_hand_rotation[:, 1]
        target_y = source_y - np.dot(source_y, target_z) * target_z
        target_y_norm = float(np.linalg.norm(target_y))
        if target_y_norm < 1e-8:
            source_z = source_hand_rotation[:, 2]
            target_y = source_z - np.dot(source_z, target_z) * target_z
            target_y_norm = float(np.linalg.norm(target_y))
        target_y /= target_y_norm
        target_x = np.cross(target_y, target_z)
        target_x /= np.linalg.norm(target_x)
        return np.column_stack((target_x, target_y, target_z))

    @classmethod
    def _within_tolerance(
        cls,
        position_error: np.ndarray,
        orientation_error: np.ndarray,
        elbow_error: np.ndarray,
    ) -> bool:
        return (
            np.linalg.norm(position_error) <= cls.POSITION_TOLERANCE_M
            and np.linalg.norm(orientation_error) <= cls.ORIENTATION_TOLERANCE_RAD
            and np.linalg.norm(elbow_error) <= cls.ELBOW_TOLERANCE_M
        )

    @staticmethod
    def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        if angle < 1e-8:
            return 0.5 * np.array(
                (
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                )
            )
        if math.pi - angle < 1e-5:
            eigenvalues, eigenvectors = np.linalg.eig(rotation)
            index = int(np.argmin(np.abs(eigenvalues - 1.0)))
            axis = np.real(eigenvectors[:, index])
            axis /= np.linalg.norm(axis)
            return angle * axis
        axis = np.array(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        ) / (2.0 * math.sin(angle))
        return angle * axis

    def _validate_contract(self) -> None:
        self._body_id(self._g1_model, self.G1_FRAME_BODY)
        self._body_id(self._tianyi_model, self.TIANYI_FRAME_BODY)
        for arm in self._arms:
            for name in arm.g1_joints:
                self._joint_id(self._g1_model, name)
            for name in arm.tianyi_joints:
                self._joint_id(self._tianyi_model, name)
            self._joint_id(self._g1_model, arm.g1_shoulder_joint)
            self._body_id(self._g1_model, arm.g1_elbow_body)
            self._body_id(self._g1_model, arm.g1_hand_body)
            self._joint_id(self._tianyi_model, arm.tianyi_shoulder_joint)
            self._body_id(self._tianyi_model, arm.tianyi_elbow_body)
            self._body_id(self._tianyi_model, arm.tianyi_hand_body)

    def _active_arms(self, source_pose_type: str) -> tuple[_ArmContract, ...]:
        if source_pose_type == "left_arm":
            return tuple(arm for arm in self._arms if arm.g1_joints == G1_LEFT_ARM_JOINT_NAMES)
        if source_pose_type == "right_arm":
            return tuple(arm for arm in self._arms if arm.g1_joints == G1_RIGHT_ARM_JOINT_NAMES)
        return self._arms

    @staticmethod
    def _arm_seed_values(
        *,
        arm: _ArmContract,
        source_joint_values: dict[str, float],
    ) -> tuple[float, ...]:
        prefix = arm.side
        return (
            source_joint_values[f"{prefix}_shoulder_pitch_joint"],
            source_joint_values[f"{prefix}_shoulder_roll_joint"],
            source_joint_values[f"{prefix}_shoulder_yaw_joint"],
            source_joint_values[f"{prefix}_elbow_joint"],
            source_joint_values[f"{prefix}_wrist_yaw_joint"],
            source_joint_values[f"{prefix}_wrist_pitch_joint"],
            source_joint_values[f"{prefix}_wrist_roll_joint"],
        )

    def _arm_contracts(self) -> tuple[_ArmContract, ...]:
        return (
            _ArmContract(
                side="left",
                g1_joints=(
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ),
                tianyi_joints=self.schema.LEFT_ARM_JOINT_NAMES,
                g1_shoulder_joint="left_shoulder_pitch_joint",
                g1_elbow_body="left_elbow_link",
                g1_hand_body="left_wrist_yaw_link",
                g1_hand_offset=(0.0415, 0.003, 0.0),
                tianyi_shoulder_joint="shoulder_pitch_l_joint",
                tianyi_elbow_body="elbow_pitch_l_link",
                tianyi_hand_body="left_tcp_link",
            ),
            _ArmContract(
                side="right",
                g1_joints=(
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ),
                tianyi_joints=self.schema.RIGHT_ARM_JOINT_NAMES,
                g1_shoulder_joint="right_shoulder_pitch_joint",
                g1_elbow_body="right_elbow_link",
                g1_hand_body="right_wrist_yaw_link",
                g1_hand_offset=(0.0415, -0.003, 0.0),
                tianyi_shoulder_joint="shoulder_pitch_r_joint",
                tianyi_elbow_body="elbow_pitch_r_link",
                tianyi_hand_body="right_tcp_link",
            ),
        )

    @staticmethod
    def _joint_id(model: mujoco.MjModel, name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name}")
        return joint_id

    @staticmethod
    def _body_id(model: mujoco.MjModel, name: str) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing body {name}")
        return body_id
