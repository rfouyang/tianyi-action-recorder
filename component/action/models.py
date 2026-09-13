"""Validated Tianyi action definitions and sampled MuJoCo trajectories."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from component.pose import PoseDefinition, PoseType, TianyiJointSchema


@dataclass(frozen=True, slots=True)
class ActionPoseReference:
    """Reference one complete head-and-arms pose used by an action."""

    pose_type: PoseType
    name: str

    ALLOWED_POSE_TYPES = frozenset({PoseType.BASE, PoseType.COMPOSED})

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pose_type, PoseType)
            or self.pose_type not in self.ALLOWED_POSE_TYPES
        ):
            raise ValueError("Action poses must be base or composed")
        object.__setattr__(self, "name", PoseDefinition.validate_name(self.name))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ActionPoseReference:
        if set(payload) != {"pose_type", "name"}:
            raise ValueError("Action pose reference must contain pose_type and name")
        if not isinstance(payload["name"], str):
            raise ValueError("Action pose reference name must be a string")
        try:
            pose_type = PoseType(str(payload["pose_type"]))
        except ValueError as error:
            raise ValueError(f"Invalid action pose type: {payload['pose_type']}") from error
        return cls(pose_type=pose_type, name=payload["name"])

    def to_dict(self) -> dict[str, str]:
        return {"pose_type": self.pose_type.value, "name": self.name}


@dataclass(frozen=True, slots=True)
class ActionTransition:
    """Move to a complete target pose, then optionally hold it."""

    target_pose: ActionPoseReference
    duration_seconds: float
    hold_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.target_pose, ActionPoseReference):
            raise ValueError("Action transition target_pose must be a pose reference")
        object.__setattr__(
            self,
            "duration_seconds",
            self._duration(self.duration_seconds, positive=True, label="duration"),
        )
        object.__setattr__(
            self,
            "hold_seconds",
            self._duration(self.hold_seconds, positive=False, label="hold"),
        )

    @staticmethod
    def _duration(value: object, *, positive: bool, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"Action transition {label} must be a real number")
        normalized = float(value)
        valid = normalized > 0.0 if positive else normalized >= 0.0
        if not math.isfinite(normalized) or not valid:
            qualifier = "greater than zero" if positive else "nonnegative"
            raise ValueError(f"Action transition {label} must be finite and {qualifier}")
        return normalized

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ActionTransition:
        allowed = {"target_pose", "duration_seconds", "hold_seconds"}
        if "target_pose" not in payload or "duration_seconds" not in payload:
            raise ValueError("Action transition is missing target_pose or duration_seconds")
        if set(payload) - allowed:
            raise ValueError("Action transition has unknown fields")
        target = payload["target_pose"]
        if not isinstance(target, Mapping):
            raise ValueError("Action transition target_pose must be an object")
        return cls(
            target_pose=ActionPoseReference.from_dict(target),
            duration_seconds=payload["duration_seconds"],
            hold_seconds=payload.get("hold_seconds", 0.0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_pose": self.target_pose.to_dict(),
            "duration_seconds": self.duration_seconds,
            "hold_seconds": self.hold_seconds,
        }


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    """A concierge-init-bounded action with complete intermediate poses."""

    schema_version: int
    name: str
    robot_model_id: str
    initial_pose: ActionPoseReference
    transitions: tuple[ActionTransition, ...]
    created_at: datetime
    notes: str = ""

    SCHEMA_VERSION = 1
    HOME_POSE_NAME: ClassVar[str] = "concierge_init"

    def __post_init__(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported action schema version: {self.schema_version}")
        object.__setattr__(self, "name", PoseDefinition.validate_name(self.name))
        if not isinstance(self.robot_model_id, str) or not self.robot_model_id:
            raise ValueError("Action robot_model_id cannot be empty")
        if not isinstance(self.initial_pose, ActionPoseReference):
            raise ValueError("Action initial_pose must be a pose reference")
        if self.initial_pose.pose_type is not PoseType.BASE:
            raise ValueError("Action home pose must be a base pose")
        if self.initial_pose.name != self.HOME_POSE_NAME:
            raise ValueError(
                f"Action start and finish pose must be base/{self.HOME_POSE_NAME}"
            )
        transitions = tuple(self.transitions)
        if len(transitions) < 2:
            raise ValueError("Action requires an intermediate pose and a return home")
        if not all(isinstance(item, ActionTransition) for item in transitions):
            raise ValueError("Action transitions must contain only ActionTransition values")
        if transitions[-1].target_pose != self.initial_pose:
            raise ValueError(f"Action must finish at base/{self.HOME_POSE_NAME}")
        if transitions[-1].hold_seconds != 0.0:
            raise ValueError("The final home transition cannot add a hold")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("Action created_at must include a timezone")
        if not isinstance(self.notes, str):
            raise ValueError("Action notes must be a string")
        object.__setattr__(self, "transitions", transitions)

    @property
    def total_duration_seconds(self) -> float:
        return sum(item.duration_seconds + item.hold_seconds for item in self.transitions)

    @property
    def pose_sequence(self) -> tuple[ActionPoseReference, ...]:
        return (self.initial_pose,) + tuple(item.target_pose for item in self.transitions)

    @classmethod
    def create(
        cls,
        *,
        schema: TianyiJointSchema,
        name: str,
        initial_pose: ActionPoseReference,
        transitions: Sequence[ActionTransition],
        notes: str = "",
        created_at: datetime | None = None,
    ) -> ActionDefinition:
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            name=name,
            robot_model_id=schema.model_id,
            initial_pose=initial_pose,
            transitions=tuple(transitions),
            created_at=created_at or datetime.now(timezone.utc),
            notes=notes,
        )

    @classmethod
    def from_dict(
        cls,
        *,
        schema: TianyiJointSchema,
        payload: Mapping[str, object],
    ) -> ActionDefinition:
        required = {
            "schema_version", "name", "robot_model_id", "initial_pose", "transitions", "created_at"
        }
        allowed = required | {"notes"}
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - allowed)
        if missing:
            raise ValueError(f"Action payload is missing fields: {missing}")
        if unknown:
            raise ValueError(f"Action payload has unknown fields: {unknown}")
        if payload["robot_model_id"] != schema.model_id:
            raise ValueError(
                f"Action model {payload['robot_model_id']} does not match {schema.model_id}"
            )
        initial = payload["initial_pose"]
        transition_payloads = payload["transitions"]
        if not isinstance(payload["name"], str):
            raise ValueError("Action name must be a string")
        if not isinstance(initial, Mapping) or not isinstance(transition_payloads, list):
            raise ValueError("Action initial_pose must be an object and transitions an array")
        try:
            created_at = datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Invalid action created_at: {error}") from error
        transitions = []
        for index, item in enumerate(transition_payloads):
            if not isinstance(item, Mapping):
                raise ValueError(f"Action transition {index} must be an object")
            transitions.append(ActionTransition.from_dict(item))
        notes = payload.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("Action notes must be a string")
        return cls.create(
            schema=schema,
            name=payload["name"],
            initial_pose=ActionPoseReference.from_dict(initial),
            transitions=transitions,
            notes=notes,
            created_at=created_at,
        )

    @classmethod
    def from_json(cls, *, schema: TianyiJointSchema, content: str) -> ActionDefinition:
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Action JSON root must be an object")
        return cls.from_dict(schema=schema, payload=payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "robot_model_id": self.robot_model_id,
            "initial_pose": self.initial_pose.to_dict(),
            "transitions": [item.to_dict() for item in self.transitions],
            "created_at": self.created_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ActionTrajectory:
    """A direct joint-space trajectory for the 17 authored Tianyi joints only."""

    action_name: str
    robot_model_id: str
    joint_names: tuple[str, ...]
    timestamps: NDArray[np.float64]
    joint_positions: NDArray[np.float64]
    keyframe_sample_indices: tuple[int, ...]
    source_pose_names: tuple[str, ...]
    keyframe_hold_seconds: tuple[float, ...]
    requested_sample_frequency_hz: float

    @property
    def sample_count(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(self.timestamps[-1])
