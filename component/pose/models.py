"""Serializable pose definitions shared by Tianyi pose capabilities."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType

from component.pose.tianyi_joint_schema import PoseType, TianyiJointSchema
from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper

LOGGER = logging.getLogger(__name__)


class PoseSource(str, Enum):
    """Origin of a pose in the simulation-first workflow."""

    SIMULATION = "simulation"
    COMPOSITION = "composition"
    RETARGETING = "retargeting"


@dataclass(frozen=True, slots=True)
class PoseDefinition:
    """A validated, immutable set of named Tianyi joint positions."""

    schema_version: int
    name: str
    pose_type: PoseType
    robot_model_id: str
    joint_values: Mapping[str, float]
    source: PoseSource
    created_at: datetime
    source_parts: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    SCHEMA_VERSION = 1
    COMPOSED_SOURCE_KEYS = frozenset({"base", "left_arm", "right_arm"})

    def __post_init__(self) -> None:
        normalized_name = self.validate_name(self.name)
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError(f"Unsupported pose schema version: {self.schema_version}")
        if not isinstance(self.pose_type, PoseType):
            raise ValueError("Pose pose_type must be a PoseType")
        if not isinstance(self.source, PoseSource):
            raise ValueError("Pose source must be a PoseSource")
        if not isinstance(self.robot_model_id, str) or not self.robot_model_id:
            raise ValueError("Pose robot_model_id cannot be empty")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("Pose created_at must include a timezone")
        if not isinstance(self.notes, str):
            raise ValueError("Pose notes must be a string")

        normalized_source_parts = self._validate_source_parts(
            pose_type=self.pose_type,
            source=self.source,
            source_parts=self.source_parts,
        )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "joint_values", MappingProxyType(dict(self.joint_values)))
        object.__setattr__(self, "source_parts", MappingProxyType(normalized_source_parts))

    @classmethod
    def validate_name(cls, name: str) -> str:
        """Return a normalized Unicode pose name that is safe as a filename."""
        if not isinstance(name, str):
            raise ValueError("Pose name must be a string")
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Pose name cannot be empty")
        if len(normalized_name) > 100:
            raise ValueError("Pose name cannot exceed 100 characters")
        if normalized_name in {".", ".."} or any(
            character in normalized_name for character in '<>:"/\\|?*'
        ):
            raise ValueError("Pose name contains a path-unsafe character")
        if any(ord(character) < 32 for character in normalized_name):
            raise ValueError("Pose name contains a control character")
        return normalized_name

    @classmethod
    def create(
        cls,
        *,
        schema: TianyiJointSchema,
        name: str,
        pose_type: PoseType,
        joint_values: Mapping[str, object],
        source: PoseSource,
        created_at: datetime | None = None,
        source_parts: Mapping[str, str] | None = None,
        notes: str = "",
    ) -> PoseDefinition:
        validated_values = schema.validate_joint_values(
            pose_type=pose_type,
            joint_values=joint_values,
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            name=name,
            pose_type=pose_type,
            robot_model_id=schema.model_id,
            joint_values=validated_values,
            source=source,
            created_at=created_at or datetime.now(timezone.utc),
            source_parts=source_parts or {},
            notes=notes,
        )

    @classmethod
    def from_dict(
        cls,
        *,
        schema: TianyiJointSchema,
        payload: Mapping[str, object],
    ) -> PoseDefinition:
        required_keys = {
            "schema_version",
            "name",
            "pose_type",
            "robot_model_id",
            "joint_values",
            "source",
            "created_at",
        }
        allowed_keys = required_keys | {"source_parts", "notes"}
        missing_keys = sorted(required_keys - set(payload))
        unknown_keys = sorted(set(payload) - allowed_keys)
        if missing_keys:
            raise ValueError(f"Pose payload is missing fields: {missing_keys}")
        if unknown_keys:
            raise ValueError(f"Pose payload has unknown fields: {unknown_keys}")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported pose schema version: {payload['schema_version']}")
        if payload["robot_model_id"] != schema.model_id:
            raise ValueError(
                f"Pose model {payload['robot_model_id']} does not match {schema.model_id}"
            )

        name = payload["name"]
        joint_values = payload["joint_values"]
        source_parts = payload.get("source_parts", {})
        if not isinstance(name, str):
            raise ValueError("Pose name must be a string")
        if not isinstance(joint_values, Mapping):
            raise ValueError("Pose joint_values must be an object")
        if not isinstance(source_parts, Mapping):
            raise ValueError("Pose source_parts must be an object")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in source_parts.items()
        ):
            raise ValueError("Pose source_parts keys and values must be strings")
        notes = payload.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError("Pose notes must be a string")

        try:
            created_at = datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00"))
            pose_type = PoseType(str(payload["pose_type"]))
            source = PoseSource(str(payload["source"]))
        except ValueError as error:
            raise ValueError(f"Invalid pose metadata: {error}") from error

        return cls.create(
            schema=schema,
            name=name,
            pose_type=pose_type,
            joint_values=joint_values,
            source=source,
            created_at=created_at,
            source_parts=dict(source_parts),
            notes=notes,
        )

    @classmethod
    def from_json(cls, *, schema: TianyiJointSchema, content: str) -> PoseDefinition:
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise ValueError("Pose JSON root must be an object")
        return cls.from_dict(schema=schema, payload=payload)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "pose_type": self.pose_type.value,
            "robot_model_id": self.robot_model_id,
            "joint_values": dict(self.joint_values),
            "source": self.source.value,
            "created_at": self.created_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "source_parts": dict(self.source_parts),
            "notes": self.notes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def _validate_source_parts(
        cls,
        *,
        pose_type: PoseType,
        source: PoseSource,
        source_parts: Mapping[str, str],
    ) -> dict[str, str]:
        if not isinstance(source_parts, Mapping):
            raise ValueError("Pose source_parts must be a mapping")
        normalized = {str(key): cls.validate_name(value) for key, value in source_parts.items()}
        if pose_type is PoseType.COMPOSED:
            if source is not PoseSource.COMPOSITION:
                raise ValueError("A composed pose must have composition as its source")
            unknown_keys = sorted(set(normalized) - cls.COMPOSED_SOURCE_KEYS)
            if unknown_keys:
                raise ValueError(f"Composed pose has unknown source parts: {unknown_keys}")
            if "base" not in normalized:
                raise ValueError("A composed pose must identify its base source pose")
        elif source is PoseSource.COMPOSITION:
            raise ValueError("Only a composed pose may have composition as its source")
        elif normalized:
            raise ValueError("A non-composed pose cannot have source parts")
        return normalized


def demo_pose_definition() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = AppSettings()
    schema = TianyiJointSchema(
        asset_helper=TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
    )
    pose = PoseDefinition.create(
        schema=schema,
        name="simulated_neutral_base",
        pose_type=PoseType.BASE,
        joint_values=schema.neutral_values(PoseType.BASE),
        source=PoseSource.SIMULATION,
        notes="Simulation-only typed-pose demo",
    )
    restored_pose = PoseDefinition.from_json(schema=schema, content=pose.to_json())
    LOGGER.info(
        "Serialized and restored %s with %d joints",
        restored_pose.name,
        len(restored_pose.joint_values),
    )


def main() -> None:
    demo_pose_definition()


if __name__ == "__main__":
    main()
