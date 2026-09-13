"""Transport schemas for fixed-base Tianyi MuJoCo state."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from component.simulation import SimulationSnapshot


class JointPositionCommand(BaseModel):
    """Validated transport envelope for partial head or arm targets."""

    model_config = ConfigDict(extra="forbid")

    joint_positions: dict[str, float] = Field(min_length=1)


class SimulationStateResponse(BaseModel):
    """Complete simulation state suitable for REST or WebSocket telemetry."""

    revision: int
    updated_at: float
    model_id: str
    joint_positions: dict[str, float]
    joint_position_targets: dict[str, float]
    base_position: tuple[float, float, float]
    base_wxyz: tuple[float, float, float, float]
    locked_joint_names: tuple[str, ...]
    locked_joint_positions: dict[str, float]
    locked: bool

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SimulationSnapshot,
        *,
        locked_joint_positions: dict[str, float],
    ) -> SimulationStateResponse:
        positions = snapshot.joint_position_map()
        locked_joint_names = tuple(locked_joint_positions)
        return cls(
            revision=snapshot.revision,
            updated_at=snapshot.updated_at,
            model_id=snapshot.robot_model_id,
            joint_positions=positions,
            joint_position_targets=snapshot.joint_position_target_map(),
            base_position=snapshot.base_position,
            base_wxyz=snapshot.base_wxyz,
            locked_joint_names=locked_joint_names,
            locked_joint_positions=locked_joint_positions,
            locked=all(
                positions[joint_name] == value
                for joint_name, value in locked_joint_positions.items()
            ),
        )
