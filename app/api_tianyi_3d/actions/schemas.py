"""Transport schemas for Tianyi action composition and preview."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from component.pose import PoseType


class ActionFrameCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pose_type: PoseType
    name: str
    duration_seconds: float = Field(gt=0.0)
    hold_seconds: float = Field(default=0.0, ge=0.0)


class ActionDefinitionCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    frames: tuple[ActionFrameCommand, ...] = Field(min_length=1)
    return_duration_seconds: float = Field(gt=0.0)
    notes: str = ""
    overwrite: bool = False


class CompileActionCommand(ActionDefinitionCommand):
    sample_frequency_hz: float = Field(default=25.0, gt=0.0)


class ActionDefinitionResponse(BaseModel):
    name: str
    home_pose: str
    frames: tuple[ActionFrameCommand, ...]
    return_duration_seconds: float
    total_duration_seconds: float
    notes: str


class ActionCatalogResponse(BaseModel):
    definitions: tuple[str, ...]
    trajectories: tuple[str, ...]
    base_poses: tuple[str, ...]
    complete_poses: tuple[dict[str, str], ...]


class TrajectoryResponse(BaseModel):
    action_name: str
    sample_count: int
    sample_frequency_hz: float
    duration_seconds: float
    joint_names: tuple[str, ...]


class PlayPreviewCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_name: str
    loop: bool = False


class PlaybackResponse(BaseModel):
    revision: int
    state: str
    action_name: str | None
    sample_index: int
    sample_count: int
    elapsed_seconds: float
    duration_seconds: float
    progress: float
    loop: bool
    error: str | None
