"""Transport schemas for saved-pose discovery and composition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from component.pose import PoseType


class PreviewSavedPoseCommand(BaseModel):
    """Saved pose selected as a MuJoCo editing starting point."""

    model_config = ConfigDict(extra="forbid")

    pose_type: PoseType
    name: str


class SavedPosePreviewResponse(BaseModel):
    """Loaded pose metadata and the resulting simulation revision."""

    pose_name: str
    pose_type: PoseType
    edit_pose_type: PoseType
    joint_positions: dict[str, float]
    notes: str
    revision: int


class PreviewPoseCompositionCommand(BaseModel):
    """Named pose parts used for an unsaved MuJoCo composition preview."""

    model_config = ConfigDict(extra="forbid")

    base_pose: str
    left_arm_pose: str | None = None
    right_arm_pose: str | None = None


class SavePoseCompositionCommand(PreviewPoseCompositionCommand):
    """Named pose parts and metadata for one persisted composed pose."""

    name: str
    notes: str = ""
    overwrite: bool = False


class PoseCatalogResponse(BaseModel):
    """Saved pose names grouped by exact pose shape."""

    base: tuple[str, ...]
    left_arm: tuple[str, ...]
    right_arm: tuple[str, ...]
    composed: tuple[str, ...]


class ComposedPoseResponse(BaseModel):
    """Composition result and the MuJoCo revision showing it."""

    pose_name: str
    pose_type: str
    source_parts: dict[str, str]
    revision: int
    saved: bool


class MirrorArmCommand(BaseModel):
    """An anatomical source arm to capture from the live MuJoCo editor."""

    model_config = ConfigDict(extra="forbid")

    source_pose_type: PoseType


class MirrorArmResponse(BaseModel):
    """Mirrored destination values and the MuJoCo revision showing them."""

    source_pose_type: PoseType
    target_pose_type: PoseType
    source_joint_positions: dict[str, float]
    joint_positions: dict[str, float]
    revision: int
