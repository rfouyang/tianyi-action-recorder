"""REST routes for Tianyi saved-pose discovery and composition."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api_tianyi_3d.dependencies import get_application
from app.api_tianyi_3d.poses.schemas import (
    ComposedPoseResponse,
    MirrorArmCommand,
    MirrorArmResponse,
    PoseCatalogResponse,
    PreviewPoseCompositionCommand,
    PreviewSavedPoseCommand,
    SavedPosePreviewResponse,
    SavePoseCompositionCommand,
)
from app.application import TianyiApplication
from component.pose import PoseDefinition, PoseType

router = APIRouter(prefix="/poses", tags=["poses"])


@router.get("", response_model=PoseCatalogResponse)
async def list_poses(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> PoseCatalogResponse:
    """Return saved pose names grouped by their exact Tianyi pose shape."""
    return PoseCatalogResponse.model_validate(application.pose_service.list_pose_names())


@router.post("/preview", response_model=SavedPosePreviewResponse)
async def preview_saved_pose(
    command: PreviewSavedPoseCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> SavedPosePreviewResponse:
    """Load a saved pose and apply it as an editable MuJoCo starting point."""
    try:
        pose = application.pose_service.load_pose(
            pose_type=command.pose_type,
            name=command.name,
        )
        snapshot = application.simulation.apply_pose(pose)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    edit_pose_type = (
        PoseType.BASE if command.pose_type is PoseType.COMPOSED else command.pose_type
    )
    return SavedPosePreviewResponse(
        pose_name=pose.name,
        pose_type=pose.pose_type,
        edit_pose_type=edit_pose_type,
        joint_positions=dict(pose.joint_values),
        notes=pose.notes,
        revision=snapshot.revision,
    )


@router.post("/compositions/preview", response_model=ComposedPoseResponse)
async def preview_composition(
    command: PreviewPoseCompositionCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> ComposedPoseResponse:
    """Compose named sources and apply the unsaved result only to MuJoCo."""
    try:
        pose = application.pose_service.compose_pose_from_names(
            name="unsaved_composed_preview",
            base_pose_name=command.base_pose,
            left_arm_pose_name=command.left_arm_pose,
            right_arm_pose_name=command.right_arm_pose,
        )
        snapshot = application.simulation.apply_pose(pose)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _response(pose=pose, revision=snapshot.revision, saved=False)


@router.post("/compositions", response_model=ComposedPoseResponse)
async def save_composition(
    command: SavePoseCompositionCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> ComposedPoseResponse:
    """Compose, persist, and apply a complete head-and-arms pose."""
    try:
        pose = application.pose_service.compose_pose_from_names(
            name=command.name,
            base_pose_name=command.base_pose,
            left_arm_pose_name=command.left_arm_pose,
            right_arm_pose_name=command.right_arm_pose,
            notes=command.notes,
        )
        application.pose_service.save_pose(pose=pose, overwrite=command.overwrite)
        snapshot = application.simulation.apply_pose(pose)
    except FileExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Pose {command.name!r} already exists. Enable replacement to save over it."
            ),
        ) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _response(pose=pose, revision=snapshot.revision, saved=True)


@router.post("/arms/mirror", response_model=MirrorArmResponse)
async def mirror_arm(
    command: MirrorArmCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> MirrorArmResponse:
    """Reflect current arm values and apply only the opposite arm in MuJoCo."""
    try:
        target_pose_type, source_values, mirrored, snapshot = (
            application.simulation.mirror_current_arm(
                source_pose_type=command.source_pose_type,
            )
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return MirrorArmResponse(
        source_pose_type=command.source_pose_type,
        target_pose_type=target_pose_type,
        source_joint_positions=source_values,
        joint_positions=mirrored,
        revision=snapshot.revision,
    )


def _response(
    *,
    pose: PoseDefinition,
    revision: int,
    saved: bool,
) -> ComposedPoseResponse:
    return ComposedPoseResponse(
        pose_name=pose.name,
        pose_type=pose.pose_type.value,
        source_parts=dict(pose.source_parts),
        revision=revision,
        saved=saved,
    )
