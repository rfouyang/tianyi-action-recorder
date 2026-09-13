"""Simulation-only G1-to-Tianyi pose retargeting REST routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api_tianyi_3d.dependencies import get_application
from app.api_tianyi_3d.retargeting.schemas import (
    ArmRetargetReportResponse,
    RetargetPoseCommand,
    RetargetPoseResponse,
    SaveRetargetedPoseCommand,
)
from app.application import TianyiApplication
from component.pose import PoseType
from component.retarget import PoseRetargetResult

router = APIRouter(prefix="/pose-retargeting", tags=["pose-retargeting"])


@router.post("/preview", response_model=RetargetPoseResponse)
async def preview_retargeted_pose(
    command: RetargetPoseCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> RetargetPoseResponse:
    try:
        result = application.pose_retargeting.retarget(
            source_pose_json=command.source_pose_json,
            source_file_name=command.source_file_name,
        )
        snapshot = application.simulation.apply_pose_values(
            pose_type=PoseType.BASE,
            joint_values=result.joint_values,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _response(result=result, revision=snapshot.revision, saved=False)


@router.post("", response_model=RetargetPoseResponse)
async def save_retargeted_pose(
    command: SaveRetargetedPoseCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> RetargetPoseResponse:
    try:
        result = application.pose_retargeting.retarget(
            source_pose_json=command.source_pose_json,
            source_file_name=command.source_file_name,
        )
        source_reference = (
            f"G1 {result.source_pose_type}/{result.source_name} "
            f"from {command.source_file_name}"
        )
        side_label = "robot-left" if command.output_pose_type == "left_arm" else "robot-right"
        notes = f"Retargeted {side_label} arm in MuJoCo from {source_reference}."
        if command.notes.strip():
            notes = f"{notes} {command.notes.strip()}"
        output_pose_type = PoseType(command.output_pose_type)
        arm_values = {
            joint_name: result.joint_values[joint_name]
            for joint_name in application.joint_schema.joint_names(output_pose_type)
        }
        pose = application.pose_service.create_retargeted_arm_pose(
            name=command.name,
            pose_type=output_pose_type,
            joint_values=arm_values,
            notes=notes,
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
    return _response(
        result=result,
        revision=snapshot.revision,
        saved=True,
        saved_pose_name=pose.name,
        saved_pose_type=pose.pose_type.value,
    )


def _response(
    *,
    result: PoseRetargetResult,
    revision: int,
    saved: bool,
    saved_pose_name: str | None = None,
    saved_pose_type: str | None = None,
) -> RetargetPoseResponse:
    return RetargetPoseResponse(
        source_pose_type=result.source_pose_type,
        source_name=result.source_name,
        source_model_id=result.source_model_id,
        target_model_id=result.target_model_id,
        converged=result.converged,
        arm_reports=tuple(
            ArmRetargetReportResponse.model_validate(report, from_attributes=True)
            for report in result.arm_reports
        ),
        contact_count=result.contact_count,
        joint_positions=dict(result.joint_values),
        revision=revision,
        saved=saved,
        saved_pose_name=saved_pose_name,
        saved_pose_type=saved_pose_type,
    )
