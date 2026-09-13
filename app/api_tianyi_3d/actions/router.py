"""REST routes over the shared Tianyi Action Composer capability."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api_tianyi_3d.actions.schemas import (
    ActionCatalogResponse,
    ActionDefinitionCommand,
    ActionDefinitionResponse,
    CompileActionCommand,
    PlaybackResponse,
    PlayPreviewCommand,
    TrajectoryResponse,
)
from app.api_tianyi_3d.dependencies import get_application
from app.application import TianyiApplication
from component.action import ActionDefinition, ActionPoseReference, ActionTransition
from component.pose import PoseType

router = APIRouter(prefix="/actions", tags=["actions"])


@router.get("", response_model=ActionCatalogResponse)
async def catalog(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> ActionCatalogResponse:
    return ActionCatalogResponse.model_validate(_catalog(application))


@router.get("/definitions/{name}", response_model=ActionDefinitionResponse)
async def load_definition(
    name: str,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> ActionDefinitionResponse:
    try:
        return _definition_response(application.action_service.load_action(name=name))
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/definitions", response_model=ActionDefinitionResponse)
async def save_definition(
    command: ActionDefinitionCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> ActionDefinitionResponse:
    try:
        action = _create_action(application, command)
        application.action_service.save_action(action=action, overwrite=command.overwrite)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail="Action definition already exists") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _definition_response(action)


@router.post("/trajectories", response_model=TrajectoryResponse)
async def compile_trajectory(
    command: CompileActionCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> TrajectoryResponse:
    try:
        action = _create_action(application, command)
        trajectory = application.action_service.generate_trajectory(
            action=action,
            sample_frequency_hz=command.sample_frequency_hz,
        )
        application.action_service.save_trajectory(
            trajectory=trajectory,
            overwrite=command.overwrite,
        )
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail="Compiled trajectory already exists") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _trajectory_response(trajectory)


@router.get("/preview", response_model=PlaybackResponse)
async def preview_status(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> PlaybackResponse:
    return _playback_response(application)


@router.post("/preview/play", response_model=PlaybackResponse)
async def play_preview(
    command: PlayPreviewCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> PlaybackResponse:
    try:
        trajectory = application.action_service.load_trajectory(name=command.action_name)
        application.action_playback.load(trajectory)
        application.action_playback.play(loop=command.loop)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _playback_response(application)


@router.post("/preview/pause", response_model=PlaybackResponse)
async def pause_preview(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> PlaybackResponse:
    try:
        application.action_playback.pause()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _playback_response(application)


@router.post("/preview/resume", response_model=PlaybackResponse)
async def resume_preview(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> PlaybackResponse:
    try:
        application.action_playback.resume()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _playback_response(application)


@router.post("/preview/stop", response_model=PlaybackResponse)
async def stop_preview(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> PlaybackResponse:
    application.action_playback.stop()
    return _playback_response(application)


def _create_action(
    application: TianyiApplication,
    command: ActionDefinitionCommand,
) -> ActionDefinition:
    frames = tuple(
        ActionTransition(
            target_pose=ActionPoseReference(frame.pose_type, frame.name),
            duration_seconds=frame.duration_seconds,
            hold_seconds=frame.hold_seconds,
        )
        for frame in command.frames
    )
    return application.action_service.create_action(
        name=command.name,
        frames=frames,
        return_duration_seconds=command.return_duration_seconds,
        notes=command.notes,
    )


def _catalog(application: TianyiApplication) -> dict[str, object]:
    complete = []
    for pose_type, label in ((PoseType.BASE, "Base"), (PoseType.COMPOSED, "Composed")):
        for pose in application.pose_service.list_poses(pose_type=pose_type):
            complete.append(
                {
                    "pose_type": pose_type.value,
                    "name": pose.name,
                    "value": f"{pose_type.value}/{pose.name}",
                    "label": f"{label} · {pose.name}",
                }
            )
    return {
        "definitions": tuple(item.name for item in application.action_service.list_actions()),
        "trajectories": application.action_service.list_trajectories(),
        "base_poses": tuple(
            item.name for item in application.pose_service.list_poses(pose_type=PoseType.BASE)
        ),
        "complete_poses": tuple(complete),
    }


def _definition_response(action: ActionDefinition) -> ActionDefinitionResponse:
    return ActionDefinitionResponse(
        name=action.name,
        home_pose=action.initial_pose.name,
        frames=tuple(
            {
                "pose_type": transition.target_pose.pose_type,
                "name": transition.target_pose.name,
                "duration_seconds": transition.duration_seconds,
                "hold_seconds": transition.hold_seconds,
            }
            for transition in action.transitions[:-1]
        ),
        return_duration_seconds=action.transitions[-1].duration_seconds,
        total_duration_seconds=action.total_duration_seconds,
        notes=action.notes,
    )


def _trajectory_response(trajectory: object) -> TrajectoryResponse:
    return TrajectoryResponse(
        action_name=trajectory.action_name,
        sample_count=trajectory.sample_count,
        sample_frequency_hz=trajectory.requested_sample_frequency_hz,
        duration_seconds=trajectory.duration_seconds,
        joint_names=trajectory.joint_names,
    )


def _playback_response(application: TianyiApplication) -> PlaybackResponse:
    snapshot = application.action_playback.snapshot()
    return PlaybackResponse(
        revision=snapshot.revision,
        state=snapshot.state.value,
        action_name=snapshot.action_name,
        sample_index=snapshot.sample_index,
        sample_count=snapshot.sample_count,
        elapsed_seconds=snapshot.elapsed_seconds,
        duration_seconds=snapshot.duration_seconds,
        progress=snapshot.progress,
        loop=snapshot.loop,
        error=snapshot.error,
    )
