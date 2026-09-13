"""HTML routes for the Tianyi 3D motion-authoring console."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from app.api_tianyi_3d.actions.schemas import (
    ActionDefinitionCommand,
    CompileActionCommand,
    PlayPreviewCommand,
)
from app.api_tianyi_3d.poses.schemas import MirrorArmCommand
from app.ui_tianyi_3d.context import UIContext
from app.ui_tianyi_3d.panels.action_composer import ActionComposerPanel
from app.ui_tianyi_3d.panels.pose_composer import (
    ComposePoseCommand,
    PoseComposerPanel,
)
from app.ui_tianyi_3d.panels.pose_recorder import PoseRecorderPanel
from app.ui_tianyi_3d.panels.pose_retargeting import (
    PoseRetargetingPanel,
    RetargetPoseCommand,
)
from app.ui_tianyi_3d.viser_manager import RobotCameraView, ViewerMode
from app.ui_tianyi_3d.websocket import ui_simulation_websocket
from component.pose import PoseDefinition, PoseType

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=TEMPLATE_DIR)
router = APIRouter(include_in_schema=False)
router.add_api_websocket_route(
    "/ui/tianyi-3d/simulation/ws",
    ui_simulation_websocket,
)


class SavePoseCommand(BaseModel):
    """Browser command for persisting one simulated pose group."""

    model_config = ConfigDict(extra="forbid")

    pose_type: PoseType
    name: str
    notes: str = ""
    joint_positions: dict[str, float] = Field(min_length=1)
    overwrite: bool = False


class PreviewActionPoseCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pose_type: PoseType
    name: str


class LoadRecorderPoseCommand(BaseModel):
    """Saved pose selected as an editable Pose Recorder starting point."""

    model_config = ConfigDict(extra="forbid")

    pose_type: PoseType
    name: str


@router.get("/", response_class=HTMLResponse, name="tianyi_3d_home")
def home(request: Request) -> HTMLResponse:
    """Render the simulation-only Pose Recorder and 3D viewer."""
    context = UIContext.from_request(request)
    template_context = {
        "model_id": context.app.joint_schema.model_id,
        "viser_port": context.viser.port,
        **PoseRecorderPanel.template_context(context),
        **PoseComposerPanel.template_context(context),
        **ActionComposerPanel.template_context(context),
    }
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context=template_context,
    )


@router.post("/ui/tianyi-3d/viewer/camera/{camera_view}")
def set_camera_view(
    request: Request,
    camera_view: RobotCameraView,
) -> dict[str, object]:
    """Move the embedded Viser client to a robot-relative view."""
    context = UIContext.from_request(request)
    client_count = context.viser.set_camera_view(camera_view)
    return {
        "camera_view": camera_view.value,
        "connected_clients": client_count,
        "transition_seconds": context.viser.camera_transition_seconds,
    }


@router.post("/ui/tianyi-3d/viewer/mode/{viewer_mode}")
def set_viewer_mode(
    request: Request,
    viewer_mode: ViewerMode,
) -> dict[str, object]:
    """Select the single-robot or retarget-comparison scene arrangement."""
    context = UIContext.from_request(request)
    return {
        "viewer_mode": viewer_mode.value,
        "comparison_visible": context.viser.set_viewer_mode(viewer_mode),
        "connected_clients": context.viser.connected_client_count,
        "transition_seconds": context.viser.camera_transition_seconds,
    }


@router.post("/ui/tianyi-3d/poses")
def save_pose(request: Request, command: SavePoseCommand) -> dict[str, object]:
    """Save a validated pose from the browser's current MuJoCo targets."""
    context = UIContext.from_request(request)
    try:
        pose, revision = PoseRecorderPanel.save_simulation_pose(
            context=context,
            pose_type=command.pose_type,
            name=command.name,
            notes=command.notes,
            joint_positions=command.joint_positions,
            overwrite=command.overwrite,
        )
    except FileExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=(f"Pose {command.name!r} already exists. Enable replacement to save over it."),
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "pose_name": pose.name,
        "pose_type": pose.pose_type.value,
        "revision": revision,
    }


@router.get("/ui/tianyi-3d/poses")
def list_poses(request: Request) -> dict[str, tuple[str, ...]]:
    """Return current saved pose names for browser workspace refreshes."""
    return PoseComposerPanel.pose_names(UIContext.from_request(request))


@router.post("/ui/tianyi-3d/pose-recorder/load")
def load_recorder_pose(
    request: Request,
    command: LoadRecorderPoseCommand,
) -> dict[str, object]:
    """Apply a saved pose to MuJoCo and return values for recorder editing."""
    try:
        pose, edit_pose_type, revision = PoseRecorderPanel.load_saved_pose(
            context=UIContext.from_request(request),
            pose_type=command.pose_type,
            name=command.name,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "pose_name": pose.name,
        "pose_type": pose.pose_type.value,
        "edit_pose_type": edit_pose_type.value,
        "joint_positions": dict(pose.joint_values),
        "notes": pose.notes,
        "revision": revision,
    }


@router.post("/ui/tianyi-3d/pose-retargeting/preview")
def preview_retargeted_pose(
    request: Request,
    command: RetargetPoseCommand,
) -> dict[str, object]:
    """Solve and apply an unsaved Tianyi pose to shared MuJoCo state."""
    try:
        result, revision = PoseRetargetingPanel.preview(
            context=UIContext.from_request(request),
            command=command,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _retargeting_response(result=result, revision=revision, saved=False)


@router.post("/ui/tianyi-3d/pose-retargeting/save")
def save_retargeted_pose(
    request: Request,
    command: RetargetPoseCommand,
) -> dict[str, object]:
    """Solve, persist one selected Tianyi arm, and apply it in MuJoCo."""
    try:
        result, pose, revision = PoseRetargetingPanel.save(
            context=UIContext.from_request(request),
            command=command,
        )
    except FileExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Pose {command.name!r} already exists. Enable replacement to save over it."
            ),
        ) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _retargeting_response(
        result=result,
        revision=revision,
        saved=True,
        saved_pose_name=pose.name,
        saved_pose_type=pose.pose_type,
    )


def _retargeting_response(
    *,
    result,
    revision: int,
    saved: bool,
    saved_pose_name: str | None = None,
    saved_pose_type: PoseType | None = None,
) -> dict[str, object]:
    return {
        "source_pose_type": result.source_pose_type,
        "source_name": result.source_name,
        "source_model_id": result.source_model_id,
        "target_model_id": result.target_model_id,
        "source_joint_positions": dict(result.source_joint_values),
        "converged": result.converged,
        "arm_reports": [
            {
                "side": report.side,
                "converged": report.converged,
                "iterations": report.iterations,
                "hand_position_error_m": report.hand_position_error_m,
                "hand_orientation_error_rad": report.hand_orientation_error_rad,
                "elbow_position_error_m": report.elbow_position_error_m,
                "upper_arm_direction_error_rad": report.upper_arm_direction_error_rad,
                "forearm_direction_error_rad": report.forearm_direction_error_rad,
                "reached_joint_limits": report.reached_joint_limits,
            }
            for report in result.arm_reports
        ],
        "contact_count": result.contact_count,
        "joint_positions": dict(result.joint_values),
        "revision": revision,
        "saved": saved,
        "saved_pose_name": saved_pose_name,
        "saved_pose_type": saved_pose_type.value if saved_pose_type else None,
    }


@router.post("/ui/tianyi-3d/pose-recorder/mirror-arm")
def mirror_recorder_arm(
    request: Request,
    command: MirrorArmCommand,
) -> dict[str, object]:
    """Mirror live recorder values into the opposite MuJoCo arm."""
    try:
        target_pose_type, source_values, mirrored, revision = PoseRecorderPanel.mirror_arm(
            context=UIContext.from_request(request),
            source_pose_type=command.source_pose_type,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "source_pose_type": command.source_pose_type.value,
        "target_pose_type": target_pose_type.value,
        "source_joint_positions": source_values,
        "joint_positions": mirrored,
        "revision": revision,
    }


@router.post("/ui/tianyi-3d/pose-composer/preview")
def preview_composed_pose(
    request: Request,
    command: ComposePoseCommand,
) -> dict[str, object]:
    """Compose selected sources and apply the unsaved result to MuJoCo."""
    context = UIContext.from_request(request)
    try:
        pose, revision = PoseComposerPanel.preview(context=context, command=command)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _composed_pose_response(pose=pose, revision=revision)


@router.post("/ui/tianyi-3d/pose-composer/save")
def save_composed_pose(
    request: Request,
    command: ComposePoseCommand,
) -> dict[str, object]:
    """Compose, persist, and apply a final head-and-arms pose."""
    context = UIContext.from_request(request)
    try:
        pose, revision = PoseComposerPanel.save(context=context, command=command)
    except FileExistsError as error:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Pose {command.name!r} already exists. Enable replacement to save over it."
            ),
        ) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _composed_pose_response(pose=pose, revision=revision)


def _composed_pose_response(
    *,
    pose: PoseDefinition,
    revision: int,
) -> dict[str, object]:
    return {
        "pose_name": pose.name,
        "pose_type": pose.pose_type.value,
        "source_parts": dict(pose.source_parts),
        "revision": revision,
    }


@router.get("/ui/tianyi-3d/action/sources")
def action_sources(request: Request) -> dict[str, object]:
    return ActionComposerPanel.sources(UIContext.from_request(request))


@router.get("/ui/tianyi-3d/action/definitions/{name}")
def load_action(request: Request, name: str) -> dict[str, object]:
    try:
        return ActionComposerPanel.load(context=UIContext.from_request(request), name=name)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/ui/tianyi-3d/action/save")
def save_action(request: Request, command: ActionDefinitionCommand) -> dict[str, object]:
    try:
        action = ActionComposerPanel.save(
            context=UIContext.from_request(request),
            command=command,
        )
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail="Action definition already exists") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "action_name": action.name,
        "keyframe_count": len(action.pose_sequence),
        "total_duration_seconds": action.total_duration_seconds,
    }


@router.post("/ui/tianyi-3d/action/compile")
def compile_action(request: Request, command: CompileActionCommand) -> dict[str, object]:
    try:
        trajectory = ActionComposerPanel.compile(
            context=UIContext.from_request(request),
            command=command,
        )
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail="Compiled trajectory already exists") from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "action_name": trajectory.action_name,
        "sample_count": trajectory.sample_count,
        "sample_frequency_hz": trajectory.requested_sample_frequency_hz,
        "duration_seconds": trajectory.duration_seconds,
        "download_url": f"/ui/tianyi-3d/action/trajectories/{trajectory.action_name}/download",
    }


@router.get("/ui/tianyi-3d/action/trajectories/{name}/download")
def download_trajectory(request: Request, name: str) -> FileResponse:
    try:
        path = UIContext.from_request(request).app.action_service.trajectory_path(name=name)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.post("/ui/tianyi-3d/action/preview-pose")
def preview_action_pose(
    request: Request,
    command: PreviewActionPoseCommand,
) -> dict[str, object]:
    try:
        revision = ActionComposerPanel.preview_pose(
            context=UIContext.from_request(request),
            pose_type=command.pose_type,
            name=command.name,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"revision": revision}


@router.get("/ui/tianyi-3d/action/playback")
def action_playback_status(request: Request) -> dict[str, object]:
    return _playback_payload(UIContext.from_request(request))


@router.post("/ui/tianyi-3d/action/playback/play")
def play_action_preview(request: Request, command: PlayPreviewCommand) -> dict[str, object]:
    context = UIContext.from_request(request)
    try:
        context.app.action_playback.load(
            context.app.action_service.load_trajectory(name=command.action_name)
        )
        context.app.action_playback.play(loop=command.loop)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _playback_payload(context)


@router.post("/ui/tianyi-3d/action/playback/{command}")
def control_action_preview(request: Request, command: str) -> dict[str, object]:
    context = UIContext.from_request(request)
    try:
        if command == "pause":
            context.app.action_playback.pause()
        elif command == "resume":
            context.app.action_playback.resume()
        elif command == "stop":
            context.app.action_playback.stop()
        else:
            raise ValueError(f"Unknown playback command: {command}")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _playback_payload(context)


def _playback_payload(context: UIContext) -> dict[str, object]:
    snapshot = context.app.action_playback.snapshot()
    return {
        "revision": snapshot.revision,
        "state": snapshot.state.value,
        "action_name": snapshot.action_name,
        "sample_index": snapshot.sample_index,
        "sample_count": snapshot.sample_count,
        "elapsed_seconds": snapshot.elapsed_seconds,
        "duration_seconds": snapshot.duration_seconds,
        "progress": snapshot.progress,
        "loop": snapshot.loop,
        "error": snapshot.error,
    }
