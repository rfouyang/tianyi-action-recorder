"""FastAPI and browser UI entrypoint for Tianyi MuJoCo pose inspection."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from app.application import TianyiApplication
from component.visualization import CameraView
from config.settings import AppSettings

LOGGER = logging.getLogger(__name__)


class JointUpdateCommand(BaseModel):
    """Partial browser edit for authored head or arm joints."""

    model_config = ConfigDict(extra="forbid")

    joint_positions: dict[str, StrictFloat] = Field(min_length=1)


def create_app(*, application: TianyiApplication | None = None) -> FastAPI:
    settings = application.settings if application is not None else AppSettings()
    owns_application = application is None
    index_html = (settings.static_dir / "index.html").read_text(encoding="utf-8")
    app_css = (settings.static_dir / "app.css").read_text(encoding="utf-8")
    app_javascript = (settings.static_dir / "app.js").read_text(encoding="utf-8")

    @asynccontextmanager
    async def lifespan(web_app: FastAPI) -> AsyncIterator[None]:
        if not hasattr(web_app.state, "tianyi"):
            web_app.state.tianyi = TianyiApplication.build(settings=settings)
        try:
            yield
        finally:
            if owns_application:
                web_app.state.tianyi.close()

    web_app = FastAPI(
        title="Tianyi MuJoCo Pose API",
        version="0.1.0",
        lifespan=lifespan,
    )
    if application is not None:
        web_app.state.tianyi = application

    @web_app.get("/", include_in_schema=False)
    async def home() -> Response:
        return Response(content=index_html, media_type="text/html")

    @web_app.get("/static/app.css", include_in_schema=False)
    async def static_css() -> Response:
        return Response(content=app_css, media_type="text/css")

    @web_app.get("/static/app.js", include_in_schema=False)
    async def static_javascript() -> Response:
        return Response(content=app_javascript, media_type="text/javascript")

    @web_app.get("/api/v1/health")
    async def health(request: Request) -> dict[str, object]:
        app_state = _application(request)
        return {
            "status": "ok",
            "mode": "mujoco_simulation_only",
            "model_id": app_state.joint_schema.model_id,
            "revision": app_state.simulation.snapshot().revision,
        }

    @web_app.get("/api/v1/model")
    async def model_contract(request: Request) -> dict[str, object]:
        app_state = _application(request)
        schema = app_state.joint_schema
        report = app_state.visual_asset_report
        return {
            "model_id": schema.model_id,
            "mode": "mujoco_simulation_only",
            "pose_joint_counts": {
                "base": len(schema.BASE_JOINT_NAMES),
                "left_arm": len(schema.LEFT_ARM_JOINT_NAMES),
                "right_arm": len(schema.RIGHT_ARM_JOINT_NAMES),
                "composed": len(schema.BASE_JOINT_NAMES),
            },
            "joint_groups": {
                "head": schema.HEAD_JOINT_NAMES,
                "left_arm": schema.LEFT_ARM_JOINT_NAMES,
                "right_arm": schema.RIGHT_ARM_JOINT_NAMES,
                "locked": schema.LOCKED_JOINT_NAMES,
            },
            "locked_joint_motor_ids": dict(schema.LOCKED_JOINT_MOTOR_IDS),
            "locked_joint_positions": schema.locked_values(),
            "joints": [
                {
                    "name": definition.name,
                    "model_index": definition.model_index,
                    "group": definition.group.value,
                    "lower_limit": definition.lower_limit,
                    "upper_limit": definition.upper_limit,
                    "authored": definition.authored,
                }
                for definition in schema.definitions
            ],
            "cameras": [camera.value for camera in CameraView],
            "visualization": {
                "mesh_count": report.mesh_count,
                "source_triangles": report.source_triangle_count,
                "optimized_triangles": report.optimized_triangle_count,
                "triangle_reduction_percent": round(
                    report.triangle_reduction_ratio * 100,
                    1,
                ),
                "geom_count": report.geom_count,
            },
        }

    @web_app.get("/api/v1/simulation")
    async def simulation_state(request: Request) -> dict[str, object]:
        return _snapshot_payload(_application(request))

    @web_app.patch("/api/v1/simulation/joints")
    async def update_joints(
        request: Request,
        command: JointUpdateCommand,
    ) -> dict[str, object]:
        app_state = _application(request)
        try:
            app_state.simulation.update_joint_positions(command.joint_positions)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _snapshot_payload(app_state)

    @web_app.post("/api/v1/simulation/reset")
    async def reset_simulation(request: Request) -> dict[str, object]:
        app_state = _application(request)
        app_state.simulation.reset_to_default()
        return _snapshot_payload(app_state)

    @web_app.get("/api/v1/render/{camera_view}.png")
    async def render_pose(
        request: Request,
        camera_view: CameraView,
        width: int = 720,
        height: int = 720,
    ) -> Response:
        app_state = _application(request)
        try:
            image = app_state.renderer.render_png(
                snapshot=app_state.simulation.snapshot(),
                camera_view=camera_view,
                width=width,
                height=height,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return Response(
            content=image.png_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Simulation-Revision": str(image.revision),
                "X-Render-Milliseconds": f"{image.render_seconds * 1000:.1f}",
            },
        )

    return web_app


def _application(request: Request) -> TianyiApplication:
    return request.app.state.tianyi


def _snapshot_payload(application: TianyiApplication) -> dict[str, object]:
    snapshot = application.simulation.snapshot()
    positions = snapshot.joint_position_map()
    locked_joint_positions = application.simulation.calibration_joint_positions()
    return {
        "revision": snapshot.revision,
        "updated_at": snapshot.updated_at,
        "model_id": snapshot.robot_model_id,
        "joint_positions": positions,
        "joint_position_targets": snapshot.joint_position_target_map(),
        "base_position": snapshot.base_position,
        "base_wxyz": snapshot.base_wxyz,
        "locked_joint_names": application.joint_schema.LOCKED_JOINT_NAMES,
        "locked_joint_positions": locked_joint_positions,
        "locked": all(
            positions[joint_name] == value
            for joint_name, value in locked_joint_positions.items()
        ),
    }


app = create_app()


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("app.main:app", host="127.0.0.1", port=7900, reload=False)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
