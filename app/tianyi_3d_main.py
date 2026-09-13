"""Run the combined host for Tianyi API and shared Viser infrastructure."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Make direct IDE/file execution behave like ``python -m app.tianyi_3d_main``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from app.api_tianyi_3d.api_main import router as api_router  # noqa: E402
from app.application import TianyiApplication  # noqa: E402
from app.ui_tianyi_3d.ui_main import router as ui_router  # noqa: E402
from app.ui_tianyi_3d.viser_manager import ViserManager  # noqa: E402
from config.settings import AppSettings  # noqa: E402
from util.tailwind_asset_helper import TailwindAssetHelper  # noqa: E402

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "ui_tianyi_3d" / "static"
FRONTEND_ASSETS = TailwindAssetHelper.for_tianyi_3d(project_root=PROJECT_ROOT)


def create_web_app(
    *,
    robot_application: TianyiApplication | None = None,
    viser_manager: ViserManager | None = None,
) -> FastAPI:
    """Create the host while keeping API and future UI routers independent."""
    owns_application = robot_application is None

    @asynccontextmanager
    async def lifespan(web_app: FastAPI) -> AsyncIterator[None]:
        FRONTEND_ASSETS.ensure_built()
        application = robot_application or TianyiApplication.build()
        manager = viser_manager or ViserManager(
            simulation=application.simulation,
            schema=application.joint_schema,
            urdf_path=application.settings.tianyi_visual_urdf_path,
            g1_urdf_path=application.settings.g1_asset_dir / "g1_29dof_fake_hand.urdf",
            host=application.settings.viser_host,
            port=application.settings.viser_port,
        )
        manager.start()
        web_app.state.tianyi = application
        web_app.state.viser = manager
        try:
            yield
        finally:
            manager.stop()
            if owns_application:
                application.close()

    web_app = FastAPI(
        title="Tianyi Action Recorder",
        version="0.1.0",
        lifespan=lifespan,
    )
    web_app.mount(
        "/static/tianyi-3d",
        StaticFiles(directory=STATIC_DIR),
        name="tianyi_3d_static",
    )
    if robot_application is not None:
        web_app.state.tianyi = robot_application
    if viser_manager is not None:
        web_app.state.viser = viser_manager
    web_app.include_router(api_router)
    web_app.include_router(ui_router)
    return web_app


web_app = create_web_app()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = AppSettings()
    LOGGER.info(
        "Starting Tianyi simulation console at http://%s:%d",
        settings.tianyi_3d_host,
        settings.tianyi_3d_port,
    )
    uvicorn.run(
        web_app,
        host=settings.tianyi_3d_host,
        port=settings.tianyi_3d_port,
    )


if __name__ == "__main__":
    main()
