"""Render synchronized Tianyi MuJoCo snapshots from robot-relative cameras."""

from __future__ import annotations

import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mujoco
from PIL import Image

from component.pose import TianyiJointSchema
from component.simulation import SimulationSnapshot

LOGGER = logging.getLogger(__name__)


class CameraView(str, Enum):
    """Camera locations named from the robot's anatomical frame."""

    FRONT = "front"
    FRONT_LEFT = "front_left"
    LEFT = "left"
    RIGHT = "right"
    FRONT_RIGHT = "front_right"
    BACK = "back"


@dataclass(frozen=True, slots=True)
class RenderedPoseImage:
    camera_view: CameraView
    revision: int
    width: int
    height: int
    png_bytes: bytes
    render_seconds: float


class TianyiRenderService:
    """Own MuJoCo GL resources on one worker thread for stable repeated rendering."""

    DEFAULT_WIDTH = 720
    DEFAULT_HEIGHT = 720
    CAMERA_ELEVATION = -8.0
    CAMERA_DISTANCE = 2.15
    CAMERA_LOOKAT = (0.0, 0.0, 0.72)
    CAMERA_AZIMUTHS = {
        CameraView.FRONT: 180.0,
        CameraView.FRONT_LEFT: 225.0,
        CameraView.LEFT: 270.0,
        CameraView.RIGHT: 90.0,
        CameraView.FRONT_RIGHT: 135.0,
        CameraView.BACK: 0.0,
    }

    def __init__(self, *, schema: TianyiJointSchema, mjcf_path: Path) -> None:
        self.schema = schema
        self.mjcf_path = mjcf_path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tianyi-render")
        self._closed = False
        self._executor.submit(self._initialize_on_worker).result()

    def render_png(
        self,
        *,
        snapshot: SimulationSnapshot,
        camera_view: CameraView,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> RenderedPoseImage:
        if self._closed:
            raise RuntimeError("Tianyi renderer is closed")
        self._validate_image_size(width=width, height=height)
        return self._executor.submit(
            self._render_on_worker,
            snapshot,
            camera_view,
            width,
            height,
        ).result()

    def close(self) -> None:
        if self._closed:
            return
        self._executor.submit(self._close_on_worker).result()
        self._executor.shutdown(wait=True)
        self._closed = True

    @classmethod
    def create_camera(cls, camera_view: CameraView) -> mujoco.MjvCamera:
        if not isinstance(camera_view, CameraView):
            raise ValueError("camera_view must be a CameraView")
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = cls.CAMERA_LOOKAT
        camera.distance = cls.CAMERA_DISTANCE
        camera.azimuth = cls.CAMERA_AZIMUTHS[camera_view]
        camera.elevation = cls.CAMERA_ELEVATION
        return camera

    @staticmethod
    def _validate_image_size(*, width: int, height: int) -> None:
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Render {name} must be an integer")
            if not 160 <= value <= 1280:
                raise ValueError(f"Render {name} must be between 160 and 1280 pixels")

    def _initialize_on_worker(self) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        self._data = mujoco.MjData(self._model)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._qpos_addresses = self._load_qpos_addresses()
        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        self._scene_option.geomgroup[0] = 1
        self._scene_option.geomgroup[2] = 1

    def _load_qpos_addresses(self) -> tuple[int, ...]:
        addresses = []
        for joint_name in self.schema.MODEL_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"Visual MJCF is missing Tianyi joint: {joint_name}")
            addresses.append(int(self._model.jnt_qposadr[joint_id]))
        if self._model.nq != len(addresses) or len(set(addresses)) != len(addresses):
            raise ValueError("Visual MJCF does not preserve the fixed-base Tianyi joint layout")
        return tuple(addresses)

    def _render_on_worker(
        self,
        snapshot: SimulationSnapshot,
        camera_view: CameraView,
        width: int,
        height: int,
    ) -> RenderedPoseImage:
        if snapshot.robot_model_id != self.schema.model_id:
            raise ValueError(
                f"Snapshot model {snapshot.robot_model_id} does not match {self.schema.model_id}"
            )
        if snapshot.joint_names != self.schema.MODEL_JOINT_NAMES:
            raise ValueError("Snapshot joint order does not match the Tianyi visualization model")

        started_at = time.perf_counter()
        for address, value in zip(
            self._qpos_addresses,
            snapshot.joint_positions,
            strict=True,
        ):
            self._data.qpos[address] = value
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        renderer_key = (width, height)
        renderer = self._renderers.get(renderer_key)
        if renderer is None:
            renderer = mujoco.Renderer(self._model, height=height, width=width)
            self._renderers[renderer_key] = renderer
        renderer.update_scene(
            self._data,
            camera=self.create_camera(camera_view),
            scene_option=self._scene_option,
        )
        pixels = renderer.render().copy()
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format="PNG", compress_level=3)
        return RenderedPoseImage(
            camera_view=camera_view,
            revision=snapshot.revision,
            width=width,
            height=height,
            png_bytes=output.getvalue(),
            render_seconds=time.perf_counter() - started_at,
        )

    def _close_on_worker(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
