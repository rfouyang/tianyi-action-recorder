"""Shared Viser runtime for the live Tianyi MuJoCo scene."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from component.pose import TianyiJointSchema
from component.simulation import SimulationService, SimulationSnapshot
from util.g1_asset_helper import G1AssetHelper
from util.g1_pose_adapter import G1_VISUAL_JOINT_NAMES
from util.tianyi_visual_urdf_helper import TianyiVisualUrdfHelper

LOGGER = logging.getLogger(__name__)


class RobotCameraView(str, Enum):
    """Robot-relative camera presets exposed by the browser UI."""

    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    FRONT_LEFT_45 = "front_left_45"
    FRONT_RIGHT_45 = "front_right_45"


class ViewerMode(str, Enum):
    """Visible robot arrangement selected by the active browser workspace."""

    TIANYI = "tianyi"
    RETARGET_COMPARISON = "retarget-comparison"


@dataclass(frozen=True, slots=True)
class CameraPreset:
    """Position and focus point for one fixed Viser camera view."""

    position: tuple[float, float, float]
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.72)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vertical_fov: float = math.radians(45.0)


class ViserManager:
    """Load both visual models once and synchronize the active scene layout."""

    # Tianyi coordinates match G1: front is +X, robot-left is +Y, and up is +Z.
    _CAMERA_DISTANCE = 2.15
    _CAMERA_DIAGONAL = _CAMERA_DISTANCE / math.sqrt(2.0)
    _CAMERA_HEIGHT = 0.92
    CAMERA_PRESETS = {
        RobotCameraView.FRONT: CameraPreset((_CAMERA_DISTANCE, 0.0, _CAMERA_HEIGHT)),
        RobotCameraView.BACK: CameraPreset((-_CAMERA_DISTANCE, 0.0, _CAMERA_HEIGHT)),
        RobotCameraView.LEFT: CameraPreset((0.0, _CAMERA_DISTANCE, _CAMERA_HEIGHT)),
        RobotCameraView.RIGHT: CameraPreset((0.0, -_CAMERA_DISTANCE, _CAMERA_HEIGHT)),
        RobotCameraView.FRONT_LEFT_45: CameraPreset(
            (_CAMERA_DIAGONAL, _CAMERA_DIAGONAL, _CAMERA_HEIGHT)
        ),
        RobotCameraView.FRONT_RIGHT_45: CameraPreset(
            (_CAMERA_DIAGONAL, -_CAMERA_DIAGONAL, _CAMERA_HEIGHT)
        ),
    }
    _COMPARISON_CAMERA_DISTANCE = 3.25
    _COMPARISON_CAMERA_DIAGONAL = _COMPARISON_CAMERA_DISTANCE / math.sqrt(2.0)
    COMPARISON_CAMERA_PRESETS = {
        RobotCameraView.FRONT: CameraPreset(
            (_COMPARISON_CAMERA_DISTANCE, 0.0, 1.05),
            look_at=(0.0, 0.0, 0.72),
        ),
        RobotCameraView.BACK: CameraPreset(
            (-_COMPARISON_CAMERA_DISTANCE, 0.0, 1.05),
            look_at=(0.0, 0.0, 0.72),
        ),
        RobotCameraView.LEFT: CameraPreset(
            (0.0, _COMPARISON_CAMERA_DISTANCE, 1.05),
            look_at=(0.0, 0.0, 0.72),
        ),
        RobotCameraView.RIGHT: CameraPreset(
            (0.0, -_COMPARISON_CAMERA_DISTANCE, 1.05),
            look_at=(0.0, 0.0, 0.72),
        ),
        RobotCameraView.FRONT_LEFT_45: CameraPreset(
            (_COMPARISON_CAMERA_DIAGONAL, _COMPARISON_CAMERA_DIAGONAL, 1.05),
            look_at=(0.0, 0.0, 0.72),
        ),
        RobotCameraView.FRONT_RIGHT_45: CameraPreset(
            (_COMPARISON_CAMERA_DIAGONAL, -_COMPARISON_CAMERA_DIAGONAL, 1.05),
            look_at=(0.0, 0.0, 0.72),
        ),
    }
    # In Front, -Y is screen-left and +Y is screen-right under the shared
    # robot-relative convention. Keep source on the left and result on the right.
    _G1_COMPARISON_POSITION = (0.0, -0.78, 0.793)
    _TIANYI_COMPARISON_OFFSET = (0.0, 0.78, 0.0)

    def __init__(
        self,
        *,
        simulation: SimulationService,
        schema: TianyiJointSchema,
        urdf_path: Path,
        g1_urdf_path: Path | None = None,
        host: str,
        port: int,
        update_hz: float = 30.0,
        camera_transition_seconds: float = 0.7,
        camera_transition_hz: float = 60.0,
    ) -> None:
        if update_hz <= 0:
            raise ValueError("Viser update frequency must be positive")
        if camera_transition_seconds < 0:
            raise ValueError("Camera transition duration cannot be negative")
        if camera_transition_hz <= 0:
            raise ValueError("Camera transition frequency must be positive")
        self.simulation = simulation
        self.schema = schema
        self.urdf_path = urdf_path
        self.g1_urdf_path = g1_urdf_path
        self.host = host
        self.requested_port = port
        self.update_hz = update_hz
        self.camera_transition_seconds = camera_transition_seconds
        self.camera_transition_hz = camera_transition_hz
        self._lifecycle_lock = threading.Lock()
        self._scene_state_lock = threading.RLock()
        self._camera_transition_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._server: viser.ViserServer | None = None
        self._robot_root: viser.FrameHandle | None = None
        self._robot: ViserUrdf | None = None
        self._g1_root: viser.FrameHandle | None = None
        self._g1_robot: ViserUrdf | None = None
        self._g1_label: viser.LabelHandle | None = None
        self._tianyi_label: viser.LabelHandle | None = None
        self._sync_thread: threading.Thread | None = None
        self._camera_transition_generation = 0
        self._camera_transition_threads: set[threading.Thread] = set()
        self._visual_actuated_joint_names: tuple[str, ...] = ()
        self._g1_visual_joint_names: tuple[str, ...] = ()
        self._g1_source_values: dict[str, float] | None = None
        self._retarget_workspace_active = False
        self._comparison_visible = False
        self._camera_view = RobotCameraView.FRONT
        self._synced_revision = -1

    @property
    def port(self) -> int:
        """Return the actual bound Viser port."""
        if self._server is None:
            raise RuntimeError("Viser runtime has not been started")
        return self._server.get_port()

    @property
    def connected_client_count(self) -> int:
        if self._server is None:
            return 0
        return len(self._server.get_clients())

    @property
    def synced_revision(self) -> int:
        return self._synced_revision

    def start(self) -> None:
        """Start Viser, build the scene once, and begin revision synchronization."""
        with self._lifecycle_lock:
            if self._server is not None:
                return
            visual_report = TianyiVisualUrdfHelper(asset_dir=self.urdf_path.parent).validate()
            if self.g1_urdf_path is None:
                raise ValueError("A pinned G1 visual URDF is required for retarget previews")
            G1AssetHelper(asset_dir=self.g1_urdf_path.parent).validate()
            server = viser.ViserServer(
                host=self.host,
                port=self.requested_port,
                label="Tianyi Action Recorder Scene",
                verbose=False,
            )
            try:
                self._configure_scene(server)
                robot_root = server.scene.add_frame("/scene/robot", show_axes=False)
                robot = ViserUrdf(
                    server,
                    self.urdf_path,
                    root_node_name="/scene/robot",
                )
                visual_actuated_joint_names = robot.get_actuated_joint_names()
                if visual_actuated_joint_names != visual_report.actuated_joint_names:
                    raise ValueError("Viser URDF actuated joints do not match generated metadata")
                if visual_report.body_joint_names != self.schema.MODEL_JOINT_NAMES:
                    raise ValueError("Viser URDF body joints do not match the Tianyi schema")
                g1_root = server.scene.add_frame(
                    "/scene/g1",
                    show_axes=False,
                    position=self._G1_COMPARISON_POSITION,
                    visible=False,
                )
                g1_robot = ViserUrdf(
                    server,
                    self.g1_urdf_path,
                    root_node_name="/scene/g1",
                )
                g1_visual_joint_names = g1_robot.get_actuated_joint_names()
                if g1_visual_joint_names != G1_VISUAL_JOINT_NAMES:
                    raise ValueError("Viser G1 joints do not match the pinned model contract")
                g1_robot.show_visual = False
                g1_label = server.scene.add_label(
                    "/scene/comparison/g1-label",
                    "G1 · ORIGINAL",
                    position=(0.0, self._G1_COMPARISON_POSITION[1], 1.47),
                    anchor="bottom-center",
                    visible=False,
                )
                tianyi_label = server.scene.add_label(
                    "/scene/comparison/tianyi-label",
                    "TIANYI · RETARGETED",
                    position=(0.0, self._TIANYI_COMPARISON_OFFSET[1], 1.47),
                    anchor="bottom-center",
                    visible=False,
                )

                self._server = server
                self._robot_root = robot_root
                self._robot = robot
                self._g1_root = g1_root
                self._g1_robot = g1_robot
                self._g1_label = g1_label
                self._tianyi_label = tianyi_label
                self._visual_actuated_joint_names = visual_actuated_joint_names
                self._g1_visual_joint_names = g1_visual_joint_names
                snapshot = self.simulation.snapshot()
                self._apply_snapshot(snapshot)
                self._install_client_defaults(server)
                self._stop_event.clear()
                self._sync_thread = threading.Thread(
                    target=self._sync_loop,
                    args=(snapshot.revision,),
                    name="tianyi-viser-sync",
                    daemon=True,
                )
                self._sync_thread.start()
                LOGGER.info("Viser Tianyi scene ready at http://%s:%d", self.host, self.port)
            except Exception:
                self._server = None
                self._robot_root = None
                self._robot = None
                self._g1_root = None
                self._g1_robot = None
                self._g1_label = None
                self._tianyi_label = None
                server.stop()
                raise

    def stop(self) -> None:
        """Stop synchronization and release the Viser server."""
        with self._lifecycle_lock:
            server = self._server
            thread = self._sync_thread
            if server is None:
                return
            self._stop_event.set()
            with self._camera_transition_lock:
                self._camera_transition_generation += 1
                camera_threads = tuple(self._camera_transition_threads)
            if thread is not None:
                thread.join(timeout=max(1.0, 2.0 / self.update_hz))
            for camera_thread in camera_threads:
                camera_thread.join(timeout=max(0.1, 2.0 / self.camera_transition_hz))
            server.stop()
            self._server = None
            self._robot_root = None
            self._robot = None
            self._g1_root = None
            self._g1_robot = None
            self._g1_label = None
            self._tianyi_label = None
            self._visual_actuated_joint_names = ()
            self._g1_visual_joint_names = ()
            self._g1_source_values = None
            self._retarget_workspace_active = False
            self._comparison_visible = False
            self._sync_thread = None
            with self._camera_transition_lock:
                self._camera_transition_threads.clear()
            self._synced_revision = -1

    def set_camera_view(self, camera_view: RobotCameraView) -> int:
        """Apply a robot-relative preset to every connected Viser client."""
        if self._server is None:
            raise RuntimeError("Viser runtime has not been started")
        self._camera_view = camera_view
        preset = self._camera_presets()[camera_view]
        clients = tuple(self._server.get_clients().values())
        with self._camera_transition_lock:
            self._camera_transition_generation += 1
            generation = self._camera_transition_generation
        for client in clients:
            self._start_camera_transition(
                client=client,
                preset=preset,
                generation=generation,
            )
        return len(clients)

    def set_viewer_mode(self, mode: ViewerMode) -> bool:
        """Show comparison only while its workspace is active and has a source."""
        with self._scene_state_lock:
            self._retarget_workspace_active = mode is ViewerMode.RETARGET_COMPARISON
            comparison_visible = (
                self._retarget_workspace_active and self._g1_source_values is not None
            )
            changed = comparison_visible != self._comparison_visible
            self._apply_layout_locked(comparison_visible)
        if changed:
            self.set_camera_view(self._camera_view)
        return comparison_visible

    def update_retarget_preview(self, source_joint_values: Mapping[str, float]) -> None:
        """Update the read-only G1 reference pose without recreating its meshes."""
        if self._server is None or self._g1_robot is None:
            raise RuntimeError("Viser runtime has not been started")
        unknown = sorted(set(source_joint_values) - set(self._g1_visual_joint_names))
        if unknown:
            raise ValueError(f"G1 preview contains unknown joints: {unknown}")
        normalized = {name: float(value) for name, value in source_joint_values.items()}
        if not all(math.isfinite(value) for value in normalized.values()):
            raise ValueError("G1 preview joint values must be finite")
        with self._scene_state_lock:
            self._g1_source_values = normalized
            comparison_visible = self._retarget_workspace_active
            changed = comparison_visible != self._comparison_visible
            with self._server.atomic():
                self._g1_robot.update_cfg(self._g1_visual_configuration(normalized))
            self._apply_layout_locked(comparison_visible)
        if changed:
            self.set_camera_view(self._camera_view)

    def _camera_presets(self) -> dict[RobotCameraView, CameraPreset]:
        return (
            self.COMPARISON_CAMERA_PRESETS
            if self._comparison_visible
            else self.CAMERA_PRESETS
        )

    def _apply_layout_locked(self, comparison_visible: bool) -> None:
        if (
            self._server is None
            or self._robot_root is None
            or self._g1_root is None
            or self._g1_robot is None
            or self._g1_label is None
            or self._tianyi_label is None
        ):
            self._comparison_visible = comparison_visible
            return
        snapshot = self.simulation.snapshot()
        with self._server.atomic():
            self._robot_root.position = self._tianyi_root_position(
                snapshot,
                comparison_visible=comparison_visible,
            )
            self._g1_root.visible = comparison_visible
            self._g1_robot.show_visual = comparison_visible
            self._g1_label.visible = comparison_visible
            self._tianyi_label.visible = comparison_visible
        self._comparison_visible = comparison_visible

    def _configure_scene(self, server: viser.ViserServer) -> None:
        server.gui.configure_theme(
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            brand_color=(71, 85, 105),
        )
        server.gui.main_panel.minimize()
        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = False
        server.scene.add_grid(
            "/scene/ground",
            width=8.0,
            height=8.0,
            cell_size=0.25,
            section_size=1.0,
            cell_color=(65, 70, 78),
            section_color=(100, 108, 120),
            plane_color=(16, 19, 24),
            plane_opacity=0.85,
            shadow_opacity=0.35,
        )
        front = self.CAMERA_PRESETS[RobotCameraView.FRONT]
        server.initial_camera.position = front.position
        server.initial_camera.look_at = front.look_at
        server.initial_camera.up = front.up
        server.initial_camera.fov = front.vertical_fov
        server.initial_camera.near = 0.01
        server.initial_camera.far = 100.0

    def _install_client_defaults(self, server: viser.ViserServer) -> None:
        @server.on_client_connect
        def initialize_client(client: viser.ClientHandle) -> None:
            self._apply_camera(
                client=client,
                preset=self._camera_presets()[self._camera_view],
            )

    def _apply_camera(
        self,
        *,
        client: viser.ClientHandle,
        preset: CameraPreset,
    ) -> None:
        with client.atomic():
            client.camera.position = preset.position
            client.camera.look_at = preset.look_at
            client.camera.up_direction = preset.up
            client.camera.fov = preset.vertical_fov

    def _start_camera_transition(
        self,
        *,
        client: viser.ClientHandle,
        preset: CameraPreset,
        generation: int,
    ) -> None:
        if self.camera_transition_seconds == 0:
            self._apply_camera(client=client, preset=preset)
            return
        thread = threading.Thread(
            target=self._animate_camera,
            args=(client, preset, generation),
            name="tianyi-viser-camera",
            daemon=True,
        )
        with self._camera_transition_lock:
            self._camera_transition_threads.add(thread)
        thread.start()

    def _animate_camera(
        self,
        client: viser.ClientHandle,
        preset: CameraPreset,
        generation: int,
    ) -> None:
        current_thread = threading.current_thread()
        try:
            start_position = np.asarray(client.camera.position, dtype=float).copy()
            start_look_at = np.asarray(client.camera.look_at, dtype=float).copy()
            start_up = np.asarray(client.camera.up_direction, dtype=float).copy()
            started_at = time.monotonic()
            frame_seconds = 1.0 / self.camera_transition_hz
            while True:
                if self._camera_transition_cancelled(generation):
                    return
                elapsed = time.monotonic() - started_at
                progress = min(1.0, elapsed / self.camera_transition_seconds)
                position, look_at, up = self._interpolate_camera(
                    start_position=start_position,
                    start_look_at=start_look_at,
                    start_up=start_up,
                    preset=preset,
                    progress=progress,
                )
                self._apply_camera(
                    client=client,
                    preset=CameraPreset(
                        position=tuple(position),
                        look_at=tuple(look_at),
                        up=tuple(up),
                        vertical_fov=preset.vertical_fov,
                    ),
                )
                if progress >= 1.0 or self._stop_event.wait(frame_seconds):
                    return
        except (AssertionError, ValueError):
            LOGGER.debug("Could not animate a Viser camera", exc_info=True)
            if not self._camera_transition_cancelled(generation):
                self._apply_camera(client=client, preset=preset)
        finally:
            with self._camera_transition_lock:
                self._camera_transition_threads.discard(current_thread)

    def _camera_transition_cancelled(self, generation: int) -> bool:
        if self._stop_event.is_set():
            return True
        with self._camera_transition_lock:
            return generation != self._camera_transition_generation

    @staticmethod
    def _interpolate_camera(
        *,
        start_position: np.ndarray,
        start_look_at: np.ndarray,
        start_up: np.ndarray,
        preset: CameraPreset,
        progress: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate along an eased orbit that stays clear of the robot."""
        eased = progress * progress * (3.0 - 2.0 * progress)
        end_position = np.asarray(preset.position, dtype=float)
        end_look_at = np.asarray(preset.look_at, dtype=float)
        end_up = np.asarray(preset.up, dtype=float)
        look_at = start_look_at + eased * (end_look_at - start_look_at)

        start_offset = start_position - start_look_at
        end_offset = end_position - end_look_at
        start_radius = float(np.linalg.norm(start_offset[:2]))
        end_radius = float(np.linalg.norm(end_offset[:2]))
        if start_radius > 1e-6 and end_radius > 1e-6:
            start_angle = math.atan2(start_offset[1], start_offset[0])
            end_angle = math.atan2(end_offset[1], end_offset[0])
            angle_delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
            if math.isclose(angle_delta, -math.pi):
                angle_delta = math.pi
            angle = start_angle + eased * angle_delta
            radius = start_radius + eased * (end_radius - start_radius)
            height = start_offset[2] + eased * (end_offset[2] - start_offset[2])
            position = look_at + np.array(
                [radius * math.cos(angle), radius * math.sin(angle), height]
            )
        else:
            position = start_position + eased * (end_position - start_position)

        up = start_up + eased * (end_up - start_up)
        up_norm = float(np.linalg.norm(up))
        up = end_up if up_norm < 1e-6 else up / up_norm
        return position, look_at, up

    def _sync_loop(self, revision: int) -> None:
        timeout = 1.0 / self.update_hz
        while not self._stop_event.is_set():
            snapshot = self.simulation.wait_for_revision(
                after_revision=revision,
                timeout=timeout,
            )
            if self._stop_event.is_set():
                break
            if snapshot.revision == revision:
                continue
            self._apply_snapshot(snapshot)
            revision = snapshot.revision

    def _apply_snapshot(self, snapshot: SimulationSnapshot) -> None:
        if self._server is None or self._robot is None or self._robot_root is None:
            return
        with self._server.atomic():
            self._robot_root.position = self._tianyi_root_position(
                snapshot,
                comparison_visible=self._comparison_visible,
            )
            self._robot_root.wxyz = snapshot.base_wxyz
            self._robot.update_cfg(self._visual_configuration(snapshot))
        self._synced_revision = snapshot.revision

    @classmethod
    def _tianyi_root_position(
        cls,
        snapshot: SimulationSnapshot,
        *,
        comparison_visible: bool,
    ) -> tuple[float, float, float]:
        offset = cls._TIANYI_COMPARISON_OFFSET if comparison_visible else (0.0, 0.0, 0.0)
        return tuple(
            float(position + delta)
            for position, delta in zip(snapshot.base_position, offset, strict=True)
        )

    def _g1_visual_configuration(self, values: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [values.get(joint_name, 0.0) for joint_name in self._g1_visual_joint_names],
            dtype=float,
        )

    def _visual_configuration(self, snapshot: SimulationSnapshot) -> np.ndarray:
        """Map MuJoCo body state into Viser while holding Inspire hands neutral."""
        body_positions = snapshot.joint_position_map()
        visual_joint_names = self._visual_actuated_joint_names or snapshot.joint_names
        return np.asarray(
            [body_positions.get(joint_name, 0.0) for joint_name in visual_joint_names],
            dtype=float,
        )
