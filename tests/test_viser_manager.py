"""Tests for Tianyi Viser synchronization and robot-relative cameras."""

from __future__ import annotations

import math
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app.ui_tianyi_3d.viser_manager import RobotCameraView, ViewerMode, ViserManager
from component.simulation import SimulationSnapshot


class ViserManagerTest(unittest.TestCase):
    def test_presets_use_the_shared_robot_anatomical_frame(self) -> None:
        presets = ViserManager.CAMERA_PRESETS

        self.assertGreater(presets[RobotCameraView.FRONT].position[0], 0.0)
        self.assertLess(presets[RobotCameraView.BACK].position[0], 0.0)
        self.assertGreater(presets[RobotCameraView.LEFT].position[1], 0.0)
        self.assertLess(presets[RobotCameraView.RIGHT].position[1], 0.0)
        self.assertGreater(presets[RobotCameraView.FRONT_LEFT_45].position[1], 0.0)
        self.assertLess(presets[RobotCameraView.FRONT_RIGHT_45].position[1], 0.0)

    def test_camera_preset_updates_connected_client_atomically(self) -> None:
        manager = self._manager(camera_transition_seconds=0.0)
        camera = SimpleNamespace(
            position=None,
            look_at=None,
            up_direction=None,
            fov=None,
        )
        client = Mock()
        client.camera = camera
        client.atomic.return_value = nullcontext()
        server = Mock()
        server.get_clients.return_value = {1: client}
        manager._server = server

        count = manager.set_camera_view(RobotCameraView.FRONT_LEFT_45)

        preset = manager.CAMERA_PRESETS[RobotCameraView.FRONT_LEFT_45]
        self.assertEqual(count, 1)
        self.assertEqual(camera.position, preset.position)
        self.assertEqual(camera.look_at, preset.look_at)
        self.assertEqual(camera.up_direction, preset.up)
        self.assertEqual(camera.fov, preset.vertical_fov)
        client.atomic.assert_called_once_with()

    def test_snapshot_updates_root_and_joint_configuration_atomically(self) -> None:
        manager = self._manager()
        server = Mock()
        server.atomic.return_value = nullcontext()
        robot = Mock()
        root = SimpleNamespace(position=None, wxyz=None)
        manager._server = server
        manager._robot = robot
        manager._robot_root = root
        snapshot = SimulationSnapshot(
            revision=4,
            updated_at=1.0,
            robot_model_id="tianyi",
            joint_names=tuple(f"joint_{index}" for index in range(21)),
            joint_positions=tuple(index / 100 for index in range(21)),
            joint_position_targets=tuple(index / 100 for index in range(21)),
            base_position=(0.1, 0.2, 0.3),
            base_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        manager._visual_actuated_joint_names = snapshot.joint_names

        manager._apply_snapshot(snapshot)

        self.assertEqual(root.position, snapshot.base_position)
        self.assertEqual(root.wxyz, snapshot.base_wxyz)
        np.testing.assert_allclose(
            robot.update_cfg.call_args.args[0],
            snapshot.joint_positions,
        )
        self.assertEqual(manager.synced_revision, 4)
        server.atomic.assert_called_once_with()

    def test_retarget_workspace_places_both_models_side_by_side(self) -> None:
        manager = self._manager(camera_transition_seconds=0.0)
        snapshot = SimulationSnapshot(
            revision=1,
            updated_at=1.0,
            robot_model_id="tianyi",
            joint_names=(),
            joint_positions=(),
            joint_position_targets=(),
            base_position=(0.0, 0.0, 0.0),
            base_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        manager.simulation.snapshot.return_value = snapshot
        server = Mock()
        server.atomic.return_value = nullcontext()
        server.get_clients.return_value = {}
        manager._server = server
        manager._robot_root = SimpleNamespace(position=None)
        manager._g1_root = SimpleNamespace(visible=False)
        manager._g1_robot = SimpleNamespace(show_visual=False)
        manager._g1_label = SimpleNamespace(visible=False)
        manager._tianyi_label = SimpleNamespace(visible=False)
        manager._g1_source_values = {"left_elbow_joint": 1.2}

        visible = manager.set_viewer_mode(ViewerMode.RETARGET_COMPARISON)

        self.assertTrue(visible)
        self.assertEqual(
            manager._robot_root.position,
            manager._TIANYI_COMPARISON_OFFSET,
        )
        self.assertTrue(manager._g1_root.visible)
        self.assertTrue(manager._g1_robot.show_visual)
        self.assertTrue(manager._g1_label.visible)
        self.assertTrue(manager._tianyi_label.visible)

        manager.set_viewer_mode(ViewerMode.TIANYI)
        self.assertEqual(manager._robot_root.position, (0.0, 0.0, 0.0))
        self.assertFalse(manager._g1_root.visible)

    def test_comparison_camera_is_wider_than_single_robot_camera(self) -> None:
        single = ViserManager.CAMERA_PRESETS[RobotCameraView.FRONT]
        comparison = ViserManager.COMPARISON_CAMERA_PRESETS[RobotCameraView.FRONT]

        self.assertGreater(comparison.position[0], single.position[0])

    def test_camera_transition_uses_orbit_and_reaches_exact_preset(self) -> None:
        front = ViserManager.CAMERA_PRESETS[RobotCameraView.FRONT]
        left = ViserManager.CAMERA_PRESETS[RobotCameraView.LEFT]
        midpoint, midpoint_look_at, _ = ViserManager._interpolate_camera(
            start_position=np.asarray(front.position),
            start_look_at=np.asarray(front.look_at),
            start_up=np.asarray(front.up),
            preset=left,
            progress=0.5,
        )
        endpoint, endpoint_look_at, endpoint_up = ViserManager._interpolate_camera(
            start_position=np.asarray(front.position),
            start_look_at=np.asarray(front.look_at),
            start_up=np.asarray(front.up),
            preset=left,
            progress=1.0,
        )

        expected_radius = np.linalg.norm(
            (np.asarray(front.position) - np.asarray(front.look_at))[:2]
        )
        self.assertAlmostEqual(
            np.linalg.norm((midpoint - midpoint_look_at)[:2]),
            expected_radius,
            places=6,
        )
        np.testing.assert_allclose(endpoint, left.position, atol=1e-12)
        np.testing.assert_allclose(endpoint_look_at, left.look_at, atol=1e-12)
        np.testing.assert_allclose(endpoint_up, left.up, atol=1e-12)
        self.assertAlmostEqual(math.degrees(left.vertical_fov), 45.0)

    @staticmethod
    def _manager(*, camera_transition_seconds: float = 0.7) -> ViserManager:
        return ViserManager(
            simulation=Mock(),
            schema=Mock(),
            urdf_path=Path("robot.urdf"),
            host="127.0.0.1",
            port=7901,
            camera_transition_seconds=camera_transition_seconds,
        )


def demo_test_viser_manager() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ViserManagerTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_viser_manager()


if __name__ == "__main__":
    main()
