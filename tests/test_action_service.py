"""Tests for action persistence, direct interpolation, and MuJoCo preview."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from app.application import TianyiApplication
from component.action import ActionPoseReference, ActionTransition
from component.pose import PoseSource, PoseType


class ActionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.application = TianyiApplication.build(
            pose_dir=root / "poses",
            action_definition_dir=root / "actions",
            action_trajectory_dir=root / "trajectories",
        )
        self._save_complete_pose("concierge_init", head_yaw=0.0)
        self._save_complete_pose("look_left", head_yaw=0.4)

    def tearDown(self) -> None:
        self.application.close()
        self.temporary_directory.cleanup()

    def test_compile_save_and_load_uses_only_seventeen_authored_joints(self) -> None:
        action = self.application.action_service.create_action(
            name="turn_head",
            frames=(
                ActionTransition(
                    ActionPoseReference(PoseType.BASE, "look_left"),
                    duration_seconds=0.2,
                    hold_seconds=0.2,
                ),
            ),
            return_duration_seconds=0.2,
        )
        trajectory = self.application.action_service.generate_trajectory(
            action=action,
            sample_frequency_hz=10.0,
        )
        self.application.action_service.save_action(action=action)
        self.application.action_service.save_trajectory(trajectory=trajectory)
        loaded = self.application.action_service.load_trajectory(name="turn_head")

        self.assertEqual(trajectory.sample_count, 7)
        self.assertAlmostEqual(trajectory.duration_seconds, 0.6)
        self.assertEqual(loaded.joint_names, self.application.joint_schema.BASE_JOINT_NAMES)
        self.assertEqual(loaded.joint_positions.shape, (7, 17))
        self.assertTrue(np.array_equal(loaded.joint_positions, trajectory.joint_positions))
        self.assertFalse(
            set(loaded.joint_names) & set(self.application.joint_schema.LOCKED_JOINT_NAMES)
        )

    def test_zero_hold_adds_no_stationary_samples(self) -> None:
        action = self.application.action_service.create_action(
            name="no_hold",
            frames=(
                ActionTransition(
                    ActionPoseReference(PoseType.BASE, "look_left"),
                    duration_seconds=0.2,
                    hold_seconds=0.0,
                ),
            ),
            return_duration_seconds=0.2,
        )
        trajectory = self.application.action_service.generate_trajectory(
            action=action,
            sample_frequency_hz=10.0,
        )

        self.assertEqual(trajectory.sample_count, 5)
        self.assertEqual(trajectory.keyframe_hold_seconds, (0.0, 0.0, 0.0))

    def test_playback_preserves_current_ui_calibration(self) -> None:
        action = self.application.action_service.create_action(
            name="calibration_safe",
            frames=(
                ActionTransition(ActionPoseReference(PoseType.BASE, "look_left"), 0.03),
            ),
            return_duration_seconds=0.03,
        )
        trajectory = self.application.action_service.generate_trajectory(
            action=action,
            sample_frequency_hz=50.0,
        )
        calibration = {
            "first_leg_pitch_joint": 0.1,
            "second_leg_pitch_joint": -0.3,
            "waist_pitch_joint": 0.2,
            "waist_yaw_joint": 0.1,
        }
        self.application.simulation.update_calibration_joint_positions(calibration)
        self.application.action_playback.load(trajectory)
        self.application.action_playback.play()
        time.sleep(0.1)

        state = self.application.simulation.snapshot().joint_position_map()
        self.assertEqual(
            {name: state[name] for name in calibration},
            calibration,
        )

    def _save_complete_pose(self, name: str, *, head_yaw: float) -> None:
        values = self.application.joint_schema.neutral_values(PoseType.BASE)
        values["head_yaw_joint"] = head_yaw
        pose = self.application.pose_service.create_simulated_pose(
            name=name,
            pose_type=PoseType.BASE,
            joint_values=values,
        )
        self.assertIs(pose.source, PoseSource.SIMULATION)
        self.application.pose_service.save_pose(pose=pose, overwrite=True)


if __name__ == "__main__":
    unittest.main()
