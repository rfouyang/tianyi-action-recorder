"""Tests for the fixed-base Tianyi MuJoCo state capability."""

from __future__ import annotations

import unittest
from dataclasses import replace

from component.pose import PoseDefinition, PoseSource, PoseType, TianyiJointSchema
from component.simulation import SimulationService
from config.settings import AppSettings
from util.pose_file_helper import PoseFileHelper
from util.tianyi_asset_helper import TianyiAssetHelper


class SimulationServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.assets = TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
        cls.schema = TianyiJointSchema(asset_helper=cls.assets)
        cls.default_pose = PoseDefinition.from_dict(
            schema=cls.schema,
            payload=PoseFileHelper().read_json(
                path=settings.pose_dir / "base" / "concierge_init.json"
            ),
        )

    def setUp(self) -> None:
        self.service = SimulationService(
            schema=self.schema,
            mjcf_path=self.assets.mjcf_path,
        )

    def test_initial_snapshot_has_neutral_authored_and_calibrated_locked_state(self) -> None:
        snapshot = self.service.snapshot()

        self.assertEqual(snapshot.revision, 0)
        self.assertEqual(snapshot.robot_model_id, self.schema.model_id)
        self.assertEqual(snapshot.joint_names, self.schema.MODEL_JOINT_NAMES)
        self.assertEqual(len(snapshot.joint_positions), 21)
        positions = snapshot.joint_position_map()
        self.assertTrue(
            all(positions[joint_name] == 0.0 for joint_name in self.schema.BASE_JOINT_NAMES)
        )
        self.assertEqual(
            {name: positions[name] for name in self.schema.LOCKED_JOINT_NAMES},
            self.schema.locked_values(),
        )
        self.assertEqual(snapshot.joint_positions, snapshot.joint_position_targets)
        self.assertEqual(snapshot.base_position, (0.0, 0.0, 0.0))
        self.assertEqual(snapshot.base_wxyz, (1.0, 0.0, 0.0, 0.0))

    def test_configured_base_pose_is_used_for_startup_and_default_reset(self) -> None:
        service = SimulationService(
            schema=self.schema,
            mjcf_path=self.assets.mjcf_path,
            default_pose=self.default_pose,
        )

        self.assertEqual(service.default_pose_name, "concierge_init")
        self.assertEqual(
            {
                name: service.snapshot().joint_position_map()[name]
                for name in self.schema.BASE_JOINT_NAMES
            },
            dict(self.default_pose.joint_values),
        )
        service.update_joint_positions({"head_yaw_joint": 0.5})
        reset = service.reset_to_default()
        self.assertEqual(
            {
                name: reset.joint_position_map()[name]
                for name in self.schema.BASE_JOINT_NAMES
            },
            dict(self.default_pose.joint_values),
        )

    def test_partial_authored_update_keeps_locked_state_at_calibration(self) -> None:
        snapshot = self.service.update_joint_positions(
            {"head_yaw_joint": 0.2, "elbow_pitch_l_joint": -0.8}
        )
        positions = snapshot.joint_position_map()
        targets = snapshot.joint_position_target_map()

        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(positions["head_yaw_joint"], 0.2)
        self.assertEqual(positions["elbow_pitch_l_joint"], -0.8)
        self.assertEqual(positions, targets)
        for joint_name in self.schema.LOCKED_JOINT_NAMES:
            with self.subTest(joint_name=joint_name):
                expected = self.schema.locked_values()[joint_name]
                self.assertEqual(positions[joint_name], expected)
                self.assertEqual(targets[joint_name], expected)

    def test_ui_calibration_survives_pose_updates_but_never_enters_capture(self) -> None:
        calibrated = self.service.update_calibration_joint_positions(
            {
                "waist_pitch_joint": 0.5,
                "second_leg_pitch_joint": 0.3,
            }
        )
        updated = self.service.update_joint_positions({"head_yaw_joint": 0.2})
        captured = self.service.capture_pose_values(PoseType.BASE)

        self.assertEqual(calibrated.joint_position_map()["waist_pitch_joint"], 0.5)
        self.assertEqual(updated.joint_position_map()["waist_pitch_joint"], 0.5)
        self.assertEqual(updated.joint_position_map()["second_leg_pitch_joint"], 0.3)
        self.assertEqual(
            self.service.calibration_joint_positions()["waist_pitch_joint"],
            0.5,
        )
        self.assertEqual(tuple(captured), self.schema.BASE_JOINT_NAMES)
        self.assertTrue(set(captured).isdisjoint(self.schema.LOCKED_JOINT_NAMES))

    def test_calibration_update_accepts_only_bounded_locked_joints(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Tianyi calibration joints"):
            self.service.update_calibration_joint_positions({"head_yaw_joint": 0.1})
        with self.assertRaisesRegex(ValueError, "outside"):
            self.service.update_calibration_joint_positions({"waist_pitch_joint": 100.0})
        with self.assertRaisesRegex(ValueError, "real number"):
            self.service.update_calibration_joint_positions({"waist_yaw_joint": True})

    def test_apply_partial_arm_pose_preserves_head_and_other_arm(self) -> None:
        base_values = self.schema.neutral_values(PoseType.BASE)
        base_values["head_pitch_joint"] = 0.2
        base_values["shoulder_yaw_r_joint"] = 0.4
        self.service.apply_pose_values(
            pose_type=PoseType.BASE,
            joint_values=base_values,
        )

        left_values = self.schema.neutral_values(PoseType.LEFT_ARM)
        left_values["shoulder_roll_l_joint"] = 0.5
        left_values["elbow_pitch_l_joint"] = -1.0
        snapshot = self.service.apply_pose_values(
            pose_type=PoseType.LEFT_ARM,
            joint_values=left_values,
        )
        positions = snapshot.joint_position_map()

        self.assertEqual(positions["head_pitch_joint"], 0.2)
        self.assertEqual(positions["shoulder_yaw_r_joint"], 0.4)
        self.assertEqual(positions["shoulder_roll_l_joint"], 0.5)
        self.assertEqual(positions["elbow_pitch_l_joint"], -1.0)

    def test_mirror_current_arm_captures_live_state_and_preserves_body_boundary(self) -> None:
        left_values = dict(
            zip(
                self.schema.LEFT_ARM_JOINT_NAMES,
                (0.4, 0.35, -0.3, -0.8, 0.25, -0.2, 0.3),
                strict=True,
            )
        )
        self.service.update_joint_positions(
            {"head_yaw_joint": 0.2, **left_values}
        )
        self.service.update_calibration_joint_positions({"waist_yaw_joint": 0.1})

        target_type, captured, mirrored, snapshot = self.service.mirror_current_arm(
            source_pose_type=PoseType.LEFT_ARM
        )

        self.assertIs(target_type, PoseType.RIGHT_ARM)
        self.assertEqual(captured, left_values)
        self.assertEqual(mirrored["shoulder_pitch_r_joint"], 0.4)
        self.assertEqual(mirrored["shoulder_roll_r_joint"], -0.35)
        self.assertEqual(mirrored["elbow_yaw_r_joint"], -0.25)
        positions = snapshot.joint_position_map()
        self.assertEqual(positions["head_yaw_joint"], 0.2)
        self.assertEqual(positions["waist_yaw_joint"], 0.1)

    def test_capture_returns_only_the_requested_pose_shape(self) -> None:
        self.service.update_joint_positions(
            {
                "head_yaw_joint": 0.1,
                "shoulder_pitch_l_joint": 0.2,
                "shoulder_pitch_r_joint": -0.2,
            }
        )

        base = self.service.capture_pose_values(PoseType.BASE)
        left = self.service.capture_pose_values(PoseType.LEFT_ARM)
        right = self.service.capture_pose_values(PoseType.RIGHT_ARM)
        composed = self.service.capture_pose_values(PoseType.COMPOSED)

        self.assertEqual(tuple(base), self.schema.BASE_JOINT_NAMES)
        self.assertEqual(tuple(left), self.schema.LEFT_ARM_JOINT_NAMES)
        self.assertEqual(tuple(right), self.schema.RIGHT_ARM_JOINT_NAMES)
        self.assertEqual(composed, base)
        self.assertNotIn("head_yaw_joint", left)
        self.assertNotIn("head_yaw_joint", right)

    def test_typed_pose_application_checks_model_identity(self) -> None:
        values = self.schema.neutral_values(PoseType.COMPOSED)
        values["head_roll_joint"] = 0.1
        pose = PoseDefinition.create(
            schema=self.schema,
            name="composed_test",
            pose_type=PoseType.COMPOSED,
            joint_values=values,
            source=PoseSource.COMPOSITION,
            source_parts={"base": "neutral_base"},
        )

        snapshot = self.service.apply_pose(pose)
        self.assertEqual(snapshot.joint_position_map()["head_roll_joint"], 0.1)

        wrong_model_pose = replace(pose, robot_model_id="different_robot")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.service.apply_pose(wrong_model_pose)

    def test_locked_unknown_empty_and_invalid_updates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            self.service.update_joint_positions({})
        with self.assertRaisesRegex(ValueError, "Locked Tianyi joints"):
            self.service.update_joint_positions({"waist_yaw_joint": 0.1})
        with self.assertRaisesRegex(ValueError, "Unknown Tianyi authored joints"):
            self.service.update_joint_positions({"not_a_joint": 0.0})
        with self.assertRaisesRegex(ValueError, "outside"):
            self.service.update_joint_positions({"head_pitch_joint": 100.0})

    def test_reset_restores_all_positions_and_targets(self) -> None:
        changed = self.service.update_joint_positions(
            {"head_yaw_joint": 0.2, "elbow_pitch_r_joint": -0.9}
        )
        reset = self.service.reset_to_neutral()

        self.assertGreater(reset.revision, changed.revision)
        positions = reset.joint_position_map()
        self.assertTrue(
            all(positions[joint_name] == 0.0 for joint_name in self.schema.BASE_JOINT_NAMES)
        )
        self.assertEqual(
            {name: positions[name] for name in self.schema.LOCKED_JOINT_NAMES},
            self.schema.locked_values(),
        )
        self.assertEqual(reset.joint_positions, reset.joint_position_targets)
        self.assertEqual(self.service.calibration_joint_positions(), self.schema.locked_values())


def demo_test_simulation_service() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SimulationServiceTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_simulation_service()


if __name__ == "__main__":
    main()
