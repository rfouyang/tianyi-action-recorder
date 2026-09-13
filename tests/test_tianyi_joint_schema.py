"""Tests for the Tianyi model and saved-pose joint schema."""

from __future__ import annotations

import math
import unittest

from component.pose import JointGroup, PoseType, TianyiJointSchema
from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper


class TianyiJointSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.schema = TianyiJointSchema(
            asset_helper=TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
        )

    def test_model_indices_groups_and_authored_boundary(self) -> None:
        self.assertEqual(len(self.schema.definitions), 21)
        expected = {
            "first_leg_pitch_joint": (0, JointGroup.LEG_MECHANISM, False),
            "waist_yaw_joint": (3, JointGroup.WAIST, False),
            "head_yaw_joint": (4, JointGroup.HEAD, True),
            "head_roll_joint": (6, JointGroup.HEAD, True),
            "shoulder_pitch_l_joint": (7, JointGroup.LEFT_ARM, True),
            "wrist_roll_l_joint": (13, JointGroup.LEFT_ARM, True),
            "shoulder_pitch_r_joint": (14, JointGroup.RIGHT_ARM, True),
            "wrist_roll_r_joint": (20, JointGroup.RIGHT_ARM, True),
        }
        for joint_name, (model_index, group, authored) in expected.items():
            with self.subTest(joint_name=joint_name):
                definition = self.schema.definition(joint_name)
                self.assertEqual(definition.model_index, model_index)
                self.assertEqual(definition.group, group)
                self.assertEqual(definition.authored, authored)

    def test_pose_joint_sets_follow_the_approved_contract(self) -> None:
        self.assertEqual(
            self.schema.joint_names(PoseType.BASE),
            self.schema.HEAD_JOINT_NAMES
            + self.schema.LEFT_ARM_JOINT_NAMES
            + self.schema.RIGHT_ARM_JOINT_NAMES,
        )
        self.assertEqual(len(self.schema.joint_names(PoseType.BASE)), 17)
        self.assertEqual(len(self.schema.joint_names(PoseType.LEFT_ARM)), 7)
        self.assertEqual(len(self.schema.joint_names(PoseType.RIGHT_ARM)), 7)
        self.assertEqual(
            self.schema.joint_names(PoseType.COMPOSED),
            self.schema.joint_names(PoseType.BASE),
        )
        self.assertEqual(
            self.schema.LOCKED_JOINT_NAMES,
            (
                "first_leg_pitch_joint",
                "second_leg_pitch_joint",
                "waist_pitch_joint",
                "waist_yaw_joint",
            ),
        )
        self.assertTrue(
            set(self.schema.BASE_JOINT_NAMES).isdisjoint(self.schema.LOCKED_JOINT_NAMES)
        )

    def test_extract_from_model_positions_uses_pinned_model_order(self) -> None:
        model_positions = [0.0] * 21
        model_positions[4] = 0.25
        model_positions[7] = -0.3
        model_positions[13] = 0.4
        values = self.schema.extract_from_model_positions(
            model_positions=model_positions,
            pose_type=PoseType.BASE,
        )

        self.assertEqual(values["head_yaw_joint"], 0.25)
        self.assertEqual(values["shoulder_pitch_l_joint"], -0.3)
        self.assertEqual(values["wrist_roll_l_joint"], 0.4)
        self.assertNotIn("waist_yaw_joint", values)

    def test_extract_rejects_state_with_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 21"):
            self.schema.extract_from_model_positions(
                model_positions=[0.0] * 20,
                pose_type=PoseType.BASE,
            )

    def test_validation_rejects_inexact_or_invalid_values(self) -> None:
        missing_joint = self.schema.neutral_values(PoseType.LEFT_ARM)
        missing_joint.pop("elbow_pitch_l_joint")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.schema.validate_joint_values(
                pose_type=PoseType.LEFT_ARM,
                joint_values=missing_joint,
            )

        unknown_joint = self.schema.neutral_values(PoseType.LEFT_ARM)
        unknown_joint["waist_yaw_joint"] = 0.0
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.schema.validate_joint_values(
                pose_type=PoseType.LEFT_ARM,
                joint_values=unknown_joint,
            )

        boolean_value = self.schema.neutral_values(PoseType.LEFT_ARM)
        boolean_value["elbow_pitch_l_joint"] = True
        with self.assertRaisesRegex(ValueError, "real number"):
            self.schema.validate_joint_values(
                pose_type=PoseType.LEFT_ARM,
                joint_values=boolean_value,
            )

        non_finite = self.schema.neutral_values(PoseType.LEFT_ARM)
        non_finite["elbow_pitch_l_joint"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.schema.validate_joint_values(
                pose_type=PoseType.LEFT_ARM,
                joint_values=non_finite,
            )

        out_of_range = self.schema.neutral_values(PoseType.LEFT_ARM)
        out_of_range["elbow_pitch_l_joint"] = 100.0
        with self.assertRaisesRegex(ValueError, "outside"):
            self.schema.validate_joint_values(
                pose_type=PoseType.LEFT_ARM,
                joint_values=out_of_range,
            )

    def test_neutral_values_keep_calibrated_locked_joints_separate(self) -> None:
        self.assertEqual(
            self.schema.locked_values(),
            {
                "first_leg_pitch_joint": 0.14,
                "second_leg_pitch_joint": -0.4,
                "waist_pitch_joint": 0.24,
                "waist_yaw_joint": 0.0,
            },
        )
        self.assertEqual(
            dict(self.schema.LOCKED_JOINT_MOTOR_IDS),
            {
                "first_leg_pitch_joint": 51,
                "second_leg_pitch_joint": 52,
                "waist_pitch_joint": 32,
                "waist_yaw_joint": 31,
            },
        )
        for pose_type in PoseType:
            with self.subTest(pose_type=pose_type):
                self.assertTrue(
                    set(self.schema.neutral_values(pose_type)).isdisjoint(
                        self.schema.LOCKED_JOINT_NAMES
                    )
                )

    def test_calibration_limits_match_urdf_for_confirmed_baseline(self) -> None:
        lower_limit, upper_limit = self.schema.calibration_limits(
            "first_leg_pitch_joint"
        )

        definition = self.schema.definition("first_leg_pitch_joint")
        self.assertEqual(lower_limit, definition.lower_limit)
        self.assertEqual(upper_limit, definition.upper_limit)
        self.assertEqual(
            self.schema.validate_calibration_joint_value(
                joint_name="first_leg_pitch_joint",
                value=0.14,
            ),
            0.14,
        )


def demo_test_tianyi_joint_schema() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TianyiJointSchemaTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_tianyi_joint_schema()


if __name__ == "__main__":
    main()
