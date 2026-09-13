"""Tests for G1-to-Tianyi simulation pose retargeting."""

from __future__ import annotations

import json
import unittest

from app.application import TianyiApplication
from component.pose import PoseType
from config.settings import AppSettings
from util.g1_pose_adapter import G1PoseAdapter


class G1PoseAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = G1PoseAdapter(
            asset_dir=AppSettings().g1_asset_dir
        )

    def test_uploaded_baseline_parses_as_complete_two_arm_pose(self) -> None:
        path = self.adapter.retarget_baseline_path
        pose = self.adapter.parse_pose(
            content=path.read_text(encoding="utf-8"),
            file_name=path.name,
        )

        self.assertEqual(pose.name, "concierge_init")
        self.assertEqual(pose.pose_type, "base")
        self.assertEqual(len(pose.joint_values), 17)

    def test_load_validates_model_and_exact_joint_shape(self) -> None:
        path = self.adapter.retarget_baseline_path
        content = path.read_text(encoding="utf-8")
        pose = self.adapter.parse_pose(content=content, file_name=path.name)

        self.assertEqual(pose.pose_type, "base")
        self.assertEqual(len(pose.joint_values), 17)
        payload = json.loads(content)
        payload["pose_type"] = "calibration"
        with self.assertRaisesRegex(ValueError, "supports base, composed"):
            self.adapter.parse_pose(
                content=json.dumps(payload),
                file_name="calibration.json",
            )
        with self.assertRaisesRegex(ValueError, "filename is invalid"):
            self.adapter.parse_pose(content=content, file_name="../concierge_init.json")


class PoseRetargetServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = TianyiApplication.build()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.application.close()

    def test_custom_g1_home_is_retargeted_as_geometry_not_tianyi_home(self) -> None:
        upload = self._uploaded_pose()
        result = self.application.pose_retargeting.retarget(
            source_pose_json=upload,
            source_file_name="concierge_init.json",
        )

        self.assertTrue(result.converged)
        self.assertEqual(len(result.source_joint_values), 17)
        self.assertEqual(result.source_joint_values["left_elbow_joint"], 1.4)
        self.assertEqual(tuple(result.joint_values), self.application.joint_schema.BASE_JOINT_NAMES)
        expected = self.application.simulation.default_pose_values()
        self.assertTrue(
            any(
                abs(result.joint_values[joint_name] - expected[joint_name]) > 0.2
                for joint_name in self.application.joint_schema.LEFT_ARM_JOINT_NAMES
            )
        )
        self.assertTrue(
            all(report.upper_arm_direction_error_rad < 0.01 for report in result.arm_reports)
        )
        self.assertEqual(result.contact_count, 0)

    def test_expressive_pose_solves_both_arms_within_limits(self) -> None:
        upload = self._uploaded_pose(
            name="concierge_present_left",
            pose_type="composed",
            changes={
                "left_shoulder_pitch_joint": -0.2592,
                "left_shoulder_roll_joint": 0.3718,
                "left_shoulder_yaw_joint": 1.372,
                "left_elbow_joint": 0.4628,
                "left_wrist_roll_joint": -0.942222054,
            },
        )
        result = self.application.pose_retargeting.retarget(
            source_pose_json=upload,
            source_file_name="concierge_present_left.json",
        )

        self.assertTrue(result.converged)
        self.assertEqual({report.side for report in result.arm_reports}, {"left", "right"})
        self.assertTrue(all(report.hand_position_error_m < 0.04 for report in result.arm_reports))
        self.assertTrue(
            all(report.hand_orientation_error_rad < 0.09 for report in result.arm_reports)
        )
        self.assertTrue(
            all(report.elbow_position_error_m < 0.03 for report in result.arm_reports)
        )
        self.assertTrue(
            all(report.upper_arm_direction_error_rad < 0.03 for report in result.arm_reports)
        )
        self.assertTrue(
            all(report.forearm_direction_error_rad < 0.07 for report in result.arm_reports)
        )
        self.assertNotEqual(
            result.joint_values["shoulder_yaw_l_joint"],
            self.application.simulation.default_pose_values()["shoulder_yaw_l_joint"],
        )
        self.application.joint_schema.validate_joint_values(
            pose_type=PoseType.BASE,
            joint_values=result.joint_values,
        )

    def test_single_g1_arm_pose_keeps_other_tianyi_arm_at_home(self) -> None:
        full_payload = json.loads(
            self._uploaded_pose(
                name="left_wave",
                pose_type="base",
                changes={"left_shoulder_yaw_joint": 0.8},
            )
        )
        full_payload["pose_type"] = "left_arm"
        full_payload["joint_values"] = {
            name: full_payload["joint_values"][name]
            for name in (
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            )
        }

        result = self.application.pose_retargeting.retarget(
            source_pose_json=json.dumps(full_payload),
            source_file_name="left_wave.json",
        )

        default = self.application.simulation.default_pose_values()
        self.assertEqual(result.source_joint_values["left_shoulder_yaw_joint"], 0.8)
        self.assertEqual(result.source_joint_values["right_elbow_joint"], 0.0)
        self.assertEqual(len(result.arm_reports), 1)
        for joint_name in self.application.joint_schema.RIGHT_ARM_JOINT_NAMES:
            self.assertAlmostEqual(result.joint_values[joint_name], default[joint_name])

    def test_real_concierge_speak_pose_preserves_both_arm_directions(self) -> None:
        upload = self._uploaded_pose(
            name="concierge_speak_1",
            pose_type="composed",
            changes={
                "left_shoulder_pitch_joint": 0.17905409679450535,
                "left_shoulder_roll_joint": 0.2872737258249055,
                "left_shoulder_yaw_joint": -0.3463983192030868,
                "left_elbow_joint": -0.32874391056641183,
                "left_wrist_roll_joint": -0.469298287599577,
                "left_wrist_pitch_joint": 0.07712766006119287,
                "left_wrist_yaw_joint": -0.08413212028087041,
                "right_shoulder_pitch_joint": 0.09150377360879608,
                "right_shoulder_roll_joint": -0.31253895139264376,
                "right_shoulder_yaw_joint": 0.22807356351757355,
                "right_elbow_joint": -0.1579423687810253,
                "right_wrist_roll_joint": 0.3575773698686636,
                "right_wrist_pitch_joint": -0.037209658519417245,
                "right_wrist_yaw_joint": -0.18744567612301632,
            },
        )

        result = self.application.pose_retargeting.retarget(
            source_pose_json=upload,
            source_file_name="concierge_speak_1.json",
        )

        self.assertTrue(result.converged)
        self.assertTrue(
            all(report.upper_arm_direction_error_rad < 0.02 for report in result.arm_reports)
        )
        self.assertTrue(
            all(report.forearm_direction_error_rad < 0.07 for report in result.arm_reports)
        )
        self.assertTrue(
            all(report.hand_position_error_m < 0.01 for report in result.arm_reports)
        )

    @classmethod
    def _uploaded_pose(
        cls,
        *,
        name: str = "concierge_init",
        pose_type: str = "base",
        changes: dict[str, float] | None = None,
    ) -> str:
        path = cls.application.settings.g1_asset_dir / "retarget_baseline.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["name"] = name
        payload["pose_type"] = pose_type
        payload["joint_values"].update(changes or {})
        return json.dumps(payload)
