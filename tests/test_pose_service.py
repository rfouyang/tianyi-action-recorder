"""Tests for simulation pose creation and atomic JSON persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from component.pose import PoseService, PoseType, TianyiJointSchema
from config.settings import AppSettings
from util.pose_file_helper import PoseFileHelper
from util.tianyi_asset_helper import TianyiAssetHelper


class PoseServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.schema = TianyiJointSchema(
            asset_helper=TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
        )

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.pose_dir = Path(self.temporary_directory.name)
        self.service = PoseService(
            schema=self.schema,
            pose_dir=self.pose_dir,
            file_helper=PoseFileHelper(),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_save_load_and_list_preserve_exact_arm_shape(self) -> None:
        values = self.schema.neutral_values(PoseType.LEFT_ARM)
        values["elbow_pitch_l_joint"] = -0.6
        pose = self.service.create_simulated_pose(
            name="左手问候",
            pose_type=PoseType.LEFT_ARM,
            joint_values=values,
            notes="MuJoCo only",
        )

        path = self.service.save_pose(pose=pose)
        restored = self.service.load_pose(
            pose_type=PoseType.LEFT_ARM,
            name="左手问候",
        )

        self.assertEqual(path, self.pose_dir / "left_arm" / "左手问候.json")
        self.assertEqual(tuple(restored.joint_values), self.schema.LEFT_ARM_JOINT_NAMES)
        self.assertEqual(restored.joint_values["elbow_pitch_l_joint"], -0.6)
        self.assertEqual(self.service.list_poses(pose_type=PoseType.LEFT_ARM), (restored,))

    def test_existing_pose_requires_explicit_overwrite(self) -> None:
        pose = self.service.create_simulated_pose(
            name="neutral",
            pose_type=PoseType.BASE,
            joint_values=self.schema.neutral_values(PoseType.BASE),
        )
        self.service.save_pose(pose=pose)

        with self.assertRaises(FileExistsError):
            self.service.save_pose(pose=pose)

        self.assertEqual(
            self.service.save_pose(pose=pose, overwrite=True),
            self.pose_dir / "base" / "neutral.json",
        )

    def test_composed_pose_cannot_be_recorded_directly_from_simulation(self) -> None:
        with self.assertRaisesRegex(ValueError, "pose composition"):
            self.service.create_simulated_pose(
                name="invalid",
                pose_type=PoseType.COMPOSED,
                joint_values=self.schema.neutral_values(PoseType.COMPOSED),
            )

    def test_composition_keeps_head_and_unselected_arm_from_base(self) -> None:
        base_values = self.schema.neutral_values(PoseType.BASE)
        base_values["head_yaw_joint"] = 0.2
        base_values["elbow_pitch_l_joint"] = -0.1
        base_values["elbow_pitch_r_joint"] = -0.2
        base = self.service.create_simulated_pose(
            name="attentive_base",
            pose_type=PoseType.BASE,
            joint_values=base_values,
        )
        left_values = self.schema.neutral_values(PoseType.LEFT_ARM)
        left_values["elbow_pitch_l_joint"] = -0.8
        left = self.service.create_simulated_pose(
            name="left_wave",
            pose_type=PoseType.LEFT_ARM,
            joint_values=left_values,
        )

        composed = self.service.compose_pose(
            name="attentive_wave",
            base=base,
            left_arm=left,
            notes="Base head and right arm, overridden robot-left arm",
        )

        self.assertEqual(tuple(composed.joint_values), self.schema.BASE_JOINT_NAMES)
        self.assertEqual(composed.joint_values["head_yaw_joint"], 0.2)
        self.assertEqual(composed.joint_values["elbow_pitch_l_joint"], -0.8)
        self.assertEqual(composed.joint_values["elbow_pitch_r_joint"], -0.2)
        self.assertEqual(
            dict(composed.source_parts),
            {"base": "attentive_base", "left_arm": "left_wave"},
        )

    def test_compose_from_saved_names_and_list_names(self) -> None:
        base = self.service.create_simulated_pose(
            name="neutral_base",
            pose_type=PoseType.BASE,
            joint_values=self.schema.neutral_values(PoseType.BASE),
        )
        right_values = self.schema.neutral_values(PoseType.RIGHT_ARM)
        right_values["elbow_pitch_r_joint"] = -0.7
        right = self.service.create_simulated_pose(
            name="right_wave",
            pose_type=PoseType.RIGHT_ARM,
            joint_values=right_values,
        )
        self.service.save_pose(pose=base)
        self.service.save_pose(pose=right)

        composed = self.service.compose_pose_from_names(
            name="right_greeting",
            base_pose_name="neutral_base",
            right_arm_pose_name="right_wave",
        )

        self.assertEqual(composed.joint_values["elbow_pitch_r_joint"], -0.7)
        self.assertEqual(
            self.service.list_pose_names(),
            {
                "base": ("neutral_base",),
                "left_arm": (),
                "right_arm": ("right_wave",),
                "composed": (),
            },
        )

    def test_composition_rejects_wrong_source_pose_type(self) -> None:
        left = self.service.create_simulated_pose(
            name="left_only",
            pose_type=PoseType.LEFT_ARM,
            joint_values=self.schema.neutral_values(PoseType.LEFT_ARM),
        )

        with self.assertRaisesRegex(ValueError, "expected base"):
            self.service.compose_pose(name="invalid", base=left)

    def test_left_right_arm_mirror_and_double_mirror(self) -> None:
        left_values = dict(
            zip(
                self.schema.LEFT_ARM_JOINT_NAMES,
                (0.4, 0.35, -0.3, -0.8, 0.25, -0.2, 0.3),
                strict=True,
            )
        )

        target_type, right_values = self.service.mirror_arm_values(
            source_pose_type=PoseType.LEFT_ARM,
            joint_values=left_values,
        )

        self.assertIs(target_type, PoseType.RIGHT_ARM)
        self.assertEqual(
            right_values,
            dict(
                zip(
                    self.schema.RIGHT_ARM_JOINT_NAMES,
                    (0.4, -0.35, 0.3, -0.8, -0.25, -0.2, -0.3),
                    strict=True,
                )
            ),
        )
        restored_type, restored_values = self.service.mirror_arm_values(
            source_pose_type=target_type,
            joint_values=right_values,
        )
        self.assertIs(restored_type, PoseType.LEFT_ARM)
        self.assertEqual(restored_values, left_values)

    def test_arm_mirror_rejects_partial_and_non_arm_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            self.service.mirror_arm_values(
                source_pose_type=PoseType.LEFT_ARM,
                joint_values={"shoulder_pitch_l_joint": 0.2},
            )
        with self.assertRaisesRegex(ValueError, "Only left_arm and right_arm"):
            self.service.mirror_arm_values(
                source_pose_type=PoseType.BASE,
                joint_values=self.schema.neutral_values(PoseType.BASE),
            )

    def test_arm_mirror_matches_mujoco_sagittal_reflection(self) -> None:
        left_values = dict(
            zip(
                self.schema.LEFT_ARM_JOINT_NAMES,
                (0.4, 0.35, -0.3, -0.8, 0.25, -0.2, 0.3),
                strict=True,
            )
        )
        _, right_values = self.service.mirror_arm_values(
            source_pose_type=PoseType.LEFT_ARM,
            joint_values=left_values,
        )
        model = mujoco.MjModel.from_xml_path(
            str(AppSettings().tianyi_asset_dir / "tianyi2_pos.xml")
        )
        data = mujoco.MjData(model)
        for joint_name, value in {**left_values, **right_values}.items():
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)

        reflection = np.diag((1.0, -1.0, 1.0))
        for stem in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "elbow_yaw",
            "wrist_pitch",
            "wrist_roll",
        ):
            left_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{stem}_l_link",
            )
            right_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{stem}_r_link",
            )
            np.testing.assert_allclose(
                data.xpos[right_id],
                reflection @ data.xpos[left_id],
                atol=1e-9,
            )
            np.testing.assert_allclose(
                data.xmat[right_id].reshape(3, 3),
                reflection @ data.xmat[left_id].reshape(3, 3) @ reflection,
                atol=1e-9,
            )
