"""Tests for the locally vendored G1 retargeting model."""

from __future__ import annotations

import unittest

import mujoco

from config.settings import AppSettings
from util.g1_asset_helper import G1AssetHelper


class G1AssetHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = AppSettings()
        cls.assets = G1AssetHelper(asset_dir=cls.settings.g1_asset_dir)

    def test_pinned_assets_are_local_and_match_reviewed_hashes(self) -> None:
        self.assets.validate()

        self.assertTrue(self.assets.mjcf_path.is_relative_to(self.settings.project_root))
        self.assertTrue(self.assets.urdf_path.is_relative_to(self.settings.project_root))
        self.assertTrue(self.assets.retarget_baseline_path.is_file())

    def test_vendored_mjcf_loads_required_arm_contract(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(self.assets.mjcf_path))

        self.assertEqual(model.njnt, 30)
        for joint_name in (
            "left_shoulder_pitch_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_elbow_joint",
        ):
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name),
                0,
            )
