"""Tests for the pinned Tianyi 2.0 body-only assets."""

from __future__ import annotations

import unittest

import mujoco

from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper


class TianyiAssetHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.helper = TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)

    def test_urdf_and_mjcf_contract(self) -> None:
        report = self.helper.validate()

        self.assertEqual(report.body_joint_count, 21)
        self.assertEqual(report.base_pose_joint_count, 17)
        self.assertEqual(report.actuator_count, 21)
        self.assertEqual(report.mesh_count, 25)

    def test_joint_specs_include_all_body_groups(self) -> None:
        self.helper.validate()
        specs = self.helper.load_joint_specs()

        self.assertEqual(len(specs), 21)
        self.assertEqual(sum(name.startswith("head_") for name in specs), 3)
        self.assertEqual(sum(name.startswith("waist_") for name in specs), 2)
        self.assertEqual(sum("_l_joint" in name for name in specs), 7)
        self.assertEqual(sum("_r_joint" in name for name in specs), 7)

    def test_position_model_can_run_forward_kinematics(self) -> None:
        self.helper.validate()
        model = mujoco.MjModel.from_xml_path(str(self.helper.mjcf_path))
        data = mujoco.MjData(model)

        mujoco.mj_forward(model, data)

        self.assertEqual(model.njnt, 21)
        self.assertEqual(model.nu, 21)
        self.assertTrue(all(value == value for value in data.qpos))


def demo_test_tianyi_assets() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TianyiAssetHelperTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_tianyi_assets()


if __name__ == "__main__":
    main()
