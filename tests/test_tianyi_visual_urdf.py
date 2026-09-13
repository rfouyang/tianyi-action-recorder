"""Tests for the generated Viser-oriented Tianyi URDF."""

from __future__ import annotations

import unittest

from component.pose import TianyiJointSchema
from config.settings import AppSettings
from util.build_tianyi_visual_urdf import render_visual_urdf
from util.tianyi_visual_urdf_helper import TianyiVisualUrdfHelper


class TianyiVisualUrdfHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.helper = TianyiVisualUrdfHelper(asset_dir=settings.tianyi_asset_dir)

    def test_generated_urdf_includes_neutral_hands_and_exact_body_order(self) -> None:
        report = self.helper.validate()

        self.assertEqual(report.link_count, 51)
        self.assertEqual(report.joint_count, 50)
        self.assertEqual(report.body_joint_names, TianyiJointSchema.MODEL_JOINT_NAMES)
        self.assertEqual(report.actuated_joint_names, TianyiJointSchema.MODEL_JOINT_NAMES)
        self.assertEqual(len(report.hand_joint_names), 24)
        self.assertEqual(len(report.source_hand_actuated_joint_names), 12)
        finger_names = ("thumb", "index", "middle", "ring", "little")
        self.assertTrue(
            all(any(finger in name for finger in finger_names) for name in report.hand_joint_names)
        )
        self.assertTrue(set(report.hand_joint_names).isdisjoint(TianyiJointSchema.BASE_JOINT_NAMES))
        self.assertEqual(report.visual_mesh_count, 51)
        self.assertEqual(report.collision_element_count, 0)

    def test_checked_in_urdf_is_reproducible(self) -> None:
        generated = render_visual_urdf(asset_dir=self.helper.asset_dir)

        self.assertEqual(generated, self.helper.visual_urdf_path.read_bytes())


def demo_test_tianyi_visual_urdf() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TianyiVisualUrdfHelperTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_tianyi_visual_urdf()


if __name__ == "__main__":
    main()
