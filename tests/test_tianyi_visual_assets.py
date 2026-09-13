"""Tests for the derived render-only Tianyi asset set."""

from __future__ import annotations

import unittest

import mujoco

from config.settings import AppSettings
from util.tianyi_visual_asset_helper import TianyiVisualAssetHelper


class TianyiVisualAssetHelperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.helper = TianyiVisualAssetHelper(asset_dir=settings.tianyi_asset_dir)

    def test_generated_asset_hashes_and_model_contract(self) -> None:
        report = self.helper.validate()

        self.assertEqual(report.mesh_count, 25)
        self.assertEqual(report.geom_count, 26)
        self.assertGreater(report.source_triangle_count, 1_000_000)
        self.assertLess(report.optimized_triangle_count, 350_000)
        self.assertGreater(report.triangle_reduction_ratio, 0.7)

    def test_visual_model_removes_duplicate_collision_geometry(self) -> None:
        source_model = mujoco.MjModel.from_xml_path(str(self.helper.source_mjcf_path))
        visual_model = mujoco.MjModel.from_xml_path(str(self.helper.visual_mjcf_path))

        self.assertEqual(source_model.ngeom, 51)
        self.assertEqual(visual_model.ngeom, 26)
        self.assertEqual(visual_model.nq, source_model.nq)
        self.assertEqual(visual_model.nu, source_model.nu)
        self.assertEqual(visual_model.nlight, 2)


def demo_test_tianyi_visual_assets() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TianyiVisualAssetHelperTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_tianyi_visual_assets()


if __name__ == "__main__":
    main()
