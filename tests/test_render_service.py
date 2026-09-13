"""Tests for optimized robot-relative MuJoCo rendering."""

from __future__ import annotations

import io
import unittest
from dataclasses import replace

import mujoco
import numpy as np
from PIL import Image

from component.pose import TianyiJointSchema
from component.simulation import SimulationService
from component.visualization import CameraView, TianyiRenderService
from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper


class TianyiRenderServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.assets = TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
        cls.schema = TianyiJointSchema(asset_helper=cls.assets)
        cls.simulation = SimulationService(
            schema=cls.schema,
            mjcf_path=cls.assets.mjcf_path,
        )
        cls.renderer = TianyiRenderService(
            schema=cls.schema,
            mjcf_path=settings.tianyi_visual_mjcf_path,
        )
        cls.visual_mjcf_path = settings.tianyi_visual_mjcf_path

    @classmethod
    def tearDownClass(cls) -> None:
        cls.renderer.close()

    def test_camera_azimuths_match_robot_relative_convention(self) -> None:
        expected_azimuths = {
            CameraView.FRONT: 180.0,
            CameraView.FRONT_LEFT: 225.0,
            CameraView.LEFT: 270.0,
            CameraView.RIGHT: 90.0,
            CameraView.FRONT_RIGHT: 135.0,
            CameraView.BACK: 0.0,
        }
        for camera_view, expected_azimuth in expected_azimuths.items():
            with self.subTest(camera_view=camera_view):
                self.assertEqual(
                    self.renderer.create_camera(camera_view).azimuth,
                    expected_azimuth,
                )

    def test_camera_positions_are_on_the_named_robot_side(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(self.visual_mjcf_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera_positions = {}
        for camera_view in CameraView:
            scene = mujoco.MjvScene(model, maxgeom=10_000)
            mujoco.mjv_updateScene(
                model,
                data,
                mujoco.MjvOption(),
                mujoco.MjvPerturb(),
                self.renderer.create_camera(camera_view),
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            camera_positions[camera_view] = np.mean(
                (scene.camera[0].pos, scene.camera[1].pos),
                axis=0,
            )

        self.assertGreater(camera_positions[CameraView.FRONT][0], 0.0)
        self.assertLess(camera_positions[CameraView.BACK][0], 0.0)
        self.assertGreater(camera_positions[CameraView.LEFT][1], 0.0)
        self.assertLess(camera_positions[CameraView.RIGHT][1], 0.0)
        self.assertGreater(camera_positions[CameraView.FRONT_LEFT][1], 0.0)
        self.assertLess(camera_positions[CameraView.FRONT_RIGHT][1], 0.0)

    def test_png_render_uses_current_simulation_revision(self) -> None:
        snapshot = self.simulation.update_joint_positions(
            {"head_yaw_joint": 0.2, "elbow_pitch_l_joint": -0.8}
        )
        result = self.renderer.render_png(
            snapshot=snapshot,
            camera_view=CameraView.FRONT,
            width=720,
            height=720,
        )

        with Image.open(io.BytesIO(result.png_bytes)) as image:
            self.assertEqual(image.size, (720, 720))
            self.assertEqual(image.mode, "RGB")
        self.assertEqual(result.revision, snapshot.revision)
        self.assertGreater(len(result.png_bytes), 10_000)

    def test_invalid_size_and_snapshot_contract_are_rejected(self) -> None:
        snapshot = self.simulation.snapshot()
        with self.assertRaisesRegex(ValueError, "between 160 and 1280"):
            self.renderer.render_png(
                snapshot=snapshot,
                camera_view=CameraView.FRONT,
                width=100,
                height=320,
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.renderer.render_png(
                snapshot=replace(snapshot, robot_model_id="different_robot"),
                camera_view=CameraView.FRONT,
                width=320,
                height=320,
            )


def demo_test_tianyi_render_service() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TianyiRenderServiceTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_tianyi_render_service()


if __name__ == "__main__":
    main()
