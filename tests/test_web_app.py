"""API and browser-shell tests for the Tianyi pose workbench."""

from __future__ import annotations

import io
import unittest

import httpx
from PIL import Image

from app.application import TianyiApplication
from app.main import create_app


class TianyiWebAppTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = TianyiApplication.build()
        cls.web_app = create_app(application=cls.application)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.application.close()

    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.web_app),
            base_url="http://testserver",
        )
        await self.client.post("/api/v1/simulation/reset")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_home_is_a_daisyui_pose_workbench(self) -> None:
        response = await self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Tianyi pose workbench", response.text)
        self.assertIn("cdn.jsdelivr.net/npm/daisyui@5", response.text)
        self.assertIn('class="card card-border', response.text)
        self.assertIn(
            'range.className = "range range-sm"',
            (await self.client.get("/static/app.js")).text,
        )
        self.assertIn("Front-left", response.text)
        self.assertIn("Back", response.text)

    async def test_model_endpoint_exposes_pose_and_visual_contracts(self) -> None:
        response = await self.client.get("/api/v1/model")
        payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["pose_joint_counts"]["base"], 17)
        self.assertEqual(payload["pose_joint_counts"]["left_arm"], 7)
        self.assertEqual(len(payload["joint_groups"]["locked"]), 4)
        self.assertEqual(
            payload["locked_joint_positions"],
            self.application.joint_schema.locked_values(),
        )
        self.assertEqual(
            payload["cameras"],
            ["front", "front_left", "left", "right", "front_right", "back"],
        )
        self.assertGreater(payload["visualization"]["triangle_reduction_percent"], 70)

    async def test_joint_update_and_locked_joint_rejection(self) -> None:
        updated = await self.client.patch(
            "/api/v1/simulation/joints",
            json={"joint_positions": {"head_yaw_joint": 0.3}},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["joint_positions"]["head_yaw_joint"], 0.3)
        self.assertTrue(updated.json()["locked"])

        rejected = await self.client.patch(
            "/api/v1/simulation/joints",
            json={"joint_positions": {"waist_yaw_joint": 0.2}},
        )
        self.assertEqual(rejected.status_code, 422)
        self.assertIn("Locked Tianyi joints", rejected.json()["detail"])

    async def test_camera_endpoint_returns_png_and_revision_headers(self) -> None:
        response = await self.client.get(
            "/api/v1/render/front_left.png",
            params={"width": 320, "height": 320},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertIn("x-simulation-revision", response.headers)
        self.assertIn("x-render-milliseconds", response.headers)
        with Image.open(io.BytesIO(response.content)) as image:
            self.assertEqual(image.size, (320, 320))

    async def test_health_identifies_simulation_only_mode(self) -> None:
        response = await self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "mujoco_simulation_only")


def demo_test_tianyi_web_app() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TianyiWebAppTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_tianyi_web_app()


if __name__ == "__main__":
    main()
