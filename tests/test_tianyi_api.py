"""Tests for the independent Tianyi REST and WebSocket API surface."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import WebSocketDisconnect

from app.api_tianyi_3d.simulation.websocket import simulation_websocket
from app.application import TianyiApplication
from app.tianyi_3d_main import create_web_app
from component.pose import PoseType
from config.settings import AppSettings


def _g1_upload(*, name: str, pose_type: str, changes: dict[str, float]) -> dict[str, str]:
    path = AppSettings().g1_asset_dir / "retarget_baseline.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["name"] = name
    payload["pose_type"] = pose_type
    payload["joint_values"].update(changes)
    return {
        "source_file_name": f"{name}.json",
        "source_pose_json": json.dumps(payload),
    }


class FakeViserManager:
    """Small lifecycle fake that keeps API tests independent of a server port."""

    def __init__(self) -> None:
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

    def stop(self) -> None:
        self.stop_count += 1


class FakeWebSocket:
    """Drive one API WebSocket command without Starlette's TestClient."""

    def __init__(self, *, application: TianyiApplication, payload: object) -> None:
        self.app = SimpleNamespace(state=SimpleNamespace(tianyi=application))
        self.payload = payload
        self.accepted = False
        self.sent_messages: list[dict[str, object]] = []
        self.receive_count = 0

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message: dict[str, object]) -> None:
        self.sent_messages.append(message)

    async def receive_json(self) -> object:
        if self.receive_count == 0:
            self.receive_count += 1
            return self.payload
        raise WebSocketDisconnect()


class TianyiApiTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temporary_directory.name)
        cls.pose_dir = root / "poses"
        cls.application = TianyiApplication.build(
            pose_dir=cls.pose_dir,
            action_definition_dir=root / "actions",
            action_trajectory_dir=root / "trajectories",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.application.close()
        cls.temporary_directory.cleanup()

    async def asyncSetUp(self) -> None:
        self.application.simulation.reset_to_default()
        self.viser = FakeViserManager()
        self.web_app = create_web_app(
            robot_application=self.application,
            viser_manager=self.viser,
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.web_app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_lifespan_owns_viser_start_and_stop(self) -> None:
        async with self.web_app.router.lifespan_context(self.web_app):
            self.assertEqual(self.viser.start_count, 1)
            self.assertIs(self.web_app.state.tianyi, self.application)
            self.assertIs(self.web_app.state.viser, self.viser)

        self.assertEqual(self.viser.stop_count, 1)

    async def test_health_and_state_expose_simulation_contract(self) -> None:
        health = await self.client.get("/api/tianyi/system/health")
        state = await self.client.get("/api/tianyi/simulation/state")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["runtime"], "simulation")
        self.assertEqual(health.json()["model_id"], self.application.joint_schema.model_id)
        self.assertEqual(state.status_code, 200)
        self.assertEqual(len(state.json()["joint_positions"]), 21)
        self.assertEqual(len(state.json()["locked_joint_names"]), 4)
        self.assertEqual(self.application.simulation.default_pose_name, "concierge_init")
        expected_default = self.application.simulation.default_pose_values()
        self.assertEqual(
            {name: state.json()["joint_positions"][name] for name in expected_default},
            expected_default,
        )
        self.assertEqual(
            state.json()["locked_joint_positions"],
            self.application.joint_schema.locked_values(),
        )
        self.assertTrue(state.json()["locked"])

    async def test_rest_updates_authored_joint_and_rejects_locked_joint(self) -> None:
        updated = await self.client.put(
            "/api/tianyi/simulation/joints",
            json={"joint_positions": {"head_yaw_joint": 0.25}},
        )
        rejected = await self.client.put(
            "/api/tianyi/simulation/joints",
            json={"joint_positions": {"waist_yaw_joint": 0.2}},
        )

        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["joint_positions"]["head_yaw_joint"], 0.25)
        self.assertTrue(updated.json()["locked"])
        self.assertEqual(rejected.status_code, 422)
        self.assertIn("Locked Tianyi joints", rejected.json()["detail"])

    async def test_arm_mirror_api_reflects_only_the_destination_arm(self) -> None:
        self.application.simulation.update_joint_positions({"head_yaw_joint": 0.2})
        self.application.simulation.update_calibration_joint_positions(
            {"waist_yaw_joint": 0.1}
        )
        left_values = dict(
            zip(
                self.application.joint_schema.LEFT_ARM_JOINT_NAMES,
                (0.4, 0.35, -0.3, -0.8, 0.25, -0.2, 0.3),
                strict=True,
            )
        )
        self.application.simulation.update_joint_positions(left_values)

        response = await self.client.post(
            "/api/tianyi/poses/arms/mirror",
            json={"source_pose_type": "left_arm"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target_pose_type"], "right_arm")
        self.assertEqual(response.json()["source_joint_positions"], left_values)
        self.assertEqual(response.json()["joint_positions"]["shoulder_roll_r_joint"], -0.35)
        self.assertEqual(response.json()["joint_positions"]["elbow_pitch_r_joint"], -0.8)
        state = self.application.simulation.snapshot().joint_position_map()
        self.assertEqual(state["head_yaw_joint"], 0.2)
        self.assertEqual(state["waist_yaw_joint"], 0.1)
        self.assertEqual(state["shoulder_roll_l_joint"], 0.35)
        self.assertEqual(state["shoulder_roll_r_joint"], -0.35)

    async def test_websocket_streams_state_and_accepts_authored_joint_target(self) -> None:
        websocket = FakeWebSocket(
            application=self.application,
            payload={"joint_positions": {"head_pitch_joint": 0.1}},
        )

        await simulation_websocket(websocket)

        self.assertTrue(websocket.accepted)
        self.assertEqual(len(websocket.sent_messages), 2)
        initial, updated = websocket.sent_messages
        self.assertEqual(initial["type"], "simulation_state")
        self.assertEqual(updated["type"], "simulation_state")
        self.assertGreater(updated["revision"], initial["revision"])
        self.assertEqual(updated["joint_positions"]["head_pitch_joint"], 0.1)
        self.assertTrue(updated["locked"])

    async def test_pose_composition_api_previews_and_saves_exact_authored_shape(self) -> None:
        self._save_composer_sources()
        catalog = await self.client.get("/api/tianyi/poses")
        preview = await self.client.post(
            "/api/tianyi/poses/compositions/preview",
            json={
                "base_pose": "api_attentive_base",
                "left_arm_pose": "api_left_wave",
                "right_arm_pose": None,
            },
        )
        command = {
            "base_pose": "api_attentive_base",
            "left_arm_pose": "api_left_wave",
            "right_arm_pose": None,
            "name": "api_composed_wave",
            "notes": "API composition",
            "overwrite": False,
        }
        saved = await self.client.post("/api/tianyi/poses/compositions", json=command)
        duplicate = await self.client.post("/api/tianyi/poses/compositions", json=command)

        self.assertEqual(catalog.status_code, 200)
        self.assertIn("api_attentive_base", catalog.json()["base"])
        self.assertIn("api_left_wave", catalog.json()["left_arm"])
        self.assertEqual(preview.status_code, 200)
        self.assertFalse(preview.json()["saved"])
        self.assertEqual(
            preview.json()["source_parts"],
            {"base": "api_attentive_base", "left_arm": "api_left_wave"},
        )
        self.assertEqual(saved.status_code, 200)
        self.assertTrue(saved.json()["saved"])
        self.assertEqual(duplicate.status_code, 409)
        pose = self.application.pose_service.load_pose(
            pose_type=PoseType.COMPOSED,
            name="api_composed_wave",
        )
        self.assertEqual(tuple(pose.joint_values), self.application.joint_schema.BASE_JOINT_NAMES)
        self.assertEqual(len(pose.joint_values), 17)
        self.assertFalse(any("finger" in name or "thumb" in name for name in pose.joint_values))
        snapshot = self.application.simulation.snapshot().joint_position_map()
        self.assertEqual(snapshot["head_yaw_joint"], 0.2)
        self.assertEqual(snapshot["elbow_pitch_l_joint"], -0.8)

    async def test_pose_retargeting_api_lists_previews_and_saves_g1_pose(self) -> None:
        upload = _g1_upload(
            name="concierge_present_right",
            pose_type="composed",
            changes={
                "right_shoulder_pitch_joint": -0.2592,
                "right_shoulder_roll_joint": -0.3718,
                "right_shoulder_yaw_joint": -1.372,
                "right_elbow_joint": 0.4628,
                "right_wrist_roll_joint": 0.942222054,
            },
        )
        preview = await self.client.post(
            "/api/tianyi/pose-retargeting/preview",
            json={
                **upload,
            },
        )
        command = {
            **upload,
            "output_pose_type": "right_arm",
            "name": "api_g1_present_right_arm",
            "notes": "API retarget test",
            "overwrite": True,
        }
        saved = await self.client.post("/api/tianyi/pose-retargeting", json=command)

        self.assertEqual(preview.status_code, 200)
        self.assertTrue(preview.json()["converged"])
        self.assertEqual(len(preview.json()["joint_positions"]), 17)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["saved_pose_name"], "api_g1_present_right_arm")
        self.assertEqual(saved.json()["saved_pose_type"], "right_arm")
        pose = self.application.pose_service.load_pose(
            pose_type=PoseType.RIGHT_ARM,
            name="api_g1_present_right_arm",
        )
        self.assertEqual(pose.source.value, "retargeting")
        self.assertIn("API retarget test", pose.notes)
        self.assertEqual(len(pose.joint_values), 7)

    async def test_saved_pose_preview_api_applies_pose_for_editing(self) -> None:
        self._save_composer_sources()

        response = await self.client.post(
            "/api/tianyi/poses/preview",
            json={"pose_type": "left_arm", "name": "api_left_wave"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["edit_pose_type"], "left_arm")
        self.assertEqual(len(response.json()["joint_positions"]), 7)
        self.assertEqual(
            self.application.simulation.snapshot().joint_position_map()[
                "elbow_pitch_l_joint"
            ],
            -0.8,
        )

    async def test_action_api_rejects_partial_arm_and_compiles_complete_poses(self) -> None:
        self._save_composer_sources()
        command = {
            "name": "api_wave_action",
            "frames": [
                {
                    "pose_type": "base",
                    "name": "api_attentive_base",
                    "duration_seconds": 0.1,
                    "hold_seconds": 0,
                }
            ],
            "return_duration_seconds": 0.1,
            "notes": "API simulation action",
            "overwrite": True,
        }
        saved = await self.client.post("/api/tianyi/actions/definitions", json=command)
        compiled = await self.client.post(
            "/api/tianyi/actions/trajectories",
            json={**command, "sample_frequency_hz": 10},
        )
        rejected = await self.client.post(
            "/api/tianyi/actions/definitions",
            json={
                **command,
                "name": "invalid_partial_action",
                "frames": [
                    {
                        "pose_type": "left_arm",
                        "name": "api_left_wave",
                        "duration_seconds": 1,
                        "hold_seconds": 0,
                    }
                ],
            },
        )

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["home_pose"], "concierge_init")
        self.assertEqual(compiled.status_code, 200)
        self.assertEqual(len(compiled.json()["joint_names"]), 17)
        self.assertEqual(rejected.status_code, 422)
        self.assertIn("base or composed", rejected.json()["detail"])
        rejected_home_override = await self.client.post(
            "/api/tianyi/actions/definitions",
            json={**command, "name": "invalid_home_override", "home_pose": "other"},
        )
        self.assertEqual(rejected_home_override.status_code, 422)
        self.assertIn("Extra inputs are not permitted", rejected_home_override.text)

    def _save_composer_sources(self) -> None:
        base_values = self.application.joint_schema.neutral_values(PoseType.BASE)
        base_values["head_yaw_joint"] = 0.2
        base_values["elbow_pitch_l_joint"] = -0.1
        base = self.application.pose_service.create_simulated_pose(
            name="api_attentive_base",
            pose_type=PoseType.BASE,
            joint_values=base_values,
        )
        left_values = self.application.joint_schema.neutral_values(PoseType.LEFT_ARM)
        left_values["elbow_pitch_l_joint"] = -0.8
        left = self.application.pose_service.create_simulated_pose(
            name="api_left_wave",
            pose_type=PoseType.LEFT_ARM,
            joint_values=left_values,
        )
        self.application.pose_service.save_pose(pose=base, overwrite=True)
        self.application.pose_service.save_pose(pose=left, overwrite=True)


def demo_test_tianyi_api() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TianyiApiTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_tianyi_api()


if __name__ == "__main__":
    main()
