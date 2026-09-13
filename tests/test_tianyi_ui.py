"""Tests for the independent Tianyi HTML Pose Recorder surface."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from app.application import TianyiApplication
from app.tianyi_3d_main import create_web_app
from app.ui_tianyi_3d.websocket import _apply_simulation_update
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
    def __init__(self) -> None:
        self.port = 7901
        self.connected_client_count = 1
        self.camera_transition_seconds = 0.0
        self.camera_views: list[object] = []
        self.viewer_modes: list[object] = []
        self.retarget_previews: list[dict[str, float]] = []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def set_camera_view(self, camera_view: object) -> int:
        self.camera_views.append(camera_view)
        return 1

    def set_viewer_mode(self, viewer_mode: object) -> bool:
        self.viewer_modes.append(viewer_mode)
        return (
            getattr(viewer_mode, "value", viewer_mode) == "retarget-comparison"
            and bool(self.retarget_previews)
        )

    def update_retarget_preview(self, source_joint_values: dict[str, float]) -> None:
        self.retarget_previews.append(dict(source_joint_values))


class TianyiUiTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_home_renders_wireframe_recorder_and_embedded_viser(self) -> None:
        response = await self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('data-theme="wireframe"', response.text)
        self.assertIn('data-theme="dark"', response.text)
        self.assertIn('data-viser-port="7901"', response.text)
        self.assertIn("Base · head + both arms (17 joints)", response.text)
        self.assertIn("motor IDs 31, 32, 51, and 52", response.text)
        self.assertEqual(response.text.count("data-joint-row"), 21)
        self.assertIn("Calibration only · motors 31, 32, 51, 52", response.text)
        self.assertIn("Never recorded", response.text)
        self.assertIn("motor 31", response.text)
        self.assertIn("motor 32", response.text)
        self.assertIn("motor 51", response.text)
        self.assertIn("motor 52", response.text)
        self.assertIn('data-panel-target="pose-composer"', response.text)
        self.assertIn('data-panel-target="pose-retargeting"', response.text)
        self.assertIn("Pose Retargeting", response.text)
        self.assertIn("Retarget and preview", response.text)
        self.assertIn("<th>Upper</th><th>Forearm</th><th>Wrist</th>", response.text)
        self.assertIn("G1 · Original pose", response.text)
        self.assertIn("Tianyi · Retargeted pose", response.text)
        self.assertIn('type="file"', response.text)
        self.assertIn('accept=".json,application/json"', response.text)
        self.assertIn("pose_retargeting.js", response.text)
        self.assertLess(
            response.text.index('data-panel-target="pose-composer"'),
            response.text.index('data-panel-target="pose-retargeting"'),
        )
        self.assertIn("Load saved pose", response.text)
        self.assertIn("Load for editing", response.text)
        self.assertIn("Robot-left arm override · optional", response.text)
        self.assertIn("Inspire hands", response.text)
        self.assertIn("pose_composer.js", response.text)
        self.assertIn('data-panel-target="action-composer"', response.text)
        self.assertIn("Action Composer", response.text)
        self.assertIn("Motion timeline", response.text)
        self.assertIn("Action Player", response.text)
        self.assertIn("action_composer.js", response.text)
        self.assertNotIn('id="action-home"', response.text)
        self.assertIn("Fixed start + finish pose", response.text)
        self.assertGreaterEqual(response.text.count("base/concierge_init"), 3)
        self.assertIn("Mirror arm edit", response.text)
        self.assertIn('data-mirror-arm="left_arm"', response.text)
        self.assertIn('data-mirror-arm="right_arm"', response.text)
        self.assertIn("Keep sign: shoulder pitch, elbow pitch, wrist pitch", response.text)
        self.assertIn("Wrist Yaw", response.text)
        self.assertIn('data-default-pose-name="concierge_init"', response.text)
        self.assertIn("Reset to concierge_init", response.text)
        self.assertIn('data-reset-value="-0.800060"', response.text)

        css = await self.client.get("/static/tianyi-3d/css/app.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn("--color-secondary:#2563eb", css.text)

    async def test_pose_retargeting_previews_both_and_saves_one_arm_pose(self) -> None:
        upload = _g1_upload(
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
        preview = await self.client.post(
            "/ui/tianyi-3d/pose-retargeting/preview",
            json={
                **upload,
                "output_pose_type": None,
                "name": "",
                "notes": "",
                "overwrite": False,
            },
        )
        saved = await self.client.post(
            "/ui/tianyi-3d/pose-retargeting/save",
            json={
                **upload,
                "output_pose_type": "left_arm",
                "name": "ui_g1_present_left_arm",
                "notes": "Checked from Front camera",
                "overwrite": True,
            },
        )

        self.assertEqual(preview.status_code, 200)
        self.assertTrue(preview.json()["converged"])
        self.assertEqual(len(preview.json()["source_joint_positions"]), 17)
        self.assertEqual(len(preview.json()["arm_reports"]), 2)
        self.assertIn(
            "upper_arm_direction_error_rad",
            preview.json()["arm_reports"][0],
        )
        self.assertIn(
            "forearm_direction_error_rad",
            preview.json()["arm_reports"][0],
        )
        self.assertEqual(preview.json()["contact_count"], 0)
        self.assertEqual(len(self.viser.retarget_previews), 2)
        self.assertEqual(
            self.viser.retarget_previews[0]["left_shoulder_yaw_joint"],
            1.372,
        )
        self.assertEqual(saved.status_code, 200)
        pose = self.application.pose_service.load_pose(
            pose_type=PoseType.LEFT_ARM,
            name="ui_g1_present_left_arm",
        )
        self.assertEqual(pose.source.value, "retargeting")
        self.assertIn("G1 composed/concierge_present_left", pose.notes)
        self.assertEqual(len(pose.joint_values), 7)
        self.assertEqual(
            tuple(pose.joint_values),
            self.application.joint_schema.LEFT_ARM_JOINT_NAMES,
        )
        state = self.application.simulation.snapshot().joint_position_map()
        self.assertTrue(
            all(
                state[name] == value
                for name, value in self.application.joint_schema.locked_values().items()
            )
        )

    async def test_pose_recorder_loads_saved_pose_as_editable_mujoco_state(self) -> None:
        self._save_composer_sources()

        response = await self.client.post(
            "/ui/tianyi-3d/pose-recorder/load",
            json={"pose_type": "left_arm", "name": "ui_left_wave"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["edit_pose_type"], "left_arm")
        self.assertEqual(len(response.json()["joint_positions"]), 7)
        state = self.application.simulation.snapshot().joint_position_map()
        self.assertEqual(state["elbow_pitch_l_joint"], -0.75)
        self.assertTrue(
            all(
                state[name] == value
                for name, value in self.application.joint_schema.locked_values().items()
            )
        )

    async def test_save_left_arm_pose_uses_exact_seven_joint_contract(self) -> None:
        values = self.application.joint_schema.neutral_values(PoseType.LEFT_ARM)
        values["elbow_pitch_l_joint"] = -0.5
        response = await self.client.post(
            "/ui/tianyi-3d/poses",
            json={
                "pose_type": "left_arm",
                "name": "ui_left_wave",
                "notes": "recorded in the UI test",
                "joint_positions": values,
                "overwrite": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        pose = self.application.pose_service.load_pose(
            pose_type=PoseType.LEFT_ARM,
            name="ui_left_wave",
        )
        self.assertEqual(
            tuple(pose.joint_values),
            self.application.joint_schema.LEFT_ARM_JOINT_NAMES,
        )
        self.assertEqual(pose.joint_values["elbow_pitch_l_joint"], -0.5)

    async def test_ui_calibration_command_is_separate_from_pose_commands(self) -> None:
        snapshot = _apply_simulation_update(
            self.application,
            {"calibration_joint_positions": {"waist_pitch_joint": 0.55}},
        )

        self.assertEqual(snapshot.joint_position_map()["waist_pitch_joint"], 0.55)
        self.assertEqual(
            self.application.simulation.calibration_joint_positions()["waist_pitch_joint"],
            0.55,
        )
        with self.assertRaisesRegex(ValueError, "Locked Tianyi joints"):
            _apply_simulation_update(
                self.application,
                {"joint_positions": {"waist_pitch_joint": 0.6}},
            )

        rejected_save = await self.client.post(
            "/ui/tianyi-3d/poses",
            json={
                "pose_type": "base",
                "name": "must_not_record_calibration",
                "joint_positions": {
                    **self.application.joint_schema.neutral_values(PoseType.BASE),
                    "waist_pitch_joint": 0.55,
                },
            },
        )
        self.assertEqual(rejected_save.status_code, 422)
        self.assertIn("unknown", rejected_save.json()["detail"])

    async def test_camera_endpoint_uses_robot_relative_preset(self) -> None:
        response = await self.client.post("/ui/tianyi-3d/viewer/camera/front_left_45")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["camera_view"], "front_left_45")
        self.assertEqual(response.json()["connected_clients"], 1)
        self.assertEqual(len(self.viser.camera_views), 1)

    async def test_viewer_comparison_mode_requires_a_retarget_preview(self) -> None:
        before = await self.client.post(
            "/ui/tianyi-3d/viewer/mode/retarget-comparison"
        )
        self.viser.retarget_previews.append({"left_elbow_joint": 0.4})
        after = await self.client.post(
            "/ui/tianyi-3d/viewer/mode/retarget-comparison"
        )
        standard = await self.client.post("/ui/tianyi-3d/viewer/mode/tianyi")

        self.assertFalse(before.json()["comparison_visible"])
        self.assertTrue(after.json()["comparison_visible"])
        self.assertFalse(standard.json()["comparison_visible"])

    async def test_pose_recorder_mirrors_live_arm_targets_without_saving(self) -> None:
        right_values = dict(
            zip(
                self.application.joint_schema.RIGHT_ARM_JOINT_NAMES,
                (0.2, -0.3, 0.4, -0.7, -0.25, 0.1, -0.2),
                strict=True,
            )
        )
        self.application.simulation.update_joint_positions(right_values)

        response = await self.client.post(
            "/ui/tianyi-3d/pose-recorder/mirror-arm",
            json={"source_pose_type": "right_arm"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target_pose_type"], "left_arm")
        self.assertEqual(response.json()["source_joint_positions"], right_values)
        self.assertEqual(response.json()["joint_positions"]["shoulder_roll_l_joint"], 0.3)
        self.assertEqual(response.json()["joint_positions"]["elbow_yaw_l_joint"], 0.25)
        state = self.application.simulation.snapshot().joint_position_map()
        self.assertEqual(state["shoulder_roll_l_joint"], 0.3)
        self.assertEqual(state["shoulder_roll_r_joint"], -0.3)

    async def test_pose_composer_preview_updates_only_head_and_arm_simulation(self) -> None:
        self._save_composer_sources()

        response = await self.client.post(
            "/ui/tianyi-3d/pose-composer/preview",
            json={
                "base_pose": "ui_attentive_base",
                "left_arm_pose": "ui_left_wave",
                "right_arm_pose": None,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["source_parts"],
            {"base": "ui_attentive_base", "left_arm": "ui_left_wave"},
        )
        state = self.application.simulation.snapshot().joint_position_map()
        self.assertEqual(state["head_pitch_joint"], 0.15)
        self.assertEqual(state["elbow_pitch_l_joint"], -0.75)
        self.assertTrue(
            all(
                state[name] == value
                for name, value in self.application.joint_schema.locked_values().items()
            )
        )

    async def test_pose_composer_saves_provenance_and_protects_existing_file(self) -> None:
        self._save_composer_sources()
        command = {
            "base_pose": "ui_attentive_base",
            "left_arm_pose": "ui_left_wave",
            "right_arm_pose": None,
            "name": "ui_composed_wave",
            "notes": "Keeps the base robot-right arm",
            "overwrite": False,
        }

        response = await self.client.post(
            "/ui/tianyi-3d/pose-composer/save",
            json=command,
        )
        duplicate = await self.client.post(
            "/ui/tianyi-3d/pose-composer/save",
            json=command,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(duplicate.status_code, 409)
        pose = self.application.pose_service.load_pose(
            pose_type=PoseType.COMPOSED,
            name="ui_composed_wave",
        )
        self.assertEqual(
            dict(pose.source_parts),
            {"base": "ui_attentive_base", "left_arm": "ui_left_wave"},
        )
        self.assertEqual(pose.notes, "Keeps the base robot-right arm")
        self.assertEqual(len(pose.joint_values), 17)

    async def test_action_composer_saves_and_compiles_simulation_only_trajectory(self) -> None:
        self._save_composer_sources()
        payload = {
            "name": "ui_wave_action",
            "frames": [
                {
                    "pose_type": "base",
                    "name": "ui_attentive_base",
                    "duration_seconds": 0.1,
                    "hold_seconds": 0,
                }
            ],
            "return_duration_seconds": 0.1,
            "notes": "simulation only",
            "overwrite": True,
        }

        saved = await self.client.post("/ui/tianyi-3d/action/save", json=payload)
        compiled = await self.client.post(
            "/ui/tianyi-3d/action/compile",
            json={**payload, "sample_frequency_hz": 10},
        )

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(compiled.status_code, 200)
        self.assertEqual(compiled.json()["sample_count"], 3)
        action = self.application.action_service.load_action(name="ui_wave_action")
        self.assertEqual(action.initial_pose.name, "concierge_init")
        self.assertEqual(action.transitions[-1].target_pose.name, "concierge_init")
        trajectory = self.application.action_service.load_trajectory(name="ui_wave_action")
        self.assertEqual(trajectory.joint_names, self.application.joint_schema.BASE_JOINT_NAMES)

    def _save_composer_sources(self) -> None:
        base_values = self.application.joint_schema.neutral_values(PoseType.BASE)
        base_values["head_pitch_joint"] = 0.15
        base = self.application.pose_service.create_simulated_pose(
            name="ui_attentive_base",
            pose_type=PoseType.BASE,
            joint_values=base_values,
        )
        left_values = self.application.joint_schema.neutral_values(PoseType.LEFT_ARM)
        left_values["elbow_pitch_l_joint"] = -0.75
        left = self.application.pose_service.create_simulated_pose(
            name="ui_left_wave",
            pose_type=PoseType.LEFT_ARM,
            joint_values=left_values,
        )
        self.application.pose_service.save_pose(pose=base, overwrite=True)
        self.application.pose_service.save_pose(pose=left, overwrite=True)
