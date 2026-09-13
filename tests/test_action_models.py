"""Tests for Tianyi action definition invariants."""

from __future__ import annotations

import unittest

from component.action import ActionDefinition, ActionPoseReference, ActionTransition
from component.pose import PoseType, TianyiJointSchema
from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper


class ActionDefinitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.schema = TianyiJointSchema(
            asset_helper=TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
        )

    def test_action_requires_base_home_complete_frame_and_exact_return(self) -> None:
        home = ActionPoseReference(PoseType.BASE, "concierge_init")
        action = ActionDefinition.create(
            schema=self.schema,
            name="wave",
            initial_pose=home,
            transitions=(
                ActionTransition(
                    ActionPoseReference(PoseType.COMPOSED, "wave_left"),
                    1.0,
                    0.25,
                ),
                ActionTransition(home, 0.5),
            ),
        )

        self.assertEqual(action.total_duration_seconds, 1.75)
        self.assertEqual(action.pose_sequence[-1], home)
        with self.assertRaisesRegex(ValueError, "base or composed"):
            ActionPoseReference(PoseType.LEFT_ARM, "partial_arm")
        with self.assertRaisesRegex(ValueError, "finish"):
            ActionDefinition.create(
                schema=self.schema,
                name="bad_return",
                initial_pose=home,
                transitions=(
                    ActionTransition(ActionPoseReference(PoseType.BASE, "other"), 1.0),
                    ActionTransition(ActionPoseReference(PoseType.BASE, "other"), 1.0),
                ),
            )
        with self.assertRaisesRegex(ValueError, "base/concierge_init"):
            other_home = ActionPoseReference(PoseType.BASE, "other_home")
            ActionDefinition.create(
                schema=self.schema,
                name="bad_home",
                initial_pose=other_home,
                transitions=(
                    ActionTransition(ActionPoseReference(PoseType.BASE, "other"), 1.0),
                    ActionTransition(other_home, 1.0),
                ),
            )

    def test_zero_hold_is_valid_and_json_round_trip_preserves_it(self) -> None:
        home = ActionPoseReference(PoseType.BASE, "concierge_init")
        action = ActionDefinition.create(
            schema=self.schema,
            name="nod",
            initial_pose=home,
            transitions=(
                ActionTransition(ActionPoseReference(PoseType.BASE, "look_down"), 0.4),
                ActionTransition(home, 0.4),
            ),
        )
        restored = ActionDefinition.from_dict(schema=self.schema, payload=action.to_dict())

        self.assertEqual(restored.transitions[0].hold_seconds, 0.0)
        self.assertEqual(restored.to_dict(), action.to_dict())


if __name__ == "__main__":
    unittest.main()
