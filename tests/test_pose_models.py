"""Tests for serializable Tianyi pose definitions."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from component.pose import PoseDefinition, PoseSource, PoseType, TianyiJointSchema
from config.settings import AppSettings
from util.tianyi_asset_helper import TianyiAssetHelper


class PoseDefinitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = AppSettings()
        cls.schema = TianyiJointSchema(
            asset_helper=TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir)
        )

    def test_each_pose_type_has_the_expected_shape(self) -> None:
        expected_counts = {
            PoseType.BASE: 17,
            PoseType.LEFT_ARM: 7,
            PoseType.RIGHT_ARM: 7,
        }
        for pose_type, expected_count in expected_counts.items():
            with self.subTest(pose_type=pose_type):
                pose = PoseDefinition.create(
                    schema=self.schema,
                    name=f"neutral_{pose_type.value}",
                    pose_type=pose_type,
                    joint_values=self.schema.neutral_values(pose_type),
                    source=PoseSource.SIMULATION,
                )
                self.assertEqual(len(pose.joint_values), expected_count)

        composed = PoseDefinition.create(
            schema=self.schema,
            name="neutral_composed",
            pose_type=PoseType.COMPOSED,
            joint_values=self.schema.neutral_values(PoseType.COMPOSED),
            source=PoseSource.COMPOSITION,
            source_parts={"base": "neutral_base"},
        )
        self.assertEqual(len(composed.joint_values), 17)

    def test_json_round_trip_preserves_unicode_and_metadata(self) -> None:
        created_at = datetime(2026, 9, 13, 8, 30, tzinfo=timezone.utc)
        pose = PoseDefinition.create(
            schema=self.schema,
            name="迎宾组合姿态",
            pose_type=PoseType.COMPOSED,
            joint_values=self.schema.neutral_values(PoseType.COMPOSED),
            source=PoseSource.COMPOSITION,
            created_at=created_at,
            source_parts={"base": "自然站姿", "left_arm": "左手问候"},
            notes="MuJoCo 模拟姿态",
        )
        restored = PoseDefinition.from_json(schema=self.schema, content=pose.to_json())

        self.assertEqual(restored, pose)
        self.assertIn("迎宾组合姿态", pose.to_json())
        self.assertEqual(restored.created_at, created_at)
        self.assertEqual(restored.source_parts["left_arm"], "左手问候")

    def test_pose_mappings_are_immutable_copies(self) -> None:
        joint_values = self.schema.neutral_values(PoseType.LEFT_ARM)
        pose = PoseDefinition.create(
            schema=self.schema,
            name="immutable_left_arm",
            pose_type=PoseType.LEFT_ARM,
            joint_values=joint_values,
            source=PoseSource.SIMULATION,
        )
        joint_values["elbow_pitch_l_joint"] = -0.5
        self.assertEqual(pose.joint_values["elbow_pitch_l_joint"], 0.0)
        with self.assertRaises(TypeError):
            pose.joint_values["elbow_pitch_l_joint"] = -0.5  # type: ignore[index]

    def test_invalid_name_and_timestamp_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "path-unsafe"):
            PoseDefinition.create(
                schema=self.schema,
                name="../unsafe",
                pose_type=PoseType.LEFT_ARM,
                joint_values=self.schema.neutral_values(PoseType.LEFT_ARM),
                source=PoseSource.SIMULATION,
            )
        with self.assertRaisesRegex(ValueError, "timezone"):
            PoseDefinition.create(
                schema=self.schema,
                name="naive_time",
                pose_type=PoseType.LEFT_ARM,
                joint_values=self.schema.neutral_values(PoseType.LEFT_ARM),
                source=PoseSource.SIMULATION,
                created_at=datetime(2026, 9, 13),
            )

    def test_composed_pose_requires_valid_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "composition as its source"):
            PoseDefinition.create(
                schema=self.schema,
                name="wrong_source",
                pose_type=PoseType.COMPOSED,
                joint_values=self.schema.neutral_values(PoseType.COMPOSED),
                source=PoseSource.SIMULATION,
                source_parts={"base": "neutral"},
            )
        with self.assertRaisesRegex(ValueError, "identify its base"):
            PoseDefinition.create(
                schema=self.schema,
                name="missing_base",
                pose_type=PoseType.COMPOSED,
                joint_values=self.schema.neutral_values(PoseType.COMPOSED),
                source=PoseSource.COMPOSITION,
                source_parts={"left_arm": "wave"},
            )
        with self.assertRaisesRegex(ValueError, "cannot have source parts"):
            PoseDefinition.create(
                schema=self.schema,
                name="left_with_parts",
                pose_type=PoseType.LEFT_ARM,
                joint_values=self.schema.neutral_values(PoseType.LEFT_ARM),
                source=PoseSource.SIMULATION,
                source_parts={"base": "neutral"},
            )

    def test_wrong_schema_model_and_non_object_json_are_rejected(self) -> None:
        pose = PoseDefinition.create(
            schema=self.schema,
            name="model_check",
            pose_type=PoseType.RIGHT_ARM,
            joint_values=self.schema.neutral_values(PoseType.RIGHT_ARM),
            source=PoseSource.SIMULATION,
        )
        payload = pose.to_dict()
        payload["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "Unsupported pose schema version"):
            PoseDefinition.from_dict(schema=self.schema, payload=payload)

        payload = pose.to_dict()
        payload["robot_model_id"] = "different_robot"
        with self.assertRaisesRegex(ValueError, "does not match"):
            PoseDefinition.from_dict(schema=self.schema, payload=payload)

        payload = pose.to_dict()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            PoseDefinition.from_dict(schema=self.schema, payload=payload)

        payload = pose.to_dict()
        payload["source_parts"] = {"base": 123}
        with self.assertRaisesRegex(ValueError, "keys and values must be strings"):
            PoseDefinition.from_dict(schema=self.schema, payload=payload)

        with self.assertRaisesRegex(ValueError, "root must be an object"):
            PoseDefinition.from_json(schema=self.schema, content=json.dumps([]))


def demo_test_pose_definition() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PoseDefinitionTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    demo_test_pose_definition()


if __name__ == "__main__":
    main()
