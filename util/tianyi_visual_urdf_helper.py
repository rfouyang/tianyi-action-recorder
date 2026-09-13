"""Validate the generated collision-free Tianyi URDF for Viser."""

from __future__ import annotations

import hashlib
import json
import logging
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path

from config.settings import AppSettings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TianyiVisualUrdfReport:
    """Summary of the checked Viser-oriented robot description."""

    link_count: int
    joint_count: int
    movable_joint_names: tuple[str, ...]
    actuated_joint_names: tuple[str, ...]
    body_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]
    source_hand_actuated_joint_names: tuple[str, ...]
    visual_mesh_count: int
    collision_element_count: int


class TianyiVisualUrdfHelper:
    """Verify provenance, kinematics, and optimized mesh use in the Viser URDF."""

    EXPECTED_LINK_COUNT = 51
    EXPECTED_JOINT_COUNT = 50
    EXPECTED_MOVABLE_JOINT_COUNT = 21
    EXPECTED_ACTUATED_JOINT_COUNT = 21
    EXPECTED_BODY_JOINT_COUNT = 21
    EXPECTED_HAND_JOINT_COUNT = 24
    EXPECTED_SOURCE_HAND_ACTUATED_JOINT_COUNT = 12
    EXPECTED_VISUAL_MESH_COUNT = 51
    EXPECTED_BODY_LIMIT_OVERRIDE_NAMES = (
        "first_leg_pitch_joint",
        "second_leg_pitch_joint",
        "waist_pitch_joint",
    )
    OPTIMIZED_ROBOT_NAME = "tianyi2_visual_optimized"
    OPTIMIZED_MESH_DIRECTORY = "meshes_visual_optimized"

    def __init__(self, *, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self.source_urdf_path = asset_dir / "tianyi2.0_urdf_with_hands.urdf"
        self.canonical_body_urdf_path = asset_dir / "tianyi2.0_URDF.urdf"
        self.visual_urdf_path = asset_dir / "tianyi2_visual_optimized.urdf"
        self.mesh_metadata_path = asset_dir / "visual_urdf_mesh_metadata.json"
        self.metadata_path = asset_dir / "visual_urdf_metadata.json"

    def validate(self) -> TianyiVisualUrdfReport:
        """Raise a descriptive error if the generated visual URDF has drifted."""
        metadata = self._load_object(self.metadata_path)
        self._validate_hashes(metadata)
        source_root = element_tree.parse(self.source_urdf_path).getroot()
        canonical_body_root = element_tree.parse(self.canonical_body_urdf_path).getroot()
        visual_root = element_tree.parse(self.visual_urdf_path).getroot()

        if visual_root.attrib.get("name") != self.OPTIMIZED_ROBOT_NAME:
            raise ValueError("Optimized visual URDF has the wrong robot name")
        if self._link_names(source_root) != self._link_names(visual_root):
            raise ValueError("Optimized visual URDF does not preserve the source links")
        self._validate_joint_contracts(
            source_root=source_root,
            canonical_body_root=canonical_body_root,
            visual_root=visual_root,
        )

        movable_joint_names = self._movable_joint_names(visual_root)
        actuated_joint_names = self._actuated_joint_names(visual_root)
        body_joint_names = self._movable_joint_names(canonical_body_root)
        source_movable_joint_names = self._movable_joint_names(source_root)
        hand_joint_names = tuple(
            name for name in source_movable_joint_names if name not in set(body_joint_names)
        )
        source_hand_actuated_joint_names = tuple(
            name
            for name in self._actuated_joint_names(source_root)
            if name in set(hand_joint_names)
        )
        visual_mesh_references = self._visual_mesh_references(visual_root)
        collision_count = len(visual_root.findall(".//collision"))
        self._validate_counts(
            link_count=len(visual_root.findall("link")),
            joint_count=len(visual_root.findall("joint")),
            movable_joint_count=len(movable_joint_names),
            actuated_joint_count=len(actuated_joint_names),
            body_joint_count=len(body_joint_names),
            hand_joint_count=len(hand_joint_names),
            source_hand_actuated_joint_count=len(source_hand_actuated_joint_names),
            visual_hand_actuated_joint_count=0,
            visual_mesh_count=len(visual_mesh_references),
            collision_element_count=collision_count,
        )
        self._validate_mesh_references(visual_mesh_references)
        self._validate_metadata_contract(
            metadata=metadata,
            movable_joint_names=movable_joint_names,
            actuated_joint_names=actuated_joint_names,
            body_joint_names=body_joint_names,
            hand_joint_names=hand_joint_names,
            source_hand_actuated_joint_names=source_hand_actuated_joint_names,
            body_limit_overrides=self._body_limit_overrides(
                source_root=source_root,
                canonical_body_root=canonical_body_root,
            ),
        )
        return TianyiVisualUrdfReport(
            link_count=len(visual_root.findall("link")),
            joint_count=len(visual_root.findall("joint")),
            movable_joint_names=movable_joint_names,
            actuated_joint_names=actuated_joint_names,
            body_joint_names=body_joint_names,
            hand_joint_names=hand_joint_names,
            source_hand_actuated_joint_names=source_hand_actuated_joint_names,
            visual_mesh_count=len(visual_mesh_references),
            collision_element_count=collision_count,
        )

    def _validate_hashes(self, metadata: dict[str, object]) -> None:
        expected_hashes = {
            self.source_urdf_path: metadata.get("source_urdf_sha256"),
            self.canonical_body_urdf_path: metadata.get("canonical_body_urdf_sha256"),
            self.visual_urdf_path: metadata.get("optimized_visual_urdf_sha256"),
            self.mesh_metadata_path: metadata.get("optimized_mesh_metadata_sha256"),
        }
        for path, expected in expected_hashes.items():
            if not isinstance(expected, str) or not expected:
                raise ValueError(f"Visual URDF metadata lacks a SHA-256 for {path.name}")
            if not path.is_file():
                raise ValueError(f"Missing Tianyi visual URDF asset: {path.name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Visual URDF asset hash mismatch for {path.name}")

    def _validate_counts(self, **actual_counts: int) -> None:
        expected_counts = {
            "link_count": self.EXPECTED_LINK_COUNT,
            "joint_count": self.EXPECTED_JOINT_COUNT,
            "movable_joint_count": self.EXPECTED_MOVABLE_JOINT_COUNT,
            "actuated_joint_count": self.EXPECTED_ACTUATED_JOINT_COUNT,
            "body_joint_count": self.EXPECTED_BODY_JOINT_COUNT,
            "hand_joint_count": self.EXPECTED_HAND_JOINT_COUNT,
            "source_hand_actuated_joint_count": (self.EXPECTED_SOURCE_HAND_ACTUATED_JOINT_COUNT),
            "visual_hand_actuated_joint_count": 0,
            "visual_mesh_count": self.EXPECTED_VISUAL_MESH_COUNT,
            "collision_element_count": 0,
        }
        for name, expected in expected_counts.items():
            actual = actual_counts[name]
            if actual != expected:
                raise ValueError(f"Optimized visual URDF {name}={actual}; expected {expected}")

    def _validate_mesh_references(self, references: tuple[str, ...]) -> None:
        mesh_metadata = self._load_object(self.mesh_metadata_path)
        expected_mesh_metadata = {
            "schema_version": 1,
            "purpose": "viser_body_and_inspire_hands",
            "source_urdf": self.source_urdf_path.name,
            "source_urdf_sha256": hashlib.sha256(self.source_urdf_path.read_bytes()).hexdigest(),
            "mesh_count": self.EXPECTED_VISUAL_MESH_COUNT,
        }
        for name, expected_value in expected_mesh_metadata.items():
            if mesh_metadata.get(name) != expected_value:
                raise ValueError(
                    f"Viser mesh metadata {name}={mesh_metadata.get(name)!r}; "
                    f"expected {expected_value!r}"
                )
        records = mesh_metadata.get("meshes")
        if not isinstance(records, list) or len(records) != self.EXPECTED_VISUAL_MESH_COUNT:
            raise ValueError(
                "Viser mesh metadata must describe exactly "
                f"{self.EXPECTED_VISUAL_MESH_COUNT} meshes"
            )
        expected_references: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Visual mesh metadata entry must be an object")
            output_name = record.get("output")
            if not isinstance(output_name, str) or Path(output_name).name != output_name:
                raise ValueError("Visual mesh metadata has an invalid output name")
            source_name = record.get("source")
            if not isinstance(source_name, str) or Path(source_name).name != source_name:
                raise ValueError("Visual mesh metadata has an invalid source name")
            self._validate_file_hash(
                path=self.asset_dir / "meshes" / source_name,
                expected=record.get("source_sha256"),
            )
            self._validate_file_hash(
                path=self.asset_dir / self.OPTIMIZED_MESH_DIRECTORY / output_name,
                expected=record.get("sha256"),
            )
            expected_references.add(f"{self.OPTIMIZED_MESH_DIRECTORY}/{output_name}")

        if set(references) != expected_references:
            raise ValueError("Optimized visual URDF mesh references do not match metadata")
        if len(references) != len(set(references)):
            raise ValueError("Optimized visual URDF contains duplicate mesh references")
        for reference in references:
            relative_path = Path(reference)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.parent != Path(self.OPTIMIZED_MESH_DIRECTORY)
            ):
                raise ValueError(f"Unsafe optimized visual mesh reference: {reference}")
            if not (self.asset_dir / relative_path).is_file():
                raise ValueError(f"Missing optimized visual mesh: {reference}")

    def _validate_metadata_contract(
        self,
        *,
        metadata: dict[str, object],
        movable_joint_names: tuple[str, ...],
        actuated_joint_names: tuple[str, ...],
        body_joint_names: tuple[str, ...],
        hand_joint_names: tuple[str, ...],
        source_hand_actuated_joint_names: tuple[str, ...],
        body_limit_overrides: dict[str, dict[str, dict[str, str]]],
    ) -> None:
        expected = {
            "schema_version": 2,
            "purpose": "viser_body_and_inspire_hands",
            "source_urdf": self.source_urdf_path.name,
            "canonical_body_urdf": self.canonical_body_urdf_path.name,
            "optimized_visual_urdf": self.visual_urdf_path.name,
            "optimized_mesh_directory": self.OPTIMIZED_MESH_DIRECTORY,
            "optimized_mesh_metadata": self.mesh_metadata_path.name,
            "generator": "util/build_tianyi_visual_urdf.py",
            "link_count": self.EXPECTED_LINK_COUNT,
            "joint_count": self.EXPECTED_JOINT_COUNT,
            "movable_joint_count": self.EXPECTED_MOVABLE_JOINT_COUNT,
            "movable_joint_names": list(movable_joint_names),
            "actuated_joint_count": self.EXPECTED_ACTUATED_JOINT_COUNT,
            "actuated_joint_names": list(actuated_joint_names),
            "body_joint_count": self.EXPECTED_BODY_JOINT_COUNT,
            "body_joint_names": list(body_joint_names),
            "hand_joint_count": self.EXPECTED_HAND_JOINT_COUNT,
            "hand_joint_names": list(hand_joint_names),
            "source_hand_actuated_joint_count": (self.EXPECTED_SOURCE_HAND_ACTUATED_JOINT_COUNT),
            "source_hand_actuated_joint_names": list(source_hand_actuated_joint_names),
            "visual_hand_actuated_joint_count": 0,
            "hand_joint_mode": "fixed_neutral",
            "hand_default_position": 0.0,
            "body_limit_overrides": body_limit_overrides,
            "visual_mesh_count": self.EXPECTED_VISUAL_MESH_COUNT,
            "collision_element_count": 0,
        }
        for name, expected_value in expected.items():
            if metadata.get(name) != expected_value:
                raise ValueError(
                    f"Visual URDF metadata {name}={metadata.get(name)!r}; "
                    f"expected {expected_value!r}"
                )

        if tuple(body_limit_overrides) != self.EXPECTED_BODY_LIMIT_OVERRIDE_NAMES:
            raise ValueError(
                "Unexpected body-limit differences between the hand and canonical URDFs"
            )

    @staticmethod
    def _link_names(root: element_tree.Element) -> tuple[str, ...]:
        return tuple(link.attrib["name"] for link in root.findall("link"))

    @staticmethod
    def _actuated_joint_names(root: element_tree.Element) -> tuple[str, ...]:
        return tuple(
            joint.attrib["name"]
            for joint in root.findall("joint")
            if joint.attrib["type"] not in {"fixed", "floating"} and joint.find("mimic") is None
        )

    @staticmethod
    def _movable_joint_names(root: element_tree.Element) -> tuple[str, ...]:
        return tuple(
            joint.attrib["name"]
            for joint in root.findall("joint")
            if joint.attrib["type"] not in {"fixed", "floating"}
        )

    def _validate_joint_contracts(
        self,
        *,
        source_root: element_tree.Element,
        canonical_body_root: element_tree.Element,
        visual_root: element_tree.Element,
    ) -> None:
        """Validate canonical body joints and untouched Inspire-hand joints."""
        source_names = tuple(joint.attrib["name"] for joint in source_root.findall("joint"))
        visual_names = tuple(joint.attrib["name"] for joint in visual_root.findall("joint"))
        if visual_names != source_names:
            raise ValueError("Optimized visual URDF does not preserve source joint order")

        source_signatures = self._joint_signatures(source_root)
        body_signatures = self._joint_signatures(canonical_body_root)
        visual_signatures = self._joint_signatures(visual_root)
        for name, signature in body_signatures.items():
            if visual_signatures.get(name) != signature:
                raise ValueError(
                    f"Optimized visual URDF body joint differs from canonical model: {name}"
                )
        source_joints = {joint.attrib["name"]: joint for joint in source_root.findall("joint")}
        visual_joints = {joint.attrib["name"]: joint for joint in visual_root.findall("joint")}
        for name in set(source_signatures) - set(body_signatures):
            source_joint = source_joints[name]
            visual_joint = visual_joints[name]
            if visual_joint.attrib["type"] != "fixed":
                raise ValueError(f"Inspire-hand joint is not fixed in Viser: {name}")
            if self._joint_topology_signature(visual_joint) != (
                self._joint_topology_signature(source_joint)
            ):
                raise ValueError(f"Optimized visual URDF changes Inspire-hand topology: {name}")
            if any(
                visual_joint.find(child_name) is not None
                for child_name in ("axis", "limit", "mimic")
            ):
                raise ValueError(f"Fixed Inspire-hand joint retains controls: {name}")

    @staticmethod
    def _joint_signatures(
        root: element_tree.Element,
    ) -> dict[str, tuple[object, ...]]:
        return {
            joint.attrib["name"]: (
                joint.attrib["type"],
                tuple((child.tag, tuple(sorted(child.attrib.items()))) for child in joint),
            )
            for joint in root.findall("joint")
        }

    @staticmethod
    def _body_limit_overrides(
        *,
        source_root: element_tree.Element,
        canonical_body_root: element_tree.Element,
    ) -> dict[str, dict[str, dict[str, str]]]:
        source_joints = {joint.attrib["name"]: joint for joint in source_root.findall("joint")}
        overrides: dict[str, dict[str, dict[str, str]]] = {}
        for canonical_joint in canonical_body_root.findall("joint"):
            name = canonical_joint.attrib["name"]
            source_limit = source_joints[name].find("limit")
            canonical_limit = canonical_joint.find("limit")
            if source_limit is None or canonical_limit is None:
                continue
            source_range = {key: source_limit.attrib[key] for key in ("lower", "upper")}
            canonical_range = {key: canonical_limit.attrib[key] for key in ("lower", "upper")}
            if source_range != canonical_range:
                overrides[name] = {
                    "hand_urdf": source_range,
                    "canonical_body": canonical_range,
                }
        return overrides

    @staticmethod
    def _joint_topology_signature(
        joint: element_tree.Element,
    ) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
        return tuple(
            (child.tag, tuple(sorted(child.attrib.items())))
            for child in joint
            if child.tag in {"origin", "parent", "child"}
        )

    @staticmethod
    def _visual_mesh_references(root: element_tree.Element) -> tuple[str, ...]:
        references = []
        for mesh in root.findall("./link/visual/geometry/mesh"):
            reference = mesh.attrib.get("filename")
            if reference is None:
                raise ValueError("Optimized visual URDF mesh lacks a filename")
            references.append(reference)
        return tuple(references)

    @staticmethod
    def _load_object(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Metadata root must be an object: {path.name}")
        return payload

    @staticmethod
    def _validate_file_hash(*, path: Path, expected: object) -> None:
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"Visual mesh metadata lacks a SHA-256 for {path.name}")
        if not path.is_file():
            raise ValueError(f"Missing Tianyi visual mesh asset: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Visual mesh asset hash mismatch for {path.name}")


def demo_tianyi_visual_urdf_helper() -> None:
    logging.basicConfig(level=logging.INFO)
    report = TianyiVisualUrdfHelper(asset_dir=AppSettings().tianyi_asset_dir).validate()
    LOGGER.info("Validated Tianyi visual URDF: %s", report)


def main() -> None:
    demo_tianyi_visual_urdf_helper()


if __name__ == "__main__":
    main()
