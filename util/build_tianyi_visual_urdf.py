"""Build the collision-free Tianyi URDF used by the Viser scene.

Run from the project root after regenerating the optimized visual meshes:

    uv run python util/build_tianyi_visual_urdf.py
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import xml.etree.ElementTree as element_tree
from copy import deepcopy
from pathlib import Path

from config.settings import AppSettings

LOGGER = logging.getLogger(__name__)
SOURCE_MESH_PREFIX = "package://tianyi2_urdf/meshes/"
OPTIMIZED_MESH_DIRECTORY = "meshes_visual_optimized"
OUTPUT_URDF_NAME = "tianyi2_visual_optimized.urdf"
OUTPUT_METADATA_NAME = "visual_urdf_metadata.json"
SOURCE_URDF_NAME = "tianyi2.0_urdf_with_hands.urdf"
CANONICAL_BODY_URDF_NAME = "tianyi2.0_URDF.urdf"
URDF_MESH_METADATA_NAME = "visual_urdf_mesh_metadata.json"


def render_visual_urdf(*, asset_dir: Path) -> bytes:
    """Return the deterministic optimized visual URDF document."""
    source_urdf_path = asset_dir / SOURCE_URDF_NAME
    canonical_body_urdf_path = asset_dir / CANONICAL_BODY_URDF_NAME
    optimized_mesh_records = _optimized_mesh_records(asset_dir=asset_dir)
    tree = element_tree.parse(source_urdf_path)
    root = tree.getroot()
    root.attrib["name"] = "tianyi2_visual_optimized"
    canonical_body_root = element_tree.parse(canonical_body_urdf_path).getroot()
    _replace_body_joint_contract(
        root=root,
        canonical_body_root=canonical_body_root,
    )
    _lock_hand_joints(root=root, canonical_body_root=canonical_body_root)

    for link in root.findall("link"):
        for collision in tuple(link.findall("collision")):
            link.remove(collision)
        for visual in link.findall("visual"):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                raise ValueError(f"Tianyi visual link {link.attrib['name']} has no mesh")
            source_reference = mesh.attrib.get("filename")
            if source_reference is None or not source_reference.startswith(SOURCE_MESH_PREFIX):
                raise ValueError(f"Unsupported Tianyi visual mesh reference: {source_reference}")
            source_name = source_reference.removeprefix(SOURCE_MESH_PREFIX)
            if Path(source_name).name != source_name:
                raise ValueError(f"Nested Tianyi visual mesh path is unsupported: {source_name}")
            try:
                output_name = optimized_mesh_records[source_name]
            except KeyError as error:
                raise ValueError(
                    f"No optimized visual mesh is recorded for {source_name}"
                ) from error
            optimized_path = asset_dir / OPTIMIZED_MESH_DIRECTORY / output_name
            if not optimized_path.is_file():
                raise ValueError(f"Missing optimized Tianyi visual mesh: {output_name}")
            mesh.attrib["filename"] = f"{OPTIMIZED_MESH_DIRECTORY}/{output_name}"

    element_tree.indent(tree, space="  ")
    output = io.BytesIO()
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output.getvalue()


def build_visual_urdf(*, asset_dir: Path) -> tuple[Path, Path]:
    """Write the optimized visual URDF and its generated provenance metadata."""
    source_urdf_path = asset_dir / SOURCE_URDF_NAME
    canonical_body_urdf_path = asset_dir / CANONICAL_BODY_URDF_NAME
    optimized_mesh_metadata_path = asset_dir / URDF_MESH_METADATA_NAME
    output_urdf_path = asset_dir / OUTPUT_URDF_NAME
    output_metadata_path = asset_dir / OUTPUT_METADATA_NAME

    urdf_bytes = render_visual_urdf(asset_dir=asset_dir)
    _write_if_changed(output_urdf_path, urdf_bytes)

    root = element_tree.fromstring(urdf_bytes)
    source_root = element_tree.parse(source_urdf_path).getroot()
    canonical_body_root = element_tree.parse(canonical_body_urdf_path).getroot()
    movable_joints = tuple(
        joint
        for joint in root.findall("joint")
        if joint.attrib["type"] not in {"fixed", "floating"}
    )
    movable_joint_names = tuple(joint.attrib["name"] for joint in movable_joints)
    actuated_joint_names = tuple(
        joint.attrib["name"] for joint in movable_joints if joint.find("mimic") is None
    )
    canonical_body_joint_names = tuple(
        joint.attrib["name"]
        for joint in canonical_body_root.findall("joint")
        if joint.attrib["type"] not in {"fixed", "floating"}
    )
    source_movable_joints = tuple(
        joint
        for joint in source_root.findall("joint")
        if joint.attrib["type"] not in {"fixed", "floating"}
    )
    source_movable_joint_names = tuple(joint.attrib["name"] for joint in source_movable_joints)
    hand_joint_names = tuple(
        name for name in source_movable_joint_names if name not in canonical_body_joint_names
    )
    source_hand_actuated_joint_names = tuple(
        joint.attrib["name"]
        for joint in source_movable_joints
        if joint.attrib["name"] in set(hand_joint_names) and joint.find("mimic") is None
    )
    metadata = {
        "schema_version": 2,
        "purpose": "viser_body_and_inspire_hands",
        "source_urdf": source_urdf_path.name,
        "source_urdf_sha256": _sha256(source_urdf_path.read_bytes()),
        "canonical_body_urdf": canonical_body_urdf_path.name,
        "canonical_body_urdf_sha256": _sha256(canonical_body_urdf_path.read_bytes()),
        "optimized_visual_urdf": output_urdf_path.name,
        "optimized_visual_urdf_sha256": _sha256(urdf_bytes),
        "optimized_mesh_directory": OPTIMIZED_MESH_DIRECTORY,
        "optimized_mesh_metadata": optimized_mesh_metadata_path.name,
        "optimized_mesh_metadata_sha256": _sha256(optimized_mesh_metadata_path.read_bytes()),
        "generator": "util/build_tianyi_visual_urdf.py",
        "link_count": len(root.findall("link")),
        "joint_count": len(root.findall("joint")),
        "movable_joint_count": len(movable_joint_names),
        "movable_joint_names": movable_joint_names,
        "actuated_joint_count": len(actuated_joint_names),
        "actuated_joint_names": actuated_joint_names,
        "body_joint_count": len(canonical_body_joint_names),
        "body_joint_names": canonical_body_joint_names,
        "hand_joint_count": len(hand_joint_names),
        "hand_joint_names": hand_joint_names,
        "source_hand_actuated_joint_count": len(source_hand_actuated_joint_names),
        "source_hand_actuated_joint_names": source_hand_actuated_joint_names,
        "visual_hand_actuated_joint_count": 0,
        "hand_joint_mode": "fixed_neutral",
        "hand_default_position": 0.0,
        "body_limit_overrides": _body_limit_overrides(
            source_root=source_root,
            canonical_body_root=canonical_body_root,
        ),
        "visual_mesh_count": len(root.findall("./link/visual/geometry/mesh")),
        "collision_element_count": len(root.findall(".//collision")),
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    _write_if_changed(output_metadata_path, metadata_bytes)
    return output_urdf_path, output_metadata_path


def _optimized_mesh_records(*, asset_dir: Path) -> dict[str, str]:
    metadata_path = asset_dir / URDF_MESH_METADATA_NAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("meshes")
    if not isinstance(records, list):
        raise ValueError("Visual model metadata meshes must be a list")

    mapping: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Visual model mesh record must be an object")
        source_name = record.get("source")
        output_name = record.get("output")
        if not isinstance(source_name, str) or not isinstance(output_name, str):
            raise ValueError("Visual model mesh record lacks source or output")
        if Path(source_name).name != source_name or Path(output_name).name != output_name:
            raise ValueError("Visual model mesh record contains an unsafe path")
        mapping[source_name] = output_name
    return mapping


def _replace_body_joint_contract(
    *,
    root: element_tree.Element,
    canonical_body_root: element_tree.Element,
) -> None:
    """Use canonical MuJoCo-aligned body joints inside the hand visual tree."""
    canonical_joints = {
        joint.attrib["name"]: joint for joint in canonical_body_root.findall("joint")
    }
    replaced_names: set[str] = set()
    for index, child in enumerate(tuple(root)):
        if child.tag != "joint" or child.attrib["name"] not in canonical_joints:
            continue
        root[index] = deepcopy(canonical_joints[child.attrib["name"]])
        replaced_names.add(child.attrib["name"])
    if replaced_names != set(canonical_joints):
        missing = sorted(set(canonical_joints) - replaced_names)
        raise ValueError(f"Inspire-hand URDF is missing canonical body joints: {missing}")


def _lock_hand_joints(
    *,
    root: element_tree.Element,
    canonical_body_root: element_tree.Element,
) -> None:
    """Keep Inspire hand geometry while removing all finger degrees of freedom."""
    body_joint_names = {joint.attrib["name"] for joint in canonical_body_root.findall("joint")}
    for joint in root.findall("joint"):
        if joint.attrib["name"] in body_joint_names:
            continue
        joint.attrib["type"] = "fixed"
        for child_name in ("axis", "limit", "mimic", "dynamics", "safety_controller"):
            child = joint.find(child_name)
            if child is not None:
                joint.remove(child)


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


def _write_if_changed(path: Path, content: bytes) -> None:
    if path.is_file() and path.read_bytes() == content:
        return
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def demo_build_visual_urdf() -> None:
    logging.basicConfig(level=logging.INFO)
    output_urdf, output_metadata = build_visual_urdf(asset_dir=AppSettings().tianyi_asset_dir)
    LOGGER.info("Generated %s and %s", output_urdf, output_metadata)


def main() -> None:
    demo_build_visual_urdf()


if __name__ == "__main__":
    main()
