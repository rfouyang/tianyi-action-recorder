"""Validate the pinned Tianyi 2.0 body-only URDF and MuJoCo assets."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from config.settings import AppSettings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TianyiAssetReport:
    """Summary of the validated Tianyi model contract."""

    body_joint_count: int
    base_pose_joint_count: int
    actuator_count: int
    mesh_count: int


@dataclass(frozen=True, slots=True)
class TianyiModelJointSpec:
    """Joint information read from the canonical Tianyi URDF."""

    name: str
    axis: tuple[float, float, float]
    lower_limit: float
    upper_limit: float
    effort_limit: float
    velocity_limit: float


class TianyiAssetHelper:
    """Check pinned hashes and agreement between the URDF, MJCF, and meshes."""

    EXPECTED_BODY_JOINT_COUNT = 21
    EXPECTED_BASE_POSE_JOINT_COUNT = 17
    EXPECTED_ACTUATOR_COUNT = 21
    EXPECTED_MESH_COUNT = 25

    def __init__(self, *, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self.urdf_path = asset_dir / "tianyi2.0_URDF.urdf"
        self.mjcf_path = asset_dir / "tianyi2_pos.xml"
        self.metadata_path = asset_dir / "model_metadata.json"

    def validate(self) -> TianyiAssetReport:
        """Raise a descriptive error if any part of the pinned contract has drifted."""
        metadata = self.load_metadata()
        self._validate_hashes(metadata)

        urdf_root = element_tree.parse(self.urdf_path).getroot()
        model = mujoco.MjModel.from_xml_path(str(self.mjcf_path))
        urdf_joints = self._urdf_movable_joints(urdf_root)
        mjcf_joints = self._mjcf_movable_joints(model)

        if len(urdf_joints) != self.EXPECTED_BODY_JOINT_COUNT:
            raise ValueError(
                f"URDF has {len(urdf_joints)} movable joints; "
                f"expected {self.EXPECTED_BODY_JOINT_COUNT}"
            )
        if len(mjcf_joints) != self.EXPECTED_BODY_JOINT_COUNT:
            raise ValueError(
                f"MJCF has {len(mjcf_joints)} movable joints; "
                f"expected {self.EXPECTED_BODY_JOINT_COUNT}"
            )
        if set(urdf_joints) != set(mjcf_joints):
            missing = sorted(set(urdf_joints) - set(mjcf_joints))
            unknown = sorted(set(mjcf_joints) - set(urdf_joints))
            raise ValueError(
                f"URDF and MJCF movable joints differ: missing={missing}, unknown={unknown}"
            )

        for joint_name, urdf_joint in urdf_joints.items():
            self._validate_joint_contract(joint_name, urdf_joint, mjcf_joints[joint_name])

        actuator_joint_names = self._actuator_joint_names(model)
        if len(actuator_joint_names) != self.EXPECTED_ACTUATOR_COUNT:
            raise ValueError(
                f"MJCF has {len(actuator_joint_names)} joint actuators; "
                f"expected {self.EXPECTED_ACTUATOR_COUNT}"
            )
        if set(actuator_joint_names) != set(mjcf_joints):
            raise ValueError("MJCF actuators do not cover exactly the 21 body joints")

        referenced_meshes = self._referenced_meshes(urdf_root)
        if len(referenced_meshes) != self.EXPECTED_MESH_COUNT:
            raise ValueError(
                f"Model references {len(referenced_meshes)} meshes; "
                f"expected {self.EXPECTED_MESH_COUNT}"
            )
        self._validate_metadata_counts(metadata, mesh_count=len(referenced_meshes))

        return TianyiAssetReport(
            body_joint_count=len(urdf_joints),
            base_pose_joint_count=self.EXPECTED_BASE_POSE_JOINT_COUNT,
            actuator_count=len(actuator_joint_names),
            mesh_count=len(referenced_meshes),
        )

    def load_joint_specs(self) -> dict[str, TianyiModelJointSpec]:
        """Load movable joint axes and limits from the pinned URDF."""
        urdf_root = element_tree.parse(self.urdf_path).getroot()
        return {
            name: TianyiModelJointSpec(
                name=name,
                axis=tuple(float(value) for value in axis),
                lower_limit=limits[0],
                upper_limit=limits[1],
                effort_limit=limits[2],
                velocity_limit=limits[3],
            )
            for name, (axis, limits) in self._urdf_movable_joints(urdf_root).items()
        }

    def load_metadata(self) -> dict[str, object]:
        """Load the checked-in model metadata document."""
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Model metadata root must be an object")
        return payload

    def _validate_hashes(self, metadata: dict[str, object]) -> None:
        hashes = metadata.get("sha256")
        if not isinstance(hashes, dict):
            raise ValueError("Model metadata sha256 must be an object")
        for path in (self.urdf_path, self.mjcf_path):
            expected = hashes.get(path.name)
            if not isinstance(expected, str) or not expected:
                raise ValueError(f"Model metadata lacks a SHA-256 for {path.name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Pinned asset hash mismatch for {path.name}")

    def _validate_metadata_counts(
        self,
        metadata: dict[str, object],
        *,
        mesh_count: int,
    ) -> None:
        expected_values = {
            "body_joint_count": self.EXPECTED_BODY_JOINT_COUNT,
            "base_pose_joint_count": self.EXPECTED_BASE_POSE_JOINT_COUNT,
            "actuator_count": self.EXPECTED_ACTUATOR_COUNT,
            "mesh_count": mesh_count,
        }
        for name, expected in expected_values.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"Model metadata {name}={metadata.get(name)!r}; expected {expected}"
                )

    def _urdf_movable_joints(
        self,
        urdf_root: element_tree.Element,
    ) -> dict[str, tuple[np.ndarray, tuple[float, float, float, float]]]:
        joints: dict[str, tuple[np.ndarray, tuple[float, float, float, float]]] = {}
        for joint in urdf_root.findall("joint"):
            if joint.attrib["type"] in {"fixed", "floating"}:
                continue
            axis_element = joint.find("axis")
            limit_element = joint.find("limit")
            if axis_element is None or limit_element is None:
                raise ValueError(f"URDF joint {joint.attrib['name']} lacks axis or limits")
            joints[joint.attrib["name"]] = (
                np.fromstring(axis_element.attrib["xyz"], sep=" "),
                (
                    float(limit_element.attrib["lower"]),
                    float(limit_element.attrib["upper"]),
                    float(limit_element.attrib["effort"]),
                    float(limit_element.attrib["velocity"]),
                ),
            )
        return joints

    def _mjcf_movable_joints(
        self,
        model: mujoco.MjModel,
    ) -> dict[str, tuple[np.ndarray, tuple[float, float]]]:
        joints: dict[str, tuple[np.ndarray, tuple[float, float]]] = {}
        for joint_id in range(model.njnt):
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is None:
                raise ValueError(f"MJCF joint {joint_id} has no name")
            joints[joint_name] = (
                model.jnt_axis[joint_id].copy(),
                tuple(float(value) for value in model.jnt_range[joint_id]),
            )
        return joints

    def _validate_joint_contract(
        self,
        joint_name: str,
        urdf_joint: tuple[np.ndarray, tuple[float, float, float, float]],
        mjcf_joint: tuple[np.ndarray, tuple[float, float]],
    ) -> None:
        urdf_axis, urdf_limits = urdf_joint
        mjcf_axis, mjcf_limits = mjcf_joint
        if not np.allclose(urdf_axis, mjcf_axis, atol=1e-7):
            raise ValueError(f"Joint axis mismatch for {joint_name}")
        if not all(
            math.isclose(urdf_value, mjcf_value, abs_tol=1e-5)
            for urdf_value, mjcf_value in zip(urdf_limits[:2], mjcf_limits, strict=True)
        ):
            raise ValueError(f"Joint limit mismatch for {joint_name}")

    def _actuator_joint_names(self, model: mujoco.MjModel) -> tuple[str, ...]:
        names: list[str] = []
        for joint_id in model.actuator_trnid[:, 0]:
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(joint_id),
            )
            if name is None:
                raise ValueError("MJCF actuator references an unnamed joint")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("MJCF contains duplicate actuators for one or more joints")
        return tuple(names)

    def _referenced_meshes(self, urdf_root: element_tree.Element) -> set[str]:
        referenced_names: set[str] = set()
        for mesh in urdf_root.findall(".//mesh"):
            reference = mesh.attrib.get("filename")
            if reference is None:
                continue
            prefix = "package://tianyi2_urdf/"
            if not reference.startswith(prefix):
                raise ValueError(f"Unsupported URDF mesh reference: {reference}")
            referenced_names.add(Path(reference).name)
            self._require_asset_path(reference.removeprefix(prefix))

        mjcf_root = element_tree.parse(self.mjcf_path).getroot()
        for mesh in mjcf_root.findall(".//mesh"):
            reference = mesh.attrib.get("file")
            if reference is None:
                continue
            normalized = reference.removeprefix("./")
            referenced_names.add(Path(normalized).name)
            self._require_asset_path(normalized)
        return referenced_names

    def _require_asset_path(self, relative_path: str) -> None:
        candidate = (self.asset_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.asset_dir.resolve())
        except ValueError as error:
            raise ValueError(
                f"Asset reference escapes the asset directory: {relative_path}"
            ) from error
        if not candidate.is_file():
            raise ValueError(f"Missing Tianyi asset: {relative_path}")


def demo_tianyi_asset_helper() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = AppSettings()
    report = TianyiAssetHelper(asset_dir=settings.tianyi_asset_dir).validate()
    LOGGER.info("Validated Tianyi assets: %s", report)


def main() -> None:
    demo_tianyi_asset_helper()


if __name__ == "__main__":
    main()
