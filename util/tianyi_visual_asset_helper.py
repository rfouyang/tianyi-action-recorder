"""Validate the generated Tianyi render-only MJCF and optimized meshes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco


@dataclass(frozen=True, slots=True)
class TianyiVisualAssetReport:
    mesh_count: int
    source_triangle_count: int
    optimized_triangle_count: int
    geom_count: int

    @property
    def triangle_reduction_ratio(self) -> float:
        return 1.0 - (self.optimized_triangle_count / self.source_triangle_count)


class TianyiVisualAssetHelper:
    """Verify reproducibility and integrity of the derived visualization model."""

    EXPECTED_MESH_COUNT = 25

    def __init__(self, *, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self.source_mjcf_path = asset_dir / "tianyi2_pos.xml"
        self.visual_mjcf_path = asset_dir / "tianyi2_visual.xml"
        self.metadata_path = asset_dir / "visual_model_metadata.json"

    def validate(self) -> TianyiVisualAssetReport:
        metadata = self.load_metadata()
        self._validate_file_hash(
            path=self.source_mjcf_path,
            expected=metadata.get("source_mjcf_sha256"),
        )
        self._validate_file_hash(
            path=self.visual_mjcf_path,
            expected=metadata.get("visual_mjcf_sha256"),
        )

        mesh_records = metadata.get("meshes")
        if not isinstance(mesh_records, list) or len(mesh_records) != self.EXPECTED_MESH_COUNT:
            raise ValueError(
                f"Visual metadata must describe exactly {self.EXPECTED_MESH_COUNT} meshes"
            )
        for record in mesh_records:
            if not isinstance(record, dict):
                raise ValueError("Each visual mesh metadata entry must be an object")
            output_name = record.get("output")
            if not isinstance(output_name, str) or not output_name:
                raise ValueError("Visual mesh metadata entry lacks an output filename")
            source_name = record.get("source")
            if not isinstance(source_name, str) or not source_name:
                raise ValueError("Visual mesh metadata entry lacks a source filename")
            self._validate_file_hash(
                path=self.asset_dir / "meshes" / source_name,
                expected=record.get("source_sha256"),
            )
            self._validate_file_hash(
                path=self.asset_dir / "meshes_visual_optimized" / output_name,
                expected=record.get("sha256"),
            )

        source_triangles = metadata.get("source_triangle_count")
        optimized_triangles = metadata.get("optimized_triangle_count")
        if not isinstance(source_triangles, int) or source_triangles <= 0:
            raise ValueError("Visual metadata source_triangle_count must be positive")
        if not isinstance(optimized_triangles, int) or optimized_triangles <= 0:
            raise ValueError("Visual metadata optimized_triangle_count must be positive")
        if optimized_triangles >= source_triangles:
            raise ValueError("Optimized visual meshes must contain fewer triangles than source")

        model = mujoco.MjModel.from_xml_path(str(self.visual_mjcf_path))
        if model.nmesh != self.EXPECTED_MESH_COUNT:
            raise ValueError(
                f"Visual MJCF has {model.nmesh} meshes; expected {self.EXPECTED_MESH_COUNT}"
            )
        if model.nq != 21 or model.nv != 21 or model.nu != 21:
            raise ValueError("Visual MJCF must preserve the fixed-base 21-joint model contract")
        return TianyiVisualAssetReport(
            mesh_count=model.nmesh,
            source_triangle_count=source_triangles,
            optimized_triangle_count=optimized_triangles,
            geom_count=model.ngeom,
        )

    def load_metadata(self) -> dict[str, object]:
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Visual model metadata root must be an object")
        return payload

    @staticmethod
    def _validate_file_hash(*, path: Path, expected: object) -> None:
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"Visual model metadata lacks a SHA-256 for {path.name}")
        if not path.is_file():
            raise ValueError(f"Missing generated visual asset: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Generated visual asset hash mismatch for {path.name}")
