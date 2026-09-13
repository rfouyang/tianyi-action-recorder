"""Central paths for the Tianyi Action Recorder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Resolve project paths from one explicit project root."""

    project_root: Path = PROJECT_ROOT
    tianyi_3d_host: str = "127.0.0.1"
    tianyi_3d_port: int = 7900
    viser_host: str = "127.0.0.1"
    viser_port: int = 7901
    default_base_pose_name: str = "concierge_init"

    @property
    def asset_dir(self) -> Path:
        return self.project_root / "asset"

    @property
    def tianyi_asset_dir(self) -> Path:
        return self.asset_dir / "tianyi2"

    @property
    def tianyi_visual_mjcf_path(self) -> Path:
        return self.tianyi_asset_dir / "tianyi2_visual.xml"

    @property
    def tianyi_visual_metadata_path(self) -> Path:
        return self.tianyi_asset_dir / "visual_model_metadata.json"

    @property
    def tianyi_visual_urdf_path(self) -> Path:
        return self.tianyi_asset_dir / "tianyi2_visual_optimized.urdf"

    @property
    def tianyi_visual_urdf_metadata_path(self) -> Path:
        return self.tianyi_asset_dir / "visual_urdf_metadata.json"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def pose_dir(self) -> Path:
        return self.data_dir / "poses"

    @property
    def action_definition_dir(self) -> Path:
        return self.data_dir / "actions"

    @property
    def action_trajectory_dir(self) -> Path:
        return self.data_dir / "trajectories"

    @property
    def static_dir(self) -> Path:
        return self.project_root / "app" / "static"

    @property
    def g1_asset_dir(self) -> Path:
        """Pinned G1 kinematic assets used by the local retargeter."""
        return self.asset_dir / "g1"
