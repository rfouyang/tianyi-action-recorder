"""Construct the shared capabilities used by API and UI entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from component.action import ActionPlaybackService, ActionService
from component.pose import PoseService, PoseType, TianyiJointSchema
from component.retarget import PoseRetargetService
from component.simulation import SimulationService
from component.visualization import TianyiRenderService
from config.settings import AppSettings
from util.g1_pose_adapter import G1PoseAdapter
from util.numpy_archive_helper import NumpyArchiveHelper
from util.pose_file_helper import PoseFileHelper
from util.tianyi_asset_helper import TianyiAssetHelper
from util.tianyi_visual_asset_helper import (
    TianyiVisualAssetHelper,
    TianyiVisualAssetReport,
)
from util.tianyi_visual_urdf_helper import (
    TianyiVisualUrdfHelper,
    TianyiVisualUrdfReport,
)


@dataclass(slots=True)
class TianyiApplication:
    settings: AppSettings
    joint_schema: TianyiJointSchema
    simulation: SimulationService
    pose_service: PoseService
    pose_retargeting: PoseRetargetService
    action_service: ActionService
    action_playback: ActionPlaybackService
    renderer: TianyiRenderService
    visual_asset_report: TianyiVisualAssetReport
    visual_urdf_report: TianyiVisualUrdfReport

    @classmethod
    def build(
        cls,
        *,
        settings: AppSettings | None = None,
        pose_dir: Path | None = None,
        action_definition_dir: Path | None = None,
        action_trajectory_dir: Path | None = None,
    ) -> TianyiApplication:
        resolved_settings = settings or AppSettings()
        assets = TianyiAssetHelper(asset_dir=resolved_settings.tianyi_asset_dir)
        schema = TianyiJointSchema(asset_helper=assets)
        visual_assets = TianyiVisualAssetHelper(asset_dir=resolved_settings.tianyi_asset_dir)
        visual_report = visual_assets.validate()
        visual_urdf_report = TianyiVisualUrdfHelper(
            asset_dir=resolved_settings.tianyi_asset_dir
        ).validate()
        pose_service = PoseService(
            schema=schema,
            pose_dir=pose_dir or resolved_settings.pose_dir,
            file_helper=PoseFileHelper(),
        )
        try:
            default_base_pose = pose_service.load_pose(
                pose_type=PoseType.BASE,
                name=resolved_settings.default_base_pose_name,
            )
        except FileNotFoundError:
            canonical_pose_service = PoseService(
                schema=schema,
                pose_dir=resolved_settings.pose_dir,
                file_helper=PoseFileHelper(),
            )
            default_base_pose = canonical_pose_service.load_pose(
                pose_type=PoseType.BASE,
                name=resolved_settings.default_base_pose_name,
            )
            pose_service.save_pose(pose=default_base_pose)
        simulation = SimulationService(
            schema=schema,
            mjcf_path=assets.mjcf_path,
            default_pose=default_base_pose,
        )
        pose_retargeting = PoseRetargetService(
            schema=schema,
            g1_poses=G1PoseAdapter(
                asset_dir=resolved_settings.g1_asset_dir
            ),
            tianyi_mjcf_path=assets.mjcf_path,
            tianyi_default_values=default_base_pose.joint_values,
            tianyi_locked_values=schema.locked_values(),
        )
        action_service = ActionService(
            schema=schema,
            pose_service=pose_service,
            definition_dir=action_definition_dir or resolved_settings.action_definition_dir,
            trajectory_dir=action_trajectory_dir or resolved_settings.action_trajectory_dir,
            file_helper=PoseFileHelper(),
            archive_helper=NumpyArchiveHelper(),
        )
        return cls(
            settings=resolved_settings,
            joint_schema=schema,
            simulation=simulation,
            pose_service=pose_service,
            pose_retargeting=pose_retargeting,
            action_service=action_service,
            action_playback=ActionPlaybackService(simulation=simulation),
            renderer=TianyiRenderService(
                schema=schema,
                mjcf_path=visual_assets.visual_mjcf_path,
            ),
            visual_asset_report=visual_report,
            visual_urdf_report=visual_urdf_report,
        )

    def close(self) -> None:
        self.action_playback.close()
        self.renderer.close()
