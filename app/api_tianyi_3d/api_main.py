"""Assemble domain routers for the Tianyi simulation API."""

from __future__ import annotations

from fastapi import APIRouter

from app.api_tianyi_3d.actions.router import router as actions_router
from app.api_tianyi_3d.poses.router import router as poses_router
from app.api_tianyi_3d.retargeting.router import router as retargeting_router
from app.api_tianyi_3d.simulation.router import router as simulation_router
from app.api_tianyi_3d.system.router import router as system_router

router = APIRouter(prefix="/api/tianyi")
router.include_router(system_router)
router.include_router(simulation_router)
router.include_router(poses_router)
router.include_router(retargeting_router)
router.include_router(actions_router)
