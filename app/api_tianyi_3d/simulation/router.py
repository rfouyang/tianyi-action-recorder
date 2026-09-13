"""REST and WebSocket routes for the shared Tianyi MuJoCo state."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api_tianyi_3d.dependencies import get_application
from app.api_tianyi_3d.simulation.schemas import (
    JointPositionCommand,
    SimulationStateResponse,
)
from app.api_tianyi_3d.simulation.websocket import simulation_websocket
from app.application import TianyiApplication

router = APIRouter(prefix="/simulation", tags=["simulation"])
router.add_api_websocket_route("/ws", simulation_websocket)


@router.get("/state", response_model=SimulationStateResponse)
async def get_state(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> SimulationStateResponse:
    """Return the latest complete fixed-base MuJoCo state."""
    return _response(application)


@router.put("/joints", response_model=SimulationStateResponse)
async def update_joints(
    command: JointPositionCommand,
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> SimulationStateResponse:
    """Apply safe, model-limited head or arm positions to simulation only."""
    try:
        application.simulation.update_joint_positions(command.joint_positions)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return _response(application)


@router.post("/reset", response_model=SimulationStateResponse)
async def reset(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> SimulationStateResponse:
    """Restore the configured base pose and locked calibration state."""
    application.simulation.reset_to_default()
    return _response(application)


def _response(application: TianyiApplication) -> SimulationStateResponse:
    return SimulationStateResponse.from_snapshot(
        application.simulation.snapshot(),
        locked_joint_positions=application.simulation.calibration_joint_positions(),
    )
