"""Realtime Tianyi simulation telemetry and target transport."""

from __future__ import annotations

import asyncio

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api_tianyi_3d.dependencies import get_websocket_application
from app.api_tianyi_3d.simulation.schemas import (
    JointPositionCommand,
    SimulationStateResponse,
)
from app.application import TianyiApplication
from component.simulation import SimulationSnapshot


async def simulation_websocket(websocket: WebSocket) -> None:
    """Stream state changes and accept simulation-only head or arm targets."""
    await websocket.accept()
    application = get_websocket_application(websocket)
    sent_revision = -1
    try:
        while True:
            snapshot = application.simulation.snapshot()
            if snapshot.revision != sent_revision:
                await websocket.send_json(
                    _state_message(application=application, snapshot=snapshot)
                )
                sent_revision = snapshot.revision

            try:
                payload = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue

            try:
                command = JointPositionCommand.model_validate(payload)
                snapshot = application.simulation.update_joint_positions(
                    command.joint_positions
                )
                await websocket.send_json(
                    _state_message(application=application, snapshot=snapshot)
                )
                sent_revision = snapshot.revision
            except (ValidationError, ValueError) as error:
                await websocket.send_json({"type": "error", "detail": str(error)})
    except WebSocketDisconnect:
        return


def _state_message(
    *,
    application: TianyiApplication,
    snapshot: SimulationSnapshot,
) -> dict[str, object]:
    response = SimulationStateResponse.from_snapshot(
        snapshot,
        locked_joint_positions=application.simulation.calibration_joint_positions(),
    )
    return {"type": "simulation_state", **response.model_dump()}
