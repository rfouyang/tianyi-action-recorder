"""Browser-only realtime transport for the Tianyi 3D UI."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast

from fastapi import WebSocket, WebSocketDisconnect

from app.application import TianyiApplication
from component.simulation import SimulationSnapshot


async def ui_simulation_websocket(websocket: WebSocket) -> None:
    """Stream MuJoCo state and accept UI target changes."""
    await websocket.accept()
    application = cast(TianyiApplication, websocket.app.state.tianyi)
    sent_revision = -1
    try:
        while True:
            snapshot = application.simulation.snapshot()
            if snapshot.revision != sent_revision:
                await websocket.send_json(_snapshot_message(snapshot))
                sent_revision = snapshot.revision

            try:
                payload = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            try:
                snapshot = _apply_simulation_update(application, payload)
                await websocket.send_json(_snapshot_message(snapshot))
                sent_revision = snapshot.revision
            except ValueError as error:
                await websocket.send_json({"type": "error", "detail": str(error)})
    except WebSocketDisconnect:
        return


def _apply_simulation_update(
    application: TianyiApplication,
    payload: object,
) -> SimulationSnapshot:
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Expected one joint position command")
    if "joint_positions" in payload:
        return application.simulation.update_joint_positions(
            _joint_positions(payload["joint_positions"])
        )
    if "calibration_joint_positions" in payload:
        return application.simulation.update_calibration_joint_positions(
            _joint_positions(payload["calibration_joint_positions"])
        )
    raise ValueError("Expected joint_positions or calibration_joint_positions")


def _joint_positions(joint_positions: object) -> Mapping[str, object]:
    if not isinstance(joint_positions, dict) or not joint_positions:
        raise ValueError("At least one joint position is required")
    return joint_positions


def _snapshot_message(snapshot: SimulationSnapshot) -> dict[str, object]:
    return {
        "type": "simulation_state",
        "revision": snapshot.revision,
        "updated_at": snapshot.updated_at,
        "joint_positions": snapshot.joint_position_map(),
        "joint_position_targets": snapshot.joint_position_target_map(),
        "base_position": snapshot.base_position,
        "base_wxyz": snapshot.base_wxyz,
    }
