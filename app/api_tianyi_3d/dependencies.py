"""FastAPI dependency providers for the Tianyi simulation API."""

from __future__ import annotations

from typing import cast

from fastapi import Request, WebSocket

from app.application import TianyiApplication


async def get_application(request: Request) -> TianyiApplication:
    """Return the process-wide application assembled by the combined host."""
    return cast(TianyiApplication, request.app.state.tianyi)


def get_websocket_application(websocket: WebSocket) -> TianyiApplication:
    """Return the shared application for one API WebSocket connection."""
    return cast(TianyiApplication, websocket.app.state.tianyi)
