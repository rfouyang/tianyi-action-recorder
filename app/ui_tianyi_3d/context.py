"""UI-only dependencies for the Tianyi 3D presentation surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from fastapi import Request

from app.application import TianyiApplication
from app.ui_tianyi_3d.viser_manager import ViserManager


@dataclass(frozen=True, slots=True)
class UIContext:
    """Expose shared capabilities without making the UI depend on the REST API."""

    app: TianyiApplication
    viser: ViserManager

    @classmethod
    def from_request(cls, request: Request) -> UIContext:
        return cls(
            app=cast(TianyiApplication, request.app.state.tianyi),
            viser=cast(ViserManager, request.app.state.viser),
        )
