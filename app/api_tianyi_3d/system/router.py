"""System status endpoints for the Tianyi simulation API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api_tianyi_3d.dependencies import get_application
from app.api_tianyi_3d.system.schemas import HealthResponse
from app.application import TianyiApplication

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(
    application: Annotated[TianyiApplication, Depends(get_application)],
) -> HealthResponse:
    """Report that the shared application and fixed-base model are ready."""
    snapshot = application.simulation.snapshot()
    return HealthResponse(
        model_id=application.joint_schema.model_id,
        revision=snapshot.revision,
    )
