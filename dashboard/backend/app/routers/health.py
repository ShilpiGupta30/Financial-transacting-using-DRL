from datetime import datetime, timezone

from fastapi import APIRouter

from ..config import get_settings
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return liveness information without making a database connection."""
    return HealthResponse(
        status="ok",
        service="irm-dashboard-backend",
        version=get_settings().version,
        time_utc=datetime.now(timezone.utc),
    )
