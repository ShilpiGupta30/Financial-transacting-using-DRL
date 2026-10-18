from fastapi import APIRouter, HTTPException

from ..schemas import ErrorBody, ErrorResponse, SimulationRequest

router = APIRouter()

SIMULATION_UNAVAILABLE_REASON = (
    "The existing C++ executable only prints a startup message and has no CLI or "
    "structured simulation output. A minimal C++ simulation CLI/JSON output change "
    "would be needed before this endpoint can run simulations."
)


@router.post(
    "/simulations/run",
    status_code=200,
    responses={501: {"model": ErrorResponse}},
)
def run_simulation(request: SimulationRequest) -> dict:
    """Keep the contract stable while honestly reporting the unavailable runner."""
    del request
    raise HTTPException(
        status_code=501,
        detail=ErrorBody(
            code="NOT_IMPLEMENTED",
            message=SIMULATION_UNAVAILABLE_REASON,
            details={"simulation_run": False},
        ).model_dump(),
    )


@router.get(
    "/simulations/latest",
    responses={404: {"model": ErrorResponse}},
)
def latest_simulation() -> dict:
    """No real simulation results are persisted or cached by this MVP."""
    raise HTTPException(
        status_code=404,
        detail=ErrorBody(
            code="NO_SIMULATION_RESULTS",
            message="No simulation result is available.",
            details=None,
        ).model_dump(),
    )
