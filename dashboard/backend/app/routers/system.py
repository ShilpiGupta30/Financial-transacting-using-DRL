from fastapi import APIRouter, Depends

from ..config import get_settings
from ..db import DatabaseUnavailable, MySQLReader, get_database
from ..schemas import (
    BackendState,
    CapabilityState,
    Capabilities,
    ComponentState,
    DatabaseState,
    SystemStatusResponse,
)

router = APIRouter()


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status(database: MySQLReader = Depends(get_database)) -> SystemStatusResponse:
    settings = get_settings()
    connected = False
    database_error = None
    try:
        database.ping()
        connected = True
    except DatabaseUnavailable:
        database_error = "Database connection is unavailable or credentials are missing."

    executable_found = settings.cpp_engine_path.is_file()
    engine_reason = (
        None
        if executable_found
        else "The C++ executable was not found at the configured path."
    )
    if executable_found:
        engine_reason = (
            "The current executable only prints a startup message; it has no simulation CLI "
            "or structured simulation output."
        )
    return SystemStatusResponse(
        backend=BackendState(status="ok"),
        database=DatabaseState(
            connected=connected,
            name=settings.db_name,
            error=database_error,
        ),
        cpp_engine=ComponentState(
            available=False,
            executable_found=executable_found,
            reason=engine_reason,
        ),
        drl=CapabilityState(
            available=False,
            reason=(
                "No verified C++ to Python inference path or trained model artifact is available."
            ),
        ),
        capabilities=Capabilities(
            transactions_read=connected,
            scheduler_list=True,
            simulation_run=False,
            simulation_results=False,
            resource_metrics=False,
        ),
    )
