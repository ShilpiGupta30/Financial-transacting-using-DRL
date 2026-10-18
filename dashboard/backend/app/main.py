"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_settings
from .routers import health, schedulers, simulations, system, transactions
from .schemas import ErrorBody

app = FastAPI(title="IRM Dashboard Backend", version=get_settings().version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(system.router, prefix="/api", tags=["system"])
app.include_router(transactions.router, prefix="/api", tags=["transactions"])
app.include_router(schedulers.router, prefix="/api", tags=["schedulers"])
app.include_router(simulations.router, prefix="/api", tags=["simulations"])


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exception: RequestValidationError
) -> JSONResponse:
    del request
    details = [
        {"location": [str(part) for part in error.get("loc", ())],
         "message": error.get("msg", "Invalid value")}
        for error in exception.errors()
    ]
    body = ErrorBody(
        code="VALIDATION_ERROR",
        message="Request validation failed.",
        details={"errors": details},
    )
    return JSONResponse(status_code=422, content={"error": body.model_dump()})


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exception: StarletteHTTPException) -> JSONResponse:
    del request
    detail = exception.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        body = ErrorBody.model_validate(detail)
    else:
        code = "NOT_FOUND" if exception.status_code == 404 else "HTTP_ERROR"
        message = "Resource was not found." if exception.status_code == 404 else "Request could not be completed."
        body = ErrorBody(code=code, message=message)
    return JSONResponse(
        status_code=exception.status_code,
        content={"error": body.model_dump()},
        headers=exception.headers,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exception: Exception) -> JSONResponse:
    del request, exception
    body = ErrorBody(code="INTERNAL_ERROR", message="An unexpected server error occurred.")
    return JSONResponse(status_code=500, content={"error": body.model_dump()})
