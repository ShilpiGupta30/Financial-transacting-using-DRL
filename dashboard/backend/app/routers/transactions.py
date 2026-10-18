from fastapi import APIRouter, Depends, HTTPException, Query

from ..db import DatabaseUnavailable, MySQLReader, VALID_STATUSES, get_database
from ..schemas import (
    ErrorBody,
    ErrorResponse,
    TransactionListResponse,
    TransactionRecord,
    TransactionSummary,
)

router = APIRouter()


def _transaction_response(row: dict) -> TransactionRecord:
    values = dict(row)
    # Preserve MySQL DECIMAL precision in JSON rather than converting to float.
    values["Amount"] = str(values["Amount"])
    return TransactionRecord.model_validate(values)


def _db_failure() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=ErrorBody(
            code="DB_UNAVAILABLE",
            message="Transaction data is currently unavailable.",
            details=None,
        ).model_dump(),
    )


@router.get("/transactions/summary", response_model=TransactionSummary)
def transaction_summary(
    database: MySQLReader = Depends(get_database),
) -> TransactionSummary:
    try:
        by_status = database.transaction_summary()
    except DatabaseUnavailable:
        raise _db_failure()
    return TransactionSummary(total=sum(by_status.values()), by_status=by_status)


@router.get(
    "/transactions",
    response_model=TransactionListResponse,
    responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def list_transactions(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    sort: str = Query(default="TransactionID"),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
    database: MySQLReader = Depends(get_database),
) -> TransactionListResponse:
    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=ErrorBody(
                code="VALIDATION_ERROR",
                message="status must be one of the defined transaction statuses.",
                details={"allowed": list(VALID_STATUSES)},
            ).model_dump(),
        )
    if sort not in {"TransactionID", "TransactionType", "SourceAccountID",
                    "DestinationAccountID", "Amount", "Status", "Priority",
                    "CreatedAt", "CompletedAt"}:
        raise HTTPException(
            status_code=422,
            detail=ErrorBody(
                code="VALIDATION_ERROR",
                message="sort must name a transaction column.",
                details=None,
            ).model_dump(),
        )
    try:
        rows, total = database.list_transactions(status, limit, offset, sort, order)
    except DatabaseUnavailable:
        raise _db_failure()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=ErrorBody("VALIDATION_ERROR", "sort must name a transaction column.").model_dump(),
        )
    return TransactionListResponse(
        items=[_transaction_response(row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/transactions/{transaction_id}",
    response_model=TransactionRecord,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def get_transaction(
    transaction_id: int,
    database: MySQLReader = Depends(get_database),
) -> TransactionRecord:
    try:
        row = database.get_transaction(transaction_id)
    except DatabaseUnavailable:
        raise _db_failure()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorBody(
                code="NOT_FOUND",
                message="Transaction was not found.",
                details={"transaction_id": transaction_id},
            ).model_dump(),
        )
    return _transaction_response(row)
