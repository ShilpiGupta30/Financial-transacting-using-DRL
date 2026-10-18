"""Pydantic response and request models for the documented API contract."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    time_utc: datetime


class BackendState(BaseModel):
    status: str


class DatabaseState(BaseModel):
    connected: bool
    name: str
    error: Optional[str]


class ComponentState(BaseModel):
    available: bool
    executable_found: bool
    reason: Optional[str] = None


class CapabilityState(BaseModel):
    available: bool
    reason: str


class Capabilities(BaseModel):
    transactions_read: bool
    scheduler_list: bool
    simulation_run: bool
    simulation_results: bool
    resource_metrics: bool


class SystemStatusResponse(BaseModel):
    backend: BackendState
    database: DatabaseState
    cpp_engine: ComponentState
    drl: CapabilityState
    capabilities: Capabilities


class TransactionRecord(BaseModel):
    """A Transactions row, retaining the column names from schema.sql."""

    TransactionID: int
    TransactionType: str
    SourceAccountID: int
    DestinationAccountID: Optional[int]
    Amount: str
    Status: str
    Priority: int
    CreatedAt: Optional[datetime]
    CompletedAt: Optional[datetime]

    @field_validator("Amount", mode="before")
    @classmethod
    def decimal_as_string(cls, value: object) -> str:
        if isinstance(value, Decimal):
            return format(value, "f")
        return str(value)


class TransactionListResponse(BaseModel):
    items: list[TransactionRecord]
    total: int
    limit: int
    offset: int


class TransactionSummary(BaseModel):
    total: int
    by_status: dict[str, int]


class SchedulerItem(BaseModel):
    id: str
    name: str
    type: str
    available: bool
    supports_time_quantum: bool
    description: str
    reason: Optional[str] = None


class SchedulerListResponse(BaseModel):
    items: list[SchedulerItem]


class SchedulingAlgorithm(str, Enum):
    FCFS = "fcfs"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: SchedulingAlgorithm
    time_quantum: Optional[int] = Field(default=None, ge=1, strict=True)
    limit: Optional[int] = Field(default=None, ge=1, le=200, strict=True)

    @model_validator(mode="after")
    def quantum_matches_algorithm(self) -> "SimulationRequest":
        if self.algorithm == SchedulingAlgorithm.ROUND_ROBIN and self.time_quantum is None:
            raise ValueError("time_quantum is required for round_robin")
        if self.algorithm != SchedulingAlgorithm.ROUND_ROBIN and self.time_quantum is not None:
            raise ValueError("time_quantum is only valid for round_robin")
        return self


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
