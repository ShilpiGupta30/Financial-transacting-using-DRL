"""Persistent simulation history stored separately from financial tables."""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from .config import Settings, get_settings


class HistoryStorageUnavailable(Exception):
    """History tables could not be read or written."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported history JSON value: {type(value).__name__}")


def _decode_json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value


def _timestamp(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, (datetime, date)) else value


class MySQLSimulationHistoryStore:
    """Write and read the additive SimulationHistory tables only."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _connect(self):
        settings = self.settings
        if not settings.db_user or settings.db_password is None:
            raise HistoryStorageUnavailable("Database credentials are not configured.")
        try:
            return pymysql.connect(
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                database=settings.db_name,
                charset="utf8mb4",
                cursorclass=DictCursor,
                connect_timeout=3,
                read_timeout=5,
                write_timeout=5,
                autocommit=False,
            )
        except Exception:
            raise HistoryStorageUnavailable("Could not connect to the history database.") from None

    def create(
        self,
        *,
        algorithm: str,
        parameters: dict[str, Any],
        workload: list[dict[str, Any]],
        summary: dict[str, Any],
        results: list[dict[str, Any]],
        comparison_id: str | None = None,
    ) -> int:
        """Atomically insert one completed run and its per-transaction metrics."""
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO SimulationHistory (
                        ComparisonID, Algorithm, Parameters, InputWorkload,
                        TransactionCount, Makespan, AverageWaitingTime,
                        AverageTurnaroundTime, AverageResponseTime,
                        CpuUtilization, Throughput
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        comparison_id,
                        algorithm,
                        json.dumps(parameters, default=_json_value),
                        json.dumps(workload, default=_json_value),
                        summary["transactionCount"],
                        summary["makespan"],
                        summary["averageWaitingTime"],
                        summary["averageTurnaroundTime"],
                        summary["averageResponseTime"],
                        summary.get("cpuUtilization"),
                        summary.get("throughput"),
                    ),
                )
                simulation_id = int(cursor.lastrowid)
                cursor.executemany(
                    """INSERT INTO SimulationHistoryResults (
                        SimulationID, TransactionID, ProcessID, BurstTime,
                        CompletionTime, WaitingTime, TurnaroundTime, ResponseTime
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (
                            simulation_id,
                            row["transactionID"],
                            row["processID"],
                            row["burstTime"],
                            row["completionTime"],
                            row["waitingTime"],
                            row["turnaroundTime"],
                            row["responseTime"],
                        )
                        for row in results
                    ],
                )
            connection.commit()
            return simulation_id
        except HistoryStorageUnavailable:
            connection.rollback()
            raise
        except Exception:
            connection.rollback()
            raise HistoryStorageUnavailable(
                "Could not save simulation history. Confirm the additive history migration is applied."
            ) from None
        finally:
            connection.close()

    def list_recent(
        self, *, algorithm: str | None, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                where = " WHERE Algorithm = %s" if algorithm else ""
                args: tuple[Any, ...] = (algorithm,) if algorithm else ()
                cursor.execute("SELECT COUNT(*) AS total FROM SimulationHistory" + where, args)
                total = int(cursor.fetchone()["total"])
                cursor.execute(
                    """SELECT SimulationID, ComparisonID, CreatedAt, Algorithm,
                        Parameters, TransactionCount, Makespan, AverageWaitingTime,
                        AverageTurnaroundTime, AverageResponseTime, CpuUtilization, Throughput
                    FROM SimulationHistory""" + where +
                    " ORDER BY CreatedAt DESC, SimulationID DESC LIMIT %s OFFSET %s",
                    args + (limit, offset),
                )
                rows = list(cursor.fetchall())
            return [self._summary_record(row) for row in rows], total
        except Exception:
            raise HistoryStorageUnavailable("Could not read simulation history.") from None
        finally:
            connection.close()

    def get(self, simulation_id: int) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT SimulationID, ComparisonID, CreatedAt, Algorithm,
                        Parameters, InputWorkload, TransactionCount, Makespan,
                        AverageWaitingTime, AverageTurnaroundTime, AverageResponseTime,
                        CpuUtilization, Throughput
                    FROM SimulationHistory WHERE SimulationID = %s""",
                    (simulation_id,),
                )
                header = cursor.fetchone()
                if header is None:
                    return None
                cursor.execute(
                    """SELECT TransactionID, ProcessID, BurstTime, CompletionTime,
                        WaitingTime, TurnaroundTime, ResponseTime
                    FROM SimulationHistoryResults WHERE SimulationID = %s
                    ORDER BY TransactionID ASC""",
                    (simulation_id,),
                )
                results = list(cursor.fetchall())
            summary = self._summary_record(header)
            summary["inputWorkload"] = _decode_json(header["InputWorkload"])
            full_parameters = _decode_json(header["Parameters"])
            summary["parameters"] = full_parameters
            summary["timeline"] = full_parameters.get("executionTimeline", [])
            summary["results"] = [
                {
                    "transactionID": row["TransactionID"],
                    "processID": row["ProcessID"],
                    "burstTime": row["BurstTime"],
                    "completionTime": row["CompletionTime"],
                    "waitingTime": row["WaitingTime"],
                    "turnaroundTime": row["TurnaroundTime"],
                    "responseTime": row["ResponseTime"],
                    "state": "COMPLETED",
                }
                for row in results
            ]
            return summary
        except Exception:
            raise HistoryStorageUnavailable("Could not read the simulation detail.") from None
        finally:
            connection.close()

    def latest(self) -> dict[str, Any] | None:
        records, _total = self.list_recent(algorithm=None, limit=1, offset=0)
        return self.get(records[0]["simulationId"]) if records else None

    @staticmethod
    def _summary_record(row: dict[str, Any]) -> dict[str, Any]:
        parameters = _decode_json(row["Parameters"])
        if isinstance(parameters, dict):
            parameters = {key: value for key, value in parameters.items()
                          if key != "executionTimeline"}
        return {
            "simulationId": int(row["SimulationID"]),
            "comparisonId": row.get("ComparisonID"),
            "createdAt": _timestamp(row["CreatedAt"]),
            "algorithm": row["Algorithm"],
            "parameters": parameters,
            "transactionCount": int(row["TransactionCount"]),
            "makespan": int(row["Makespan"]),
            "averageWaitingTime": float(row["AverageWaitingTime"]),
            "averageTurnaroundTime": float(row["AverageTurnaroundTime"]),
            "averageResponseTime": float(row["AverageResponseTime"]),
            "cpuUtilization": (
                float(row["CpuUtilization"]) if row.get("CpuUtilization") is not None else None
            ),
            "throughput": float(row["Throughput"]) if row.get("Throughput") is not None else None,
        }


def get_simulation_history_store() -> MySQLSimulationHistoryStore:
    return MySQLSimulationHistoryStore()
