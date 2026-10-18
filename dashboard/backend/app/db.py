"""Read-only MySQL access for dashboard API endpoints."""

from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from .config import Settings, get_settings

VALID_STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "ROLLED_BACK")
SORT_COLUMNS = {
    "TransactionID": "TransactionID",
    "TransactionType": "TransactionType",
    "SourceAccountID": "SourceAccountID",
    "DestinationAccountID": "DestinationAccountID",
    "Amount": "Amount",
    "Status": "Status",
    "Priority": "Priority",
    "CreatedAt": "CreatedAt",
    "CompletedAt": "CompletedAt",
}
TRANSACTION_COLUMNS = (
    "TransactionID, TransactionType, SourceAccountID, DestinationAccountID, "
    "Amount, Status, Priority, CreatedAt, CompletedAt"
)


class DatabaseUnavailable(Exception):
    """Raised without exposing driver messages or connection details."""


class MySQLReader:
    """Opens short-lived MySQL connections and performs SELECT statements only."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _connect(self):
        if not self.settings.db_user or self.settings.db_password is None:
            raise DatabaseUnavailable("Database credentials are not configured.")
        try:
            return pymysql.connect(
                host=self.settings.db_host,
                port=self.settings.db_port,
                user=self.settings.db_user,
                password=self.settings.db_password,
                database=self.settings.db_name,
                charset="utf8mb4",
                cursorclass=DictCursor,
                connect_timeout=3,
                read_timeout=5,
                write_timeout=5,
                autocommit=True,
            )
        except Exception:
            raise DatabaseUnavailable("Could not connect to the configured database.") from None

    def ping(self) -> None:
        connection = self._connect()
        connection.close()

    def list_transactions(
        self, status: str | None, limit: int, offset: int, sort: str, order: str
    ) -> tuple[list[dict[str, Any]], int]:
        if sort not in SORT_COLUMNS:
            raise ValueError("Unsupported sort column")
        column = SORT_COLUMNS[sort]
        direction = "ASC" if order == "asc" else "DESC"
        where = " WHERE Status = %s" if status is not None else ""
        values: tuple[Any, ...] = (status,) if status is not None else ()
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS total FROM Transactions" + where, values)
                total = int(cursor.fetchone()["total"])
                query = (
                    f"SELECT {TRANSACTION_COLUMNS} FROM Transactions{where} "
                    f"ORDER BY {column} {direction} LIMIT %s OFFSET %s"
                )
                cursor.execute(query, values + (limit, offset))
                rows = list(cursor.fetchall())
            return rows, total
        except Exception:
            raise DatabaseUnavailable("Could not read transaction records.") from None
        finally:
            connection.close()

    def get_transaction(self, transaction_id: int) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {TRANSACTION_COLUMNS} FROM Transactions WHERE TransactionID = %s",
                    (transaction_id,),
                )
                return cursor.fetchone()
        except Exception:
            raise DatabaseUnavailable("Could not read the transaction record.") from None
        finally:
            connection.close()

    def transaction_summary(self) -> dict[str, int]:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT Status, COUNT(*) AS total FROM Transactions GROUP BY Status")
                results = {row["Status"]: int(row["total"]) for row in cursor.fetchall()}
            return {status: results.get(status, 0) for status in VALID_STATUSES}
        except Exception:
            raise DatabaseUnavailable("Could not summarize transaction records.") from None
        finally:
            connection.close()


def get_database() -> MySQLReader:
    """FastAPI dependency, replaceable with a test double."""
    return MySQLReader()
