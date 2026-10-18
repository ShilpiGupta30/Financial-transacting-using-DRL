from datetime import datetime, timezone
from decimal import Decimal

from fastapi.testclient import TestClient

from dashboard.backend.app.db import DatabaseUnavailable, get_database
from dashboard.backend.app.main import app

client = TestClient(app)


def sample_transaction(transaction_id: int = 7, status: str = "PENDING") -> dict:
    return {
        "TransactionID": transaction_id,
        "TransactionType": "DEPOSIT",
        "SourceAccountID": 1,
        "DestinationAccountID": None,
        "Amount": Decimal("125.50"),
        "Status": status,
        "Priority": 2,
        "CreatedAt": datetime(2024, 1, 2, tzinfo=timezone.utc),
        "CompletedAt": None,
    }


class FakeDatabase:
    def ping(self):
        return None

    def list_transactions(self, status, limit, offset, sort, order):
        rows = [sample_transaction(7, "PENDING"), sample_transaction(8, "COMPLETED")]
        if status:
            rows = [row for row in rows if row["Status"] == status]
        return rows[offset:offset + limit], len(rows)

    def get_transaction(self, transaction_id):
        return sample_transaction(transaction_id) if transaction_id == 7 else None

    def transaction_summary(self):
        return {"PENDING": 1, "RUNNING": 0, "COMPLETED": 1, "FAILED": 0, "ROLLED_BACK": 0}


class OfflineDatabase:
    def ping(self):
        raise DatabaseUnavailable("safe internal reason")

    def list_transactions(self, status, limit, offset, sort, order):
        raise DatabaseUnavailable("safe internal reason")


def setup_function():
    app.dependency_overrides[get_database] = lambda: FakeDatabase()


def teardown_function():
    app.dependency_overrides.clear()


def test_health_does_not_require_database():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "irm-dashboard-backend"
    assert body["time_utc"].endswith("Z") or "+00:00" in body["time_utc"]


def test_system_status_returns_200_when_database_is_down():
    app.dependency_overrides[get_database] = lambda: OfflineDatabase()
    response = client.get("/api/system/status")
    assert response.status_code == 200
    body = response.json()
    assert body["database"]["connected"] is False
    assert body["database"]["error"]
    assert body["capabilities"]["transactions_read"] is False
    assert body["capabilities"]["scheduler_list"] is True
    serialized = response.text.lower()
    assert "password" not in serialized
    assert "user" not in serialized


def test_transaction_list_filter_pagination_and_decimal_string():
    response = client.get("/api/transactions?status=PENDING&limit=1&offset=0")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["TransactionID"] == 7
    assert body["items"][0]["Amount"] == "125.50"
    assert body["items"][0]["DestinationAccountID"] is None


def test_transaction_list_rejects_invalid_status_limit_and_sort():
    for query in (
        "/api/transactions?status=INVALID",
        "/api/transactions?limit=201",
        "/api/transactions?sort=Status%20DESC",
        "/api/transactions?order=sideways",
    ):
        response = client.get(query)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_transaction_detail_and_not_found_error_shape():
    found = client.get("/api/transactions/7")
    missing = client.get("/api/transactions/999")
    assert found.status_code == 200
    assert found.json()["TransactionID"] == 7
    assert missing.status_code == 404
    assert set(missing.json()) == {"error"}
    assert missing.json()["error"]["code"] == "NOT_FOUND"


def test_summary_aggregates_database_status_counts():
    response = client.get("/api/transactions/summary")
    assert response.status_code == 200
    assert response.json() == {
        "total": 2,
        "by_status": {
            "PENDING": 1,
            "RUNNING": 0,
            "COMPLETED": 1,
            "FAILED": 0,
            "ROLLED_BACK": 0,
        },
    }


def test_scheduler_list_matches_supported_algorithms_and_reports_dqn_unavailable():
    response = client.get("/api/schedulers")
    assert response.status_code == 200
    items = {item["id"]: item for item in response.json()["items"]}
    assert items["fcfs"]["available"] is True
    assert items["round_robin"]["supports_time_quantum"] is True
    assert items["priority"]["available"] is True
    assert items["dqn"]["available"] is False
    assert items["dqn"]["reason"]


def test_simulation_validates_body_then_reports_unavailable():
    invalid_unknown_algorithm = client.post(
        "/api/simulations/run", json={"algorithm": "sjf"}
    )
    missing_quantum = client.post(
        "/api/simulations/run", json={"algorithm": "round_robin"}
    )
    invalid_quantum = client.post(
        "/api/simulations/run", json={"algorithm": "round_robin", "time_quantum": 0}
    )
    unavailable = client.post("/api/simulations/run", json={"algorithm": "fcfs"})
    assert invalid_unknown_algorithm.status_code == 422
    assert missing_quantum.status_code == 422
    assert invalid_quantum.status_code == 422
    assert unavailable.status_code == 501
    assert unavailable.json()["error"]["code"] == "NOT_IMPLEMENTED"


def test_latest_simulation_reports_no_real_cached_result():
    response = client.get("/api/simulations/latest")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NO_SIMULATION_RESULTS"


def test_all_non_success_responses_use_standard_error_envelope():
    response = client.get("/api/transactions?limit=0")
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
    assert {"code", "message", "details"} <= set(response.json()["error"])


def test_unavailable_database_reads_return_safe_503():
    app.dependency_overrides[get_database] = lambda: OfflineDatabase()
    response = client.get("/api/transactions")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DB_UNAVAILABLE"
    assert "safe internal reason" not in response.text
