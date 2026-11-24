from decimal import Decimal
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app.main import app
from dashboard.backend.app import synthetic_data
from dashboard.backend.app.db import get_database
from dashboard.backend.app.history_store import get_simulation_history_store


class NoReadDatabase:
    def list_transactions(self, **_kwargs):
        raise AssertionError("synthetic path must not read MySQL")


class MemoryHistory:
    def create(self, **values):
        return 71


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic_data, "GENERATED_DATA_DIR", tmp_path / "synthetic")
    app.dependency_overrides[get_database] = NoReadDatabase
    app.dependency_overrides[get_simulation_history_store] = MemoryHistory
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def stable_payload(dataset):
    return dataset["accounts"], dataset["transactions"], dataset["scheduling_workload"]


def test_profiles_generate_valid_reproducible_data():
    for profile in ("normal", "peak", "high_contention", "custom"):
        first = synthetic_data.generate_dataset(dataset_type="combined", scenario=profile,
            count=100, seed=43, burst_min=1, burst_max=8)
        second = synthetic_data.generate_dataset(dataset_type="combined", scenario=profile,
            count=100, seed=43, burst_min=1, burst_max=8)
        assert stable_payload(first) == stable_payload(second)
        assert len(first["transactions"]) == len(first["scheduling_workload"]) == 100
        for row, process in zip(first["transactions"], first["scheduling_workload"]):
            assert row["TransactionID"] == process["transaction_id"]
            assert Decimal(row["Amount"]) > 0
            assert Decimal(row["Amount"]).as_tuple().exponent == -2
            assert row["SourceAccountID"] >= 1_000_000_000
            assert process["arrival_time"] >= 0 and 1 <= process["burst_time"] <= 8
            if row["TransactionType"] == "TRANSFER":
                assert row["DestinationAccountID"] != row["SourceAccountID"]
            else:
                assert row["DestinationAccountID"] is None
        if profile == "high_contention":
            assert all(row["arrival_time"] == 0 for row in first["scheduling_workload"])
            assert all(row["resource_requirements"] == ["CPU_IO"] for row in first["scheduling_workload"])


@pytest.mark.parametrize("count", [100, 500, 1000, 37])
def test_supported_dataset_sizes(count):
    data = synthetic_data.generate_dataset(dataset_type="scheduling", scenario="peak",
        count=count, seed=5)
    assert data["count"] == len(data["scheduling_workload"]) == count


def test_invalid_generation_and_safe_selected_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic_data, "GENERATED_DATA_DIR", tmp_path / "safe")
    with pytest.raises(ValueError):
        synthetic_data.generate_dataset(dataset_type="combined", scenario="normal", count=1001, seed=0)
    with pytest.raises(ValueError):
        synthetic_data.get_dataset("..")
    one = synthetic_data.save_dataset(synthetic_data.generate_dataset(
        dataset_type="combined", scenario="normal", count=4, seed=3))
    two = synthetic_data.save_dataset(synthetic_data.generate_dataset(
        dataset_type="combined", scenario="normal", count=4, seed=4))
    assert synthetic_data.delete_dataset(one["dataset_id"])
    assert synthetic_data.get_dataset(one["dataset_id"]) is None
    assert synthetic_data.get_dataset(two["dataset_id"]) is not None


def test_dataset_api_export_preview_and_clear(client):
    response = client.post("/api/synthetic-data/generate", json={
        "dataset_type": "combined", "scenario": "high_contention", "count": 100, "seed": 2,
        "burst_min": 1, "burst_max": 8,
    })
    assert response.status_code == 201
    payload = response.json()
    assert payload["label"] == "SYNTHETIC / DEMO DATA"
    assert len(payload["transactions_preview"]) == 20
    identifier = payload["dataset_id"]
    assert client.get(f"/api/synthetic-data/{identifier}/export?format=csv").status_code == 200
    assert client.get(f"/api/synthetic-data/{identifier}/export?format=json").status_code == 200
    assert client.delete(f"/api/synthetic-data/{identifier}").json()["deleted"] is True
    assert client.get(f"/api/synthetic-data/{identifier}").status_code == 404


def test_synthetic_simulation_uses_arrival_aware_cpp_contract(client, monkeypatch, tmp_path):
    executable = tmp_path / "engine.exe"
    executable.write_bytes(b"mock")
    monkeypatch.setenv("CPP_ENGINE_PATH", str(executable))
    dataset = synthetic_data.save_dataset(synthetic_data.generate_dataset(
        dataset_type="combined", scenario="peak", count=5, seed=2, burst_min=2, burst_max=2))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        parsed = [line.split(",") for line in kwargs["input"].splitlines()]
        assert all(len(row) == 8 for row in parsed)
        results = []
        current = 0
        for index, row in enumerate(parsed):
            txn, arrival, burst = int(row[0]), int(row[6]), int(row[7])
            current = max(current, arrival) + burst
            results.append({"transactionID": txn, "processID": index + 1,
                "completionTime": current, "waitingTime": current - arrival - burst,
                "turnaroundTime": current - arrival})
        output = json.dumps({"algorithm": command[2], "results": results})
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    from dashboard.backend.app.routers import simulations
    monkeypatch.setattr(simulations.subprocess, "run", fake_run)
    response = client.post("/api/simulations/run", json={
        "algorithm": "fcfs", "data_source": "synthetic", "synthetic_dataset_id": dataset["dataset_id"], "limit": 5,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dataSource"] == "synthetic"
    assert len(body["results"]) == 5 and len(body["timeline"]) == 5
    assert body["summary"]["transactionCount"] == 5
    assert calls[0][0][1:3] == ["schedule_workload", "fcfs"]


def test_synthetic_run_rejects_unavailable_dqn_contract(client):
    dataset = synthetic_data.save_dataset(synthetic_data.generate_dataset(
        dataset_type="combined", scenario="normal", count=10, seed=6))
    response = client.post("/api/simulations/run", json={
        "algorithm": "priority_free_dqn", "data_source": "synthetic",
        "synthetic_dataset_id": dataset["dataset_id"], "limit": 3, "burst_times": [1, 1, 1],
    })
    assert response.status_code == 422
    assert "zero arrivals" in response.json()["error"]["message"]


def test_synthetic_comparison_uses_same_arrival_burst_records(client, monkeypatch, tmp_path):
    executable = tmp_path / "comparison-engine.exe"
    executable.write_bytes(b"mock")
    monkeypatch.setenv("CPP_ENGINE_PATH", str(executable))
    dataset = synthetic_data.save_dataset(synthetic_data.generate_dataset(
        dataset_type="combined", scenario="high_contention", count=4, seed=8,
        burst_min=2, burst_max=2))
    from dashboard.backend.app.routers import simulations

    def fake_run(command, **kwargs):
        lines = [line.split(",") for line in kwargs["input"].splitlines()]
        now = 0
        results = []
        for index, fields in enumerate(lines):
            transaction_id, arrival, burst = int(fields[0]), int(fields[6]), int(fields[7])
            now = max(now, arrival) + burst
            results.append({"transactionID": transaction_id, "processID": index + 1,
                "completionTime": now, "waitingTime": now - arrival - burst,
                "turnaroundTime": now - arrival})
        return subprocess.CompletedProcess(command, 0,
            stdout=json.dumps({"algorithm": command[2], "results": results}), stderr="")

    monkeypatch.setattr(simulations.subprocess, "run", fake_run)
    response = client.post("/api/simulations/compare", json={
        "algorithms": ["fcfs", "priority"], "burst_times": [2, 2],
        "data_source": "synthetic", "synthetic_dataset_id": dataset["dataset_id"],
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dataSource"] == "synthetic"
    assert len(payload["results"]) == 2
    assert all(result["status"] == "completed" for result in payload["results"])
    assert all(result["summary"]["transactionCount"] == 2 for result in payload["results"])


def test_custom_profile_can_generate_exact_model_contract():
    data = synthetic_data.generate_dataset(dataset_type="combined", scenario="custom",
        count=3, seed=10, custom_burst_times=[3, 2, 1], custom_arrival_times=[0, 0, 0])
    assert [row["burst_time"] for row in data["scheduling_workload"]] == [3, 2, 1]
    assert [row["arrival_time"] for row in data["scheduling_workload"]] == [0, 0, 0]
