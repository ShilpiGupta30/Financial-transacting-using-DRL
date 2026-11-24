"""Isolated synthetic demo data generation and file storage."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import io
import json
import math
from pathlib import Path
import random
import re
import uuid

from .config import REPOSITORY_ROOT

GENERATED_DATA_DIR = REPOSITORY_ROOT / "generated_data" / "synthetic"
MAX_DATASET_SIZE = 1000
_BASE_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)
_TYPES = ("DEPOSIT", "WITHDRAWAL", "TRANSFER", "LOAN_PAYMENT")


def generate_dataset(*, dataset_type: str, scenario: str, count: int, seed: int,
                     account_count: int = 50, burst_min: int = 1,
                     burst_max: int = 20, arrival_span: int | None = None,
                     custom_burst_times: list[int] | None = None,
                     custom_arrival_times: list[int] | None = None) -> dict:
    if dataset_type not in {"banking", "scheduling", "combined"}:
        raise ValueError("dataset_type must be banking, scheduling, or combined")
    if scenario not in {"normal", "peak", "high_contention", "custom"}:
        raise ValueError("scenario must be normal, peak, high_contention, or custom")
    if type(count) is not int or not 1 <= count <= MAX_DATASET_SIZE:
        raise ValueError("count must be an integer from 1 to 1000")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 2 <= account_count <= 10000:
        raise ValueError("account_count must be from 2 to 10000")
    if not 1 <= burst_min <= burst_max <= 1000:
        raise ValueError("burst bounds must satisfy 1 <= burst_min <= burst_max <= 1000")
    if arrival_span is not None and (type(arrival_span) is not int or not 0 <= arrival_span <= 1000000):
        raise ValueError("arrival_span must be an integer from 0 to 1000000")
    for values, label, minimum in ((custom_burst_times, "custom burst times", 1), (custom_arrival_times, "custom arrival times", 0)):
        if values is not None and (len(values) != count or any(type(value) is not int or value < minimum for value in values)):
            raise ValueError(f"{label} must contain count valid integers")
    if scenario != "custom" and (custom_burst_times is not None or custom_arrival_times is not None):
        raise ValueError("custom times are only valid for the custom scenario")

    rng = random.Random(seed)
    base = 2_000_000_000
    accounts = [1_000_000_000 + i for i in range(account_count)]
    if scenario == "high_contention":
        span = 0
    elif scenario == "peak":
        span = max(1, count // 10)
    elif scenario == "normal":
        span = max(1, count // 2)
    else:
        span = arrival_span if arrival_span is not None else max(1, count // 2)

    transactions = []
    workload = []
    for index in range(count):
        transaction_id = base + index
        transaction_type = rng.choices(_TYPES, weights=(30, 25, 35, 10), k=1)[0]
        source = accounts[rng.randrange(len(accounts))]
        destination = None
        if transaction_type == "TRANSFER":
            candidates = [account for account in accounts if account != source]
            destination = candidates[rng.randrange(len(candidates))]
        # A log-normal distribution produces many routine values and fewer large values.
        cents = max(1, min(999_999_999, int(round(rng.lognormvariate(math.log(75), 0.9) * 100))))
        amount = Decimal(cents) / Decimal(100)
        arrival = (custom_arrival_times[index] if custom_arrival_times is not None
                   else rng.randint(0, span) if span else 0)
        created = (_BASE_TIME + timedelta(seconds=arrival, microseconds=index)).isoformat()
        transactions.append({
            "TransactionID": transaction_id,
            "TransactionType": transaction_type,
            "SourceAccountID": source,
            "DestinationAccountID": destination,
            "Amount": format(amount, ".2f"),
            "Status": "PENDING",
            "Priority": rng.randint(1, 10),
            "CreatedAt": created,
            "CompletedAt": None,
        })
        workload.append({
            "transaction_id": transaction_id,
            "arrival_time": arrival,
            "burst_time": (custom_burst_times[index] if custom_burst_times is not None
                           else rng.randint(burst_min, burst_max)),
            "priority": transactions[-1]["Priority"],
            # The current contention simulator models one shared CPU_IO resource.
            "resource_requirements": ["CPU_IO"] if scenario == "high_contention" else [],
        })

    if dataset_type == "banking":
        workload = []
    identifier = str(uuid.uuid4())
    return {
        "dataset_id": identifier,
        "label": "SYNTHETIC / DEMO DATA",
        "dataset_type": dataset_type,
        "scenario": scenario,
        "seed": seed,
        "count": count,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accounts": ([{"AccountID": account, "AccountNumber": f"SYNTH-{account}",
                       "AccountHolderName": f"Synthetic Demo Account {i + 1}",
                       "Balance": "10000.00", "AccountStatus": "ACTIVE"}
                      for i, account in enumerate(accounts)]),
        "transactions": transactions,
        "scheduling_workload": workload,
        "summary": {
            "transaction_count": len(transactions),
            "workload_count": len(workload),
            "account_count": len(accounts),
            "total_amount": format(sum((Decimal(item["Amount"]) for item in transactions), Decimal(0)), ".2f"),
            "average_burst_time": (round(sum(x["burst_time"] for x in workload) / len(workload), 2) if workload else 0),
            "simultaneous_arrivals": sum(1 for x in workload if x["arrival_time"] == 0),
        },
    }


def _paths(dataset_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", dataset_id):
        raise ValueError("invalid synthetic dataset ID")
    root = GENERATED_DATA_DIR.resolve()
    json_path = (root / f"{dataset_id}.json").resolve()
    csv_path = (root / f"{dataset_id}.csv").resolve()
    if json_path.parent != root or csv_path.parent != root:
        raise ValueError("synthetic dataset path is invalid")
    return json_path, csv_path


def _combined_csv(dataset: dict) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["record_type", "account_id", "account_number", "account_holder_name", "balance", "account_status", "transaction_id", "type", "source_account_id", "destination_account_id", "amount", "status", "priority", "arrival_time", "burst_time"])
    for account in dataset["accounts"]:
        writer.writerow(["account", account["AccountID"], account["AccountNumber"], account["AccountHolderName"], account["Balance"], account["AccountStatus"], "", "", "", "", "", "", "", "", ""])
    txns = {item["TransactionID"]: item for item in dataset["transactions"]}
    work = {item["transaction_id"]: item for item in dataset["scheduling_workload"]}
    for key in list(txns) + [key for key in work if key not in txns]:
        txn, item = txns.get(key, {}), work.get(key, {})
        writer.writerow(["transaction" if txn else "workload", "", "", "", "", "", key, txn.get("TransactionType", ""), txn.get("SourceAccountID", ""), txn.get("DestinationAccountID") if txn else "", txn.get("Amount", ""), txn.get("Status", ""), txn.get("Priority", item.get("priority", "")), item.get("arrival_time", ""), item.get("burst_time", "")])
    return output.getvalue()


def save_dataset(dataset: dict) -> dict:
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = _paths(dataset["dataset_id"])
    json_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    csv_path.write_text(_combined_csv(dataset), encoding="utf-8", newline="")
    return dataset


def get_dataset(dataset_id: str) -> dict | None:
    json_path, _ = _paths(dataset_id)
    if not json_path.is_file():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def list_datasets() -> list[dict]:
    if not GENERATED_DATA_DIR.exists():
        return []
    items = []
    for path in GENERATED_DATA_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append({key: data[key] for key in ("dataset_id", "label", "dataset_type", "scenario", "seed", "count", "created_at", "summary")})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


def export_csv(dataset_id: str) -> str:
    _, csv_path = _paths(dataset_id)
    if not csv_path.is_file():
        raise FileNotFoundError(dataset_id)
    return csv_path.read_text(encoding="utf-8")


def delete_dataset(dataset_id: str) -> bool:
    json_path, csv_path = _paths(dataset_id)
    found = json_path.exists() or csv_path.exists()
    for path in (json_path, csv_path):
        if path.exists() and path.is_file():
            path.unlink()
    return found


