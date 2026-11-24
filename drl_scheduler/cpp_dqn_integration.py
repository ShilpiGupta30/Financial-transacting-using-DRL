"""Run the saved Python DQN policy and replay its stable-ID trace in C++."""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKLOAD = PROJECT_ROOT / "drl_scheduler" / "examples" / "sample_workload.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "drl_scheduler" / "models" / "dqn_checkpoint.pt"
DEFAULT_CPP_EXECUTABLE = (
    PROJECT_ROOT / "core_engine" / "build" / "replay-tests" /
    "IntelligentResourceManager.exe"
)
ADAPTER_MODULE = "drl_scheduler.inference_adapter"
VALID_TRANSACTION_TYPES = {"DEPOSIT", "WITHDRAWAL", "TRANSFER", "LOAN_PAYMENT"}


class IntegrationError(ValueError):
    """Raised when workload validation or either subprocess fails."""


def _resolve_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _positive_timeout(timeout: float) -> float:
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise IntegrationError("timeout must be a positive number of seconds")
    return float(timeout)


def _load_workload(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise IntegrationError(f"could not read workload file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise IntegrationError(
            f"malformed workload JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(payload, dict) or not isinstance(payload.get("transactions"), list):
        raise IntegrationError("workload must be a JSON object containing a transactions array")
    rows = payload["transactions"]
    if len(rows) != 3:
        raise IntegrationError("the saved DQN checkpoint supports exactly three transactions")

    seen_ids = set()
    validated = []
    required = ("transaction_id", "burst_time", "priority")
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise IntegrationError(f"transaction {position} must be a JSON object")
        for field in required:
            if field not in row:
                raise IntegrationError(f"transaction {position} is missing {field}")
        for field in required:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise IntegrationError(f"transaction {position} field {field} must be an integer")
        transaction_id = row["transaction_id"]
        burst_time = row["burst_time"]
        if transaction_id in seen_ids:
            raise IntegrationError(f"duplicate transaction_id: {transaction_id}")
        if burst_time <= 0:
            raise IntegrationError(f"transaction {position} burst_time must be positive")
        seen_ids.add(transaction_id)

        transaction_type = row.get("type", "DEPOSIT")
        if not isinstance(transaction_type, str) or transaction_type not in VALID_TRANSACTION_TYPES:
            raise IntegrationError(
                f"transaction {position} type must be one of "
                f"{', '.join(sorted(VALID_TRANSACTION_TYPES))}"
            )
        source_id = row.get("source_account_id", 0)
        destination_id = row.get("destination_account_id", -1)
        amount = row.get("amount", 0.0)
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            raise IntegrationError(f"transaction {position} source_account_id must be an integer")
        if isinstance(destination_id, bool) or not isinstance(destination_id, int):
            raise IntegrationError(
                f"transaction {position} destination_account_id must be an integer"
            )
        if destination_id < -1:
            raise IntegrationError(
                f"transaction {position} destination_account_id must be -1 or non-negative"
            )
        if isinstance(amount, str):
            try:
                valid_amount = Decimal(amount).is_finite()
            except InvalidOperation:
                valid_amount = False
        else:
            try:
                valid_amount = (not isinstance(amount, bool) and
                                isinstance(amount, (int, float)) and math.isfinite(amount))
            except OverflowError:
                valid_amount = False
        if not valid_amount:
            raise IntegrationError(f"transaction {position} amount must be a finite number")

        validated.append({
            "transaction_id": transaction_id,
            "burst_time": burst_time,
            "priority": row["priority"],
            "type": transaction_type,
            "source_account_id": source_id,
            "destination_account_id": destination_id,
            "amount": amount,
        })

    if [row["burst_time"] for row in validated] != [3, 2, 1]:
        raise IntegrationError(
            "the saved DQN workload contract requires burst times 3, 2, 1 in input order"
        )
    return validated


def _csv_transactions(workload: list[dict]) -> str:
    lines = []
    for row in workload:
        amount = row["amount"]
        formatted_amount = amount if isinstance(amount, str) else f"{amount:.15g}"
        lines.append(
            f"{row['transaction_id']},{row['type']},{row['source_account_id']},"
            f"{row['destination_account_id']},{formatted_amount},{row['priority']}"
        )
    return "\n".join(lines) + "\n"


def _parse_action_trace(stdout: str, workload: list[dict]) -> list[int]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise IntegrationError(f"inference adapter returned invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("action_trace"), list):
        raise IntegrationError("inference adapter JSON is missing the action_trace array")
    trace = payload["action_trace"]
    expected_steps = sum(row["burst_time"] for row in workload)
    valid_ids = {row["transaction_id"] for row in workload}
    if (len(trace) != expected_steps or any(isinstance(item, bool) or not isinstance(item, int)
                                           or item not in valid_ids for item in trace)):
        raise IntegrationError("inference adapter returned an invalid transaction-ID trace")
    expected_counts = {row["transaction_id"]: row["burst_time"] for row in workload}
    actual_counts = {transaction_id: trace.count(transaction_id) for transaction_id in valid_ids}
    if actual_counts != expected_counts:
        raise IntegrationError("inference trace does not execute each transaction for its burst time")
    if payload.get("steps") != len(trace):
        raise IntegrationError("inference adapter step count does not match its action trace")
    completed_ids = payload.get("completed_transaction_ids")
    if (not isinstance(completed_ids, list) or len(completed_ids) != len(valid_ids)
            or any(isinstance(item, bool) or not isinstance(item, int)
                   for item in completed_ids)
            or set(completed_ids) != valid_ids):
        raise IntegrationError("inference adapter did not report completion for every transaction")
    return trace


def _make_trace_file(workload: list[dict], trace: list[int], directory: Path) -> Path:
    records = [f"BURST,{row['transaction_id']},{row['burst_time']}" for row in workload]
    records.extend(f"ACTION,{transaction_id}" for transaction_id in trace)
    trace_path = directory / "dqn_action_trace.csv"
    trace_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return trace_path


def _parse_cpp_results(stdout: str, workload: list[dict]) -> dict:
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise IntegrationError(f"C++ replay returned invalid JSON: {error.msg}") from error
    if not isinstance(result, dict) or result.get("algorithm") != "dqn_trace":
        raise IntegrationError("C++ replay JSON has an unexpected algorithm value")
    rows = result.get("results")
    expected_ids = {row["transaction_id"] for row in workload}
    required_fields = (
        "transactionID", "processID", "completionTime", "waitingTime", "turnaroundTime"
    )
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise IntegrationError("C++ replay JSON must contain one result per transaction")
    actual_ids = set()
    for row in rows:
        if not isinstance(row, dict) or any(field not in row for field in required_fields):
            raise IntegrationError("C++ replay result is missing required result fields")
        transaction_id = row["transactionID"]
        if isinstance(transaction_id, bool) or not isinstance(transaction_id, int):
            raise IntegrationError("C++ replay returned a non-integer transactionID")
        if transaction_id not in expected_ids or transaction_id in actual_ids:
            raise IntegrationError("C++ replay returned an unknown or duplicate transactionID")
        actual_ids.add(transaction_id)
        for field in required_fields[1:]:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IntegrationError(f"C++ replay returned an invalid {field}")
    if actual_ids != expected_ids:
        raise IntegrationError("C++ replay did not return every input transaction")
    return result


def _run(command: list[str], *, input_text: Optional[str], timeout: float,
         cwd: Path, env: dict) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run(
            command, input=input_text, text=True, capture_output=True, timeout=timeout,
            cwd=str(cwd), env=env, check=False,
        )
    except FileNotFoundError as error:
        raise IntegrationError(f"could not start executable {command[0]}: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise IntegrationError(f"command timed out after {timeout:g} seconds: {command[0]}") from error
    except OSError as error:
        raise IntegrationError(f"could not run {command[0]}: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no error output").strip()
        raise IntegrationError(f"command failed ({completed.returncode}): {detail}")
    return completed


def run_integration(workload_path: Path | str, *, checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
                    cpp_executable: Path | str = DEFAULT_CPP_EXECUTABLE,
                    python_executable: str = sys.executable, timeout: float = 30,
                    include_action_trace: bool = False) -> dict:
    """Run greedy inference followed by C++ trace replay and return its JSON result."""
    timeout = _positive_timeout(timeout)
    workload_file = _resolve_project_path(workload_path)
    checkpoint = _resolve_project_path(checkpoint_path)
    cpp_path = _resolve_project_path(cpp_executable)
    if not workload_file.is_file():
        raise IntegrationError(f"workload file does not exist: {workload_file}")
    if not checkpoint.is_file():
        raise IntegrationError(f"DQN checkpoint does not exist: {checkpoint}")
    if not cpp_path.is_file():
        raise IntegrationError(f"C++ CLI executable does not exist: {cpp_path}")

    workload = _load_workload(workload_file)
    env = os.environ.copy()
    msys2_bin = os.environ.get("MSYS2_UCRT64_BIN", r"C:\msys64\ucrt64\bin")
    if Path(msys2_bin).is_dir():
        env["PATH"] = msys2_bin + os.pathsep + env.get("PATH", "")

    adapter_command = [
        python_executable, "-m", ADAPTER_MODULE,
        "--input", str(workload_file), "--checkpoint", str(checkpoint),
    ]
    adapter_result = _run(adapter_command, input_text=None, timeout=timeout,
                          cwd=PROJECT_ROOT, env=env)
    trace = _parse_action_trace(adapter_result.stdout, workload)
    csv_input = _csv_transactions(workload)

    with tempfile.TemporaryDirectory(prefix="dqn_cpp_trace_") as temporary_directory:
        trace_path = _make_trace_file(workload, trace, Path(temporary_directory))
        cpp_command = [str(cpp_path), "dqn_trace", str(trace_path)]
        cpp_result = _run(cpp_command, input_text=csv_input, timeout=timeout,
                          cwd=PROJECT_ROOT, env=env)
    result = _parse_cpp_results(cpp_result.stdout, workload)
    if include_action_trace:
        result["action_trace"] = trace
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the saved DQN inference adapter and replay its trace in the C++ CLI"
    )
    parser.add_argument("--input", default=DEFAULT_WORKLOAD, type=Path,
                        help="workload JSON file (relative paths are rooted at the repository)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--cpp-executable", default=DEFAULT_CPP_EXECUTABLE, type=Path)
    parser.add_argument("--timeout", type=float, default=30,
                        help="timeout in seconds for each subprocess")
    parser.add_argument("--include-action-trace", action="store_true",
                        help="include the stable transaction-ID trace in JSON output")
    args = parser.parse_args(argv)
    try:
        result = run_integration(
            args.input, checkpoint_path=args.checkpoint,
            cpp_executable=args.cpp_executable, timeout=args.timeout,
            include_action_trace=args.include_action_trace,
        )
    except IntegrationError as error:
        parser.exit(2, f"Error: {error}\n")
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
