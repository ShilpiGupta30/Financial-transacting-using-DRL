"""Validate priority-free DQN traces against the existing C++ replay CLI.

With no ``--input`` argument, runs a deterministic workload suite at N=1, 2,
3, 4, 6, and 8. The CSV uses neutral transaction fields because this validator
checks scheduler timing only; it does not read or change financial records.
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Optional

from drl_scheduler.priority_free_inference_adapter import (
    DEFAULT_CHECKPOINT,
    AdapterError,
    create_action_trace,
    parse_workload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CPP_EXECUTABLE = (
    PROJECT_ROOT / "core_engine" / "build" / "replay-tests" /
    "IntelligentResourceManager.exe"
)
RESULT_FIELDS = ("transactionID", "processID", "completionTime", "waitingTime", "turnaroundTime")


class ValidationError(ValueError):
    """Invalid workload/trace or a failed/mismatched replay."""


def validation_workloads() -> list[list[dict[str, int]]]:
    """Return varied fixed test profiles with nonsequential, stable IDs."""
    burst_profiles = {
        1: [4],
        2: [7, 2],
        3: [3, 2, 1],
        4: [5, 1, 8, 3],
        6: [8, 2, 6, 1, 5, 3],
        8: [4, 8, 1, 7, 2, 6, 3, 5],
    }
    result = []
    for count, bursts in burst_profiles.items():
        result.append([
            {"transaction_id": 810_013 + index * 1_009 + count * 31,
             "burst_time": burst}
            for index, burst in enumerate(bursts)
        ])
    return result


def _csv_input(workload: list[dict[str, int]]) -> str:
    """Build the C++ CLI's six-column format with neutral non-scheduling fields."""
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for row in workload:
        writer.writerow((row["transaction_id"], "DEPOSIT", 0, -1, "0", 0))
    return output.getvalue()


def replay_python(workload: list[dict[str, int]], action_trace: list[int]) -> dict[int, dict[str, int]]:
    """Independently replay one unit per ID action and calculate completion metrics."""
    ids = [row["transaction_id"] for row in workload]
    if len(set(ids)) != len(ids):
        raise ValidationError("workload transaction IDs must be unique")
    remaining = {row["transaction_id"]: row["burst_time"] for row in workload}
    waiting = {transaction_id: 0 for transaction_id in ids}
    completion: dict[int, int] = {}
    elapsed = 0
    for step, transaction_id in enumerate(action_trace, start=1):
        if transaction_id not in remaining:
            raise ValidationError(f"action {step} references unknown transaction ID {transaction_id}")
        if remaining[transaction_id] <= 0:
            raise ValidationError(
                f"action {step} executes completed transaction ID {transaction_id} again"
            )
        for other_id in ids:
            if other_id != transaction_id and remaining[other_id] > 0:
                waiting[other_id] += 1
        remaining[transaction_id] -= 1
        elapsed += 1
        if remaining[transaction_id] == 0:
            completion[transaction_id] = elapsed

    unfinished = [transaction_id for transaction_id in ids if remaining[transaction_id] > 0]
    if unfinished:
        raise ValidationError(f"action trace ended before transactions completed: {unfinished}")
    expected_steps = sum(row["burst_time"] for row in workload)
    if elapsed != expected_steps or len(action_trace) != expected_steps:
        raise ValidationError(
            f"action trace has {len(action_trace)} steps; expected total burst time {expected_steps}"
        )
    return {
        transaction_id: {
            "completionTime": completion[transaction_id],
            "waitingTime": waiting[transaction_id],
            # All replayed processes arrive at time zero.
            "turnaroundTime": completion[transaction_id],
        }
        for transaction_id in ids
    }


def _validate_adapter_trace(workload: list[dict[str, int]], result: object) -> list[int]:
    if not isinstance(result, dict) or result.get("algorithm") != "priority_free_variable_dqn":
        raise ValidationError("inference adapter returned an unexpected result")
    trace = result.get("action_trace")
    if not isinstance(trace, list) or any(type(value) is not int for value in trace):
        raise ValidationError("inference adapter action_trace must be an integer ID array")
    ids = [row["transaction_id"] for row in workload]
    if len(set(ids)) != len(ids):
        raise ValidationError("input workload contains duplicate transaction IDs")
    known_ids = set(ids)
    counts: Counter[int] = Counter()
    for position, transaction_id in enumerate(trace, start=1):
        if transaction_id not in known_ids:
            raise ValidationError(f"action {position} references unknown transaction ID {transaction_id}")
        counts[transaction_id] += 1
        burst = next(row["burst_time"] for row in workload
                     if row["transaction_id"] == transaction_id)
        if counts[transaction_id] > burst:
            raise ValidationError(
                f"action {position} executes transaction ID {transaction_id} beyond its burst time"
            )
    expected_counts = {row["transaction_id"]: row["burst_time"] for row in workload}
    if dict(counts) != expected_counts:
        raise ValidationError("action trace does not complete every transaction exactly to its burst time")
    if result.get("steps") != sum(expected_counts.values()) or len(trace) != sum(expected_counts.values()):
        raise ValidationError("adapter action count does not equal the sum of burst times")

    expected_records = [
        f"BURST,{row['transaction_id']},{row['burst_time']}" for row in workload
    ] + [f"ACTION,{transaction_id}" for transaction_id in trace]
    if result.get("cpp_trace_records") != expected_records:
        raise ValidationError("adapter C++ trace records do not match the workload and action IDs")
    return trace


def _subprocess_environment(executable: Path) -> dict[str, str]:
    environment = os.environ.copy()
    directories = [executable.parent]
    configured = os.environ.get("MSYS2_UCRT64_BIN", r"C:\msys64\ucrt64\bin")
    directories.append(Path(configured))
    current_path = environment.get("PATH", "")
    seen = {entry.casefold() for entry in current_path.split(os.pathsep) if entry}
    prefix = [str(path) for path in directories if path.is_dir() and str(path).casefold() not in seen]
    environment["PATH"] = os.pathsep.join(prefix + ([current_path] if current_path else []))
    return environment


def _parse_cpp_result(stdout: str, workload: list[dict[str, int]]) -> dict[int, dict[str, int]]:
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValidationError(f"C++ replay returned invalid JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("algorithm") != "dqn_trace":
        raise ValidationError("C++ replay JSON has an unexpected algorithm value")
    rows = payload.get("results")
    expected_ids = {row["transaction_id"] for row in workload}
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValidationError("C++ replay must return exactly one metric record per input ID")

    parsed: dict[int, dict[str, int]] = {}
    process_ids: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or any(field not in row for field in RESULT_FIELDS):
            raise ValidationError("C++ replay returned a metric record with missing fields")
        if any(type(row[field]) is not int for field in RESULT_FIELDS):
            raise ValidationError("C++ replay metrics must be integers")
        transaction_id = row["transactionID"]
        process_id = row["processID"]
        if transaction_id not in expected_ids or transaction_id in parsed:
            raise ValidationError("C++ replay returned an unknown or duplicate transaction ID")
        if process_id <= 0 or process_id in process_ids:
            raise ValidationError("C++ replay returned an invalid or duplicate process ID")
        if min(row["completionTime"], row["waitingTime"], row["turnaroundTime"]) < 0:
            raise ValidationError("C++ replay returned a negative scheduling metric")
        process_ids.add(process_id)
        parsed[transaction_id] = {
            "completionTime": row["completionTime"],
            "waitingTime": row["waitingTime"],
            "turnaroundTime": row["turnaroundTime"],
        }
    if set(parsed) != expected_ids:
        raise ValidationError("C++ replay did not return every input transaction ID")
    return parsed


def validate_workload(
    workload: list[dict[str, int]],
    *,
    cpp_executable: Path | str = DEFAULT_CPP_EXECUTABLE,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    timeout: float = 30,
) -> dict:
    """Run candidate inference, validate/replay its trace, then compare C++ metrics."""
    try:
        clean_workload = parse_workload({"transactions": workload})
    except AdapterError as error:
        raise ValidationError(str(error)) from error
    executable = Path(cpp_executable).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if not executable.is_file():
        raise ValidationError(f"C++ replay executable not found: {executable}")
    if not checkpoint.is_file():
        raise ValidationError(f"priority-free checkpoint not found: {checkpoint}")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValidationError("timeout must be a positive number of seconds")

    try:
        adapter_result = create_action_trace(clean_workload, checkpoint)
    except AdapterError as error:
        raise ValidationError(f"priority-free inference failed: {error}") from error
    trace = _validate_adapter_trace(clean_workload, adapter_result)
    python_metrics = replay_python(clean_workload, trace)

    with tempfile.TemporaryDirectory(prefix="priority_free_cpp_validation_") as directory:
        trace_path = Path(directory) / "action_trace.csv"
        trace_path.write_text("\n".join(adapter_result["cpp_trace_records"]) + "\n",
                              encoding="utf-8")
        command = [str(executable), "dqn_trace", str(trace_path)]
        try:
            completed = subprocess.run(
                command,
                input=_csv_input(clean_workload),
                text=True,
                capture_output=True,
                timeout=float(timeout),
                cwd=str(PROJECT_ROOT),
                env=_subprocess_environment(executable),
                check=False,
            )
        except FileNotFoundError as error:
            raise ValidationError(f"could not start C++ executable {executable}: {error}") from error
        except subprocess.TimeoutExpired as error:
            raise ValidationError(f"C++ replay timed out after {timeout} seconds") from error
        except OSError as error:
            raise ValidationError(f"could not run C++ replay: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no error output").strip()
            raise ValidationError(f"C++ replay failed with exit code {completed.returncode}: {detail}")

    cpp_metrics = _parse_cpp_result(completed.stdout, clean_workload)
    mismatches = {
        transaction_id: {
            "python": python_metrics[transaction_id],
            "cpp": cpp_metrics[transaction_id],
        }
        for transaction_id in python_metrics
        if python_metrics[transaction_id] != cpp_metrics[transaction_id]
    }
    if mismatches:
        raise ValidationError(f"Python/C++ scheduling metrics differ by transaction: {mismatches}")
    return {
        "algorithm": "priority_free_dqn_cpp_validation",
        "transaction_count": len(clean_workload),
        "execution_steps": len(trace),
        "expected_execution_steps": sum(row["burst_time"] for row in clean_workload),
        "action_trace": trace,
        "python_results": [
            {"transactionID": row["transaction_id"], **python_metrics[row["transaction_id"]]}
            for row in clean_workload
        ],
        "cpp_results": [
            {"transactionID": row["transaction_id"], **cpp_metrics[row["transaction_id"]]}
            for row in clean_workload
        ],
        "metrics_match": True,
    }


def _load_input(path: Path) -> list[dict[str, int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValidationError(f"could not read workload file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValidationError(f"malformed workload JSON: {error}") from error
    try:
        return parse_workload(payload)
    except AdapterError as error:
        raise ValidationError(str(error)) from error


def validate_suite(
    *,
    cpp_executable: Path | str = DEFAULT_CPP_EXECUTABLE,
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    timeout: float = 30,
) -> list[dict]:
    """Run the required varied profiles and require a non-input-order action case."""
    results = [
        validate_workload(workload, cpp_executable=cpp_executable,
                          checkpoint_path=checkpoint_path, timeout=timeout)
        for workload in validation_workloads()
    ]
    has_reordered_actions = False
    for workload, result in zip(validation_workloads(), results):
        first_seen_order = list(dict.fromkeys(result["action_trace"]))
        input_order = [row["transaction_id"] for row in workload]
        if first_seen_order != input_order:
            has_reordered_actions = True
            break
    if not has_reordered_actions:
        raise ValidationError("suite did not produce any action order different from input order")
    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare priority-free DQN Python replay metrics with the existing C++ dqn_trace CLI"
    )
    parser.add_argument("--input", type=Path,
                        help="one JSON workload file; omit to run the N=1,2,3,4,6,8 validation suite")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cpp-executable", type=Path, default=DEFAULT_CPP_EXECUTABLE)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.input:
            output = validate_workload(
                _load_input(args.input), cpp_executable=args.cpp_executable,
                checkpoint_path=args.checkpoint, timeout=args.timeout,
            )
        else:
            output = {
                "suite": validate_suite(
                    cpp_executable=args.cpp_executable,
                    checkpoint_path=args.checkpoint,
                    timeout=args.timeout,
                )
            }
    except ValidationError as error:
        parser.exit(2, f"Error: {error}\n")
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
