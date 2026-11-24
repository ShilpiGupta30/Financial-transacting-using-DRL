"""Greedy inference for the normalized, priority-free variable-size DQN.

Input is a JSON object containing transactions with stable integer IDs and
explicit burst times. The returned action trace contains transaction IDs, not
ready-queue positions. Priority and IDs are never passed to the model.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

from drl_scheduler.agent import PriorityFreeFeatureEncoder
from drl_scheduler.env import SchedulingEnv
from drl_scheduler.train_priority_free_variable_size import (
    ARCHITECTURE,
    DEFAULT_CHECKPOINT,
    FEATURE_NAMES,
    load_priority_free_checkpoint,
)


MAX_TRANSACTIONS = 8
MAX_BURST_TIME = 8
CPP_INT_MAX = 2_147_483_647


class AdapterError(ValueError):
    """An invalid workload, checkpoint, or inference result."""


def parse_workload(payload: object) -> list[dict[str, int]]:
    """Validate workload records and retain only ID and burst for the model."""
    if not isinstance(payload, dict) or not isinstance(payload.get("transactions"), list):
        raise AdapterError("input must be a JSON object containing a transactions array")
    rows = payload["transactions"]
    if not rows:
        raise AdapterError("transactions must not be empty")
    if len(rows) > MAX_TRANSACTIONS:
        raise AdapterError(f"the priority-free checkpoint supports at most {MAX_TRANSACTIONS} transactions")

    workload: list[dict[str, int]] = []
    seen_ids: set[int] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AdapterError(f"transaction {position} must be a JSON object")
        missing = [field for field in ("transaction_id", "burst_time") if field not in row]
        if missing:
            raise AdapterError(
                f"transaction {position} is missing required field(s): {', '.join(missing)}"
            )
        transaction_id = row["transaction_id"]
        burst_time = row["burst_time"]
        if isinstance(transaction_id, bool) or not isinstance(transaction_id, int):
            raise AdapterError(f"transaction {position} transaction_id must be an integer")
        if not 1 <= transaction_id <= CPP_INT_MAX:
            raise AdapterError(f"transaction {position} transaction_id must fit a positive C++ int")
        if transaction_id in seen_ids:
            raise AdapterError(f"duplicate transaction_id: {transaction_id}")
        if isinstance(burst_time, bool) or not isinstance(burst_time, int):
            raise AdapterError(f"transaction {position} burst_time must be an integer")
        if not 1 <= burst_time <= MAX_BURST_TIME:
            raise AdapterError(f"transaction {position} burst_time must be between 1 and {MAX_BURST_TIME}")
        seen_ids.add(transaction_id)
        workload.append({"transaction_id": transaction_id, "burst_time": burst_time})
    return workload


def _validate_checkpoint_schema(checkpoint: object) -> dict:
    """Check the candidate's metadata before using its features or scales."""
    if not isinstance(checkpoint, dict):
        raise AdapterError("priority-free checkpoint metadata must be an object")
    expected_normalization = {
        "burst_time_scale": MAX_BURST_TIME,
        "remaining_time_scale": MAX_BURST_TIME,
        "waiting_time_scale": (MAX_TRANSACTIONS - 1) * MAX_BURST_TIME,
        "waiting_time_scale_rule": "max(1, (max_transactions - 1) * max_burst_time)",
    }
    workload_config = checkpoint.get("workload_config")
    if not isinstance(workload_config, dict):
        raise AdapterError("checkpoint workload_config metadata must be an object")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise AdapterError("checkpoint architecture is not the supported priority-free model")
    if checkpoint.get("feature_count") != 4 or checkpoint.get("feature_names") != FEATURE_NAMES:
        raise AdapterError("checkpoint feature schema is incompatible with the priority-free adapter")
    if workload_config.get("max_transactions") != MAX_TRANSACTIONS:
        raise AdapterError("checkpoint max_transactions metadata must be 8")
    if workload_config.get("max_burst_time") != MAX_BURST_TIME:
        raise AdapterError("checkpoint max_burst_time metadata must be 8")
    if checkpoint.get("normalization") != expected_normalization:
        raise AdapterError("checkpoint normalization metadata is incompatible")
    return checkpoint["normalization"]


def create_action_trace(
    workload: list[dict[str, int]],
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
) -> dict:
    """Run deterministic greedy inference and emit stable-ID C++ trace records."""
    return _create_action_trace_with_model(workload, _load_model(checkpoint_path))


def _load_model(checkpoint_path: Path | str):
    """Load and validate the checkpoint once for one-shot or worker inference."""
    try:
        agent, checkpoint = load_priority_free_checkpoint(checkpoint_path, device="cpu")
    except (OSError, RuntimeError, TypeError, ValueError, pickle.UnpicklingError) as error:
        raise AdapterError(f"could not load priority-free checkpoint: {error}") from error
    normalization = _validate_checkpoint_schema(checkpoint)
    return agent, checkpoint, normalization


def _create_action_trace_with_model(
    workload: list[dict[str, int]], loaded_model: tuple,
) -> dict:
    """Run one workload using an already loaded agent and checkpoint metadata."""
    # Validate even when called as a Python function instead of through the CLI.
    workload = parse_workload({"transactions": workload})
    agent, checkpoint, normalization = loaded_model
    workload_config = checkpoint["workload_config"]
    if len(workload) > workload_config["max_transactions"]:
        raise AdapterError("workload exceeds the checkpoint's configured transaction limit")
    if any(row["burst_time"] > workload_config["max_burst_time"] for row in workload):
        raise AdapterError("burst_time exceeds the checkpoint's configured burst limit")

    # SchedulingEnv requires a priority field, but this model does not consume it.
    # Supplying a constant makes that separation explicit and avoids using the
    # database's priority as either an input feature or a hidden scheduling rule.
    environment_workload = [
        {**row, "priority": 0} for row in workload
    ]
    env = SchedulingEnv(
        workload=environment_workload,
        max_episode_steps=sum(row["burst_time"] for row in workload),
    )
    encoder = PriorityFreeFeatureEncoder(
        max_transactions=workload_config["max_transactions"],
        max_burst_time=workload_config["max_burst_time"],
    )
    # Guard against accidental drift between this adapter and checkpoint metadata.
    if (encoder.burst_time_scale != normalization["burst_time_scale"]
            or encoder.remaining_time_scale != normalization["remaining_time_scale"]
            or encoder.waiting_time_scale != normalization["waiting_time_scale"]):
        raise AdapterError("feature encoder scales do not match checkpoint metadata")

    observation = encoder.encode(env.get_process_features())
    action_trace: list[int] = []
    terminated = truncated = False
    while not (terminated or truncated):
        ready = [process for process in env.transactions if not process.completed]
        if not ready:
            raise AdapterError("environment has no ready transaction before termination")
        try:
            # Agent returns an index into the compact ready queue, with completed
            # rows masked from greedy Q-value selection.
            ready_index = agent.select_action(observation, explore=False)
        except (RuntimeError, TypeError, ValueError) as error:
            raise AdapterError(f"DQN could not select a valid action: {error}") from error
        if (isinstance(ready_index, bool) or not isinstance(ready_index, int)
                or not 0 <= ready_index < len(ready)):
            raise AdapterError(f"DQN selected invalid ready-queue index {ready_index!r}")

        # Resolve the transient ready index to the stable external ID immediately.
        action_trace.append(ready[ready_index].transaction_id)
        _, _, terminated, truncated, info = env.step(ready_index)
        if info.get("invalid_action"):
            raise AdapterError("environment rejected a DQN-selected action")
        observation = encoder.encode(env.get_process_features())

    if truncated or not terminated:
        raise AdapterError("workload did not complete before the environment step limit")
    completed_ids = {process.transaction_id for process in env.transactions if process.completed}
    expected_ids = {row["transaction_id"] for row in workload}
    if completed_ids != expected_ids:
        raise AdapterError("action trace ended before every transaction completed")

    burst_records = [
        f"BURST,{row['transaction_id']},{row['burst_time']}" for row in workload
    ]
    cpp_trace_records = burst_records + [f"ACTION,{transaction_id}" for transaction_id in action_trace]
    return {
        "algorithm": "priority_free_variable_dqn",
        "action_trace": action_trace,
        "steps": len(action_trace),
        "completed_transaction_ids": [row["transaction_id"] for row in workload],
        "cpp_trace_records": cpp_trace_records,
    }


def run_worker(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    *,
    input_stream=None,
    output_stream=None,
) -> int:
    """Serve line-delimited JSON requests while reusing one loaded model.

    Each input line contains the same workload object accepted by the one-shot
    CLI. Valid responses retain the usual action-trace format. Invalid requests
    receive a JSON error object and do not stop later requests.
    """
    input_stream = input_stream if input_stream is not None else sys.stdin
    output_stream = output_stream if output_stream is not None else sys.stdout
    try:
        loaded_model = _load_model(checkpoint_path)
    except AdapterError as error:
        print(f"Priority-free inference worker could not start: {error}", file=sys.stderr)
        return 2

    for line in input_stream:
        try:
            payload = json.loads(line)
            workload = parse_workload(payload)
            response = _create_action_trace_with_model(workload, loaded_model)
        except json.JSONDecodeError as error:
            response = {
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": (
                        f"malformed JSON at line {error.lineno}, column "
                        f"{error.colno}: {error.msg}"
                    ),
                }
            }
        except AdapterError as error:
            response = {
                "error": {"code": "INVALID_REQUEST", "message": str(error)}
            }

        json.dump(response, output_stream, separators=(",", ":"))
        output_stream.write("\n")
        output_stream.flush()
    return 0


def _read_input(input_path: str) -> object:
    try:
        contents = sys.stdin.read() if input_path == "-" else Path(input_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AdapterError(f"could not read workload input: {error}") from error
    try:
        return json.loads(contents)
    except json.JSONDecodeError as error:
        raise AdapterError(
            f"malformed JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a greedy stable-ID trace with the priority-free variable-size DQN"
    )
    parser.add_argument("--input", default="-", metavar="PATH",
                        help="JSON workload file; use '-' for standard input")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--worker", action="store_true",
                        help="serve one line-delimited JSON request at a time and reuse the loaded model")
    args = parser.parse_args(argv)
    if args.worker:
        return run_worker(args.checkpoint)
    try:
        workload = parse_workload(_read_input(args.input))
        result = create_action_trace(workload, args.checkpoint)
    except AdapterError as error:
        parser.exit(2, f"Error: {error}\n")
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
