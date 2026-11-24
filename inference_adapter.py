"""Run greedy DQN inference for a JSON workload and emit stable transaction IDs.

The current checkpoint was trained for exactly three processes. The adapter
checks this size and the observation shape before inference; it does not resize
the network or claim that the synthetic policy generalizes to real workloads.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import torch

from drl_scheduler.agent import DQNAgent
from drl_scheduler.env import SchedulingEnv
from drl_scheduler.train import DEFAULT_MODEL_DIR, _load_checkpoint


DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "dqn_checkpoint.pt"


class AdapterError(ValueError):
    """An invalid workload, checkpoint, or inference result."""


def parse_workload(payload: object) -> list[dict]:
    """Validate JSON-decoded input and return the environment's feature fields."""
    if not isinstance(payload, dict) or not isinstance(payload.get("transactions"), list):
        raise AdapterError("input must be a JSON object containing a transactions array")
    rows = payload["transactions"]
    if not rows:
        raise AdapterError("transactions must not be empty")

    workload = []
    seen_ids = set()
    required = ("transaction_id", "burst_time", "priority")
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AdapterError(f"transaction {position} must be a JSON object")
        missing = [field for field in required if field not in row]
        if missing:
            raise AdapterError(
                f"transaction {position} is missing required field(s): {', '.join(missing)}"
            )

        transaction_id = row["transaction_id"]
        burst_time = row["burst_time"]
        priority = row["priority"]
        for name, value in zip(required, (transaction_id, burst_time, priority)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise AdapterError(f"transaction {position} field {name} must be an integer")
        if transaction_id in seen_ids:
            raise AdapterError(f"duplicate transaction_id: {transaction_id}")
        if burst_time <= 0:
            raise AdapterError(f"transaction {position} burst_time must be positive")

        seen_ids.add(transaction_id)
        workload.append({
            "transaction_id": transaction_id,
            "burst_time": burst_time,
            "priority": priority,
        })
    return workload


def create_action_trace(
    workload: list[dict],
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    *,
    seed: int = 7,
) -> dict:
    """Load a compatible checkpoint and return one stable transaction ID per action."""
    workload = parse_workload({"transactions": workload})
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AdapterError("seed must be an integer")

    try:
        checkpoint = _load_checkpoint(Path(checkpoint_path))
    except Exception as error:
        raise AdapterError(f"could not load DQN checkpoint: {error}") from error

    checkpoint_count = checkpoint.get("action_count")
    checkpoint_observation_size = checkpoint.get("observation_size")
    if isinstance(checkpoint_count, bool) or not isinstance(checkpoint_count, int):
        raise AdapterError("checkpoint has an invalid action_count")
    if checkpoint_count != len(SchedulingEnv.DEFAULT_WORKLOAD):
        raise AdapterError(
            "this adapter currently supports the saved three-transaction checkpoint only"
        )
    if len(workload) != checkpoint_count:
        raise AdapterError(
            f"checkpoint supports exactly {checkpoint_count} transactions; "
            f"received {len(workload)}. Use a matching checkpoint; the adapter "
            "does not resize or retrain the model."
        )

    # Keep checkpoint observation IDs in their training range. The external IDs
    # remain separate and are the only IDs exposed in the action trace.
    checkpoint_ids = [
        item["transaction_id"] for item in SchedulingEnv.DEFAULT_WORKLOAD
    ]
    model_workload = [
        {
            "transaction_id": checkpoint_ids[index],
            "burst_time": row["burst_time"],
            "priority": row["priority"],
        }
        for index, row in enumerate(workload)
    ]
    external_id_by_model_id = {
        model_workload[index]["transaction_id"]: row["transaction_id"]
        for index, row in enumerate(workload)
    }
    try:
        env = SchedulingEnv(workload=model_workload)
    except (TypeError, ValueError, OverflowError) as error:
        raise AdapterError(f"invalid scheduling workload: {error}") from error
    if len(env.get_observation()) != checkpoint_observation_size:
        raise AdapterError(
            "workload observation size does not match the saved DQN checkpoint"
        )

    config = checkpoint.get("agent_config")
    weights = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(weights, dict):
        raise AdapterError("checkpoint is missing its agent configuration or model weights")
    config = dict(config)
    config.update({"epsilon": 0.0, "epsilon_min": 0.0, "seed": seed, "device": "cpu"})
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        agent = DQNAgent.from_environment(env, **config)
        agent.policy_network.load_state_dict(weights)
        agent.policy_network.eval()
        for parameter in agent.policy_network.parameters():
            parameter.requires_grad_(False)
    except (RuntimeError, TypeError, ValueError) as error:
        raise AdapterError(f"checkpoint is incompatible with the scheduling environment: {error}") from error

    action_trace = []
    observation = env.reset()
    terminated = truncated = False
    with torch.inference_mode():
        while not (terminated or truncated):
            ready = [transaction for transaction in env.transactions if not transaction.completed]
            if not ready:
                raise AdapterError("environment has no ready transaction before termination")

            try:
                action_index = agent.select_action(observation, explore=False)
            except (TypeError, ValueError, RuntimeError) as error:
                raise AdapterError(f"DQN could not select a valid action: {error}") from error
            if (isinstance(action_index, bool) or not isinstance(action_index, int)
                    or not 0 <= action_index < len(ready)):
                raise AdapterError(
                    f"DQN selected invalid ready-queue index {action_index!r} "
                    f"for {len(ready)} ready transactions"
                )

            # Convert immediately while the ready-queue index still identifies this PCB.
            transaction_id = external_id_by_model_id[ready[action_index].transaction_id]
            action_trace.append(transaction_id)
            observation, _, terminated, truncated, info = env.step(action_index)
            if info.get("invalid_action"):
                raise AdapterError("environment rejected a DQN-selected action")

    if truncated or not terminated:
        raise AdapterError("workload did not finish before the environment step limit")
    completed_ids = [external_id_by_model_id[transaction.transaction_id]
                     for transaction in env.transactions
                     if transaction.completed]
    if len(completed_ids) != len(workload):
        raise AdapterError("action trace ended before every transaction completed")

    return {
        "action_trace": action_trace,
        "steps": len(action_trace),
        "completed_transaction_ids": completed_ids,
    }


def _read_input(input_path: str) -> object:
    try:
        if input_path == "-":
            contents = sys.stdin.read()
        else:
            contents = Path(input_path).read_text(encoding="utf-8")
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
        description="Generate a greedy DQN transaction-ID action trace"
    )
    parser.add_argument(
        "--input", default="-", metavar="PATH",
        help="JSON workload file; use '-' for standard input (default)",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    try:
        workload = parse_workload(_read_input(args.input))
        result = create_action_trace(workload, args.checkpoint, seed=args.seed)
    except AdapterError as error:
        parser.exit(2, f"Error: {error}\n")

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
