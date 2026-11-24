"""Evaluate a saved variable-size DQN against deterministic CPU baselines.

All policies receive exactly the same synthetic workload for each size/seed.
The environment executes one CPU unit per action; SJF is non-preemptive (once a
job is chosen it runs to completion), and Round Robin defaults to quantum 1.
"""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Optional, Sequence

from drl_scheduler.env import SchedulingEnv
from drl_scheduler.train_variable_size import (
    DEFAULT_MODEL_DIR,
    generate_workload,
    load_variable_checkpoint,
)


DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_checkpoint.pt"
DEFAULT_DETAILED_OUTPUT = DEFAULT_MODEL_DIR / "variable_evaluation_detailed.csv"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_evaluation_summary.csv"
WORKLOAD_SIZES = (1, 2, 3, 4, 6, 8)
POLICIES = ("dqn", "fcfs", "sjf", "priority", "round_robin")
DETAILED_FIELDS = (
    "workload_id",
    "seed",
    "transaction_count",
    "policy",
    "round_robin_quantum",
    "average_waiting_time",
    "average_turnaround_time",
    "makespan",
    "completed_transactions",
    "workload_json",
)
SUMMARY_FIELDS = (
    "transaction_count",
    "policy",
    "workload_runs",
    "round_robin_quantum",
    "mean_average_waiting_time",
    "mean_average_turnaround_time",
    "mean_makespan",
    "mean_completed_transactions",
)


def calculate_metrics(env: SchedulingEnv, completion_times: dict[int, int]) -> dict:
    """Calculate comparable metrics for a finished episode.

    Every generated process has arrival time zero. Turnaround is therefore its
    completion time. Waiting time uses the environment's existing convention:
    each unfinished process accrues one waiting unit for every action spent on
    another ready process. Makespan is total elapsed simulated CPU time.
    """
    processes = env.transactions
    completed = [process for process in processes if process.completed]
    if len(completed) != len(processes) or len(completion_times) != len(processes):
        raise ValueError("cannot calculate final metrics before every process completes")
    if not processes:
        raise ValueError("cannot calculate metrics for an empty workload")

    waiting_total = sum(process.waiting_time for process in processes)
    turnaround_total = sum(completion_times[process.transaction_id] for process in processes)
    return {
        "average_waiting_time": waiting_total / len(processes),
        "average_turnaround_time": turnaround_total / len(processes),
        "makespan": env.elapsed_time,
        "completed_transactions": len(completed),
    }


def _run_policy(
    workload: list[dict],
    policy: str,
    *,
    agent=None,
    round_robin_quantum: int = 1,
    max_episode_steps: int = 100,
) -> dict:
    env = SchedulingEnv(workload=workload, max_episode_steps=max_episode_steps)
    process_by_id = {process.transaction_id: process for process in env.transactions}
    completion_times: dict[int, int] = {}
    terminated = truncated = False
    rr_queue = [process.transaction_id for process in env.transactions]
    rr_slice_remaining = round_robin_quantum
    current_sjf_id: Optional[int] = None

    while not (terminated or truncated):
        ready = [process for process in env.transactions if not process.completed]
        if policy == "dqn":
            action = agent.select_action(env.get_process_features(), explore=False)
        elif policy == "fcfs":
            action = 0
        elif policy == "sjf":
            if current_sjf_id not in {process.transaction_id for process in ready}:
                selected = min(ready, key=lambda process: process.burst_time)
                current_sjf_id = selected.transaction_id
            action = next(
                index for index, process in enumerate(ready)
                if process.transaction_id == current_sjf_id
            )
        elif policy == "priority":
            # Match the existing C++ PriorityScheduler: larger numeric values
            # run first; max() preserves input order for ties.
            action = max(range(len(ready)), key=lambda index: ready[index].priority)
        elif policy == "round_robin":
            while rr_queue and process_by_id[rr_queue[0]].completed:
                rr_queue.pop(0)
            current_id = rr_queue[0]
            action = next(
                index for index, process in enumerate(ready)
                if process.transaction_id == current_id
            )
        else:
            raise ValueError(f"unknown evaluation policy: {policy}")

        _, _, terminated, truncated, info = env.step(action)
        if info.get("invalid_action"):
            raise RuntimeError(f"{policy} selected an invalid action")
        if info.get("completed_now"):
            completion_times[info["transaction_id"]] = env.elapsed_time

        if policy == "round_robin":
            rr_slice_remaining -= 1
            selected = process_by_id[info["transaction_id"]]
            if selected.completed or rr_slice_remaining == 0:
                rr_queue.pop(0)
                if not selected.completed:
                    rr_queue.append(selected.transaction_id)
                rr_slice_remaining = round_robin_quantum

    if not terminated or truncated:
        raise RuntimeError(f"{policy} did not complete its workload")
    return calculate_metrics(env, completion_times)


def _validate_checkpoint_workload_range(checkpoint: dict, workload_sizes: Sequence[int]) -> dict:
    config = checkpoint.get("workload_config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing its workload configuration")
    minimum = config.get("min_transactions")
    maximum = config.get("max_transactions")
    if (isinstance(minimum, bool) or not isinstance(minimum, int)
            or isinstance(maximum, bool) or not isinstance(maximum, int)):
        raise ValueError("checkpoint workload transaction range is invalid")
    required_integer_fields = (
        "max_burst_time", "min_priority", "max_priority", "max_episode_steps"
    )
    for field in required_integer_fields:
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"checkpoint workload setting {field} is invalid")
    if (config["max_burst_time"] <= 0
            or config["max_priority"] < config["min_priority"]
            or config["max_episode_steps"] < maximum * config["max_burst_time"]):
        raise ValueError("checkpoint workload configuration is inconsistent")
    unsupported = [size for size in workload_sizes if not minimum <= size <= maximum]
    if unsupported:
        raise ValueError(
            f"checkpoint training range is {minimum}..{maximum}; "
            f"cannot evaluate transaction counts {unsupported}"
        )
    return config


def _write_csv(rows: list[dict], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(detailed: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in detailed:
        grouped.setdefault((row["transaction_count"], row["policy"]), []).append(row)

    summary = []
    for (transaction_count, policy), rows in sorted(grouped.items()):
        mean = lambda field: round(statistics.mean(row[field] for row in rows), 6)
        quantum = rows[0]["round_robin_quantum"] if policy == "round_robin" else ""
        summary.append({
            "transaction_count": transaction_count,
            "policy": policy,
            "workload_runs": len(rows),
            "round_robin_quantum": quantum,
            "mean_average_waiting_time": mean("average_waiting_time"),
            "mean_average_turnaround_time": mean("average_turnaround_time"),
            "mean_makespan": mean("makespan"),
            "mean_completed_transactions": mean("completed_transactions"),
        })
    return summary


def evaluate_variable_size(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    *,
    detailed_output: Path | str = DEFAULT_DETAILED_OUTPUT,
    summary_output: Path | str = DEFAULT_SUMMARY_OUTPUT,
    workload_sizes: Sequence[int] = WORKLOAD_SIZES,
    workloads_per_size: int = 5,
    seed: int = 7,
    round_robin_quantum: int = 1,
) -> tuple[Path, Path, list[dict], list[dict]]:
    """Evaluate DQN and four baselines on paired seeded synthetic workloads."""
    sizes = tuple(workload_sizes)
    if not sizes or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0
                        for size in sizes):
        raise ValueError("workload_sizes must contain positive integers")
    if len(set(sizes)) != len(sizes):
        raise ValueError("workload_sizes must not contain duplicates")
    if (isinstance(workloads_per_size, bool) or not isinstance(workloads_per_size, int)
            or workloads_per_size < 2):
        raise ValueError("workloads_per_size must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (isinstance(round_robin_quantum, bool)
            or not isinstance(round_robin_quantum, int)
            or round_robin_quantum <= 0):
        raise ValueError("round_robin_quantum must be a positive integer")

    agent, checkpoint = load_variable_checkpoint(checkpoint_path, device="cpu")
    workload_config = _validate_checkpoint_workload_range(checkpoint, sizes)
    detailed = []

    for count in sizes:
        for workload_index in range(workloads_per_size):
            workload_seed = seed + workload_index
            workload = generate_workload(
                random.Random(workload_seed),
                episode_index=0,
                min_transactions=count,
                max_transactions=count,
                max_burst_time=workload_config["max_burst_time"],
                min_priority=workload_config["min_priority"],
                max_priority=workload_config["max_priority"],
            )
            workload_json = json.dumps(workload, sort_keys=True, separators=(",", ":"))
            workload_id = f"n{count}_seed{workload_seed}"
            for policy in POLICIES:
                metrics = _run_policy(
                    workload,
                    policy,
                    agent=agent if policy == "dqn" else None,
                    round_robin_quantum=round_robin_quantum,
                    max_episode_steps=workload_config["max_episode_steps"],
                )
                detailed.append({
                    "workload_id": workload_id,
                    "seed": workload_seed,
                    "transaction_count": count,
                    "policy": policy,
                    "round_robin_quantum": (
                        round_robin_quantum if policy == "round_robin" else ""
                    ),
                    **metrics,
                    "workload_json": workload_json,
                })

    summary = _aggregate(detailed)
    detailed_path = Path(detailed_output)
    summary_path = Path(summary_output)
    _write_csv(detailed, detailed_path, DETAILED_FIELDS)
    _write_csv(summary, summary_path, SUMMARY_FIELDS)
    return detailed_path, summary_path, detailed, summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the variable-size DQN against CPU scheduling baselines"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--detailed-output", type=Path, default=DEFAULT_DETAILED_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--workloads-per-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--round-robin-quantum", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        detailed_path, summary_path, _, summary = evaluate_variable_size(
            args.checkpoint,
            detailed_output=args.detailed_output,
            summary_output=args.summary_output,
            workloads_per_size=args.workloads_per_size,
            seed=args.seed,
            round_robin_quantum=args.round_robin_quantum,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print("Variable-size scheduler evaluation summary")
    print("Metrics are averaged across paired seeded workloads; priority is larger-value-first.")
    for row in summary:
        print(
            f"N={row['transaction_count']:>2} {row['policy']:<12} "
            f"wait={row['mean_average_waiting_time']:.3f} "
            f"turnaround={row['mean_average_turnaround_time']:.3f} "
            f"makespan={row['mean_makespan']:.3f} "
            f"completed={row['mean_completed_transactions']:.2f}"
        )
    print(f"Detailed CSV: {detailed_path}")
    print(f"Aggregated CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
