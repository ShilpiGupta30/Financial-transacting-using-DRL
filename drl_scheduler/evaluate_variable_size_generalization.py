"""Run a larger paired-workload evaluation of the variable-size DQN.

The default experiment uses 100 workloads for each requested transaction count
and stores policy-level detailed results, policy summary statistics, and paired
DQN-versus-SJF waiting-time differences in new CSV files.
"""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Optional, Sequence

from drl_scheduler.evaluate_variable_size import (
    POLICIES,
    WORKLOAD_SIZES,
    _run_policy,
    _validate_checkpoint_workload_range,
)
from drl_scheduler.train_variable_size import (
    DEFAULT_MODEL_DIR,
    generate_workload,
    load_variable_checkpoint,
)


DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_checkpoint.pt"
DEFAULT_DETAILED_OUTPUT = DEFAULT_MODEL_DIR / "variable_generalization_detailed.csv"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_generalization_summary.csv"
DEFAULT_PAIRED_OUTPUT = DEFAULT_MODEL_DIR / "variable_generalization_pairs.csv"
DEFAULT_PAIRED_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_generalization_paired_summary.csv"
DEFAULT_WORKLOADS_PER_SIZE = 100
# The saved model used training seed 7. A distinct default seed range avoids
# accidentally evaluating the exact deterministic training stream.
DEFAULT_EVALUATION_SEED = 10_000

DETAILED_FIELDS = (
    "workload_id", "seed", "transaction_count", "policy", "round_robin_quantum",
    "average_waiting_time", "average_turnaround_time", "makespan",
    "completed_transactions", "workload_json",
)
SUMMARY_FIELDS = (
    "transaction_count", "policy", "workload_runs", "round_robin_quantum",
    "mean_average_waiting_time", "stdev_average_waiting_time",
    "mean_average_turnaround_time", "stdev_average_turnaround_time",
    "mean_makespan", "mean_completed_transactions",
)
PAIRED_FIELDS = (
    "workload_id", "seed", "transaction_count", "dqn_average_waiting_time",
    "sjf_average_waiting_time", "dqn_minus_sjf_waiting_time", "outcome",
    "workload_json",
)
PAIRED_SUMMARY_FIELDS = (
    "transaction_count", "workload_runs", "dqn_beats_sjf", "ties", "dqn_loses_to_sjf",
    "dqn_win_rate", "tie_rate", "loss_rate",
    "mean_dqn_minus_sjf_waiting_time", "stdev_dqn_minus_sjf_waiting_time",
)


def mean_and_sample_stdev(values: Sequence[float]) -> tuple[float, float]:
    """Return a rounded mean and sample standard deviation for multiple runs."""
    if len(values) < 2:
        raise ValueError("at least two values are required to calculate sample standard deviation")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("statistical input values must be finite")
    return round(statistics.mean(values), 6), round(statistics.stdev(values), 6)


def summarize_policy_results(detailed: list[dict]) -> list[dict]:
    """Aggregate workload-level mean metrics by transaction count and policy."""
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in detailed:
        grouped.setdefault((row["transaction_count"], row["policy"]), []).append(row)

    summary = []
    for (transaction_count, policy), rows in sorted(grouped.items()):
        wait_mean, wait_sd = mean_and_sample_stdev(
            [row["average_waiting_time"] for row in rows]
        )
        turnaround_mean, turnaround_sd = mean_and_sample_stdev(
            [row["average_turnaround_time"] for row in rows]
        )
        quantum = rows[0]["round_robin_quantum"] if policy == "round_robin" else ""
        summary.append({
            "transaction_count": transaction_count,
            "policy": policy,
            "workload_runs": len(rows),
            "round_robin_quantum": quantum,
            "mean_average_waiting_time": wait_mean,
            "stdev_average_waiting_time": wait_sd,
            "mean_average_turnaround_time": turnaround_mean,
            "stdev_average_turnaround_time": turnaround_sd,
            "mean_makespan": round(statistics.mean(row["makespan"] for row in rows), 6),
            "mean_completed_transactions": round(
                statistics.mean(row["completed_transactions"] for row in rows), 6
            ),
        })
    return summary


def pair_dqn_and_sjf(detailed: list[dict]) -> list[dict]:
    """Calculate one DQN-minus-SJF waiting difference per paired workload."""
    grouped: dict[str, dict[str, dict]] = {}
    for row in detailed:
        group = grouped.setdefault(row["workload_id"], {})
        if row["policy"] in group:
            raise ValueError(f"duplicate policy {row['policy']} for {row['workload_id']}")
        group[row["policy"]] = row

    paired = []
    for workload_id, policies in sorted(grouped.items()):
        if set(policies) != set(POLICIES):
            raise ValueError(f"workload {workload_id} does not contain every policy")
        dqn = policies["dqn"]
        sjf = policies["sjf"]
        if dqn["workload_json"] != sjf["workload_json"]:
            raise ValueError(f"DQN and SJF did not receive the same workload: {workload_id}")
        difference = (
            float(dqn["average_waiting_time"])
            - float(sjf["average_waiting_time"])
        )
        if math.isclose(difference, 0.0, rel_tol=0.0, abs_tol=1e-9):
            outcome = "tie"
            difference = 0.0
        elif difference < 0.0:
            outcome = "dqn_beats_sjf"
        else:
            outcome = "dqn_loses_to_sjf"
        paired.append({
            "workload_id": workload_id,
            "seed": dqn["seed"],
            "transaction_count": dqn["transaction_count"],
            "dqn_average_waiting_time": dqn["average_waiting_time"],
            "sjf_average_waiting_time": sjf["average_waiting_time"],
            "dqn_minus_sjf_waiting_time": round(difference, 6),
            "outcome": outcome,
            "workload_json": dqn["workload_json"],
        })
    return paired


def summarize_paired_results(paired: list[dict]) -> list[dict]:
    """Count paired wins/ties/losses and summarize signed waiting differences."""
    grouped: dict[int, list[dict]] = {}
    for row in paired:
        grouped.setdefault(row["transaction_count"], []).append(row)

    summary = []
    for transaction_count, rows in sorted(grouped.items()):
        wins = sum(row["outcome"] == "dqn_beats_sjf" for row in rows)
        ties = sum(row["outcome"] == "tie" for row in rows)
        losses = sum(row["outcome"] == "dqn_loses_to_sjf" for row in rows)
        difference_mean, difference_sd = mean_and_sample_stdev(
            [row["dqn_minus_sjf_waiting_time"] for row in rows]
        )
        count = len(rows)
        summary.append({
            "transaction_count": transaction_count,
            "workload_runs": count,
            "dqn_beats_sjf": wins,
            "ties": ties,
            "dqn_loses_to_sjf": losses,
            "dqn_win_rate": round(wins / count, 6),
            "tie_rate": round(ties / count, 6),
            "loss_rate": round(losses / count, 6),
            "mean_dqn_minus_sjf_waiting_time": difference_mean,
            "stdev_dqn_minus_sjf_waiting_time": difference_sd,
        })
    return summary


def _write_csv(rows: list[dict], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_generalization(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    *,
    detailed_output: Path | str = DEFAULT_DETAILED_OUTPUT,
    summary_output: Path | str = DEFAULT_SUMMARY_OUTPUT,
    paired_output: Path | str = DEFAULT_PAIRED_OUTPUT,
    paired_summary_output: Path | str = DEFAULT_PAIRED_SUMMARY_OUTPUT,
    workload_sizes: Sequence[int] = WORKLOAD_SIZES,
    workloads_per_size: int = DEFAULT_WORKLOADS_PER_SIZE,
    seed: int = DEFAULT_EVALUATION_SEED,
    round_robin_quantum: int = 1,
) -> tuple[Path, Path, Path, Path, list[dict], list[dict], list[dict], list[dict]]:
    """Run a reproducible paired experiment and save four new result tables."""
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
        for index in range(workloads_per_size):
            workload_seed = seed + index
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

    summary = summarize_policy_results(detailed)
    paired = pair_dqn_and_sjf(detailed)
    paired_summary = summarize_paired_results(paired)
    output_paths = tuple(Path(item) for item in (
        detailed_output, summary_output, paired_output, paired_summary_output
    ))
    _write_csv(detailed, output_paths[0], DETAILED_FIELDS)
    _write_csv(summary, output_paths[1], SUMMARY_FIELDS)
    _write_csv(paired, output_paths[2], PAIRED_FIELDS)
    _write_csv(paired_summary, output_paths[3], PAIRED_SUMMARY_FIELDS)
    return (*output_paths, detailed, summary, paired, paired_summary)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a reproducible generalization experiment for the variable-size DQN"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--detailed-output", type=Path, default=DEFAULT_DETAILED_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--paired-output", type=Path, default=DEFAULT_PAIRED_OUTPUT)
    parser.add_argument("--paired-summary-output", type=Path,
                        default=DEFAULT_PAIRED_SUMMARY_OUTPUT)
    parser.add_argument("--workloads-per-size", type=int, default=DEFAULT_WORKLOADS_PER_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_EVALUATION_SEED,
                        help="workload seed base (default 10000, separate from training seed 7)")
    parser.add_argument("--round-robin-quantum", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        paths = evaluate_generalization(
            args.checkpoint,
            detailed_output=args.detailed_output,
            summary_output=args.summary_output,
            paired_output=args.paired_output,
            paired_summary_output=args.paired_summary_output,
            workloads_per_size=args.workloads_per_size,
            seed=args.seed,
            round_robin_quantum=args.round_robin_quantum,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    detailed_path, summary_path, paired_path, paired_summary_path = paths[:4]
    summary, paired_summary = paths[5], paths[7]
    print("Variable-size DQN generalization experiment")
    print(f"Workloads per transaction count: {args.workloads_per_size}; seed range: "
          f"{args.seed}..{args.seed + args.workloads_per_size - 1}")
    print("Waiting/turnaround values show mean +/- sample standard deviation.")
    for row in summary:
        print(
            f"N={row['transaction_count']:>2} {row['policy']:<12} "
            f"wait={row['mean_average_waiting_time']:.3f} +/- "
            f"{row['stdev_average_waiting_time']:.3f}; "
            f"turnaround={row['mean_average_turnaround_time']:.3f} +/- "
            f"{row['stdev_average_turnaround_time']:.3f}"
        )
    print("Paired DQN vs SJF average-waiting results:")
    for row in paired_summary:
        print(
            f"N={row['transaction_count']:>2}: beats={row['dqn_beats_sjf']}, "
            f"ties={row['ties']}, loses={row['dqn_loses_to_sjf']} "
            f"(of {row['workload_runs']}); mean DQN-SJF="
            f"{row['mean_dqn_minus_sjf_waiting_time']:.3f}"
        )
    print("Makespan is not a useful discriminator here: all arrivals are at time 0 "
          "and this single-CPU workload never idles.")
    print(f"Detailed CSV: {detailed_path}")
    print(f"Policy summary CSV: {summary_path}")
    print(f"Paired differences CSV: {paired_path}")
    print(f"Paired summary CSV: {paired_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
