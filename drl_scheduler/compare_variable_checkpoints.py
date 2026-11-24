"""Compare the gamma=0.99 and gamma=1.0 variable-size DQNs on paired workloads."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Optional, Sequence

from drl_scheduler.evaluate_variable_size import (
    POLICIES,
    WORKLOAD_SIZES,
    _run_policy,
)
from drl_scheduler.env import SchedulingEnv
from drl_scheduler.evaluate_variable_size_generalization import (
    DEFAULT_WORKLOADS_PER_SIZE,
    mean_and_sample_stdev,
)
from drl_scheduler.train_variable_size import (
    DEFAULT_MODEL_DIR,
    load_variable_checkpoint,
)


DEFAULT_WORKLOADS_CSV = DEFAULT_MODEL_DIR / "variable_generalization_detailed.csv"
DEFAULT_GAMMA_099_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_checkpoint.pt"
DEFAULT_GAMMA_1_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_gamma1_checkpoint.pt"
DEFAULT_DETAILED_OUTPUT = DEFAULT_MODEL_DIR / "variable_gamma_comparison_detailed.csv"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_gamma_comparison_summary.csv"
DEFAULT_PAIRED_OUTPUT = DEFAULT_MODEL_DIR / "variable_gamma_comparison_pairs.csv"
DEFAULT_PAIRED_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_gamma_comparison_paired_summary.csv"

BASELINE_POLICIES = ("fcfs", "sjf", "priority", "round_robin")
COMPARISON_POLICIES = ("dqn_gamma_0.99", "dqn_gamma_1.0", *BASELINE_POLICIES)

DETAILED_FIELDS = (
    "workload_id", "seed", "transaction_count", "policy", "round_robin_quantum",
    "average_waiting_time", "average_turnaround_time", "makespan",
    "completed_transactions", "workload_json",
)
SUMMARY_FIELDS = (
    "transaction_count", "policy", "workload_runs", "round_robin_quantum",
    "mean_average_waiting_time", "stdev_average_waiting_time",
    "mean_average_turnaround_time", "stdev_average_turnaround_time",
)
PAIRED_FIELDS = (
    "workload_id", "seed", "transaction_count", "comparison",
    "left_average_waiting_time", "right_average_waiting_time",
    "left_minus_right_waiting_time", "outcome", "workload_json",
)
PAIRED_SUMMARY_FIELDS = (
    "transaction_count", "comparison", "workload_runs", "wins", "ties", "losses",
    "mean_left_minus_right_waiting_time", "stdev_left_minus_right_waiting_time",
)


def load_existing_workloads(
    path: Path | str = DEFAULT_WORKLOADS_CSV,
    *,
    workload_sizes: Sequence[int] = WORKLOAD_SIZES,
    workloads_per_size: int = DEFAULT_WORKLOADS_PER_SIZE,
) -> list[dict]:
    """Read and validate the exact workload set stored in the prior experiment."""
    grouped: dict[str, list[dict]] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if not row.get("workload_id"):
                raise ValueError("generalization CSV contains a row without workload_id")
            grouped.setdefault(row["workload_id"], []).append(row)

    expected_sizes = tuple(workload_sizes)
    expected_total = len(expected_sizes) * workloads_per_size
    if len(grouped) != expected_total:
        raise ValueError(
            f"expected {expected_total} unique workloads in {path}, found {len(grouped)}"
        )

    workloads = []
    counts_by_size = {size: 0 for size in expected_sizes}
    for workload_id, rows in sorted(grouped.items()):
        if len(rows) != len(POLICIES) or {row["policy"] for row in rows} != set(POLICIES):
            raise ValueError(f"workload {workload_id} does not have exactly one row per policy")
        json_values = {row["workload_json"] for row in rows}
        if len(json_values) != 1:
            raise ValueError(f"policies have different workload data for {workload_id}")
        try:
            workload = json.loads(next(iter(json_values)))
            transaction_count = int(rows[0]["transaction_count"])
            seed = int(rows[0]["seed"])
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid workload data for {workload_id}: {error}") from error
        if (not isinstance(workload, list) or len(workload) != transaction_count
                or transaction_count not in counts_by_size):
            raise ValueError(f"workload {workload_id} has an unexpected transaction count")
        # The environment validates unique IDs and positive integer burst times.
        SchedulingEnv(workload=workload)
        counts_by_size[transaction_count] += 1
        workloads.append({
            "workload_id": workload_id,
            "seed": seed,
            "transaction_count": transaction_count,
            "workload": workload,
            "workload_json": next(iter(json_values)),
        })

    if any(count != workloads_per_size for count in counts_by_size.values()):
        raise ValueError(
            f"expected {workloads_per_size} workloads for each size; got {counts_by_size}"
        )
    return workloads


def _load_expected_agent(path: Path | str, expected_gamma: float):
    agent, checkpoint = load_variable_checkpoint(path, device="cpu")
    saved_gamma = checkpoint.get("agent_config", {}).get("gamma")
    if not isinstance(saved_gamma, (int, float)) or not math.isclose(
        float(saved_gamma), expected_gamma, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"checkpoint {path} must have gamma={expected_gamma}; found {saved_gamma!r}"
        )
    if not math.isclose(agent.gamma, expected_gamma, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"loaded agent from {path} has unexpected gamma={agent.gamma}")
    agent.policy_network.eval()
    return agent


def _summarize(detailed: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in detailed:
        grouped.setdefault((row["transaction_count"], row["policy"]), []).append(row)
    summary = []
    for (size, policy), rows in sorted(grouped.items()):
        wait_mean, wait_sd = mean_and_sample_stdev(
            [float(row["average_waiting_time"]) for row in rows]
        )
        turn_mean, turn_sd = mean_and_sample_stdev(
            [float(row["average_turnaround_time"]) for row in rows]
        )
        summary.append({
            "transaction_count": size,
            "policy": policy,
            "workload_runs": len(rows),
            "round_robin_quantum": rows[0]["round_robin_quantum"]
            if policy == "round_robin" else "",
            "mean_average_waiting_time": wait_mean,
            "stdev_average_waiting_time": wait_sd,
            "mean_average_turnaround_time": turn_mean,
            "stdev_average_turnaround_time": turn_sd,
        })
    return summary


def _pair_comparisons(detailed: list[dict]) -> list[dict]:
    by_workload: dict[str, dict[str, dict]] = {}
    for row in detailed:
        by_workload.setdefault(row["workload_id"], {})[row["policy"]] = row

    comparisons = (
        ("dqn_gamma_1.0_vs_dqn_gamma_0.99", "dqn_gamma_1.0", "dqn_gamma_0.99"),
        ("dqn_gamma_0.99_vs_sjf", "dqn_gamma_0.99", "sjf"),
        ("dqn_gamma_1.0_vs_sjf", "dqn_gamma_1.0", "sjf"),
    )
    paired = []
    for workload_id, policies in sorted(by_workload.items()):
        if set(policies) != set(COMPARISON_POLICIES):
            raise ValueError(f"workload {workload_id} has incomplete policy results")
        workload_jsons = {row["workload_json"] for row in policies.values()}
        if len(workload_jsons) != 1:
            raise ValueError(f"policies did not receive the same workload: {workload_id}")
        for name, left_policy, right_policy in comparisons:
            left = policies[left_policy]
            right = policies[right_policy]
            difference = float(left["average_waiting_time"]) - float(
                right["average_waiting_time"]
            )
            if math.isclose(difference, 0.0, rel_tol=0.0, abs_tol=1e-9):
                outcome = "tie"
                difference = 0.0
            elif difference < 0:
                outcome = "win"
            else:
                outcome = "loss"
            paired.append({
                "workload_id": workload_id,
                "seed": left["seed"],
                "transaction_count": left["transaction_count"],
                "comparison": name,
                "left_average_waiting_time": left["average_waiting_time"],
                "right_average_waiting_time": right["average_waiting_time"],
                "left_minus_right_waiting_time": round(difference, 6),
                "outcome": outcome,
                "workload_json": next(iter(workload_jsons)),
            })
    return paired


def _summarize_pairs(paired: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in paired:
        grouped.setdefault((row["transaction_count"], row["comparison"]), []).append(row)
    summary = []
    for (size, comparison), rows in sorted(grouped.items()):
        mean, stdev = mean_and_sample_stdev(
            [float(row["left_minus_right_waiting_time"]) for row in rows]
        )
        summary.append({
            "transaction_count": size,
            "comparison": comparison,
            "workload_runs": len(rows),
            "wins": sum(row["outcome"] == "win" for row in rows),
            "ties": sum(row["outcome"] == "tie" for row in rows),
            "losses": sum(row["outcome"] == "loss" for row in rows),
            "mean_left_minus_right_waiting_time": mean,
            "stdev_left_minus_right_waiting_time": stdev,
        })
    return summary


def _write_csv(rows: list[dict], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_checkpoints(
    workloads: Sequence[dict],
    *,
    gamma_099_checkpoint: Path | str = DEFAULT_GAMMA_099_CHECKPOINT,
    gamma_1_checkpoint: Path | str = DEFAULT_GAMMA_1_CHECKPOINT,
    detailed_output: Path | str = DEFAULT_DETAILED_OUTPUT,
    summary_output: Path | str = DEFAULT_SUMMARY_OUTPUT,
    paired_output: Path | str = DEFAULT_PAIRED_OUTPUT,
    paired_summary_output: Path | str = DEFAULT_PAIRED_SUMMARY_OUTPUT,
    round_robin_quantum: int = 1,
) -> tuple[Path, Path, Path, Path, list[dict], list[dict], list[dict], list[dict]]:
    """Run both greedy DQNs and four baselines on each supplied workload."""
    if isinstance(round_robin_quantum, bool) or not isinstance(round_robin_quantum, int):
        raise ValueError("round_robin_quantum must be a positive integer")
    if round_robin_quantum <= 0:
        raise ValueError("round_robin_quantum must be a positive integer")
    if len(workloads) < 2:
        raise ValueError("at least two workloads are required for sample standard deviations")

    gamma_099_agent = _load_expected_agent(gamma_099_checkpoint, 0.99)
    gamma_1_agent = _load_expected_agent(gamma_1_checkpoint, 1.0)
    detailed = []
    for item in workloads:
        workload = item["workload"]
        size = item["transaction_count"]
        workload_json = item["workload_json"]
        policy_agents = {
            "dqn_gamma_0.99": gamma_099_agent,
            "dqn_gamma_1.0": gamma_1_agent,
        }
        for policy in COMPARISON_POLICIES:
            if policy.startswith("dqn_"):
                actual_policy = "dqn"
                agent = policy_agents[policy]
            else:
                actual_policy = policy
                agent = None
            metrics = _run_policy(
                workload,
                actual_policy,
                agent=agent,
                round_robin_quantum=round_robin_quantum,
            )
            detailed.append({
                "workload_id": item["workload_id"],
                "seed": item["seed"],
                "transaction_count": size,
                "policy": policy,
                "round_robin_quantum": (
                    round_robin_quantum if policy == "round_robin" else ""
                ),
                **metrics,
                "workload_json": workload_json,
            })

    summary = _summarize(detailed)
    paired = _pair_comparisons(detailed)
    paired_summary = _summarize_pairs(paired)
    paths = tuple(Path(value) for value in (
        detailed_output, summary_output, paired_output, paired_summary_output
    ))
    _write_csv(detailed, paths[0], DETAILED_FIELDS)
    _write_csv(summary, paths[1], SUMMARY_FIELDS)
    _write_csv(paired, paths[2], PAIRED_FIELDS)
    _write_csv(paired_summary, paths[3], PAIRED_SUMMARY_FIELDS)
    return (*paths, detailed, summary, paired, paired_summary)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare gamma=0.99 and gamma=1.0 DQNs on saved paired workloads"
    )
    parser.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS_CSV)
    parser.add_argument("--gamma-099-checkpoint", type=Path,
                        default=DEFAULT_GAMMA_099_CHECKPOINT)
    parser.add_argument("--gamma-1-checkpoint", type=Path, default=DEFAULT_GAMMA_1_CHECKPOINT)
    parser.add_argument("--detailed-output", type=Path, default=DEFAULT_DETAILED_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--paired-output", type=Path, default=DEFAULT_PAIRED_OUTPUT)
    parser.add_argument("--paired-summary-output", type=Path,
                        default=DEFAULT_PAIRED_SUMMARY_OUTPUT)
    parser.add_argument("--round-robin-quantum", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        workloads = load_existing_workloads(args.workloads)
        paths = compare_checkpoints(
            workloads,
            gamma_099_checkpoint=args.gamma_099_checkpoint,
            gamma_1_checkpoint=args.gamma_1_checkpoint,
            detailed_output=args.detailed_output,
            summary_output=args.summary_output,
            paired_output=args.paired_output,
            paired_summary_output=args.paired_summary_output,
            round_robin_quantum=args.round_robin_quantum,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    detailed, summary, paired, paired_summary = paths[4], paths[5], paths[6], paths[7]
    print("Paired variable-size checkpoint comparison (greedy DQN)")
    print(f"Workloads: {len(workloads)} from {args.workloads}")
    print("Mean average waiting time and turnaround time by transaction count:")
    for row in summary:
        print(
            f"N={row['transaction_count']:>2} {row['policy']:<16} "
            f"wait={row['mean_average_waiting_time']:.3f} "
            f"turnaround={row['mean_average_turnaround_time']:.3f}"
        )
    print("Paired waiting-time win/tie/loss counts by transaction count:")
    for row in paired_summary:
        print(
            f"N={row['transaction_count']:>2} {row['comparison']:<34} "
            f"wins={row['wins']} ties={row['ties']} losses={row['losses']}"
        )
    print(f"Detailed results: {paths[0]}")
    print(f"Summary: {paths[1]}")
    print(f"Paired results: {paths[2]}")
    print(f"Paired summary: {paths[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
