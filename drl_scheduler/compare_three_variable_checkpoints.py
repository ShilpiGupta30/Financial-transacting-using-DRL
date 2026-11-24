"""Compare three variable-size DQN checkpoints on the stored 600 workloads."""

import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Sequence

from drl_scheduler.agent import PriorityFreeFeatureEncoder
from drl_scheduler.compare_variable_checkpoints import (
    DEFAULT_WORKLOADS_CSV,
    _run_policy,
    load_existing_workloads,
)
from drl_scheduler.evaluate_variable_size_generalization import mean_and_sample_stdev
from drl_scheduler.train_priority_free_variable_size import (
    ARCHITECTURE as PRIORITY_FREE_ARCHITECTURE,
    DEFAULT_CHECKPOINT as DEFAULT_PRIORITY_FREE_CHECKPOINT,
    FEATURE_NAMES as PRIORITY_FREE_FEATURE_NAMES,
    load_priority_free_checkpoint,
)
from drl_scheduler.train_variable_size import (
    ARCHITECTURE as VARIABLE_ARCHITECTURE,
    DEFAULT_MODEL_DIR,
    FEATURE_NAMES as VARIABLE_FEATURE_NAMES,
    load_variable_checkpoint,
)


DEFAULT_GAMMA_099_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_checkpoint.pt"
DEFAULT_GAMMA_1_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_gamma1_checkpoint.pt"
DEFAULT_DETAILED_OUTPUT = DEFAULT_MODEL_DIR / "variable_three_checkpoint_comparison_detailed.csv"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_three_checkpoint_comparison_summary.csv"
DEFAULT_PAIRED_OUTPUT = DEFAULT_MODEL_DIR / "variable_three_checkpoint_comparison_pairs.csv"
DEFAULT_PAIRED_SUMMARY_OUTPUT = (
    DEFAULT_MODEL_DIR / "variable_three_checkpoint_comparison_paired_summary.csv"
)
DEFAULT_ROUND_ROBIN_QUANTUM = 1

DQN_POLICIES = (
    "dqn_gamma_0.99",
    "dqn_gamma_1.0",
    "dqn_priority_free_gamma_1.0",
)
BASELINE_POLICIES = ("fcfs", "sjf", "priority", "round_robin")
COMPARISON_POLICIES = (*DQN_POLICIES, *BASELINE_POLICIES)
PAIRED_COMPARISONS = (
    ("gamma_1.0_vs_gamma_0.99", "dqn_gamma_1.0", "dqn_gamma_0.99"),
    ("priority_free_gamma_1.0_vs_gamma_1.0",
     "dqn_priority_free_gamma_1.0", "dqn_gamma_1.0"),
    ("gamma_0.99_vs_sjf", "dqn_gamma_0.99", "sjf"),
    ("gamma_1.0_vs_sjf", "dqn_gamma_1.0", "sjf"),
    ("priority_free_gamma_1.0_vs_sjf", "dqn_priority_free_gamma_1.0", "sjf"),
)

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


def validate_checkpoint_schema(
    checkpoint: dict,
    *,
    architecture: str,
    feature_names: Sequence[str],
    feature_count: int,
    expected_gamma: float,
    label: str,
) -> None:
    """Reject checkpoint metadata that does not describe the requested agent."""
    saved_features = checkpoint.get("feature_names")
    saved_count = checkpoint.get("feature_count")
    saved_architecture = checkpoint.get("architecture")
    saved_gamma = checkpoint.get("agent_config", {}).get("gamma")
    if (saved_architecture != architecture
            or saved_features != list(feature_names)
            or saved_count != feature_count):
        raise ValueError(f"{label} checkpoint feature schema does not match its agent")
    if (not isinstance(saved_gamma, (int, float)) or isinstance(saved_gamma, bool)
            or not math.isclose(float(saved_gamma), expected_gamma,
                                rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError(
            f"{label} checkpoint must record gamma={expected_gamma}; found {saved_gamma!r}"
        )


class _PriorityFreeInferenceAdapter:
    """Encode environment rows before forwarding actions to the four-feature DQN."""

    def __init__(self, agent, encoder: PriorityFreeFeatureEncoder) -> None:
        self.agent = agent
        self.encoder = encoder

    def select_action(self, process_features, *, explore: bool) -> int:
        return self.agent.select_action(
            self.encoder.encode(process_features), explore=explore
        )


def load_comparison_agents(
    gamma_099_checkpoint: Path | str = DEFAULT_GAMMA_099_CHECKPOINT,
    gamma_1_checkpoint: Path | str = DEFAULT_GAMMA_1_CHECKPOINT,
    priority_free_checkpoint: Path | str = DEFAULT_PRIORITY_FREE_CHECKPOINT,
) -> dict:
    """Load each checkpoint through its matching loader and verify its schema."""
    gamma_099, metadata_099 = load_variable_checkpoint(gamma_099_checkpoint, device="cpu")
    gamma_1, metadata_1 = load_variable_checkpoint(gamma_1_checkpoint, device="cpu")
    priority_free, metadata_priority_free = load_priority_free_checkpoint(
        priority_free_checkpoint, device="cpu"
    )

    validate_checkpoint_schema(
        metadata_099,
        architecture=VARIABLE_ARCHITECTURE,
        feature_names=VARIABLE_FEATURE_NAMES,
        feature_count=gamma_099.FEATURE_COUNT,
        expected_gamma=0.99,
        label="gamma=0.99",
    )
    validate_checkpoint_schema(
        metadata_1,
        architecture=VARIABLE_ARCHITECTURE,
        feature_names=VARIABLE_FEATURE_NAMES,
        feature_count=gamma_1.FEATURE_COUNT,
        expected_gamma=1.0,
        label="gamma=1.0",
    )
    validate_checkpoint_schema(
        metadata_priority_free,
        architecture=PRIORITY_FREE_ARCHITECTURE,
        feature_names=PRIORITY_FREE_FEATURE_NAMES,
        feature_count=priority_free.FEATURE_COUNT,
        expected_gamma=1.0,
        label="priority-free gamma=1.0",
    )
    if (gamma_099.gamma != 0.99 or gamma_1.gamma != 1.0
            or priority_free.gamma != 1.0):
        raise ValueError("a loaded agent gamma does not match its checkpoint metadata")

    for agent in (gamma_099, gamma_1, priority_free):
        agent.policy_network.eval()

    # Use the recorded workload bounds directly, including the max_transactions=1 case.
    encoder = PriorityFreeFeatureEncoder(
        max_transactions=metadata_priority_free["workload_config"]["max_transactions"],
        max_burst_time=metadata_priority_free["workload_config"]["max_burst_time"],
    )
    return {
        "dqn_gamma_0.99": gamma_099,
        "dqn_gamma_1.0": gamma_1,
        "dqn_priority_free_gamma_1.0": _PriorityFreeInferenceAdapter(
            priority_free, encoder
        ),
        "priority_free_encoder": encoder,
        "metadata": (metadata_099, metadata_1, metadata_priority_free),
    }


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


def _pair_results(detailed: list[dict]) -> list[dict]:
    grouped: dict[str, dict[str, dict]] = {}
    for row in detailed:
        policy_rows = grouped.setdefault(row["workload_id"], {})
        if row["policy"] in policy_rows:
            raise ValueError(f"duplicate policy result for {row['workload_id']}: {row['policy']}")
        policy_rows[row["policy"]] = row

    paired = []
    for workload_id, policies in sorted(grouped.items()):
        if set(policies) != set(COMPARISON_POLICIES):
            raise ValueError(f"workload {workload_id} is missing comparison policies")
        if len({row["workload_json"] for row in policies.values()}) != 1:
            raise ValueError(f"policies received different workloads for {workload_id}")
        for name, left_name, right_name in PAIRED_COMPARISONS:
            left = policies[left_name]
            right = policies[right_name]
            difference = float(left["average_waiting_time"]) - float(
                right["average_waiting_time"]
            )
            if math.isclose(difference, 0.0, rel_tol=0.0, abs_tol=1e-9):
                difference = 0.0
                outcome = "tie"
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
                "workload_json": left["workload_json"],
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


def compare_three_checkpoints(
    workloads: Sequence[dict],
    *,
    gamma_099_checkpoint: Path | str = DEFAULT_GAMMA_099_CHECKPOINT,
    gamma_1_checkpoint: Path | str = DEFAULT_GAMMA_1_CHECKPOINT,
    priority_free_checkpoint: Path | str = DEFAULT_PRIORITY_FREE_CHECKPOINT,
    detailed_output: Path | str = DEFAULT_DETAILED_OUTPUT,
    summary_output: Path | str = DEFAULT_SUMMARY_OUTPUT,
    paired_output: Path | str = DEFAULT_PAIRED_OUTPUT,
    paired_summary_output: Path | str = DEFAULT_PAIRED_SUMMARY_OUTPUT,
    round_robin_quantum: int = DEFAULT_ROUND_ROBIN_QUANTUM,
) -> tuple[Path, Path, Path, Path, list[dict], list[dict], list[dict], list[dict]]:
    """Run three greedy agents and four baselines on identical workload records."""
    if isinstance(round_robin_quantum, bool) or not isinstance(round_robin_quantum, int):
        raise ValueError("round_robin_quantum must be a positive integer")
    if round_robin_quantum <= 0:
        raise ValueError("round_robin_quantum must be a positive integer")
    if len(workloads) < 2:
        raise ValueError("at least two workloads are required for sample standard deviations")

    agents = load_comparison_agents(
        gamma_099_checkpoint, gamma_1_checkpoint, priority_free_checkpoint
    )
    detailed = []
    for item in workloads:
        for policy in COMPARISON_POLICIES:
            agent = agents.get(policy)
            metrics = _run_policy(
                item["workload"],
                "dqn" if policy in DQN_POLICIES else policy,
                agent=agent,
                round_robin_quantum=round_robin_quantum,
            )
            detailed.append({
                "workload_id": item["workload_id"],
                "seed": item["seed"],
                "transaction_count": item["transaction_count"],
                "policy": policy,
                "round_robin_quantum": (
                    round_robin_quantum if policy == "round_robin" else ""
                ),
                **metrics,
                "workload_json": item["workload_json"],
            })

    summary = _summarize(detailed)
    paired = _pair_results(detailed)
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
        description="Compare three variable-size DQN checkpoints on the stored paired workloads"
    )
    parser.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS_CSV)
    parser.add_argument("--gamma-099-checkpoint", type=Path,
                        default=DEFAULT_GAMMA_099_CHECKPOINT)
    parser.add_argument("--gamma-1-checkpoint", type=Path, default=DEFAULT_GAMMA_1_CHECKPOINT)
    parser.add_argument("--priority-free-checkpoint", type=Path,
                        default=DEFAULT_PRIORITY_FREE_CHECKPOINT)
    parser.add_argument("--detailed-output", type=Path, default=DEFAULT_DETAILED_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--paired-output", type=Path, default=DEFAULT_PAIRED_OUTPUT)
    parser.add_argument("--paired-summary-output", type=Path,
                        default=DEFAULT_PAIRED_SUMMARY_OUTPUT)
    parser.add_argument("--round-robin-quantum", type=int,
                        default=DEFAULT_ROUND_ROBIN_QUANTUM)
    args = parser.parse_args(argv)
    try:
        workloads = load_existing_workloads(args.workloads)
        paths = compare_three_checkpoints(
            workloads,
            gamma_099_checkpoint=args.gamma_099_checkpoint,
            gamma_1_checkpoint=args.gamma_1_checkpoint,
            priority_free_checkpoint=args.priority_free_checkpoint,
            detailed_output=args.detailed_output,
            summary_output=args.summary_output,
            paired_output=args.paired_output,
            paired_summary_output=args.paired_summary_output,
            round_robin_quantum=args.round_robin_quantum,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print("Three-checkpoint comparison (greedy DQN; paired workloads)")
    print(f"Workloads: {len(workloads)} from {args.workloads}")
    print("Mean +/- sample standard deviation for average waiting and turnaround:")
    for row in paths[5]:
        print(
            f"N={row['transaction_count']:>2} {row['policy']:<29} "
            f"wait={row['mean_average_waiting_time']:.3f} +/- "
            f"{row['stdev_average_waiting_time']:.3f}; "
            f"turnaround={row['mean_average_turnaround_time']:.3f} +/- "
            f"{row['stdev_average_turnaround_time']:.3f}"
        )
    print("Paired waiting-time win/tie/loss counts:")
    for row in paths[7]:
        print(
            f"N={row['transaction_count']:>2} {row['comparison']:<43} "
            f"wins={row['wins']} ties={row['ties']} losses={row['losses']}"
        )
    print("Makespan is not a useful discriminator: all arrivals are at time zero and the CPU never idles.")
    for label, path in zip(
        ("Detailed", "Summary", "Paired", "Paired summary"), paths[:4]
    ):
        print(f"{label} CSV: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
