"""Evaluate three saved variable-size DQNs on schema-aware held-out workloads.

The priority-aware models are held out by ordered (burst, priority) profiles.
The priority-free model gets a stricter cohort held out by ordered burst
sequences, matching the inputs it actually receives.
"""

import argparse
import csv
import itertools
import json
import random
import statistics
from pathlib import Path
from typing import Sequence

from drl_scheduler.compare_three_variable_checkpoints import (
    COMPARISON_POLICIES,
    DQN_POLICIES,
    load_comparison_agents,
)
from drl_scheduler.compare_variable_checkpoints import _run_policy
from drl_scheduler.evaluate_variable_size import WORKLOAD_SIZES
from drl_scheduler.train_variable_size import DEFAULT_MODEL_DIR, generate_workload


DEFAULT_HOLDOUT_SEED = 20_000
DEFAULT_WORKLOADS_PER_SIZE = 100
DEFAULT_ROUND_ROBIN_QUANTUM = 1
DEFAULT_GAMMA_099_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_checkpoint.pt"
DEFAULT_GAMMA_1_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_gamma1_checkpoint.pt"
DEFAULT_PRIORITY_FREE_CHECKPOINT = (
    DEFAULT_MODEL_DIR / "variable_dqn_priority_free_gamma1_checkpoint.pt"
)
DEFAULT_REFERENCE_CSVS = (
    DEFAULT_MODEL_DIR / "variable_evaluation_detailed.csv",
    DEFAULT_MODEL_DIR / "variable_generalization_detailed.csv",
    DEFAULT_MODEL_DIR / "variable_gamma_comparison_detailed.csv",
    DEFAULT_MODEL_DIR / "variable_three_checkpoint_comparison_detailed.csv",
)
DEFAULT_DETAILED_OUTPUT = DEFAULT_MODEL_DIR / "variable_heldout_generalization_detailed.csv"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_heldout_generalization_summary.csv"
DEFAULT_PAIRED_OUTPUT = DEFAULT_MODEL_DIR / "variable_heldout_sjf_pairs.csv"
DEFAULT_PAIRED_SUMMARY_OUTPUT = DEFAULT_MODEL_DIR / "variable_heldout_sjf_paired_summary.csv"
DEFAULT_REPORT_OUTPUT = DEFAULT_MODEL_DIR / "variable_heldout_generalization.md"

# Corrected, schema-aware outputs. Existing held-out files remain untouched.
DEFAULT_CORRECTED_DETAILED_OUTPUT = (
    DEFAULT_MODEL_DIR / "variable_heldout_corrected_detailed.csv"
)
DEFAULT_CORRECTED_SUMMARY_OUTPUT = (
    DEFAULT_MODEL_DIR / "variable_heldout_corrected_summary.csv"
)
DEFAULT_CORRECTED_PAIRED_OUTPUT = (
    DEFAULT_MODEL_DIR / "variable_heldout_corrected_sjf_pairs.csv"
)
DEFAULT_CORRECTED_PAIRED_SUMMARY_OUTPUT = (
    DEFAULT_MODEL_DIR / "variable_heldout_corrected_sjf_paired_summary.csv"
)
DEFAULT_CORRECTED_AVAILABILITY_OUTPUT = (
    DEFAULT_MODEL_DIR / "variable_heldout_corrected_availability.csv"
)
DEFAULT_CORRECTED_REPORT_OUTPUT = (
    DEFAULT_MODEL_DIR / "variable_heldout_corrected_generalization.md"
)

PRIORITY_AWARE_POLICIES = (
    "dqn_gamma_0.99", "dqn_gamma_1.0", "fcfs", "sjf", "priority", "round_robin",
)
STRICT_PRIORITY_FREE_COHORT = "strict_priority_free_input"
PRIORITY_AWARE_COHORT = "priority_aware_profile"

DETAILED_FIELDS = (
    "cohort", "workload_id", "seed_kind", "seed_value", "profile_index",
    "transaction_count", "policy", "round_robin_quantum",
    "average_waiting_time", "average_turnaround_time", "makespan",
    "completed_transactions", "workload_json",
)
SUMMARY_FIELDS = (
    "cohort", "transaction_count", "policy", "workload_runs",
    "mean_average_waiting_time", "stdev_average_waiting_time",
    "mean_average_turnaround_time", "stdev_average_turnaround_time",
)
PAIRED_FIELDS = (
    "cohort", "workload_id", "seed_kind", "seed_value", "profile_index",
    "transaction_count", "comparison",
    "dqn_average_waiting_time", "sjf_average_waiting_time",
    "dqn_minus_sjf_waiting_time", "outcome", "workload_json",
)
AVAILABILITY_FIELDS = (
    "transaction_count", "priority_aware_profile_space",
    "previous_priority_aware_profiles", "available_priority_aware_profiles",
    "available_priority_aware_profiles_after_cohort_separation",
    "priority_aware_workloads_used", "priority_free_burst_sequence_space",
    "previous_priority_free_burst_sequences",
    "available_priority_free_burst_sequences", "strict_priority_free_workloads_used",
)
PAIRED_SUMMARY_FIELDS = (
    "cohort", "transaction_count", "comparison", "workload_runs",
    "wins", "ties", "losses",
    "mean_dqn_minus_sjf_waiting_time", "stdev_dqn_minus_sjf_waiting_time",
)


def workload_signature(workload: Sequence[dict]) -> tuple[tuple[int, int], ...]:
    """Return the model-visible, input-ordered workload features."""
    return tuple((int(item["burst_time"]), int(item["priority"])) for item in workload)


def burst_signature(workload: Sequence[dict]) -> tuple[int, ...]:
    """Return the ordered burst-time sequence visible to the priority-free agent."""
    return tuple(int(item["burst_time"]) for item in workload)


def training_signatures(checkpoint_metadata: Sequence[dict]) -> set[tuple[tuple[int, int], ...]]:
    """Rebuild the exact seeded workload streams recorded by the checkpoints."""
    signatures: set[tuple[tuple[int, int], ...]] = set()
    if not checkpoint_metadata:
        raise ValueError("checkpoint metadata is required to reconstruct training workloads")
    for metadata in checkpoint_metadata:
        config = metadata.get("workload_config")
        training = metadata.get("training_config")
        if not isinstance(config, dict) or not isinstance(training, dict):
            raise ValueError("checkpoint is missing workload or training configuration")
        required = ("min_transactions", "max_transactions", "max_burst_time",
                    "min_priority", "max_priority")
        if any(field not in config for field in required):
            raise ValueError("checkpoint workload configuration is incomplete")
        episodes, seed = training.get("episodes"), training.get("seed")
        if (isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0
                or isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("checkpoint training configuration is invalid")
        rng = random.Random(seed)
        for episode in range(episodes):
            workload = generate_workload(
                rng, episode,
                min_transactions=config["min_transactions"],
                max_transactions=config["max_transactions"],
                max_burst_time=config["max_burst_time"],
                min_priority=config["min_priority"],
                max_priority=config["max_priority"],
            )
            signatures.add(workload_signature(workload))
    return signatures


def read_reference_signatures(paths: Sequence[Path | str]) -> set[tuple[tuple[int, int], ...]]:
    """Read feature signatures from existing detailed evaluation CSVs."""
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for path in paths:
        reference_path = Path(path)
        if not reference_path.is_file():
            raise FileNotFoundError(f"reference evaluation CSV not found: {reference_path}")
        with reference_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if "workload_json" not in (reader.fieldnames or ()):
                raise ValueError(f"{reference_path} has no workload_json column")
            for row in reader:
                try:
                    workload = json.loads(row["workload_json"])
                    signatures.add(workload_signature(workload))
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid workload in {reference_path}: {error}") from error
    return signatures


def read_reference_burst_signatures(paths: Sequence[Path | str]) -> set[tuple[int, ...]]:
    """Read ordered burst sequences from the same prior detailed evaluation files."""
    signatures: set[tuple[int, ...]] = set()
    for path in paths:
        reference_path = Path(path)
        if not reference_path.is_file():
            raise FileNotFoundError(f"reference evaluation CSV not found: {reference_path}")
        with reference_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if "workload_json" not in (reader.fieldnames or ()):
                raise ValueError(f"{reference_path} has no workload_json column")
            for row in reader:
                try:
                    signatures.add(burst_signature(json.loads(row["workload_json"])))
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid workload in {reference_path}: {error}") from error
    return signatures


def generate_heldout_workloads(
    *,
    seen_signatures: set[tuple[tuple[int, int], ...]],
    seen_burst_sequences: set[tuple[int, ...]] | None = None,
    excluded_burst_sequences: set[tuple[int, ...]] | None = None,
    seed: int = DEFAULT_HOLDOUT_SEED,
    workload_sizes: Sequence[int] = WORKLOAD_SIZES,
    workloads_per_size: int = DEFAULT_WORKLOADS_PER_SIZE,
    max_burst_time: int = 8,
    min_priority: int = 1,
    max_priority: int = 10,
    cohort: str = PRIORITY_AWARE_COHORT,
) -> list[dict]:
    """Generate as many unique profiles as available, up to the requested cap.

    When ``seen_burst_sequences`` is supplied, no workload is accepted if its
    priority-free input sequence has appeared previously. N=1 pair profiles
    are enumerated exactly. For other cases a deterministic seeded sampler is
    used; the corresponding availability table reports the exact finite-space
    count, while this function returns at most ``workloads_per_size`` examples.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(workloads_per_size, bool) or not isinstance(workloads_per_size, int) \
            or workloads_per_size < 2:
        raise ValueError("workloads_per_size must be an integer of at least 2")
    sizes = tuple(workload_sizes)
    if not sizes or len(set(sizes)) != len(sizes) or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sizes):
        raise ValueError("workload_sizes must contain distinct positive integers")
    if max_burst_time <= 0 or max_priority < min_priority:
        raise ValueError("invalid workload feature bounds")

    heldout = []
    held_signatures = set()
    held_bursts = set()
    strict_priority_free = seen_burst_sequences is not None
    seen_burst_sequences = seen_burst_sequences or set()
    excluded_burst_sequences = excluded_burst_sequences or set()
    for count in sizes:
        pair_space_size = (max_burst_time * (max_priority - min_priority + 1)) ** count
        previous_pair_count = sum(len(profile) == count for profile in seen_signatures)
        pair_available = max(0, pair_space_size - previous_pair_count)
        if strict_priority_free:
            burst_space_size = max_burst_time ** count
            previous_burst_count = sum(
                len(profile) == count for profile in seen_burst_sequences
            )
            available = max(0, burst_space_size - previous_burst_count)
        else:
            burst_space_size = max_burst_time ** count
            previous_burst_count = 0
            priorities_per_sequence = (max_priority - min_priority + 1) ** count
            excluded_pairs = sum(
                len(profile) == count for profile in excluded_burst_sequences
            ) * priorities_per_sequence
            available = max(0, pair_available - excluded_pairs)
        target_count = min(workloads_per_size, available)
        profiles_with_seeds: list[tuple[tuple[tuple[int, int], ...], int, str]] = []

        if count == 1 and not strict_priority_free:
            candidates = [((burst, priority),)
                          for burst, priority in itertools.product(
                              range(1, max_burst_time + 1),
                              range(min_priority, max_priority + 1))]
            ordering_seed = seed + count * 1_000_003
            random.Random(ordering_seed).shuffle(candidates)
            selected = [profile for profile in candidates
                        if profile not in seen_signatures
                        and tuple(burst for burst, _ in profile)
                        not in excluded_burst_sequences][:target_count]
            profiles_with_seeds = [
                (profile, ordering_seed, "profile_order_seed") for profile in selected
            ]
        elif target_count:
            selected_for_size = set()
            selected_bursts_for_size = set()
            candidate_seed = seed + count * 1_000_003
            attempts = 0
            while len(profiles_with_seeds) < target_count and attempts < 1_000_000:
                rng = random.Random(candidate_seed)
                workload = generate_workload(
                    rng, 0, min_transactions=count, max_transactions=count,
                    max_burst_time=max_burst_time,
                    min_priority=min_priority, max_priority=max_priority,
                )
                profile = workload_signature(workload)
                burst_profile = tuple(burst for burst, _ in profile)
                if strict_priority_free:
                    passes_burst_holdout = (
                        burst_profile not in seen_burst_sequences
                        and burst_profile not in excluded_burst_sequences
                        and burst_profile not in held_bursts
                        and burst_profile not in selected_bursts_for_size
                    )
                else:
                    passes_burst_holdout = burst_profile not in excluded_burst_sequences
                if (passes_burst_holdout and profile not in seen_signatures
                        and profile not in held_signatures
                        and profile not in selected_for_size):
                    kind = "candidate_rng_seed"
                    profiles_with_seeds.append((profile, candidate_seed, kind))
                    selected_for_size.add(profile)
                    selected_bursts_for_size.add(burst_profile)
                candidate_seed += 1
                attempts += 1
            if len(profiles_with_seeds) != target_count:
                raise RuntimeError(
                    f"generated {len(profiles_with_seeds)} of {target_count} unseen profiles "
                    f"for N={count} after {attempts} attempts"
                )

        for index, (profile, profile_seed, seed_kind) in enumerate(profiles_with_seeds):
            burst_profile = tuple(burst for burst, _ in profile)
            if (profile in seen_signatures or profile in held_signatures
                    or burst_profile in excluded_burst_sequences
                    or (strict_priority_free and
                        (burst_profile in seen_burst_sequences or burst_profile in held_bursts))):
                raise ValueError("held-out generation duplicated a prior model-visible profile")
            workload = [
                {
                    "transaction_id": (count * 1_000_000) + index * count + item_index + 1,
                    "burst_time": burst,
                    "priority": priority,
                }
                for item_index, (burst, priority) in enumerate(profile)
            ]
            workload_json = json.dumps(workload, sort_keys=True, separators=(",", ":"))
            heldout.append({
                "workload_id": f"{cohort}_n{count}_heldout_{index:04d}",
                "seed_kind": seed_kind,
                "seed": profile_seed,
                "profile_index": index if seed_kind == "profile_order_seed" else "",
                "transaction_count": count,
                "workload": workload,
                "workload_json": workload_json,
            })
            held_signatures.add(profile)
            held_bursts.add(burst_profile)
    return heldout


def validate_heldout_separation(
    workloads: Sequence[dict], seen_signatures: set[tuple[tuple[int, int], ...]],
    seen_burst_sequences: set[tuple[int, ...]] | None = None,
) -> None:
    """Reject pair-profile duplicates and, optionally, burst-only duplicates."""
    current = set()
    current_bursts = set()
    for item in workloads:
        profile = workload_signature(item["workload"])
        burst_profile = burst_signature(item["workload"])
        if profile in seen_signatures:
            raise ValueError(f"held-out workload overlaps prior data: {item['workload_id']}")
        if profile in current:
            raise ValueError(f"held-out workload is duplicated: {item['workload_id']}")
        if seen_burst_sequences is not None:
            if burst_profile in seen_burst_sequences:
                raise ValueError(
                    f"held-out burst sequence overlaps prior data: {item['workload_id']}"
                )
            if burst_profile in current_bursts:
                raise ValueError(
                    f"held-out burst sequence is duplicated: {item['workload_id']}"
                )
            current_bursts.add(burst_profile)
        current.add(profile)


def summarize_results(detailed: list[dict]) -> list[dict]:
    """Aggregate workload-level average metrics by size and algorithm."""
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in detailed:
        grouped.setdefault((row["cohort"], int(row["transaction_count"]),
                            row["policy"]), []).append(row)
    result = []
    for (cohort, size, policy), rows in sorted(grouped.items()):
        waits = [float(row["average_waiting_time"]) for row in rows]
        turns = [float(row["average_turnaround_time"]) for row in rows]
        result.append({
            "cohort": cohort,
            "transaction_count": size,
            "policy": policy,
            "workload_runs": len(rows),
            "mean_average_waiting_time": round(statistics.mean(waits), 6),
            "stdev_average_waiting_time": round(statistics.stdev(waits), 6),
            "mean_average_turnaround_time": round(statistics.mean(turns), 6),
            "stdev_average_turnaround_time": round(statistics.stdev(turns), 6),
        })
    return result


def pair_against_sjf(
    detailed: list[dict], policies: Sequence[str] = COMPARISON_POLICIES
) -> tuple[list[dict], list[dict]]:
    """Produce paired waiting-time differences for each DQN versus SJF."""
    comparisons = [
        (f"{policy}_vs_sjf", policy)
        for policy in DQN_POLICIES if policy in policies
    ]
    expected_policies = set(policies)
    grouped: dict[str, dict[str, dict]] = {}
    for row in detailed:
        bucket = grouped.setdefault(row["workload_id"], {})
        if row["policy"] in bucket:
            raise ValueError(f"duplicate policy result for {row['workload_id']}")
        bucket[row["policy"]] = row

    pairs = []
    for workload_id, policies in sorted(grouped.items()):
        if set(policies) != expected_policies:
            raise ValueError(f"incomplete policy results for {workload_id}")
        if len({row["workload_json"] for row in policies.values()}) != 1:
            raise ValueError(f"policies received different workloads for {workload_id}")
        sjf = policies["sjf"]
        for comparison, dqn_policy in comparisons:
            dqn = policies[dqn_policy]
            difference = float(dqn["average_waiting_time"]) - float(
                sjf["average_waiting_time"]
            )
            if abs(difference) <= 1e-9:
                difference, outcome = 0.0, "tie"
            elif difference < 0:
                outcome = "win"
            else:
                outcome = "loss"
            pairs.append({
                "workload_id": workload_id,
                "cohort": dqn["cohort"],
                "seed_kind": dqn["seed_kind"],
                "seed_value": dqn["seed_value"],
                "profile_index": dqn["profile_index"],
                "transaction_count": dqn["transaction_count"],
                "comparison": comparison,
                "dqn_average_waiting_time": dqn["average_waiting_time"],
                "sjf_average_waiting_time": sjf["average_waiting_time"],
                "dqn_minus_sjf_waiting_time": round(difference, 6),
                "outcome": outcome,
                "workload_json": dqn["workload_json"],
            })

    by_group: dict[tuple[str, int, str], list[dict]] = {}
    for row in pairs:
        by_group.setdefault((row["cohort"], row["transaction_count"],
                             row["comparison"]), []).append(row)
    summary = []
    for (cohort, size, comparison), rows in sorted(by_group.items()):
        differences = [float(row["dqn_minus_sjf_waiting_time"]) for row in rows]
        summary.append({
            "cohort": cohort,
            "transaction_count": size,
            "comparison": comparison,
            "workload_runs": len(rows),
            "wins": sum(row["outcome"] == "win" for row in rows),
            "ties": sum(row["outcome"] == "tie" for row in rows),
            "losses": sum(row["outcome"] == "loss" for row in rows),
            "mean_dqn_minus_sjf_waiting_time": round(statistics.mean(differences), 6),
            "stdev_dqn_minus_sjf_waiting_time": round(statistics.stdev(differences), 6),
        })
    return pairs, summary


def evaluate_workloads(
    workloads: Sequence[dict], agents: dict, *, round_robin_quantum: int = 1,
    policies: Sequence[str] = COMPARISON_POLICIES,
    cohort: str = "test",
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Run all three greedy DQNs and all four baselines on identical inputs."""
    detailed = []
    for item in workloads:
        for policy in policies:
            metrics = _run_policy(
                item["workload"], "dqn" if policy in DQN_POLICIES else policy,
                agent=agents.get(policy) if policy in DQN_POLICIES else None,
                round_robin_quantum=round_robin_quantum,
            )
            detailed.append({
                "cohort": item.get("cohort", cohort),
                "workload_id": item["workload_id"],
                "seed_kind": item.get("seed_kind", "test_seed"),
                "seed_value": item["seed"],
                "profile_index": item.get("profile_index", ""),
                "transaction_count": item["transaction_count"], "policy": policy,
                "round_robin_quantum": round_robin_quantum
                if policy == "round_robin" else "",
                **metrics, "workload_json": item["workload_json"],
            })
    summary = summarize_results(detailed)
    pairs, paired_summary = pair_against_sjf(detailed, policies=policies)
    return detailed, summary, pairs, paired_summary


def _write_csv(rows: list[dict], path: Path, fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    summary: list[dict], paired_summary: list[dict], workloads_by_size: dict[int, int],
    path: Path,
) -> None:
    """Write a compact readout while leaving all earlier reports untouched."""
    by_size_policy = {
        (int(row["transaction_count"]), row["policy"]): row for row in summary
    }
    lines = [
        "# Held-out variable-size DQN generalization",
        "",
        "All three saved checkpoints were evaluated greedily alongside FCFS, "
        "non-preemptive SJF, Priority, and Round Robin (quantum 1). Every policy "
        "received the same workload for each run. Workload separation used the "
        "ordered burst/priority feature profile, excluding transaction IDs.",
        "",
        "The holdout uses seed 20000. Training streams were reconstructed from "
        "checkpoint metadata, and previous evaluation profiles were read from "
        "the existing detailed CSVs. No profile overlaps those sources.",
        "For N=1 the full feature space contains only 80 profiles; prior work "
        "covered 71, leaving nine unique held-out cases. The other sizes use "
        "100 unique workloads each.",
        "",
        "## Mean average waiting time / turnaround time",
        "",
        "Values are workload-level means, shown as `waiting / turnaround`.",
        "",
        "| Transactions | Runs | DQN γ=.99 | DQN γ=1.0 | Priority-free γ=1.0 | FCFS | SJF | Priority | RR q=1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    policies = (
        "dqn_gamma_0.99", "dqn_gamma_1.0", "dqn_priority_free_gamma_1.0",
        "fcfs", "sjf", "priority", "round_robin",
    )
    for size in sorted(workloads_by_size):
        cells = []
        for policy in policies:
            row = by_size_policy[(size, policy)]
            cells.append(f"{row['mean_average_waiting_time']:.3f} / "
                         f"{row['mean_average_turnaround_time']:.3f}")
        lines.append(f"| {size} | {workloads_by_size[size]} | " + " | ".join(cells) + " |")

    paired_by_size = {
        (int(row["transaction_count"]), row["comparison"]): row
        for row in paired_summary
    }
    lines.extend([
        "",
        "## Paired waiting-time outcomes versus SJF",
        "",
        "Counts are DQN wins / ties / losses; lower waiting time is a win.",
        "",
        "| Transactions | DQN γ=.99 | DQN γ=1.0 | Priority-free γ=1.0 |",
        "|---:|---:|---:|---:|",
    ])
    comparisons = (
        "dqn_gamma_0.99_vs_sjf", "dqn_gamma_1.0_vs_sjf",
        "dqn_priority_free_gamma_1.0_vs_sjf",
    )
    for size in sorted(workloads_by_size):
        values = []
        for comparison in comparisons:
            row = paired_by_size[(size, comparison)]
            values.append(f"{row['wins']} / {row['ties']} / {row['losses']}")
        lines.append(f"| {size} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "No DQN checkpoint had a paired waiting-time win over SJF in this "
        "held-out sample. That is evidence about these synthetic distributions "
        "and saved policies only; it is not a claim about all workloads.",
        "All jobs arrive at time zero and the CPU does not idle, so makespan is "
        "the sum of burst times for every policy and is not a useful discriminator.",
        "",
        "Detailed per-workload metrics and paired differences are in the sibling CSV files.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_availability_rows(
    workload_sizes: Sequence[int],
    seen_signatures: set[tuple[tuple[int, int], ...]],
    seen_burst_sequences: set[tuple[int, ...]],
    profile_workloads: Sequence[dict],
    strict_workloads: Sequence[dict],
    *,
    max_burst_time: int,
    min_priority: int,
    max_priority: int,
) -> list[dict]:
    """Count exact unseen profiles in both finite initial-feature spaces."""
    rows = []
    for count in workload_sizes:
        pair_space = (max_burst_time * (max_priority - min_priority + 1)) ** count
        burst_space = max_burst_time ** count
        prior_pairs = sum(len(profile) == count for profile in seen_signatures)
        prior_bursts = sum(len(profile) == count for profile in seen_burst_sequences)
        strict_burst_count = sum(
            len(burst_signature(item["workload"])) == count for item in strict_workloads
        )
        priority_combinations_per_burst = (max_priority - min_priority + 1) ** count
        rows.append({
            "transaction_count": count,
            "priority_aware_profile_space": pair_space,
            "previous_priority_aware_profiles": prior_pairs,
            "available_priority_aware_profiles": max(0, pair_space - prior_pairs),
            "available_priority_aware_profiles_after_cohort_separation": max(
                0, pair_space - prior_pairs
                - strict_burst_count * priority_combinations_per_burst
            ),
            "priority_aware_workloads_used": sum(
                item["transaction_count"] == count for item in profile_workloads
            ),
            "priority_free_burst_sequence_space": burst_space,
            "previous_priority_free_burst_sequences": prior_bursts,
            "available_priority_free_burst_sequences": max(0, burst_space - prior_bursts),
            "strict_priority_free_workloads_used": sum(
                item["transaction_count"] == count for item in strict_workloads
            ),
        })
    return rows


def write_corrected_markdown_report(
    summary: list[dict], paired_summary: list[dict], availability: list[dict],
    path: Path,
) -> None:
    """Write the corrected, schema-aware result interpretation."""
    lines = [
        "# Corrected variable-size DQN held-out experiment",
        "",
        "All DQN policies use greedy inference. Baselines are FCFS, non-preemptive "
        "SJF, Priority, and Round Robin with quantum 1. Policies within each "
        "cohort receive identical workloads.",
        "",
        "Training workloads are reconstructed from checkpoint seed/configuration "
        "metadata. Prior evaluation inputs are read from the existing detailed "
        "CSVs. The base holdout seed is 20000.",
        "",
        "## Exact availability and sample counts",
        "",
        "The availability columns count unseen ordered profiles in each finite "
        "input space. Samples are capped at 100 and are never repeated to meet "
        "that cap.",
        "",
        "| N | Pair profiles available | After cohort separation | Pair used | Burst sequences available | Strict used |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in availability:
        lines.append(
            f"| {row['transaction_count']} | {row['available_priority_aware_profiles']} | "
            f"{row['available_priority_aware_profiles_after_cohort_separation']} | "
            f"{row['priority_aware_workloads_used']} | "
            f"{row['available_priority_free_burst_sequences']} | "
            f"{row['strict_priority_free_workloads_used']} |"
        )
    lines.extend([
        "",
        "There are no unseen single-transaction burst values, so strict "
        "priority-free generalization cannot be tested at N=1. At N=2, the "
        "strict cohort includes only the available unseen burst sequences; its "
        "sample is smaller than the 100-workload cap.",
        "",
        "## Priority-aware profile holdout",
        "",
        "The gamma=0.99 and gamma=1.0 models exclude previously seen ordered "
        "`(burst_time, priority)` profiles. Entries show mean average waiting / "
        "turnaround time and workload count.",
        "",
        "| N | Algorithm | Wait / turnaround | Runs |",
        "|---:|---|---:|---:|",
    ])
    for row in summary:
        if row["cohort"] == PRIORITY_AWARE_COHORT:
            lines.append(
                f"| {row['transaction_count']} | {row['policy']} | "
                f"{row['mean_average_waiting_time']:.3f} / "
                f"{row['mean_average_turnaround_time']:.3f} | {row['workload_runs']} |"
            )
    lines.extend([
        "",
        "## Strict priority-free, model-input-aware holdout",
        "",
        "The priority-free model excludes every ordered burst-time sequence "
        "seen in its training or prior evaluation inputs, regardless of priority. "
        "The other models and baselines are evaluated on this same strict cohort "
        "for paired comparison.",
        "",
        "| N | Algorithm | Wait / turnaround | Runs |",
        "|---:|---|---:|---:|",
    ])
    for row in summary:
        if row["cohort"] == STRICT_PRIORITY_FREE_COHORT:
            lines.append(
                f"| {row['transaction_count']} | {row['policy']} | "
                f"{row['mean_average_waiting_time']:.3f} / "
                f"{row['mean_average_turnaround_time']:.3f} | {row['workload_runs']} |"
            )
    lines.extend([
        "",
        "## Paired waiting-time outcomes against SJF",
        "",
        "Counts are DQN wins / ties / losses; lower waiting time is a win.",
        "",
        "| Cohort | N | DQN | Wins / ties / losses |",
        "|---|---:|---|---:|",
    ])
    for row in paired_summary:
        lines.append(
            f"| {row['cohort']} | {row['transaction_count']} | "
            f"{row['comparison'].removesuffix('_vs_sjf')} | "
            f"{row['wins']} / {row['ties']} / {row['losses']} |"
        )
    lines.extend([
        "",
        "These results describe the sampled synthetic workloads and saved "
        "policies only. They do not establish generalization to real transaction "
        "traffic or superiority over SJF. Since all jobs arrive at time zero and "
        "the CPU never idles, makespan is not a useful policy discriminator.",
        "",
        "For N=1 profile enumeration, CSV `seed_kind` is `profile_order_seed`; "
        "`seed_value` controls enumeration order, not an individual workload draw. "
        "For other sizes, `seed_kind` is `candidate_rng_seed`.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_heldout_experiment(
    *, seed: int = DEFAULT_HOLDOUT_SEED,
    workloads_per_size: int = DEFAULT_WORKLOADS_PER_SIZE,
    workload_sizes: Sequence[int] = WORKLOAD_SIZES,
    reference_csvs: Sequence[Path | str] = DEFAULT_REFERENCE_CSVS,
    detailed_output: Path | str = DEFAULT_CORRECTED_DETAILED_OUTPUT,
    summary_output: Path | str = DEFAULT_CORRECTED_SUMMARY_OUTPUT,
    paired_output: Path | str = DEFAULT_CORRECTED_PAIRED_OUTPUT,
    paired_summary_output: Path | str = DEFAULT_CORRECTED_PAIRED_SUMMARY_OUTPUT,
    availability_output: Path | str = DEFAULT_CORRECTED_AVAILABILITY_OUTPUT,
    report_output: Path | str = DEFAULT_CORRECTED_REPORT_OUTPUT,
    gamma_099_checkpoint: Path | str = DEFAULT_GAMMA_099_CHECKPOINT,
    gamma_1_checkpoint: Path | str = DEFAULT_GAMMA_1_CHECKPOINT,
    priority_free_checkpoint: Path | str = DEFAULT_PRIORITY_FREE_CHECKPOINT,
    round_robin_quantum: int = DEFAULT_ROUND_ROBIN_QUANTUM,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], tuple[Path, ...]]:
    if (isinstance(round_robin_quantum, bool)
            or not isinstance(round_robin_quantum, int)
            or round_robin_quantum <= 0):
        raise ValueError("round_robin_quantum must be a positive integer")
    agents = load_comparison_agents(
        gamma_099_checkpoint, gamma_1_checkpoint, priority_free_checkpoint
    )
    metadata = agents["metadata"]
    training_seen = training_signatures(metadata)
    evaluation_seen = read_reference_signatures(reference_csvs)
    seen = training_seen | evaluation_seen
    training_bursts = {
        tuple(burst for burst, _priority in profile) for profile in training_seen
    }
    evaluation_bursts = read_reference_burst_signatures(reference_csvs)
    seen_bursts = training_bursts | evaluation_bursts
    workload_config = metadata[0]["workload_config"]
    for other in metadata[1:]:
        if other["workload_config"] != workload_config:
            raise ValueError("the three checkpoints have incompatible workload configurations")
    common_options = dict(
        seed=seed, workload_sizes=workload_sizes,
        workloads_per_size=workloads_per_size,
        max_burst_time=workload_config["max_burst_time"],
        min_priority=workload_config["min_priority"],
        max_priority=workload_config["max_priority"],
    )
    strict_workloads = generate_heldout_workloads(
        seen_signatures=seen, seen_burst_sequences=seen_bursts,
        cohort=STRICT_PRIORITY_FREE_COHORT, **common_options,
    )
    strict_burst_profiles = {burst_signature(item["workload"]) for item in strict_workloads}
    profile_workloads = generate_heldout_workloads(
        seen_signatures=seen, excluded_burst_sequences=strict_burst_profiles,
        cohort=PRIORITY_AWARE_COHORT, **common_options,
    )
    validate_heldout_separation(profile_workloads, seen)
    if strict_burst_profiles & {
        burst_signature(item["workload"]) for item in profile_workloads
    }:
        raise ValueError("the two held-out cohorts share a priority-free burst sequence")
    validate_heldout_separation(strict_workloads, seen, seen_bursts)

    profile_results = evaluate_workloads(
        profile_workloads, agents, round_robin_quantum=round_robin_quantum,
        policies=PRIORITY_AWARE_POLICIES, cohort=PRIORITY_AWARE_COHORT,
    )
    strict_results = evaluate_workloads(
        strict_workloads, agents, round_robin_quantum=round_robin_quantum,
        policies=COMPARISON_POLICIES, cohort=STRICT_PRIORITY_FREE_COHORT,
    )
    detailed = profile_results[0] + strict_results[0]
    summary = profile_results[1] + strict_results[1]
    pairs = profile_results[2] + strict_results[2]
    paired_summary = profile_results[3] + strict_results[3]
    availability = build_availability_rows(
        workload_sizes, seen, seen_bursts, profile_workloads, strict_workloads,
        max_burst_time=workload_config["max_burst_time"],
        min_priority=workload_config["min_priority"],
        max_priority=workload_config["max_priority"],
    )
    paths = tuple(Path(path) for path in (
        detailed_output, summary_output, paired_output, paired_summary_output,
        availability_output, report_output,
    ))
    if len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("output paths must be distinct")
    _write_csv(detailed, paths[0], DETAILED_FIELDS)
    _write_csv(summary, paths[1], SUMMARY_FIELDS)
    _write_csv(pairs, paths[2], PAIRED_FIELDS)
    _write_csv(paired_summary, paths[3], PAIRED_SUMMARY_FIELDS)
    _write_csv(availability, paths[4], AVAILABILITY_FIELDS)
    write_corrected_markdown_report(summary, paired_summary, availability, paths[5])
    return detailed, summary, pairs, paired_summary, availability, paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--workloads-per-size", type=int, default=DEFAULT_WORKLOADS_PER_SIZE)
    parser.add_argument("--round-robin-quantum", type=int, default=1)
    parser.add_argument("--reference-csv", action="append", dest="reference_csvs")
    args = parser.parse_args(argv)
    try:
        result = run_heldout_experiment(
            seed=args.seed, workloads_per_size=args.workloads_per_size,
            reference_csvs=args.reference_csvs or DEFAULT_REFERENCE_CSVS,
            round_robin_quantum=args.round_robin_quantum,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    detailed, summary, _, paired_summary, availability, paths = result
    print("Schema-aware held-out variable-size DQN comparison (greedy inference)")
    print("Available and used profiles by transaction count:")
    for row in availability:
        print(f"N={row['transaction_count']:>2} pair profiles: "
              f"{row['available_priority_aware_profiles']} available, "
              f"{row['priority_aware_workloads_used']} used; burst sequences: "
              f"{row['available_priority_free_burst_sequences']} available, "
              f"{row['strict_priority_free_workloads_used']} used")
    print("Mean average waiting time / turnaround time by cohort, size, and policy:")
    for row in summary:
        print(f"{row['cohort']:<28} N={row['transaction_count']:>2} "
              f"{row['policy']:<30} wait={row['mean_average_waiting_time']:.3f} "
              f"turnaround={row['mean_average_turnaround_time']:.3f}")
    print("Paired DQN-versus-SJF waiting-time outcomes:")
    for row in paired_summary:
        print(f"{row['cohort']:<28} N={row['transaction_count']:>2} "
              f"{row['comparison']:<39} "
              f"wins={row['wins']} ties={row['ties']} losses={row['losses']}")
    print("All-arrival-zero, non-idling makespan is not a useful policy discriminator.")
    for label, path in zip(("Detailed", "Summary", "Paired", "Paired summary",
                            "Availability", "Markdown report"), paths):
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
