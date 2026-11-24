"""Compare the saved DQN policy with simple scheduling baselines.

Every policy runs one unit per environment action on the same default
SchedulingEnv workload. Priority ties retain the original workload order; a
larger numeric priority runs first. Round Robin uses a one-unit quantum.
"""

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Optional

import torch

from drl_scheduler.agent import DQNAgent
from drl_scheduler.env import SchedulingEnv
from drl_scheduler.train import DEFAULT_MODEL_DIR, _load_checkpoint


DEFAULT_OUTPUT = DEFAULT_MODEL_DIR / "scheduler_comparison.csv"
ROUND_ROBIN_QUANTUM = 1
RESULT_FIELDS = (
    "policy",
    "reward",
    "total_waiting_time",
    "completed_count",
    "episode_length",
)


def fcfs_action(ready: list) -> int:
    """Choose the oldest unfinished process (original workload order)."""
    if not ready:
        raise ValueError("no process is ready")
    return 0


def sjf_action(ready: list) -> int:
    """Choose the least remaining burst; ties preserve workload order."""
    if not ready:
        raise ValueError("no process is ready")
    return min(range(len(ready)), key=lambda index: ready[index].remaining_time)


def priority_action(ready: list) -> int:
    """Choose highest numeric priority first; ties preserve workload order."""
    if not ready:
        raise ValueError("no process is ready")
    return max(range(len(ready)), key=lambda index: ready[index].priority)


def _run_fixed_policy(env: SchedulingEnv, policy: str, quantum: int) -> dict:
    env.reset()
    reward_total = 0.0
    terminated = truncated = False
    rr_queue = deque(process.transaction_id for process in env.transactions)
    rr_remaining = quantum

    while not (terminated or truncated):
        ready = [process for process in env.transactions if not process.completed]
        if policy == "fcfs":
            action = fcfs_action(ready)
        elif policy == "sjf":
            action = sjf_action(ready)
        elif policy == "priority_larger_first":
            action = priority_action(ready)
        elif policy == "round_robin":
            while rr_queue and next(
                process for process in env.transactions
                if process.transaction_id == rr_queue[0]
            ).completed:
                rr_queue.popleft()
            selected_id = rr_queue[0]
            action = next(
                index for index, process in enumerate(ready)
                if process.transaction_id == selected_id
            )
        else:
            raise ValueError(f"unknown baseline policy: {policy}")

        _, reward, terminated, truncated, info = env.step(action)
        if info.get("invalid_action"):
            raise RuntimeError(f"{policy} selected an invalid action")
        reward_total += reward

        if policy == "round_robin":
            rr_remaining -= 1
            selected = next(
                process for process in env.transactions
                if process.transaction_id == info["transaction_id"]
            )
            if selected.completed or rr_remaining == 0:
                rr_queue.popleft()
                if not selected.completed:
                    rr_queue.append(selected.transaction_id)
                rr_remaining = quantum

    return _metrics(policy, env, reward_total, terminated, truncated)


def _metrics(policy: str, env: SchedulingEnv, reward: float,
             terminated: bool, truncated: bool) -> dict:
    if not terminated or truncated:
        raise RuntimeError(f"{policy} did not complete the workload")
    return {
        "policy": policy,
        "reward": round(float(reward), 6),
        "total_waiting_time": sum(process.waiting_time for process in env.transactions),
        "completed_count": sum(process.completed for process in env.transactions),
        "episode_length": env.elapsed_time,
    }


def _run_dqn(checkpoint_path: Path, seed: int) -> dict:
    checkpoint = _load_checkpoint(checkpoint_path)
    env = SchedulingEnv()
    if (len(env.get_observation()) != checkpoint["observation_size"]
            or len(env.transactions) != checkpoint["action_count"]):
        raise ValueError("checkpoint dimensions do not match the default SchedulingEnv")

    config = dict(checkpoint["agent_config"])
    config.update({"epsilon": 0.0, "epsilon_min": 0.0, "seed": seed, "device": "cpu"})
    agent = DQNAgent.from_environment(env, **config)
    agent.policy_network.load_state_dict(checkpoint["model_state_dict"])
    agent.policy_network.eval()

    observation = env.reset()
    reward_total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.select_action(observation, explore=False)
        observation, reward, terminated, truncated, info = env.step(action)
        if info.get("invalid_action"):
            raise RuntimeError("DQN selected an invalid action")
        reward_total += reward
    return _metrics("dqn", env, reward_total, terminated, truncated)


def _write_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def compare_schedulers(
    checkpoint_path: Path | str = DEFAULT_MODEL_DIR / "dqn_checkpoint.pt",
    output_path: Path | str = DEFAULT_OUTPUT,
    *,
    round_robin_quantum: int = ROUND_ROBIN_QUANTUM,
    seed: int = 7,
) -> tuple[Path, list[dict]]:
    """Run DQN and four deterministic baselines on the default workload."""
    if (isinstance(round_robin_quantum, bool)
            or not isinstance(round_robin_quantum, int)
            or round_robin_quantum <= 0):
        raise ValueError("round_robin_quantum must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    torch.manual_seed(seed)
    rows = [_run_dqn(Path(checkpoint_path), seed)]
    for policy in ("fcfs", "sjf", "priority_larger_first"):
        rows.append(_run_fixed_policy(SchedulingEnv(), policy, round_robin_quantum))
    round_robin = _run_fixed_policy(
        SchedulingEnv(), "round_robin", round_robin_quantum
    )
    round_robin["policy"] = f"round_robin_q{round_robin_quantum}"
    rows.append(round_robin)

    output = Path(output_path)
    _write_results(rows, output)
    return output, rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the saved DQN with deterministic scheduling baselines"
    )
    parser.add_argument("--checkpoint", type=Path,
                        default=DEFAULT_MODEL_DIR / "dqn_checkpoint.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--round-robin-quantum", type=int,
                        default=ROUND_ROBIN_QUANTUM,
                        help="time units per turn (default: 1)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        output, rows = compare_schedulers(
            args.checkpoint, args.output,
            round_robin_quantum=args.round_robin_quantum,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Comparison complete: {len(rows)} policies")
    for result in rows:
        print(
            f"{result['policy']}: reward={result['reward']}, "
            f"waiting={result['total_waiting_time']}, "
            f"completed={result['completed_count']}, "
            f"length={result['episode_length']}"
        )
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
