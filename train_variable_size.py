"""Train a separate DQN on reproducibly generated variable-size workloads.

This workflow writes ``variable_dqn_checkpoint.pt`` and
``variable_training_metrics.csv``. It does not read or overwrite the original
``dqn_checkpoint.pt`` produced by :mod:`drl_scheduler.train`.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import torch

from drl_scheduler.agent import VariableSizeDQNAgent
from drl_scheduler.env import SchedulingEnv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "models"
DEFAULT_AGENT_CONFIG = {
    "hidden_size": 64,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 32,
    "replay_capacity": 10_000,
    "target_update_interval": 100,
    "device": "cpu",
}
FEATURE_NAMES = ["burst_time", "priority", "remaining_time", "waiting_time", "completed"]
ARCHITECTURE = "shared_per_process_mlp_v1"
METRIC_FIELDS = (
    "episode",
    "transaction_count",
    "total_burst_time",
    "reward",
    "total_waiting_time",
    "completed_count",
    "episode_length",
    "terminated",
    "truncated",
    "epsilon",
)


def _validate_workload_settings(
    min_transactions: int,
    max_transactions: int,
    max_burst_time: int,
    min_priority: int,
    max_priority: int,
    max_episode_steps: int,
) -> None:
    settings = {
        "min_transactions": min_transactions,
        "max_transactions": max_transactions,
        "max_burst_time": max_burst_time,
        "min_priority": min_priority,
        "max_priority": max_priority,
        "max_episode_steps": max_episode_steps,
    }
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in settings.values()):
        raise TypeError("workload settings must be integers")
    if min_transactions <= 0 or max_transactions < min_transactions:
        raise ValueError("transaction counts must satisfy 1 <= min_transactions <= max_transactions")
    if max_burst_time <= 0:
        raise ValueError("max_burst_time must be positive")
    if max_priority < min_priority:
        raise ValueError("priorities must satisfy min_priority <= max_priority")
    if max_episode_steps < max_transactions * max_burst_time:
        raise ValueError(
            "max_episode_steps must cover the largest possible sum of burst times"
        )


def generate_workload(
    rng: random.Random,
    episode_index: int,
    *,
    min_transactions: int = 1,
    max_transactions: int = 8,
    max_burst_time: int = 8,
    min_priority: int = 1,
    max_priority: int = 10,
) -> list[dict]:
    """Generate one workload; counts cycle through the configured range.

    Cycling guarantees that a multi-episode run covers every configured count.
    Burst times and priorities are sampled with the supplied seeded RNG. The
    synthetic IDs are for environment bookkeeping and are never agent features.
    """
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode_index must be an integer")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    generation_settings = (
        min_transactions, max_transactions, max_burst_time, min_priority, max_priority
    )
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in generation_settings):
        raise TypeError("workload settings must be integers")
    _validate_workload_settings(
        min_transactions, max_transactions, max_burst_time,
        min_priority, max_priority, max(max_transactions * max_burst_time, 1),
    )

    count_range = max_transactions - min_transactions + 1
    transaction_count = min_transactions + (episode_index % count_range)
    id_base = (episode_index + 1) * (max_transactions + 1)
    return [
        {
            "transaction_id": id_base + index,
            "burst_time": rng.randint(1, max_burst_time),
            "priority": rng.randint(min_priority, max_priority),
        }
        for index in range(transaction_count)
    ]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _write_metrics(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train_variable_size(
    episodes: int = 300,
    seed: int = 7,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    *,
    min_transactions: int = 1,
    max_transactions: int = 8,
    max_burst_time: int = 8,
    min_priority: int = 1,
    max_priority: int = 10,
    max_episode_steps: int = 100,
    agent_config: Optional[dict] = None,
    checkpoint_filename: str = "variable_dqn_checkpoint.pt",
    metrics_filename: str = "variable_training_metrics.csv",
) -> tuple[Path, Path, list[dict]]:
    """Train on varied workload sizes and save a separate checkpoint and CSV."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _validate_workload_settings(
        min_transactions, max_transactions, max_burst_time,
        min_priority, max_priority, max_episode_steps,
    )
    for filename, suffix, label in (
        (checkpoint_filename, ".pt", "checkpoint_filename"),
        (metrics_filename, ".csv", "metrics_filename"),
    ):
        if (not isinstance(filename, str) or not filename
                or Path(filename).name != filename or not filename.endswith(suffix)):
            raise ValueError(f"{label} must be a simple {suffix} filename")

    _seed_everything(seed)
    rng = random.Random(seed)
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / checkpoint_filename
    metrics_path = output_dir / metrics_filename

    config = dict(DEFAULT_AGENT_CONFIG)
    if agent_config is not None:
        config.update(agent_config)
    config["seed"] = seed
    agent = VariableSizeDQNAgent(**config)
    rows = []

    for episode in range(episodes):
        workload = generate_workload(
            rng, episode,
            min_transactions=min_transactions,
            max_transactions=max_transactions,
            max_burst_time=max_burst_time,
            min_priority=min_priority,
            max_priority=max_priority,
        )
        env = SchedulingEnv(workload=workload, max_episode_steps=max_episode_steps)
        features = env.get_process_features()
        episode_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(features, explore=True)
            _, reward, terminated, truncated, _ = env.step(action)
            next_features = env.get_process_features()
            agent.remember(
                features, action, reward, next_features, terminated, truncated
            )
            agent.train_step()
            features = next_features
            episode_reward += reward

        rows.append({
            "episode": episode + 1,
            "transaction_count": len(workload),
            "total_burst_time": sum(item["burst_time"] for item in workload),
            "reward": round(float(episode_reward), 6),
            "total_waiting_time": sum(item.waiting_time for item in env.transactions),
            "completed_count": sum(item.completed for item in env.transactions),
            "episode_length": env.elapsed_time,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "epsilon": round(float(agent.epsilon), 8),
        })

    workload_config = {
        "min_transactions": min_transactions,
        "max_transactions": max_transactions,
        "max_burst_time": max_burst_time,
        "min_priority": min_priority,
        "max_priority": max_priority,
        "max_episode_steps": max_episode_steps,
        "count_policy": "cyclic",
    }
    checkpoint = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "feature_count": VariableSizeDQNAgent.FEATURE_COUNT,
        "feature_names": FEATURE_NAMES,
        "model_state_dict": agent.policy_network.state_dict(),
        "target_state_dict": agent.target_network.state_dict(),
        "agent_config": {key: value for key, value in config.items() if key != "seed"},
        "training_config": {"episodes": episodes, "seed": seed},
        "workload_config": workload_config,
        "training_steps": agent.training_steps,
    }
    torch.save(checkpoint, checkpoint_path)
    _write_metrics(rows, metrics_path)
    return checkpoint_path, metrics_path, rows


def load_variable_checkpoint(
    path: Path | str,
    *,
    device: str = "cpu",
) -> tuple[VariableSizeDQNAgent, dict]:
    """Load a checkpoint only if its architecture and feature schema are known."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"variable-size checkpoint not found: {checkpoint_path}")
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"could not read variable-size checkpoint: {error}") from error

    required = {
        "format_version", "architecture", "feature_count", "feature_names",
        "model_state_dict", "agent_config", "training_config", "workload_config",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("variable-size checkpoint is missing required metadata")
    if checkpoint["format_version"] != 1 or checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("unsupported variable-size checkpoint format or architecture")
    if (checkpoint["feature_count"] != VariableSizeDQNAgent.FEATURE_COUNT
            or checkpoint["feature_names"] != FEATURE_NAMES):
        raise ValueError("checkpoint feature schema does not match this agent")
    if not isinstance(checkpoint["agent_config"], dict):
        raise ValueError("checkpoint agent_config must be an object")

    config = dict(checkpoint["agent_config"])
    training_config = checkpoint["training_config"]
    if not isinstance(training_config, dict) or not isinstance(training_config.get("seed"), int):
        raise ValueError("checkpoint training_config is invalid")
    config.update({
        "epsilon": 0.0,
        "epsilon_min": 0.0,
        "seed": training_config["seed"],
        "device": device,
    })
    try:
        agent = VariableSizeDQNAgent(**config)
        agent.policy_network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "target_state_dict" in checkpoint:
            agent.target_network.load_state_dict(checkpoint["target_state_dict"], strict=True)
        else:
            agent.sync_target_network()
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"variable-size checkpoint weights are incompatible: {error}") from error
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent, checkpoint


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the separate variable-size transaction scheduling DQN"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--min-transactions", type=int, default=1)
    parser.add_argument("--max-transactions", type=int, default=8)
    parser.add_argument("--max-burst-time", type=int, default=8)
    parser.add_argument("--min-priority", type=int, default=1)
    parser.add_argument("--max-priority", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        checkpoint, metrics, rows = train_variable_size(
            episodes=args.episodes,
            seed=args.seed,
            model_dir=args.model_dir,
            min_transactions=args.min_transactions,
            max_transactions=args.max_transactions,
            max_burst_time=args.max_burst_time,
            min_priority=args.min_priority,
            max_priority=args.max_priority,
            max_episode_steps=args.max_episode_steps,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Variable-size training complete: {len(rows)} episodes")
    print(f"Checkpoint: {checkpoint}")
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
