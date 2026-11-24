"""Train a normalized, priority-free variable-size DQN as a separate ablation."""

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import torch

from drl_scheduler.agent import PriorityFreeFeatureEncoder, PriorityFreeVariableSizeDQNAgent
from drl_scheduler.env import SchedulingEnv
from drl_scheduler.train_variable_size import (
    DEFAULT_AGENT_CONFIG,
    DEFAULT_MODEL_DIR,
    METRIC_FIELDS,
    _seed_everything,
    _validate_workload_settings,
    generate_workload,
)


ARCHITECTURE = "shared_priority_free_normalized_process_mlp_v1"
FEATURE_NAMES = [
    "burst_time_normalized",
    "remaining_time_normalized",
    "waiting_time_normalized",
    "completed",
]
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "variable_dqn_priority_free_gamma1_checkpoint.pt"
DEFAULT_METRICS = DEFAULT_MODEL_DIR / "variable_priority_free_gamma1_training_metrics.csv"
GAMMA_ONE_AGENT_CONFIG = {**DEFAULT_AGENT_CONFIG, "gamma": 1.0}


def _write_metrics(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _normalization_metadata(max_transactions: int, max_burst_time: int) -> dict:
    return {
        "burst_time_scale": max_burst_time,
        "remaining_time_scale": max_burst_time,
        "waiting_time_scale": max(1, (max_transactions - 1) * max_burst_time),
        "waiting_time_scale_rule": "max(1, (max_transactions - 1) * max_burst_time)",
    }


def train_priority_free_variable_size(
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
) -> tuple[Path, Path, list[dict]]:
    """Train with the existing reward/workloads and isolated artifact names."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    _validate_workload_settings(
        min_transactions, max_transactions, max_burst_time,
        min_priority, max_priority, max_episode_steps,
    )

    _seed_everything(seed)
    workload_rng = random.Random(seed)
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / DEFAULT_CHECKPOINT.name
    metrics_path = output_dir / DEFAULT_METRICS.name
    if checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError(
            "priority-free experiment outputs already exist; choose another --model-dir"
        )

    agent_config = dict(GAMMA_ONE_AGENT_CONFIG)
    agent_config["seed"] = seed
    agent = PriorityFreeVariableSizeDQNAgent(**agent_config)
    encoder = PriorityFreeFeatureEncoder(
        max_transactions=max_transactions,
        max_burst_time=max_burst_time,
    )
    metric_rows = []

    for episode in range(episodes):
        workload = generate_workload(
            workload_rng,
            episode,
            min_transactions=min_transactions,
            max_transactions=max_transactions,
            max_burst_time=max_burst_time,
            min_priority=min_priority,
            max_priority=max_priority,
        )
        env = SchedulingEnv(workload=workload, max_episode_steps=max_episode_steps)
        features = encoder.encode(env.get_process_features())
        episode_reward = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.select_action(features, explore=True)
            _, reward, terminated, truncated, _ = env.step(action)
            next_features = encoder.encode(env.get_process_features())
            agent.remember(features, action, reward, next_features, terminated, truncated)
            agent.train_step()
            features = next_features
            episode_reward += reward

        metric_rows.append({
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
        "feature_count": PriorityFreeVariableSizeDQNAgent.FEATURE_COUNT,
        "feature_names": list(FEATURE_NAMES),
        "normalization": _normalization_metadata(max_transactions, max_burst_time),
        "model_state_dict": agent.policy_network.state_dict(),
        "target_state_dict": agent.target_network.state_dict(),
        "agent_config": {key: value for key, value in agent_config.items() if key != "seed"},
        "training_config": {"episodes": episodes, "seed": seed, "gamma": 1.0},
        "workload_config": workload_config,
        "training_steps": agent.training_steps,
    }
    torch.save(checkpoint, checkpoint_path)
    _write_metrics(metric_rows, metrics_path)
    return checkpoint_path, metrics_path, metric_rows


def load_priority_free_checkpoint(
    path: Path | str,
    *,
    device: str = "cpu",
) -> tuple[PriorityFreeVariableSizeDQNAgent, dict]:
    """Load this ablation only when its feature and normalization schemas match."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"priority-free checkpoint not found: {checkpoint_path}")
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"could not read priority-free checkpoint: {error}") from error

    required = {
        "format_version", "architecture", "feature_count", "feature_names",
        "normalization", "model_state_dict", "agent_config", "training_config",
        "workload_config",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("priority-free checkpoint is missing required metadata")
    if (checkpoint["format_version"] != 1
            or checkpoint["architecture"] != ARCHITECTURE
            or checkpoint["feature_count"] != len(FEATURE_NAMES)
            or checkpoint["feature_names"] != FEATURE_NAMES):
        raise ValueError("priority-free checkpoint architecture or feature schema is incompatible")
    config = checkpoint["workload_config"]
    if not isinstance(config, dict):
        raise ValueError("priority-free workload_config must be an object")
    for key in ("max_transactions", "max_burst_time"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"priority-free workload_config {key} must be a positive integer")
    expected_normalization = _normalization_metadata(
        config.get("max_transactions"), config.get("max_burst_time")
    )
    if checkpoint["normalization"] != expected_normalization:
        raise ValueError("priority-free checkpoint normalization does not match workload limits")
    agent_config = checkpoint["agent_config"]
    training_config = checkpoint["training_config"]
    if not isinstance(agent_config, dict) or agent_config.get("gamma") != 1.0:
        raise ValueError("priority-free agent_config must record gamma=1.0")
    if (not isinstance(training_config, dict)
            or not isinstance(training_config.get("seed"), int)
            or training_config.get("gamma") != 1.0):
        raise ValueError("priority-free training_config is invalid or has the wrong gamma")

    rebuild_config = dict(agent_config)
    rebuild_config.update({
        "epsilon": 0.0,
        "epsilon_min": 0.0,
        "seed": training_config["seed"],
        "device": device,
    })
    try:
        agent = PriorityFreeVariableSizeDQNAgent(**rebuild_config)
        agent.policy_network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "target_state_dict" in checkpoint:
            agent.target_network.load_state_dict(checkpoint["target_state_dict"], strict=True)
        else:
            agent.sync_target_network()
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"priority-free checkpoint weights are incompatible: {error}") from error
    agent.policy_network.eval()
    agent.target_network.eval()
    return agent, checkpoint


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a normalized variable-size DQN without priority features"
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
        checkpoint, metrics, rows = train_priority_free_variable_size(
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
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Priority-free gamma=1.0 training complete: {len(rows)} episodes")
    print(f"Checkpoint: {checkpoint}")
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
