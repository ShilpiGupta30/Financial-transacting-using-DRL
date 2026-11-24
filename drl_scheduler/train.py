"""Train and evaluate the project's small DQN transaction scheduler.

Run from the repository root with ``python -m drl_scheduler.train --help``.
Training artifacts are written beneath ``drl_scheduler/models`` by default.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import torch

from drl_scheduler.agent import DQNAgent
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
    "batch_size": 16,
    "replay_capacity": 10_000,
    "target_update_interval": 50,
    "device": "cpu",
}
METRIC_FIELDS = (
    "episode",
    "reward",
    "total_waiting_time",
    "completed_count",
    "episode_length",
    "terminated",
    "truncated",
    "epsilon",
)


def _set_seed(seed: int) -> None:
    """Seed the random sources used by the current environment and agent."""
    random.seed(seed)
    torch.manual_seed(seed)


def _episode_metrics(env: SchedulingEnv, episode: int, reward: float,
                     terminated: bool, truncated: bool,
                     epsilon: Optional[float]) -> dict:
    return {
        "episode": episode,
        "reward": round(float(reward), 6),
        "total_waiting_time": sum(item.waiting_time for item in env.transactions),
        "completed_count": sum(item.completed for item in env.transactions),
        "episode_length": env.elapsed_time,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "epsilon": "" if epsilon is None else round(float(epsilon), 8),
    }


def _write_metrics(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def train(
    episodes: int = 300,
    seed: int = 7,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    *,
    agent_config: Optional[dict] = None,
) -> tuple[Path, Path, list[dict]]:
    """Train for ``episodes`` and save a checkpoint and CSV metrics.

    Returns ``(checkpoint_path, metrics_path, episode_rows)``. The optional
    agent_config is useful for small, fast unit tests; normal callers can use
    the student-friendly defaults.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    _set_seed(seed)
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dqn_checkpoint.pt"
    metrics_path = output_dir / "training_metrics.csv"

    env = SchedulingEnv()
    config = dict(DEFAULT_AGENT_CONFIG)
    if agent_config is not None:
        config.update(agent_config)
    config["seed"] = seed
    agent = DQNAgent.from_environment(env, **config)
    rows = []

    for episode in range(1, episodes + 1):
        observation = env.reset()
        episode_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(observation, explore=True)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.remember(
                observation, action, reward, next_observation, terminated, truncated
            )
            agent.train_step()
            observation = next_observation
            episode_reward += reward

        rows.append(
            _episode_metrics(
                env, episode, episode_reward, terminated, truncated, agent.epsilon
            )
        )

    checkpoint = {
        "model_state_dict": agent.policy_network.state_dict(),
        "agent_config": {key: value for key, value in config.items() if key != "seed"},
        "observation_size": agent.observation_size,
        "action_count": agent.action_count,
        "training": {"episodes": episodes, "seed": seed},
    }
    torch.save(checkpoint, checkpoint_path)
    _write_metrics(rows, metrics_path)
    return checkpoint_path, metrics_path, rows


def _load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older installed PyTorch releases.
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not {
        "model_state_dict", "agent_config", "observation_size", "action_count"
    }.issubset(checkpoint):
        raise ValueError(f"invalid DQN checkpoint: {path}")
    return checkpoint


def evaluate(
    checkpoint_path: Path | str = DEFAULT_MODEL_DIR / "dqn_checkpoint.pt",
    episodes: int = 20,
    seed: int = 7,
    metrics_path: Optional[Path | str] = None,
) -> tuple[Path, list[dict]]:
    """Evaluate a saved policy greedily, without replay or training updates."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    _set_seed(seed)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    env = SchedulingEnv()
    if (len(env.get_observation()) != checkpoint["observation_size"]
            or len(env.transactions) != checkpoint["action_count"]):
        raise ValueError("checkpoint dimensions do not match the current SchedulingEnv")

    config = dict(checkpoint["agent_config"])
    config.update({"epsilon": 0.0, "epsilon_min": 0.0, "seed": seed, "device": "cpu"})
    agent = DQNAgent.from_environment(env, **config)
    agent.policy_network.load_state_dict(checkpoint["model_state_dict"])
    agent.policy_network.eval()
    rows = []

    for episode in range(1, episodes + 1):
        observation = env.reset()
        episode_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.select_action(observation, explore=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        rows.append(
            _episode_metrics(env, episode, episode_reward, terminated, truncated, None)
        )

    output_path = Path(metrics_path) if metrics_path is not None else (
        checkpoint_path.parent / "evaluation_metrics.csv"
    )
    _write_metrics(rows, output_path)
    return output_path, rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train or evaluate the scheduling DQN")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and save a DQN policy")
    train_parser.add_argument("--episodes", type=int, default=300)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)

    evaluation_parser = subparsers.add_parser("evaluate", help="evaluate a saved policy")
    evaluation_parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "dqn_checkpoint.pt"
    )
    evaluation_parser.add_argument("--episodes", type=int, default=20)
    evaluation_parser.add_argument("--seed", type=int, default=7)
    evaluation_parser.add_argument("--metrics", type=Path, default=None)

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            checkpoint, metrics, rows = train(args.episodes, args.seed, args.model_dir)
            print(f"Training complete: {len(rows)} episodes")
            print(f"Model: {checkpoint}")
            print(f"Metrics: {metrics}")
        else:
            metrics, rows = evaluate(
                args.checkpoint, args.episodes, args.seed, args.metrics
            )
            completed = sum(row["completed_count"] for row in rows)
            print(f"Evaluation complete: {len(rows)} episodes")
            print(f"Completed transactions across episodes: {completed}")
            print(f"Metrics: {metrics}")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
