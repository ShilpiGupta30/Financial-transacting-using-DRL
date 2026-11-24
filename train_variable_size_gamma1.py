"""Run a separate variable-size DQN experiment with gamma fixed at 1.0.

This entry point reuses the existing environment and trainer. It writes a
distinct checkpoint and metrics file so the earlier experiments are untouched.
"""

import argparse
from pathlib import Path
from typing import Optional

from drl_scheduler.train_variable_size import (
    DEFAULT_AGENT_CONFIG,
    DEFAULT_MODEL_DIR,
    train_variable_size,
)


GAMMA1_CHECKPOINT_FILENAME = "variable_dqn_gamma1_checkpoint.pt"
GAMMA1_METRICS_FILENAME = "variable_gamma1_training_metrics.csv"
GAMMA1_AGENT_CONFIG = {**DEFAULT_AGENT_CONFIG, "gamma": 1.0}


def train_gamma1(
    *,
    episodes: int = 300,
    seed: int = 7,
    model_dir: Path | str = DEFAULT_MODEL_DIR,
    min_transactions: int = 1,
    max_transactions: int = 8,
    max_burst_time: int = 8,
    min_priority: int = 1,
    max_priority: int = 10,
    max_episode_steps: int = 100,
):
    """Train the same model/environment setup with gamma=1 and separate files."""
    return train_variable_size(
        episodes=episodes,
        seed=seed,
        model_dir=model_dir,
        min_transactions=min_transactions,
        max_transactions=max_transactions,
        max_burst_time=max_burst_time,
        min_priority=min_priority,
        max_priority=max_priority,
        max_episode_steps=max_episode_steps,
        agent_config=dict(GAMMA1_AGENT_CONFIG),
        checkpoint_filename=GAMMA1_CHECKPOINT_FILENAME,
        metrics_filename=GAMMA1_METRICS_FILENAME,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a separate variable-size DQN experiment with gamma=1.0"
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
        checkpoint, metrics, rows = train_gamma1(
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

    print(f"Gamma=1.0 training complete: {len(rows)} episodes")
    print(f"Checkpoint: {checkpoint}")
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
