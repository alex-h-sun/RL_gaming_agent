"""Learner side: load rollouts.npz, run the PPO gradient update, save .zip.

Runs on Colab (or anywhere with torch). No live game needed: the model is
built against MockGameEnv, which shares GameEnv's spaces exactly.

Usage:
    python -m src.agent.train_colab --rollouts rollouts/rollouts.npz \
        --checkpoint checkpoints/best.zip --output checkpoints/best.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.logger import configure

from src.agent.rollout_io import fill_rollout_buffer, load_rollouts
from src.config import load_config
from src.env.mock_game_env import MockGameEnv


def train_on_rollouts(
    rollouts_path: str | Path,
    checkpoint_path: str | Path | None,
    output_path: str | Path,
    config: dict | None = None,
    device: str = "auto",
):
    from src.agent.model import load_or_init

    config = config or load_config()
    env = MockGameEnv()  # spaces donor only; never stepped during training
    model = load_or_init(checkpoint_path, env, config, device)
    model.set_logger(configure(None, ["stdout"]))

    arrays = load_rollouts(rollouts_path)
    fill_rollout_buffer(model, arrays)
    print(f"Training on {len(arrays['rewards'])} steps from {rollouts_path}")
    model.train()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Saved updated checkpoint: {output_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", default="rollouts/rollouts.npz")
    parser.add_argument("--checkpoint", default="checkpoints/best.zip")
    parser.add_argument("--output", default="checkpoints/best.zip")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    train_on_rollouts(args.rollouts, args.checkpoint, args.output, config, args.device)


if __name__ == "__main__":
    main()
