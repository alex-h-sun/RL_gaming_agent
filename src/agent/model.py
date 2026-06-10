"""PPO model factory with a CNN feature extractor for 12-channel frames.

Both collect.py (Mac) and train_colab.py (Colab) build models through this
factory so architecture and hyperparameters always match.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FrameStackCNN(BaseFeaturesExtractor):
    """Nature-CNN style extractor accepting 12 input channels (4x RGB)."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # channels-first after SB3 transpose
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations / 255.0))


def make_ppo(env, config: dict[str, Any], device: str = "auto") -> PPO:
    ppo_cfg = config["ppo"]
    return PPO(
        "CnnPolicy",
        env,
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        learning_rate=ppo_cfg["learning_rate"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        policy_kwargs={
            "features_extractor_class": FrameStackCNN,
            "features_extractor_kwargs": {"features_dim": 512},
            "normalize_images": False,  # FrameStackCNN scales internally
        },
        device=device,
        verbose=1,
    )


def load_or_init(checkpoint: str | Path | None, env, config: dict[str, Any], device: str = "auto") -> PPO:
    """Load a checkpoint if it exists, otherwise initialize fresh."""
    if checkpoint is not None and Path(checkpoint).exists():
        model = PPO.load(checkpoint, env=env, device=device)
        print(f"Loaded checkpoint: {checkpoint}")
        return model
    print("No checkpoint found, initializing fresh PPO model")
    return make_ppo(env, config, device)
