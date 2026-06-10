"""Behavioral cloning: supervised pre-training of the PPO policy.

Trains the exact same SB3 PPO policy used for live RL (built via
model.make_ppo) on (observation, action) demonstrations, then saves a
standard checkpoint .zip. That checkpoint replaces random initialization
as the starting point for live PPO collection rounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.agent.model import make_ppo
from src.env.mock_game_env import MockGameEnv


def train_bc(
    observations: np.ndarray,
    actions: np.ndarray,
    config: dict[str, Any],
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "auto",
):
    """Clone demonstrations into a fresh PPO model; returns the PPO model.

    observations: (n, 84, 84, 12) uint8 channels-last; actions: (n, 2) int64.
    """
    env = MockGameEnv()
    model = make_ppo(env, config, device=device)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    obs_tensor = torch.as_tensor(
        observations.transpose(0, 3, 1, 2), dtype=torch.float32
    )
    act_tensor = torch.as_tensor(actions, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(obs_tensor, act_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy.train()
    for epoch in range(epochs):
        total, count = 0.0, 0
        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(model.device)
            batch_act = batch_act.to(model.device)
            distribution = policy.get_distribution(batch_obs)
            loss = -distribution.log_prob(batch_act).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_act)
            count += len(batch_act)
        print(f"BC epoch {epoch + 1}/{epochs}: nll {total / count:.4f}")
    return model


def save_bc_checkpoint(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path
