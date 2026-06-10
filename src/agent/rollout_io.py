"""Serialize rollout experience to .npz and rebuild SB3 RolloutBuffers.

The .npz contract (all arrays length n_steps, n_envs == 1):
  observations   (n, 84, 84, 12) uint8   channels-last, as captured
  actions        (n, 2)          int64   MultiDiscrete [card, cell]
  rewards        (n,)            float32
  episode_starts (n,)            float32 1.0 where a new episode begins
  values         (n,)            float32 V(s) under the collecting policy
  log_probs      (n,)            float32 log pi(a|s) under the collecting policy
  last_value     ()              float32 V(s_n) for GAE bootstrap
  last_done      ()              float32 whether the final step terminated
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REQUIRED_KEYS = (
    "observations",
    "actions",
    "rewards",
    "episode_starts",
    "values",
    "log_probs",
    "last_value",
    "last_done",
)


def save_rollouts(path: str | Path, **arrays: np.ndarray) -> Path:
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Missing rollout arrays: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_rollouts(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Rollout file {path} missing arrays: {missing}")
    return arrays


def fill_rollout_buffer(model, arrays: dict[str, np.ndarray]) -> None:
    """Inject serialized experience into an SB3 PPO model's rollout buffer.

    The model's policy is NOT consulted: values/log_probs come from the
    collecting policy, exactly as SB3's own collect_rollouts would store.
    """
    import torch
    from stable_baselines3.common.buffers import RolloutBuffer

    n_steps = len(arrays["rewards"])
    buffer = RolloutBuffer(
        buffer_size=n_steps,
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=1,
    )
    observations = _to_channels_first(arrays["observations"], model.observation_space)
    for i in range(n_steps):
        buffer.add(
            obs=observations[i][None],
            action=arrays["actions"][i][None],
            reward=np.array([arrays["rewards"][i]], dtype=np.float32),
            episode_start=np.array([arrays["episode_starts"][i]], dtype=np.float32),
            value=torch.as_tensor([arrays["values"][i]], dtype=torch.float32),
            log_prob=torch.as_tensor([arrays["log_probs"][i]], dtype=torch.float32),
        )
    last_value = torch.as_tensor([[float(arrays["last_value"])]], dtype=torch.float32)
    last_done = np.array([float(arrays["last_done"])], dtype=np.float32)
    buffer.compute_returns_and_advantage(last_values=last_value, dones=last_done)
    model.rollout_buffer = buffer


def _to_channels_first(observations: np.ndarray, observation_space) -> np.ndarray:
    """Match the buffer's expected layout (SB3 transposes image spaces)."""
    if observation_space.shape == observations.shape[1:]:
        return observations
    transposed = observations.transpose(0, 3, 1, 2)
    if observation_space.shape != transposed.shape[1:]:
        raise ValueError(
            f"Observations {observations.shape[1:]} do not match "
            f"observation space {observation_space.shape}"
        )
    return transposed
