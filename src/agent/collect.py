"""Actor side: run the env with the current policy, save rollouts to .npz.

Runs on the Mac with the live game (or with MockGameEnv for pipeline tests).
The saved file is uploaded to Colab where train_colab.py consumes it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.agent.rollout_io import save_rollouts


def collect_rollouts(
    env,
    model,
    n_steps: int,
    output_path: str | Path = "rollouts/rollouts.npz",
    random_policy: bool = False,
) -> Path:
    """Step the env for n_steps under the model's policy; serialize to .npz."""
    observations = np.zeros(
        (n_steps, *env.observation_space.shape), dtype=env.observation_space.dtype
    )
    actions = np.zeros((n_steps, len(env.action_space.nvec)), dtype=np.int64)
    rewards = np.zeros(n_steps, dtype=np.float32)
    episode_starts = np.zeros(n_steps, dtype=np.float32)
    values = np.zeros(n_steps, dtype=np.float32)
    log_probs = np.zeros(n_steps, dtype=np.float32)

    obs, _ = env.reset()
    episode_start = 1.0
    episode_rewards, episode_reward = [], 0.0

    for step in range(n_steps):
        action, value, log_prob = _policy_step(model, obs, env, random_policy)

        observations[step] = obs
        actions[step] = action
        episode_starts[step] = episode_start
        values[step] = value
        log_probs[step] = log_prob

        obs, reward, terminated, truncated, _ = env.step(action)
        rewards[step] = reward
        episode_reward += reward
        episode_start = 0.0

        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()
            episode_start = 1.0

    _, last_value, _ = _policy_step(model, obs, env, random_policy)
    path = save_rollouts(
        output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        episode_starts=episode_starts,
        values=values,
        log_probs=log_probs,
        last_value=np.float32(last_value),
        last_done=np.float32(episode_start),
    )
    episodes = len(episode_rewards)
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
    print(
        f"Saved {n_steps} steps ({episodes} complete episodes, "
        f"mean episode reward {mean_reward:.2f}) to {path}"
    )
    return path


def _policy_step(model, obs: np.ndarray, env, random_policy: bool):
    """Return (action, value, log_prob) for one observation."""
    if random_policy or model is None:
        action = env.action_space.sample()
        return np.asarray(action, dtype=np.int64), 0.0, 0.0

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        action_tensor, value, log_prob = model.policy(obs_tensor)
    action = action_tensor.cpu().numpy().reshape(-1).astype(np.int64)
    return action, float(value.item()), float(log_prob.item())
