"""Run N evaluation episodes and report win rate, crowns, and efficiency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EpisodeResult:
    outcome: str  # "win" | "loss" | "draw"
    crowns_won: int
    crowns_lost: int
    total_reward: float
    length: int


def run_episode(env, model, deterministic: bool = True) -> EpisodeResult:
    obs, info = env.reset()
    total_reward, steps = 0.0, 0
    state = info.get("state")
    while True:
        if model is None:
            action = env.action_space.sample()
        else:
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                action_tensor = model.policy._predict(
                    obs_tensor, deterministic=deterministic
                )
            action = action_tensor.cpu().numpy().reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        state = info.get("state", state)
        if terminated or truncated:
            break

    crowns_won = getattr(state, "crowns_won", 0)
    crowns_lost = getattr(state, "crowns_lost", 0)
    if crowns_won > crowns_lost:
        outcome = "win"
    elif crowns_won < crowns_lost:
        outcome = "loss"
    else:
        outcome = "draw"
    return EpisodeResult(outcome, crowns_won, crowns_lost, total_reward, steps)


def evaluate(env, model, episodes: int = 20, deterministic: bool = True) -> dict:
    results = []
    for i in range(episodes):
        result = run_episode(env, model, deterministic)
        results.append(result)
        print(
            f"Episode {i + 1}/{episodes}: {result.outcome}, "
            f"crowns {result.crowns_won}-{result.crowns_lost}, "
            f"reward {result.total_reward:.2f}, length {result.length}"
        )

    wins = sum(r.outcome == "win" for r in results)
    crowns = np.array([r.crowns_won for r in results], dtype=np.float64)
    summary = {
        "episodes": episodes,
        "win_rate": wins / episodes,
        "mean_crowns": float(crowns.mean()),
        "std_crowns": float(crowns.std()),
        "mean_reward": float(np.mean([r.total_reward for r in results])),
        "mean_length": float(np.mean([r.length for r in results])),
    }
    print(
        f"\nSummary: win rate {summary['win_rate']:.0%}, "
        f"crowns {summary['mean_crowns']:.2f} +/- {summary['std_crowns']:.2f}, "
        f"mean reward {summary['mean_reward']:.2f}"
    )
    return summary
