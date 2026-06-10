"""Multi-component reward for Clash Royale.

r_total = r_pbrs + r_destroy + r_king_activate + r_elixir_waste + r_survival
Win/loss bonus applies at episode end only. Includes the Action Guidance
curriculum (shaped -> blended -> sparse) and the log crown-scaling variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.games.clash_royale.state import GameState, TowerHP


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99
    aux_tower_destroy: float = 1.0
    king_tower_destroy: float = 3.0
    king_activate: float = 0.5
    elixir_waste_coef: float = 0.1
    elixir_waste_threshold: float = 9.0
    survival: float = 0.01
    win_bonus: float = 5.0
    loss_penalty: float = -2.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RewardConfig":
        return cls(**config["reward"])


def potential(state: GameState) -> float:
    """Phi(s) = normalized enemy tower HP lost minus ours lost, in [-1, 1]."""

    def hp_lost(towers: TowerHP) -> float:
        return (3.0 - (towers.left + towers.right + towers.king)) / 3.0

    return hp_lost(state.enemy_towers) - hp_lost(state.our_towers)


def pbrs_reward(prev: GameState, curr: GameState, gamma: float) -> float:
    """Potential-based shaping: gamma * Phi(s') - Phi(s). Policy-invariant."""
    return gamma * potential(curr) - potential(prev)


def destroy_reward(prev: GameState, curr: GameState, config: RewardConfig) -> float:
    """Bonus when an enemy tower transitions to destroyed this step."""
    reward = 0.0
    if prev.enemy_towers.left > 0.0 and curr.enemy_towers.left <= 0.0:
        reward += config.aux_tower_destroy
    if prev.enemy_towers.right > 0.0 and curr.enemy_towers.right <= 0.0:
        reward += config.aux_tower_destroy
    if prev.enemy_towers.king > 0.0 and curr.enemy_towers.king <= 0.0:
        reward += config.king_tower_destroy
    return reward


def king_activate_reward(prev: GameState, curr: GameState, config: RewardConfig) -> float:
    if not prev.enemy_king_active and curr.enemy_king_active:
        return config.king_activate
    return 0.0


def elixir_waste_penalty(curr: GameState, config: RewardConfig) -> float:
    return -config.elixir_waste_coef * max(0.0, curr.elixir - config.elixir_waste_threshold)


def shaped_reward(prev: GameState, curr: GameState, config: RewardConfig) -> float:
    """Dense per-step reward (no terminal win/loss component)."""
    return (
        pbrs_reward(prev, curr, config.gamma)
        + destroy_reward(prev, curr, config)
        + king_activate_reward(prev, curr, config)
        + elixir_waste_penalty(curr, config)
        + config.survival
    )


def terminal_reward(won: bool | None, config: RewardConfig) -> float:
    """Win/loss bonus at episode end. Draw (None) gets no bonus."""
    if won is True:
        return config.win_bonus
    if won is False:
        return config.loss_penalty
    return 0.0


def crown_reward(crowns_won: int, crowns_lost: int) -> float:
    """Logarithmic crown scaling, approx [-15, +15]. Alternative terminal signal."""

    def f(crowns: int) -> float:
        return 4.9 * math.log(4.8 * crowns + 0.75) + 1.4

    return f(crowns_won) - f(crowns_lost)


def curriculum_weight(round_index: int, phase_rounds: int = 200) -> float:
    """Weight on the shaped component: 1.0 early, 0.5 mid, 0.0 late."""
    phase = min(round_index // phase_rounds, 2)
    return (1.0, 0.5, 0.0)[phase]


def blended_reward(
    shaped: float, win_loss: float, round_index: int, phase_rounds: int = 200
) -> float:
    """Action Guidance curriculum blend of shaped and sparse reward."""
    weight = curriculum_weight(round_index, phase_rounds)
    return weight * shaped + (1.0 - weight) * win_loss
