"""Reward for Brawl Stars (v2 baseline).

r_total = r_hp_pbrs + r_survival
plus a terminal victory/defeat bonus read from the results banner.

Damage-dealt and kill detection are deferred: they need enemy-side CV that
is not reliable yet. HP-based PBRS still rewards dodging and staying alive,
which is the core Showdown skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.games.brawl_stars.state import BrawlState


@dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99
    survival: float = 0.01
    win_bonus: float = 5.0
    loss_penalty: float = -2.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RewardConfig":
        return cls(**config["reward"])


def hp_pbrs(prev: BrawlState, curr: BrawlState, gamma: float) -> float:
    """Potential-based shaping on own HP: gamma * hp' - hp."""
    return gamma * curr.own_hp - prev.own_hp


def shaped_reward(prev: BrawlState, curr: BrawlState, config: RewardConfig) -> float:
    return hp_pbrs(prev, curr, config.gamma) + config.survival


def terminal_reward(won: bool | None, config: RewardConfig) -> float:
    if won is True:
        return config.win_bonus
    if won is False:
        return config.loss_penalty
    return 0.0
