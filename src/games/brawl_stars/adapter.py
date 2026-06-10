"""Brawl Stars GameAdapter.

Action space: MultiDiscrete([9, 3])
  [0]: movement — 0 = stand still, 1-8 = compass direction (N, NE, E, ...)
  [1]: attack   — 0 = none, 1 = attack (auto-aim tap), 2 = super

Movement is a short virtual-joystick drag each step; the next step re-issues
it, approximating continuous movement at the env's step rate.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

from src.games.base import Drag, Gesture, Tap
from src.games.brawl_stars.detector import Detector
from src.games.brawl_stars.reward import (
    RewardConfig,
    shaped_reward,
    terminal_reward,
)
from src.games.brawl_stars.state import BrawlState, ScreenState

NUM_MOVE = 9
NUM_ATTACK = 3


def move_offset(direction: int, radius: float) -> tuple[float, float]:
    """Direction 1-8 -> normalized (dx, dy); 1 = north, clockwise."""
    if not 1 <= direction <= 8:
        raise ValueError(f"move direction must be 1-8, got {direction}")
    angle = (direction - 1) * math.pi / 4.0
    return radius * math.sin(angle), -radius * math.cos(angle)


class BrawlStarsAdapter:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._reward_config = RewardConfig.from_config(config)
        self._detector = Detector(config)
        self._ui = config["ui_regions"]
        self._joystick_radius = config["action"]["joystick_radius"]

    def build_action_space(self) -> gym.spaces.MultiDiscrete:
        return gym.spaces.MultiDiscrete([NUM_MOVE, NUM_ATTACK])

    def action_to_gestures(self, action) -> tuple[Gesture, ...]:
        move, attack = int(action[0]), int(action[1])
        gestures: list[Gesture] = []
        if move > 0:
            center = tuple(self._ui["joystick_center"][:2])
            dx, dy = move_offset(move, self._joystick_radius)
            gestures.append(Drag(center, (center[0] + dx, center[1] + dy)))
        if attack == 1:
            gestures.append(Tap(tuple(self._ui["attack_button"][:2])))
        elif attack == 2:
            gestures.append(Tap(tuple(self._ui["super_button"][:2])))
        return tuple(gestures)

    def detect(self, frame) -> BrawlState:
        return self._detector.detect(frame)

    def is_in_battle(self, state: BrawlState) -> bool:
        return state.screen is ScreenState.IN_MATCH

    def is_terminal(self, state: BrawlState) -> bool:
        return state.screen is ScreenState.RESULTS

    def reset_gesture(self, state: BrawlState) -> Gesture | None:
        if state.screen is ScreenState.RESULTS:
            return Tap(tuple(self._ui["continue_button"][:2]))
        if state.screen is ScreenState.MAIN_MENU:
            return Tap(tuple(self._ui["play_button"][:2]))
        return None

    def shaped_reward(self, prev_state: BrawlState, curr_state: BrawlState) -> float:
        return shaped_reward(prev_state, curr_state, self._reward_config)

    def terminal_reward(
        self, last_battle_state: BrawlState, terminal_state: BrawlState
    ) -> float:
        return terminal_reward(terminal_state.won, self._reward_config)
