"""Clash Royale GameAdapter: wraps the existing detector/reward/mapping."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from src.actions.mapping import decode_action
from src.games.base import Drag, Gesture, Tap
from src.games.clash_royale.detector import Detector
from src.games.clash_royale.reward import (
    RewardConfig,
    shaped_reward,
    terminal_reward,
)
from src.games.clash_royale.state import GameState, ScreenState


class ClashRoyaleAdapter:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._reward_config = RewardConfig.from_config(config)
        self._detector = Detector(config)
        self._ui = config["ui_regions"]

    def build_action_space(self) -> gym.spaces.MultiDiscrete:
        action = self._config["action"]
        return gym.spaces.MultiDiscrete([action["num_cards"], action["num_cells"]])

    def action_to_gestures(self, action) -> tuple[Gesture, ...]:
        decoded = decode_action(action, self._config)
        if decoded is None:
            return ()
        source, target = decoded
        return (Drag(source, target),)

    def detect(self, frame) -> GameState:
        return self._detector.detect(frame)

    def is_in_battle(self, state: GameState) -> bool:
        return state.screen is ScreenState.IN_BATTLE

    def is_terminal(self, state: GameState) -> bool:
        return state.screen is ScreenState.END_SCREEN

    def reset_gesture(self, state: GameState) -> Gesture | None:
        if state.screen is ScreenState.END_SCREEN:
            return Tap(tuple(self._ui["ok_button"][:2]))
        if state.screen is ScreenState.MAIN_MENU:
            return Tap(tuple(self._ui["battle_button"][:2]))
        return None

    def shaped_reward(self, prev_state: GameState, curr_state: GameState) -> float:
        return shaped_reward(prev_state, curr_state, self._reward_config)

    def terminal_reward(
        self, last_battle_state: GameState, terminal_state: GameState
    ) -> float:
        return terminal_reward(self.infer_win(last_battle_state), self._reward_config)

    @staticmethod
    def infer_win(last_battle_state: GameState) -> bool | None:
        if last_battle_state.crowns_won > last_battle_state.crowns_lost:
            return True
        if last_battle_state.crowns_won < last_battle_state.crowns_lost:
            return False
        return None
