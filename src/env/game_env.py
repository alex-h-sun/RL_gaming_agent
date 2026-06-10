"""Live Clash Royale gymnasium environment.

Requires the game running (iPhone Mirroring on Mac, or Android via ADB).
Observation: 4-frame stack of 84x84 RGB -> (84, 84, 12) uint8.
Action: MultiDiscrete([5, 70]) -> card slot (0 = no-op) + arena grid cell.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from src.games.clash_royale.detector import Detector
from src.games.clash_royale.reward import (
    RewardConfig,
    blended_reward,
    shaped_reward,
    terminal_reward,
)
from src.games.clash_royale.state import GameState, ScreenState

RESET_TIMEOUT_S = 90.0
MAX_EPISODE_STEPS = 1500  # ~6 min at 4 FPS; matches overtime cap


class GameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        platform: str = "mac",
        curriculum_round: int = 0,
    ):
        super().__init__()
        self._config = config
        self._reward_config = RewardConfig.from_config(config)
        self._phase_rounds = config.get("curriculum", {}).get("phase_rounds", 200)
        self._curriculum_round = curriculum_round

        obs = config["observation"]
        self._frame_size = obs["frame_size"]
        self._stack_size = obs["frame_stack"]
        self._step_seconds = config["timing"]["step_seconds"]

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._frame_size, self._frame_size, 3 * self._stack_size),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [config["action"]["num_cards"], config["action"]["num_cells"]]
        )

        self._capture, self._actions = self._build_backend(platform)
        self._detector = Detector(config)
        self._frames: deque[np.ndarray] = deque(maxlen=self._stack_size)
        self._prev_state = GameState()
        self._steps = 0

    def _build_backend(self, platform: str):
        if platform == "mac":
            from src.actions.mac_actions import MacActions
            from src.capture.mac_capture import MacCapture

            capture = MacCapture(
                self._config["window"]["mac_title"], self._frame_size
            )
            return capture, MacActions(capture.rect, self._config)
        if platform == "adb":
            from src.actions.adb_actions import AdbActions
            from src.capture.adb_capture import AdbCapture

            capture = AdbCapture(frame_size=self._frame_size)
            return capture, AdbActions(capture._device, self._config)
        raise ValueError(f"Unknown platform: {platform}")

    def _observe(self) -> tuple[np.ndarray, GameState]:
        raw = self._capture.capture_raw()
        state = self._detector.detect(_to_rgb(raw))
        from src.capture.preprocess import to_observation

        self._frames.append(to_observation(raw, self._frame_size))
        return self._stacked(), state

    def _stacked(self) -> np.ndarray:
        while len(self._frames) < self._stack_size:
            self._frames.appendleft(self._frames[0].copy())
        return np.concatenate(list(self._frames), axis=2)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._frames.clear()
        self._steps = 0
        self._navigate_to_battle()
        obs, state = self._observe()
        self._prev_state = state
        return obs, {"state": state}

    def _navigate_to_battle(self) -> None:
        """END_SCREEN -> tap OK -> MAIN_MENU -> tap Battle -> IN_BATTLE."""
        ui = self._config["ui_regions"]
        deadline = time.time() + RESET_TIMEOUT_S
        while time.time() < deadline:
            raw = self._capture.capture_raw()
            screen = self._detector.detect(_to_rgb(raw)).screen
            if screen is ScreenState.IN_BATTLE:
                return
            if screen is ScreenState.END_SCREEN:
                self._actions.tap(tuple(ui["ok_button"][:2]))
            elif screen is ScreenState.MAIN_MENU:
                self._actions.tap(tuple(ui["battle_button"][:2]))
            time.sleep(1.5)
        raise TimeoutError("Could not reach IN_BATTLE within reset timeout")

    def step(self, action):
        self._actions.play(action)
        time.sleep(self._step_seconds)
        obs, state = self._observe()
        self._steps += 1

        terminated = state.screen is ScreenState.END_SCREEN
        truncated = self._steps >= MAX_EPISODE_STEPS

        if state.screen is ScreenState.IN_BATTLE:
            shaped = shaped_reward(self._prev_state, state, self._reward_config)
            win_loss = 0.0
            self._prev_state = state
        else:
            # Battle over: decide win/loss from last known crown counts.
            won = _infer_win(self._prev_state)
            shaped = 0.0
            win_loss = terminal_reward(won, self._reward_config)

        reward = blended_reward(
            shaped, win_loss, self._curriculum_round, self._phase_rounds
        )
        info = {"state": state, "steps": self._steps}
        return obs, float(reward), terminated, truncated, info


def _infer_win(last_battle_state: GameState) -> bool | None:
    if last_battle_state.crowns_won > last_battle_state.crowns_lost:
        return True
    if last_battle_state.crowns_won < last_battle_state.crowns_lost:
        return False
    return None


def _to_rgb(raw: np.ndarray) -> np.ndarray:
    """mss/Quartz give BGRA; ADB gives RGB already."""
    if raw.shape[2] == 4:
        return raw[:, :, 2::-1]
    return raw
