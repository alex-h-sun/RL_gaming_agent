"""Live gymnasium environment, game-agnostic via GameAdapter.

The config's `game:` key selects the adapter (clash_royale, brawl_stars).
Requires the game running (iPhone Mirroring on Mac, or Android via ADB).
Observation: 4-frame stack of 84x84 RGB -> (84, 84, 12) uint8.
Action space: defined by the game adapter.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from src.games.clash_royale.reward import blended_reward
from src.games.registry import make_adapter

RESET_TIMEOUT_S = 300.0  # generous: the user navigates menus manually
MAX_EPISODE_STEPS = 1500  # ~6 min at 4 FPS


class GameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        platform: str = "mac",
        curriculum_round: int = 0,
        serial: str | None = None,
    ):
        super().__init__()
        self._config = config
        self._adapter = make_adapter(config)
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
        self.action_space = self._adapter.build_action_space()

        self._capture, self._actions = self._build_backend(platform, serial)
        self._frames: deque[np.ndarray] = deque(maxlen=self._stack_size)
        self._prev_state = None
        self._in_battle = False
        self._steps = 0

    def _build_backend(self, platform: str, serial: str | None):
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

            capture = AdbCapture(serial=serial, frame_size=self._frame_size)
            return capture, AdbActions(capture.device, self._config)
        raise ValueError(f"Unknown platform: {platform}")

    def _observe(self):
        raw = self._capture.capture_raw()
        state = self._adapter.detect(_to_rgb(raw))
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
        self._wait_for_battle()
        obs, state = self._observe()
        self._prev_state = state
        self._in_battle = True
        return obs, {"state": state}

    def _wait_for_battle(self) -> None:
        """Poll until a battle is running. The agent never touches menus:
        the user navigates end screens and the main menu manually."""
        deadline = time.time() + RESET_TIMEOUT_S
        announced = False
        while time.time() < deadline:
            raw = self._capture.capture_raw()
            state = self._adapter.detect(_to_rgb(raw))
            if self._adapter.is_in_battle(state):
                if announced:
                    print("Battle detected — resuming.")
                return
            if not announced:
                print(
                    "Waiting for a battle. Navigate the game into a match "
                    "yourself; the agent only acts during battles."
                )
                announced = True
            time.sleep(1.0)
        raise TimeoutError(
            f"No battle detected within {RESET_TIMEOUT_S:.0f}s of waiting"
        )

    def step(self, action):
        # Gestures are only injected while the last observed screen was a
        # running battle; on every other screen the user has control.
        if self._in_battle:
            for gesture in self._adapter.action_to_gestures(action):
                self._actions.execute(gesture)
        time.sleep(self._step_seconds)
        obs, state = self._observe()
        self._steps += 1

        terminated = self._adapter.is_terminal(state)
        truncated = self._steps >= MAX_EPISODE_STEPS
        self._in_battle = self._adapter.is_in_battle(state)

        if self._in_battle:
            shaped = self._adapter.shaped_reward(self._prev_state, state)
            win_loss = 0.0
            self._prev_state = state
        else:
            shaped = 0.0
            win_loss = self._adapter.terminal_reward(self._prev_state, state)

        reward = blended_reward(
            shaped, win_loss, self._curriculum_round, self._phase_rounds
        )
        info = {"state": state, "steps": self._steps}
        return obs, float(reward), terminated, truncated, info


def _to_rgb(raw: np.ndarray) -> np.ndarray:
    """mss/Quartz give BGRA; ADB gives RGB already."""
    if raw.shape[2] == 4:
        return raw[:, :, 2::-1]
    return raw
