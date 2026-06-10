"""Synthetic stand-in for GameEnv: identical spaces, no game required.

Used to test collect.py, train_colab.py, evaluate.py, and the full Colab
pipeline anywhere (Mac, Colab, CI).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

FRAME_SIZE = 84
STACK_CHANNELS = 12  # 4 frames x 3 RGB channels
NUM_CARDS = 5
NUM_CELLS = 70


class MockGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, min_episode_steps: int = 50, max_episode_steps: int = 200):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(FRAME_SIZE, FRAME_SIZE, STACK_CHANNELS),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.MultiDiscrete([NUM_CARDS, NUM_CELLS])
        self._min_steps = min_episode_steps
        self._max_steps = max_episode_steps
        self._steps = 0
        self._episode_length = 0

    def _random_obs(self) -> np.ndarray:
        return self.np_random.integers(
            0, 256, size=self.observation_space.shape, dtype=np.uint8
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._steps = 0
        self._episode_length = int(
            self.np_random.integers(self._min_steps, self._max_steps + 1)
        )
        return self._random_obs(), {}

    def step(self, action):
        self._steps += 1
        # Simulated dense reward, slightly favoring non-no-op actions so a
        # learner has signal to latch onto in pipeline sanity checks.
        card = int(action[0])
        reward = float(self.np_random.normal(0.01, 0.05)) + (0.02 if card > 0 else 0.0)
        terminated = self._steps >= self._episode_length
        if terminated:
            reward += float(self.np_random.choice([5.0, -2.0]))
        return self._random_obs(), reward, terminated, False, {"steps": self._steps}
