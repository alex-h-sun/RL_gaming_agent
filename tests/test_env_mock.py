"""MockGameEnv obeys the gymnasium contract and mirrors GameEnv's spaces."""

import gymnasium as gym
import numpy as np
import pytest

from src.actions.mapping import cell_to_point, decode_action
from src.env.mock_game_env import MockGameEnv


@pytest.fixture
def env():
    return MockGameEnv(min_episode_steps=10, max_episode_steps=30)


class TestSpaces:
    def test_observation_space(self, env):
        assert env.observation_space == gym.spaces.Box(
            0, 255, shape=(84, 84, 12), dtype=np.uint8
        )

    def test_action_space(self, env):
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert list(env.action_space.nvec) == [5, 70]


class TestContract:
    def test_reset(self, env):
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self, env):
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates_in_range(self, env):
        env.reset(seed=42)
        for step in range(1, 31):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            if terminated:
                break
        assert terminated
        assert 10 <= step <= 30

    def test_resets_after_termination(self, env):
        env.reset(seed=1)
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(env.action_space.sample())
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_gymnasium_env_checker(self, env):
        from gymnasium.utils.env_checker import check_env

        check_env(env, skip_render_check=True)


class TestActionMapping:
    def test_no_op_decodes_to_none(self, config):
        assert decode_action([0, 35], config) is None

    def test_card_play_decodes_to_points(self, config):
        decoded = decode_action([1, 0], config)
        assert decoded is not None
        (sx, sy), (tx, ty) = decoded
        assert 0 <= sx <= 1 and 0 <= sy <= 1
        assert 0 <= tx <= 1 and 0 <= ty <= 1

    def test_cell_grid_corners(self):
        region = [0.0, 0.0, 1.0, 1.0]
        x0, y0 = cell_to_point(0, 10, 7, region)
        x_last, y_last = cell_to_point(69, 10, 7, region)
        assert x0 < x_last and y0 < y_last

    def test_cell_out_of_range(self):
        with pytest.raises(ValueError):
            cell_to_point(70, 10, 7, [0, 0, 1, 1])
