"""GameEnv only injects gestures while a battle is running.

Menus/end screens are never touched by the agent: reset() polls until the
user has navigated into a match, and step() executes no gestures once the
battle is over.
"""

import numpy as np
import pytest

from src.env import game_env as game_env_module
from src.env.game_env import GameEnv
from src.games.clash_royale.state import GameState, ScreenState


class FakeCapture:
    def capture_raw(self):
        return np.zeros((100, 60, 4), dtype=np.uint8)


class FakeActions:
    def __init__(self):
        self.executed = []

    def execute(self, gesture):
        self.executed.append(gesture)


class ScriptedAdapter:
    """Clash-Royale-shaped adapter whose screen sequence is scripted."""

    def __init__(self, real_adapter, screens):
        self._real = real_adapter
        self._screens = iter(screens)
        self._last = ScreenState.IN_BATTLE

    def detect(self, frame):
        self._last = next(self._screens, self._last)
        return GameState(screen=self._last)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def make_env(config, monkeypatch):
    monkeypatch.setattr(game_env_module.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        GameEnv, "_build_backend", lambda self, p, s: (FakeCapture(), FakeActions())
    )

    def _make(screens):
        env = GameEnv(config, platform="mac")
        env._adapter = ScriptedAdapter(env._adapter, screens)
        return env

    return _make


PLAY_CARD = [1, 0]


def test_reset_waits_without_touching_menus(make_env):
    # Two menu polls before the user starts a battle; reset taps nothing.
    env = make_env(
        [ScreenState.MAIN_MENU, ScreenState.MAIN_MENU, ScreenState.IN_BATTLE,
         ScreenState.IN_BATTLE]
    )
    env.reset()
    assert env._actions.executed == []


def test_step_executes_gestures_only_in_battle(make_env):
    env = make_env(
        [ScreenState.IN_BATTLE,  # reset poll
         ScreenState.IN_BATTLE,  # reset observe
         ScreenState.IN_BATTLE,  # step 1 observe: still in battle
         ScreenState.END_SCREEN]  # step 2 observe: battle over
    )
    env.reset()
    env.step(PLAY_CARD)
    assert len(env._actions.executed) == 1  # battle running: gesture injected

    _, _, terminated, _, _ = env.step(PLAY_CARD)
    assert terminated
    assert len(env._actions.executed) == 2  # this step still fired (was in battle)

    # After the end screen was observed, no further gestures are injected.
    env._steps = 0  # avoid truncation bookkeeping noise
    env.step(PLAY_CARD)
    assert len(env._actions.executed) == 2


def test_reset_times_out_without_battle(make_env, monkeypatch):
    env = make_env([ScreenState.MAIN_MENU])
    monkeypatch.setattr(game_env_module, "RESET_TIMEOUT_S", 0.0)
    with pytest.raises(TimeoutError):
        env.reset()
