"""Brawl Stars: detector on synthetic frames, reward math, adapter gestures."""

import numpy as np
import pytest

from src.config import load_config
from src.games.base import Drag, Tap
from src.games.brawl_stars.adapter import BrawlStarsAdapter, move_offset
from src.games.brawl_stars.detector import Detector, detect_screen_state
from src.games.brawl_stars.reward import (
    RewardConfig,
    hp_pbrs,
    shaped_reward,
    terminal_reward,
)
from src.games.brawl_stars.state import BrawlState, ScreenState

FRAME_W, FRAME_H = 870, 400  # landscape
ORANGE = (255, 150, 30)
GREEN = (60, 220, 80)
YELLOW = (255, 220, 40)
BLUE = (40, 100, 230)
RED = (230, 30, 30)
DARK = (25, 28, 30)


@pytest.fixture(scope="module")
def bs_config():
    return load_config("config/brawl_stars.yaml")


def _blank(color=DARK) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _fill(frame, region, color, fraction=1.0):
    x, y, w, h = region
    x0, y0 = int(x * FRAME_W), int(y * FRAME_H)
    x1 = x0 + int(w * FRAME_W * fraction)
    y1 = int((y + h) * FRAME_H)
    frame[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = color


def make_match_frame(config, hp=1.0, ammo=1.0, super_ready=False):
    frame = _blank()
    ui = config["ui_regions"]
    _fill(frame, ui["ammo_bar"], ORANGE, ammo if ammo > 0 else 0.06)
    if hp > 0:
        _fill(frame, ui["own_hp_bar"], GREEN, hp)
    if super_ready:
        _fill(frame, ui["super_button"], YELLOW)
    return frame


def make_results_frame(config, won: bool):
    frame = _blank((50, 50, 55))
    _fill(frame, config["ui_regions"]["result_banner"], BLUE if won else RED)
    return frame


class TestScreenState:
    def test_in_match(self, bs_config):
        frame = make_match_frame(bs_config)
        assert detect_screen_state(frame, bs_config["ui_regions"]) is ScreenState.IN_MATCH

    def test_results(self, bs_config):
        frame = make_results_frame(bs_config, won=True)
        assert detect_screen_state(frame, bs_config["ui_regions"]) is ScreenState.RESULTS

    def test_loading(self, bs_config):
        frame = _blank((5, 5, 8))
        assert detect_screen_state(frame, bs_config["ui_regions"]) is ScreenState.LOADING

    def test_main_menu(self, bs_config):
        frame = _blank((140, 170, 210))
        assert detect_screen_state(frame, bs_config["ui_regions"]) is ScreenState.MAIN_MENU


class TestDetector:
    def test_match_state(self, bs_config):
        detector = Detector(bs_config)
        state = detector.detect(make_match_frame(bs_config, hp=0.5, super_ready=True))
        assert state.screen is ScreenState.IN_MATCH
        assert state.own_hp == pytest.approx(0.5, abs=0.1)
        assert state.super_ready

    def test_victory_and_defeat(self, bs_config):
        detector = Detector(bs_config)
        assert detector.detect(make_results_frame(bs_config, won=True)).won is True
        assert detector.detect(make_results_frame(bs_config, won=False)).won is False


CFG = RewardConfig()


class TestReward:
    def test_taking_damage_is_negative(self):
        prev = BrawlState(screen=ScreenState.IN_MATCH, own_hp=1.0)
        curr = BrawlState(screen=ScreenState.IN_MATCH, own_hp=0.5)
        assert hp_pbrs(prev, curr, CFG.gamma) < 0

    def test_steady_state_gets_survival(self):
        state = BrawlState(screen=ScreenState.IN_MATCH, own_hp=1.0)
        reward = shaped_reward(state, state, CFG)
        # pbrs at constant hp is (gamma - 1) * hp, small negative; survival dominates
        assert reward == pytest.approx((CFG.gamma - 1.0) + CFG.survival)

    def test_terminal(self):
        assert terminal_reward(True, CFG) == pytest.approx(5.0)
        assert terminal_reward(False, CFG) == pytest.approx(-2.0)
        assert terminal_reward(None, CFG) == 0.0


class TestAdapter:
    def test_action_space(self, bs_config):
        adapter = BrawlStarsAdapter(bs_config)
        assert list(adapter.build_action_space().nvec) == [9, 3]

    def test_no_op_produces_no_gestures(self, bs_config):
        adapter = BrawlStarsAdapter(bs_config)
        assert adapter.action_to_gestures([0, 0]) == ()

    def test_move_produces_joystick_drag(self, bs_config):
        adapter = BrawlStarsAdapter(bs_config)
        gestures = adapter.action_to_gestures([3, 0])  # east
        assert len(gestures) == 1
        drag = gestures[0]
        assert isinstance(drag, Drag)
        assert drag.target[0] > drag.source[0]  # east = +x
        assert drag.target[1] == pytest.approx(drag.source[1], abs=1e-9)

    def test_attack_and_super_taps(self, bs_config):
        adapter = BrawlStarsAdapter(bs_config)
        attack = adapter.action_to_gestures([0, 1])
        super_ = adapter.action_to_gestures([0, 2])
        assert isinstance(attack[0], Tap) and isinstance(super_[0], Tap)
        assert attack[0].point != super_[0].point

    def test_move_and_attack_combine(self, bs_config):
        adapter = BrawlStarsAdapter(bs_config)
        gestures = adapter.action_to_gestures([1, 1])
        assert len(gestures) == 2

    def test_move_offset_directions(self):
        north = move_offset(1, 0.1)
        east = move_offset(3, 0.1)
        south = move_offset(5, 0.1)
        assert north[1] < 0 and abs(north[0]) < 1e-9
        assert east[0] > 0 and abs(east[1]) < 1e-9
        assert south[1] > 0

    def test_terminal_reward_reads_results_state(self, bs_config):
        adapter = BrawlStarsAdapter(bs_config)
        battle = BrawlState(screen=ScreenState.IN_MATCH)
        results = BrawlState(screen=ScreenState.RESULTS, won=True)
        assert adapter.terminal_reward(battle, results) == pytest.approx(5.0)

    def test_navigation_gestures(self, bs_config):
        adapter = BrawlStarsAdapter(bs_config)
        menu = BrawlState(screen=ScreenState.MAIN_MENU)
        results = BrawlState(screen=ScreenState.RESULTS)
        match = BrawlState(screen=ScreenState.IN_MATCH)
        assert isinstance(adapter.reset_gesture(menu), Tap)
        assert isinstance(adapter.reset_gesture(results), Tap)
        assert adapter.reset_gesture(match) is None
