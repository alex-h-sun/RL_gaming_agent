"""Detector: synthetic fixture frames -> correct state/HP/elixir extraction."""

import pytest

from src.games.clash_royale.detector import (
    Detector,
    detect_elixir,
    detect_hp_fraction,
    detect_screen_state,
)
from src.games.clash_royale.state import ScreenState
from tests.conftest import (
    make_battle_frame,
    make_end_screen_frame,
    make_loading_frame,
    make_menu_frame,
)


class TestScreenState:
    def test_in_battle(self, config):
        assert detect_screen_state(make_battle_frame(config)) is ScreenState.IN_BATTLE

    def test_main_menu(self):
        assert detect_screen_state(make_menu_frame()) is ScreenState.MAIN_MENU

    def test_battle_loading(self):
        assert detect_screen_state(make_loading_frame()) is ScreenState.BATTLE_LOADING

    def test_end_screen(self):
        assert detect_screen_state(make_end_screen_frame()) is ScreenState.END_SCREEN


class TestElixir:
    @pytest.mark.parametrize("fraction,expected", [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0)])
    def test_elixir_levels(self, config, fraction, expected):
        frame = make_battle_frame(config, elixir_fraction=fraction)
        assert detect_elixir(frame, config["ui_regions"]) == pytest.approx(
            expected, abs=0.5
        )


class TestTowerHP:
    def test_full_hp(self, config):
        frame = make_battle_frame(config)
        hp = detect_hp_fraction(frame, config["ui_regions"]["our_king_hp"])
        assert hp == pytest.approx(1.0, abs=0.1)

    def test_half_hp(self, config):
        frame = make_battle_frame(config, enemy_hp=(0.5, 1.0, 1.0))
        hp = detect_hp_fraction(frame, config["ui_regions"]["enemy_left_tower_hp"])
        assert hp == pytest.approx(0.5, abs=0.1)

    def test_destroyed_tower(self, config):
        frame = make_battle_frame(config, enemy_hp=(0.0, 1.0, 1.0))
        hp = detect_hp_fraction(frame, config["ui_regions"]["enemy_left_tower_hp"])
        assert hp == pytest.approx(0.0, abs=0.1)


class TestDetector:
    def test_battle_state_extraction(self, config):
        detector = Detector(config)
        frame = make_battle_frame(
            config, elixir_fraction=0.7, enemy_hp=(0.0, 0.6, 1.0)
        )
        state = detector.detect(frame)
        assert state.screen is ScreenState.IN_BATTLE
        assert state.elixir == pytest.approx(7.0, abs=0.5)
        assert state.crowns_won == 1
        assert state.crowns_lost == 0
        assert state.enemy_king_active

    def test_non_battle_returns_bare_state(self, config):
        detector = Detector(config)
        state = detector.detect(make_menu_frame())
        assert state.screen is ScreenState.MAIN_MENU
        assert state.elixir == 0.0

    def test_hand_without_templates(self, config):
        detector = Detector(config, templates=None)
        hand = detector.detect_hand(make_battle_frame(config))
        assert hand == [None, None, None, None]
