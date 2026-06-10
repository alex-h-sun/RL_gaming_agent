"""Reward math on synthetic game-state values."""

import pytest

from src.games.clash_royale.reward import (
    RewardConfig,
    blended_reward,
    crown_reward,
    curriculum_weight,
    destroy_reward,
    elixir_waste_penalty,
    king_activate_reward,
    pbrs_reward,
    potential,
    shaped_reward,
    terminal_reward,
)
from src.games.clash_royale.state import GameState, ScreenState, TowerHP


def battle_state(**kwargs) -> GameState:
    return GameState(screen=ScreenState.IN_BATTLE, **kwargs)


CFG = RewardConfig()


class TestPBRS:
    def test_zero_at_start(self):
        state = battle_state()
        assert potential(state) == 0.0

    def test_damaging_enemy_is_positive(self):
        prev = battle_state()
        curr = battle_state(enemy_towers=TowerHP(left=0.5))
        expected = CFG.gamma * (0.5 / 3.0) - 0.0
        assert pbrs_reward(prev, curr, CFG.gamma) == pytest.approx(expected)

    def test_taking_damage_is_negative(self):
        prev = battle_state()
        curr = battle_state(our_towers=TowerHP(left=0.5))
        assert pbrs_reward(prev, curr, CFG.gamma) < 0


class TestDestroy:
    def test_aux_tower(self):
        prev = battle_state()
        curr = battle_state(enemy_towers=TowerHP(left=0.0))
        assert destroy_reward(prev, curr, CFG) == pytest.approx(1.0)

    def test_king_tower(self):
        prev = battle_state()
        curr = battle_state(enemy_towers=TowerHP(king=0.0))
        assert destroy_reward(prev, curr, CFG) == pytest.approx(3.0)

    def test_no_double_count(self):
        prev = battle_state(enemy_towers=TowerHP(left=0.0))
        curr = battle_state(enemy_towers=TowerHP(left=0.0))
        assert destroy_reward(prev, curr, CFG) == 0.0


class TestKingActivate:
    def test_fires_on_transition(self):
        prev = battle_state()
        curr = battle_state(enemy_king_active=True)
        assert king_activate_reward(prev, curr, CFG) == pytest.approx(0.5)

    def test_fires_once(self):
        prev = battle_state(enemy_king_active=True)
        curr = battle_state(enemy_king_active=True)
        assert king_activate_reward(prev, curr, CFG) == 0.0


class TestElixirWaste:
    def test_no_penalty_below_threshold(self):
        assert elixir_waste_penalty(battle_state(elixir=8.0), CFG) == 0.0

    def test_penalty_at_cap(self):
        assert elixir_waste_penalty(battle_state(elixir=10.0), CFG) == pytest.approx(-0.1)


class TestTerminal:
    def test_win(self):
        assert terminal_reward(True, CFG) == pytest.approx(5.0)

    def test_loss(self):
        assert terminal_reward(False, CFG) == pytest.approx(-2.0)

    def test_draw(self):
        assert terminal_reward(None, CFG) == 0.0


class TestCrownReward:
    def test_zero_when_even(self):
        assert crown_reward(0, 0) == pytest.approx(0.0)
        assert crown_reward(2, 2) == pytest.approx(0.0)

    def test_first_crown_worth_more_than_third(self):
        first = crown_reward(1, 0) - crown_reward(0, 0)
        third = crown_reward(3, 0) - crown_reward(2, 0)
        assert first > third > 0

    def test_range_is_roughly_15(self):
        assert crown_reward(3, 0) == pytest.approx(15, abs=1.0)
        assert crown_reward(0, 3) == pytest.approx(-15, abs=1.0)


class TestCurriculum:
    @pytest.mark.parametrize(
        "round_index,weight", [(0, 1.0), (199, 1.0), (200, 0.5), (400, 0.0), (9999, 0.0)]
    )
    def test_phase_weights(self, round_index, weight):
        assert curriculum_weight(round_index, phase_rounds=200) == weight

    def test_blend(self):
        assert blended_reward(2.0, 10.0, round_index=200) == pytest.approx(6.0)
        assert blended_reward(2.0, 10.0, round_index=500) == pytest.approx(10.0)


class TestShapedTotal:
    def test_includes_survival_bonus(self):
        prev = battle_state(elixir=5.0)
        curr = battle_state(elixir=5.0)
        assert shaped_reward(prev, curr, CFG) == pytest.approx(CFG.survival)

    def test_tower_destruction_dominates(self):
        prev = battle_state()
        curr = battle_state(enemy_towers=TowerHP(left=0.0), enemy_king_active=True)
        reward = shaped_reward(prev, curr, CFG)
        # destroy 1.0 + activate 0.5 + pbrs ~0.33 + survival
        assert reward > 1.5
