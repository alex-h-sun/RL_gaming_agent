"""GameAdapter protocol conformance and ADB backend gesture execution."""

import pytest

from src.games.base import Drag, GameAdapter, Tap
from src.games.clash_royale.adapter import ClashRoyaleAdapter
from src.games.clash_royale.state import GameState, ScreenState
from src.games.registry import make_adapter


class TestRegistry:
    def test_clash_royale(self, config):
        assert isinstance(make_adapter(config), ClashRoyaleAdapter)

    def test_brawl_stars(self):
        from src.config import load_config
        from src.games.brawl_stars.adapter import BrawlStarsAdapter

        bs_config = load_config("config/brawl_stars.yaml")
        assert isinstance(make_adapter(bs_config), BrawlStarsAdapter)

    def test_unknown_game(self, config):
        with pytest.raises(ValueError):
            make_adapter({**config, "game": "chess"})


class TestClashRoyaleAdapter:
    def test_protocol_conformance(self, config):
        assert isinstance(ClashRoyaleAdapter(config), GameAdapter)

    def test_action_space(self, config):
        adapter = ClashRoyaleAdapter(config)
        assert list(adapter.build_action_space().nvec) == [5, 70]

    def test_no_op_gestures(self, config):
        adapter = ClashRoyaleAdapter(config)
        assert adapter.action_to_gestures([0, 35]) == ()

    def test_card_play_is_drag(self, config):
        adapter = ClashRoyaleAdapter(config)
        gestures = adapter.action_to_gestures([2, 10])
        assert len(gestures) == 1
        assert isinstance(gestures[0], Drag)

    def test_navigation(self, config):
        adapter = ClashRoyaleAdapter(config)
        end = GameState(screen=ScreenState.END_SCREEN)
        menu = GameState(screen=ScreenState.MAIN_MENU)
        battle = GameState(screen=ScreenState.IN_BATTLE)
        assert isinstance(adapter.reset_gesture(end), Tap)
        assert isinstance(adapter.reset_gesture(menu), Tap)
        assert adapter.reset_gesture(battle) is None

    def test_terminal_reward_from_crowns(self, config):
        adapter = ClashRoyaleAdapter(config)
        winning = GameState(screen=ScreenState.IN_BATTLE, crowns_won=2, crowns_lost=0)
        end = GameState(screen=ScreenState.END_SCREEN)
        assert adapter.terminal_reward(winning, end) == pytest.approx(5.0)


class FakeAdbDevice:
    """Records shell commands; mimics the adbutils device surface we use."""

    def __init__(self, width=1080, height=2340):
        self._size = (width, height)
        self.commands = []

    def window_size(self):
        return self._size

    def shell(self, command):
        self.commands.append(command)


class TestAdbActions:
    @pytest.fixture
    def backend(self, config):
        from src.actions.adb_actions import AdbActions

        device = FakeAdbDevice()
        return AdbActions(device, config), device

    def test_tap_command(self, backend):
        actions, device = backend
        actions.execute(Tap((0.5, 0.5)))
        assert device.commands == ["input tap 540 1170"]

    def test_drag_command(self, backend):
        actions, device = backend
        actions.execute(Drag((0.0, 0.0), (1.0, 1.0)))
        assert device.commands == ["input swipe 0 0 1080 2340 300"]

    def test_play_card_issues_swipe(self, backend):
        actions, device = backend
        assert actions.play([1, 0]) is True
        assert len(device.commands) == 1
        assert device.commands[0].startswith("input swipe")

    def test_no_op_issues_nothing(self, backend):
        actions, device = backend
        assert actions.play([0, 0]) is False
        assert device.commands == []
