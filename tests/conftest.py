"""Shared fixtures: synthetic Clash Royale frames drawn to the config's
UI regions, so detector tests run with no real screenshots required.

Generated frames are also written to tests/fixtures/*.png on first run for
visual inspection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import load_config

FRAME_W, FRAME_H = 400, 870  # phone-ish aspect
PINK = (255, 0, 200)  # falls in the detector's elixir HSV range
DARK = (20, 20, 25)
BRIGHT_SKY = (120, 180, 230)
BAR_RED = (220, 40, 40)
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def config():
    return load_config()


def _blank(color=DARK) -> np.ndarray:
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _fill_region(frame: np.ndarray, region, color, fraction: float = 1.0) -> None:
    x, y, w, h = region
    x0, y0 = int(x * FRAME_W), int(y * FRAME_H)
    x1 = x0 + int(w * FRAME_W * fraction)
    y1 = int((y + h) * FRAME_H)
    frame[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] = color


def make_battle_frame(
    config,
    elixir_fraction: float = 0.7,
    our_hp=(1.0, 1.0, 1.0),
    enemy_hp=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """IN_BATTLE frame: dark arena, pink elixir bar, bright HP bars."""
    frame = _blank(DARK)
    ui = config["ui_regions"]
    _fill_region(frame, ui["elixir_bar"], PINK, elixir_fraction)
    hp_regions = [
        ("our_left_tower_hp", our_hp[0]),
        ("our_right_tower_hp", our_hp[1]),
        ("our_king_hp", our_hp[2]),
        ("enemy_left_tower_hp", enemy_hp[0]),
        ("enemy_right_tower_hp", enemy_hp[1]),
        ("enemy_king_hp", enemy_hp[2]),
    ]
    for name, fill in hp_regions:
        if fill > 0.0:
            _fill_region(frame, ui[name], BAR_RED, fill)
    return frame


def make_menu_frame() -> np.ndarray:
    """MAIN_MENU: bright frame, no elixir bar."""
    return _blank(BRIGHT_SKY)


def make_loading_frame() -> np.ndarray:
    """BATTLE_LOADING: near-black frame."""
    return _blank((5, 5, 8))


def make_end_screen_frame() -> np.ndarray:
    """END_SCREEN: dim overlay with a bright center banner."""
    frame = _blank((40, 40, 45))
    frame[int(FRAME_H * 0.38):int(FRAME_H * 0.52), :] = (230, 220, 200)
    return frame


@pytest.fixture(scope="session", autouse=True)
def write_fixture_pngs(config):
    """Persist the synthetic frames as PNGs for manual inspection."""
    import cv2

    FIXTURES_DIR.mkdir(exist_ok=True)
    frames = {
        "battle.png": make_battle_frame(config),
        "menu.png": make_menu_frame(),
        "loading.png": make_loading_frame(),
        "end_screen.png": make_end_screen_frame(),
    }
    for name, frame in frames.items():
        path = FIXTURES_DIR / name
        if not path.exists():
            cv2.imwrite(str(path), frame[:, :, ::-1])  # RGB -> BGR for cv2
