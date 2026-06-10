"""CV detection for Clash Royale: screen state, elixir, tower HP, hand.

All functions operate on a full-resolution RGB frame plus normalized UI
regions from the config, so they are unit-testable against fixture images.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.games.clash_royale.state import GameState, ScreenState, TowerHP

# HSV ranges (OpenCV H in [0,180]) for the signature UI colors.
ELIXIR_PINK_LOW = np.array([140, 80, 120])
ELIXIR_PINK_HIGH = np.array([175, 255, 255])
HP_BAR_LOW = np.array([0, 0, 100])  # bright bar pixels vs dark background
MAX_ELIXIR = 10.0


def crop_region(frame: np.ndarray, region: list[float]) -> np.ndarray:
    """Crop a normalized (x, y, w, h) region from an RGB frame."""
    height, width = frame.shape[:2]
    x, y, w, h = region
    x0, y0 = int(x * width), int(y * height)
    x1, y1 = int((x + w) * width), int((y + h) * height)
    return frame[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]


def color_fill_fraction(
    strip: np.ndarray, hsv_low: np.ndarray, hsv_high: np.ndarray
) -> float:
    """Fraction of columns in a horizontal bar strip matching a color range.

    Bars fill left-to-right, so we count matching columns rather than pixels
    to be robust to bar height variations.
    """
    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    column_hit = (mask.mean(axis=0) > 127).astype(np.float64)
    return float(column_hit.mean())


def detect_elixir(frame: np.ndarray, ui_regions: dict[str, Any]) -> float:
    """Current elixir in [0, 10] from the pink elixir bar fill."""
    strip = crop_region(frame, ui_regions["elixir_bar"])
    return color_fill_fraction(strip, ELIXIR_PINK_LOW, ELIXIR_PINK_HIGH) * MAX_ELIXIR


def detect_hp_fraction(frame: np.ndarray, region: list[float]) -> float:
    """Tower HP bar fill fraction in [0, 1]. Missing bar (destroyed) -> 0."""
    strip = crop_region(frame, region)
    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    # Bright, saturated bar pixels against the dark arena background.
    mask = (hsv[:, :, 2] > 100).astype(np.uint8) * 255
    column_hit = (mask.mean(axis=0) > 127).astype(np.float64)
    return float(column_hit.mean())


def detect_screen_state(frame: np.ndarray) -> ScreenState:
    """Classify the screen with cheap global heuristics.

    IN_BATTLE: the pink elixir bar is present in the bottom strip.
    END_SCREEN: large dark overlay with a bright center banner.
    BATTLE_LOADING: mostly dark frame (loading transition).
    MAIN_MENU: everything else (bright, no elixir bar).
    """
    height = frame.shape[0]
    bottom = frame[int(height * 0.92):, :]
    hsv_bottom = cv2.cvtColor(bottom, cv2.COLOR_RGB2HSV)
    pink = cv2.inRange(hsv_bottom, ELIXIR_PINK_LOW, ELIXIR_PINK_HIGH)
    if pink.mean() > 8.0:
        return ScreenState.IN_BATTLE

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 25.0:
        return ScreenState.BATTLE_LOADING

    center = gray[int(height * 0.35):int(height * 0.55), :]
    if mean_brightness < 90.0 and float(center.mean()) > mean_brightness * 1.5:
        return ScreenState.END_SCREEN

    return ScreenState.MAIN_MENU


def detect_hand(
    frame: np.ndarray,
    ui_regions: dict[str, Any],
    templates: dict[str, np.ndarray] | None,
) -> list[str | None]:
    """Match card templates against the 4 hand slots.

    Returns the best-matching card name per slot, or None when templates are
    unavailable or confidence is low. Template loading is optional in v1:
    the agent acts by slot index, names are for logging/analysis.
    """
    if not templates:
        return [None] * 4

    hand: list[str | None] = []
    for slot_region in ui_regions["card_slots"]:
        slot = crop_region(frame, slot_region)
        best_name, best_score = None, 0.6  # confidence floor
        for name, template in templates.items():
            resized = cv2.resize(template, (slot.shape[1], slot.shape[0]))
            score = cv2.matchTemplate(
                slot, resized, cv2.TM_CCOEFF_NORMED
            ).max()
            if score > best_score:
                best_name, best_score = name, float(score)
        hand.append(best_name)
    return hand


class Detector:
    """Stateful detector: tracks crowns and king activation across steps."""

    def __init__(self, config: dict[str, Any], templates: dict[str, np.ndarray] | None = None):
        self._ui = config["ui_regions"]
        self._templates = templates
        self._prev_enemy_towers = TowerHP()

    def detect(self, frame: np.ndarray) -> GameState:
        screen = detect_screen_state(frame)
        if screen is not ScreenState.IN_BATTLE:
            return GameState(screen=screen)

        our = TowerHP(
            left=detect_hp_fraction(frame, self._ui["our_left_tower_hp"]),
            right=detect_hp_fraction(frame, self._ui["our_right_tower_hp"]),
            king=detect_hp_fraction(frame, self._ui["our_king_hp"]),
        )
        enemy = TowerHP(
            left=detect_hp_fraction(frame, self._ui["enemy_left_tower_hp"]),
            right=detect_hp_fraction(frame, self._ui["enemy_right_tower_hp"]),
            king=detect_hp_fraction(frame, self._ui["enemy_king_hp"]),
        )
        crowns_won = int(enemy.left <= 0.0) + int(enemy.right <= 0.0) + int(enemy.king <= 0.0)
        crowns_lost = int(our.left <= 0.0) + int(our.right <= 0.0) + int(our.king <= 0.0)
        king_active = enemy.left <= 0.0 or enemy.right <= 0.0 or enemy.king < 1.0

        self._prev_enemy_towers = enemy
        return GameState(
            screen=screen,
            elixir=detect_elixir(frame, self._ui),
            our_towers=our,
            enemy_towers=enemy,
            enemy_king_active=king_active,
            crowns_won=crowns_won,
            crowns_lost=crowns_lost,
        )

    def detect_hand(self, frame: np.ndarray) -> list[str | None]:
        return detect_hand(frame, self._ui, self._templates)
