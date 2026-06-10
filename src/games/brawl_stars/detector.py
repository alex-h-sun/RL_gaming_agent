"""CV detection for Brawl Stars: screen state, own HP, ammo, super.

Same approach as the Clash Royale detector: color heuristics over normalized
UI regions from the config, unit-testable on synthetic fixture frames.
Regions must be calibrated against real captures before live runs.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.games.brawl_stars.state import BrawlState, ScreenState
from src.games.clash_royale.detector import color_fill_fraction, crop_region

# HSV ranges (OpenCV H in [0,180]).
HP_GREEN_LOW = np.array([40, 80, 100])
HP_GREEN_HIGH = np.array([85, 255, 255])
AMMO_ORANGE_LOW = np.array([10, 120, 150])
AMMO_ORANGE_HIGH = np.array([30, 255, 255])
SUPER_YELLOW_LOW = np.array([20, 150, 180])
SUPER_YELLOW_HIGH = np.array([35, 255, 255])

VICTORY_BLUE_LOW = np.array([95, 120, 150])
VICTORY_BLUE_HIGH = np.array([125, 255, 255])
DEFEAT_RED_LOW = np.array([0, 120, 150])
DEFEAT_RED_HIGH = np.array([8, 255, 255])


def detect_screen_state(frame: np.ndarray, ui_regions: dict[str, Any]) -> ScreenState:
    """Classify the screen with cheap global heuristics.

    IN_MATCH: orange ammo segments present in the ammo region.
    RESULTS: large saturated blue (victory) or red (defeat) banner.
    LOADING: near-black frame.
    MAIN_MENU: everything else.
    """
    ammo_strip = crop_region(frame, ui_regions["ammo_bar"])
    if color_fill_fraction(ammo_strip, AMMO_ORANGE_LOW, AMMO_ORANGE_HIGH) > 0.05:
        return ScreenState.IN_MATCH

    banner = crop_region(frame, ui_regions["result_banner"])
    hsv = cv2.cvtColor(banner, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, VICTORY_BLUE_LOW, VICTORY_BLUE_HIGH)
    red = cv2.inRange(hsv, DEFEAT_RED_LOW, DEFEAT_RED_HIGH)
    if blue.mean() > 60.0 or red.mean() > 60.0:
        return ScreenState.RESULTS

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if float(gray.mean()) < 25.0:
        return ScreenState.LOADING
    return ScreenState.MAIN_MENU


def detect_result(frame: np.ndarray, ui_regions: dict[str, Any]) -> bool | None:
    """On RESULTS: blue banner -> win, red banner -> loss, neither -> None."""
    banner = crop_region(frame, ui_regions["result_banner"])
    hsv = cv2.cvtColor(banner, cv2.COLOR_RGB2HSV)
    blue = float(cv2.inRange(hsv, VICTORY_BLUE_LOW, VICTORY_BLUE_HIGH).mean())
    red = float(cv2.inRange(hsv, DEFEAT_RED_LOW, DEFEAT_RED_HIGH).mean())
    if blue > 60.0 and blue > red:
        return True
    if red > 60.0 and red > blue:
        return False
    return None


class Detector:
    def __init__(self, config: dict[str, Any]):
        self._ui = config["ui_regions"]

    def detect(self, frame: np.ndarray) -> BrawlState:
        screen = detect_screen_state(frame, self._ui)
        if screen is ScreenState.RESULTS:
            return BrawlState(screen=screen, won=detect_result(frame, self._ui))
        if screen is not ScreenState.IN_MATCH:
            return BrawlState(screen=screen)

        hp_strip = crop_region(frame, self._ui["own_hp_bar"])
        ammo_strip = crop_region(frame, self._ui["ammo_bar"])
        super_button = crop_region(frame, self._ui["super_button"])
        super_hsv = cv2.cvtColor(super_button, cv2.COLOR_RGB2HSV)
        super_mask = cv2.inRange(super_hsv, SUPER_YELLOW_LOW, SUPER_YELLOW_HIGH)

        return BrawlState(
            screen=screen,
            own_hp=color_fill_fraction(hp_strip, HP_GREEN_LOW, HP_GREEN_HIGH),
            ammo=color_fill_fraction(ammo_strip, AMMO_ORANGE_LOW, AMMO_ORANGE_HIGH),
            super_ready=float(super_mask.mean()) > 60.0,
        )
