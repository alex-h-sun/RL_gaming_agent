"""Brawl Stars game state types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ScreenState(Enum):
    MAIN_MENU = "main_menu"
    LOADING = "loading"
    IN_MATCH = "in_match"
    RESULTS = "results"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BrawlState:
    screen: ScreenState = ScreenState.UNKNOWN
    own_hp: float = 1.0          # HP bar fill fraction in [0, 1]
    ammo: float = 0.0            # ammo bar fill fraction in [0, 1]
    super_ready: bool = False
    won: bool | None = None      # set on RESULTS when detectable; None otherwise
