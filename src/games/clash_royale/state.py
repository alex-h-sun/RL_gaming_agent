"""Game state types shared by detector, reward, and env."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ScreenState(Enum):
    MAIN_MENU = "main_menu"
    BATTLE_LOADING = "battle_loading"
    IN_BATTLE = "in_battle"
    END_SCREEN = "end_screen"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TowerHP:
    """HP fill fractions in [0, 1]; 0 means destroyed."""

    left: float = 1.0
    right: float = 1.0
    king: float = 1.0


@dataclass(frozen=True)
class GameState:
    screen: ScreenState = ScreenState.UNKNOWN
    elixir: float = 0.0
    our_towers: TowerHP = field(default_factory=TowerHP)
    enemy_towers: TowerHP = field(default_factory=TowerHP)
    enemy_king_active: bool = False
    crowns_won: int = 0
    crowns_lost: int = 0
    won: bool | None = None  # set on END_SCREEN; None while undecided
