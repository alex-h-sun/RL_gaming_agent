"""Game-agnostic interfaces: gestures and the GameAdapter protocol.

A game plugs into GameEnv by providing an adapter that translates agent
actions into primitive gestures, frames into game state, and state
transitions into rewards. Backends (Mac, ADB) only know how to execute
gestures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Tap:
    point: tuple[float, float]  # normalized (x, y)


@dataclass(frozen=True)
class Drag:
    source: tuple[float, float]
    target: tuple[float, float]


Gesture = Tap | Drag


@runtime_checkable
class GameAdapter(Protocol):
    """Everything GameEnv needs to run one specific game."""

    def build_action_space(self) -> Any:
        """The gymnasium action space for this game."""

    def action_to_gestures(self, action) -> tuple[Gesture, ...]:
        """Translate an agent action into primitive gestures (may be empty)."""

    def detect(self, frame) -> Any:
        """Extract game state from a full-resolution RGB frame."""

    def is_in_battle(self, state) -> bool:
        """True while an episode is running."""

    def is_terminal(self, state) -> bool:
        """True when the episode just ended (results/end screen)."""

    def reset_gesture(self, state) -> Gesture | None:
        """Navigation gesture toward a new battle from a non-battle screen."""

    def shaped_reward(self, prev_state, curr_state) -> float:
        """Dense per-step reward while in battle."""

    def terminal_reward(self, last_battle_state, terminal_state) -> float:
        """Win/loss reward when the episode ends.

        Receives both the last in-battle state and the terminal-screen state;
        games read the outcome from whichever side carries it.
        """
