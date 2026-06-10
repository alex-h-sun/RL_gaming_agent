"""Inject taps/drags into the iPhone Mirroring window via pyautogui."""

from __future__ import annotations

from typing import Any

from src.actions.mapping import decode_action
from src.capture.mac_capture import WindowRect
from src.games.base import Drag, Gesture, Tap


class MacActions:
    def __init__(self, rect: WindowRect, config: dict[str, Any]):
        self._rect = rect
        self._config = config
        self._tap_duration = config["timing"]["tap_duration"]
        self._drag_duration = config["timing"]["drag_duration"]

    def _to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (
            self._rect.left + int(x * self._rect.width),
            self._rect.top + int(y * self._rect.height),
        )

    def tap(self, point: tuple[float, float]) -> None:
        import pyautogui

        x, y = self._to_screen(point)
        pyautogui.click(x, y, duration=self._tap_duration)

    def drag(self, source: tuple[float, float], target: tuple[float, float]) -> None:
        import pyautogui

        sx, sy = self._to_screen(source)
        tx, ty = self._to_screen(target)
        pyautogui.moveTo(sx, sy)
        pyautogui.dragTo(tx, ty, duration=self._drag_duration, button="left")

    def execute(self, gesture: Gesture) -> None:
        if isinstance(gesture, Tap):
            self.tap(gesture.point)
        elif isinstance(gesture, Drag):
            self.drag(gesture.source, gesture.target)
        else:
            raise TypeError(f"Unknown gesture: {gesture!r}")

    def play(self, action: tuple[int, int] | list[int]) -> bool:
        """Execute an agent action. Returns False for no-op."""
        decoded = decode_action(action, self._config)
        if decoded is None:
            return False
        source, target = decoded
        self.drag(source, target)
        return True
