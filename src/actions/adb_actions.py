"""Inject taps/swipes via adb shell input."""

from __future__ import annotations

from typing import Any

from src.actions.mapping import decode_action


class AdbActions:
    def __init__(self, device: Any, config: dict[str, Any]):
        self._device = device
        self._config = config
        self._drag_ms = int(config["timing"]["drag_duration"] * 1000)
        size = device.window_size()
        self._width, self._height = size[0], size[1]

    def _to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return int(x * self._width), int(y * self._height)

    def tap(self, point: tuple[float, float]) -> None:
        x, y = self._to_screen(point)
        self._device.shell(f"input tap {x} {y}")

    def drag(self, source: tuple[float, float], target: tuple[float, float]) -> None:
        sx, sy = self._to_screen(source)
        tx, ty = self._to_screen(target)
        self._device.shell(f"input swipe {sx} {sy} {tx} {ty} {self._drag_ms}")

    def play(self, action: tuple[int, int] | list[int]) -> bool:
        """Execute an agent action. Returns False for no-op."""
        decoded = decode_action(action, self._config)
        if decoded is None:
            return False
        source, target = decoded
        self.drag(source, target)
        return True
