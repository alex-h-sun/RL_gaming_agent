"""Capture frames from an Android emulator/device via adbutils."""

from __future__ import annotations

import numpy as np

from src.capture.preprocess import to_observation


class AdbCapture:
    def __init__(self, serial: str | None = None, frame_size: int = 84):
        import adbutils

        adb = adbutils.AdbClient()
        self._device = adb.device(serial) if serial else adb.device()
        self._frame_size = frame_size

    def capture_raw(self) -> np.ndarray:
        """Full-resolution RGB frame for the detector."""
        image = self._device.screenshot()  # PIL.Image
        return np.asarray(image.convert("RGB"))

    def capture(self) -> np.ndarray:
        """Downscaled (size, size, 3) uint8 observation frame."""
        return to_observation(self.capture_raw(), self._frame_size)
